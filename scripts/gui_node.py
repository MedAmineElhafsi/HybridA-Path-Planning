#!/usr/bin/env python3

import queue
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.parameter_client import AsyncParameterClient
from rclpy.parameter import Parameter
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import tkinter as tk
import threading
import math

# --- SPEED PRESETS ---
# (label, vel_max, vel_max_reverse, animation_speed)
SPEED_PRESETS = [
    ("SLOW\n1.5 m/s",   1.5, 0.8, 1.5),
    ("NORMAL\n4.0 m/s", 4.0, 1.5, 4.0),
    ("FAST\n6.0 m/s",   6.0, 2.0, 6.0),
]

# Thread-safe queue: Tkinter thread pushes presets, ROS timer thread consumes them
speed_change_queue: queue.Queue = queue.Queue()

# --- CONSTANTS ---
# Each GUI cell represents 2.0 m of real-world space so a 4.5 m car
# occupies ~2 cells instead of ~4.5 cells, giving proper room to maneuver.
# World size: 32 x 22 cells x 2.0 m/cell = 64 m x 44 m
CELL_SIZE = 26                  # pixels per display cell
GRID_W = 32                     # display cells wide
GRID_H = 22                     # display cells tall
DISPLAY_CELL_WORLD_SIZE = 2.0   # real-world metres per display cell
MAP_RESOLUTION = 0.25           # OccupancyGrid cell size (m) — unchanged
SUBCELLS_PER_CELL = max(1, int(round(DISPLAY_CELL_WORLD_SIZE / MAP_RESOLUTION)))
W = GRID_W * CELL_SIZE
H = GRID_H * CELL_SIZE

EMPTY = 0
WALL = 1

# --- SHARED STATE ---
lock = threading.Lock()
grid = [[EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]
start_pose = None
goal_pose = None
mouse_down = False
placing_wall = True
place_mode = None
drag_anchor_xy = None

def world_to_canvas(x, y):
    return (
        (x / DISPLAY_CELL_WORLD_SIZE) * CELL_SIZE,
        ((GRID_H * DISPLAY_CELL_WORLD_SIZE - y) / DISPLAY_CELL_WORLD_SIZE) * CELL_SIZE,
    )

def canvas_to_world(px, py):
    return (
        (px / CELL_SIZE) * DISPLAY_CELL_WORLD_SIZE,
        (GRID_H * DISPLAY_CELL_WORLD_SIZE) - ((py / CELL_SIZE) * DISPLAY_CELL_WORLD_SIZE),
    )

# --- ROS 2 NODE ---
class GuiNode(Node):
    def __init__(self):
        super().__init__('tkinter_map_editor')
        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.grid_pub = self.create_publisher(OccupancyGrid, 'astar_grid', map_qos)
        self.start_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.timer = self.create_timer(0.5, self.publish_grid)  # 2 Hz

        # Parameter clients — used to push speed changes at runtime
        self.planner_param_client = AsyncParameterClient(self, 'planner_node')
        self.viz_param_client     = AsyncParameterClient(self, 'path_vehicle_visualizer')

    def publish_grid(self):
        # Apply any pending speed change (queued from the Tkinter thread)
        try:
            _, vel_max, vel_max_rev, anim_speed = speed_change_queue.get_nowait()
            self.planner_param_client.set_parameters([
                Parameter('vel_max',         Parameter.Type.DOUBLE, vel_max),
                Parameter('vel_max_reverse', Parameter.Type.DOUBLE, vel_max_rev),
            ])
            self.viz_param_client.set_parameters([
                Parameter('animation_speed', Parameter.Type.DOUBLE, anim_speed),
            ])
            self.get_logger().info(
                f'Speed set → forward {vel_max} m/s, reverse {vel_max_rev} m/s')
        except queue.Empty:
            pass

        with lock:
            local_grid = [row[:] for row in grid]
            
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = MAP_RESOLUTION
        msg.info.width = GRID_W * SUBCELLS_PER_CELL
        msg.info.height = GRID_H * SUBCELLS_PER_CELL
        
        data = []
        for r in reversed(range(GRID_H)):
            for _ in range(SUBCELLS_PER_CELL):
                for c in range(GRID_W):
                    value = 100 if local_grid[r][c] == WALL else 0
                    data.extend([value] * SUBCELLS_PER_CELL)
        msg.data = data
        self.grid_pub.publish(msg)

    def publish_start(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.start_pub.publish(msg)

    def publish_goal(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_pub.publish(msg)

# --- TKINTER GUI ---
root = tk.Tk()
root.title("Hybrid A* C++ Frontend")
canvas = tk.Canvas(root, width=W, height=H, bg="#1E1E1E")
canvas.pack()

# --- SPEED PANEL ---
speed_frame = tk.Frame(root, bg="#1A1A1A", pady=6)
speed_frame.pack(fill=tk.X)

tk.Label(speed_frame, text="SPEED:", bg="#1A1A1A", fg="#AAAAAA",
         font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(10, 4))

speed_label = tk.Label(speed_frame, text="NORMAL  4.0 m/s",
                       bg="#1A1A1A", fg="#FFD24D",
                       font=("Helvetica", 9, "bold"), width=18)
speed_label.pack(side=tk.LEFT, padx=(0, 12))

PRESET_COLORS = ["#2255AA", "#1A7A3A", "#AA3322"]   # blue / green / red

def make_speed_button(parent, preset, color):
    label, vel_max, vel_max_rev, anim_speed = preset
    def on_click():
        speed_change_queue.put(preset)
        speed_label.config(
            text=f"{label.split(chr(10))[0]}  fwd {vel_max} m/s  rev {vel_max_rev} m/s  anim {anim_speed} m/s"
        )
    btn = tk.Button(
        parent, text=label, command=on_click,
        bg=color, fg="white", activebackground=color,
        font=("Helvetica", 8, "bold"),
        relief=tk.FLAT, padx=10, pady=2, width=10,
    )
    btn.pack(side=tk.LEFT, padx=4)
    return btn

for preset, color in zip(SPEED_PRESETS, PRESET_COLORS):
    make_speed_button(speed_frame, preset, color)

ros_node = None

def draw():
    canvas.delete("all")
    with lock:
        for r in range(GRID_H):
            for c in range(GRID_W):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                color = "#FF9900" if grid[r][c] == WALL else "#2D2D2D"
                canvas.create_rectangle(x1, y1, x1+CELL_SIZE, y1+CELL_SIZE, fill=color, outline="#333")
        
        for pose, color, label in [(start_pose, "#00D1FF", "START"), (goal_pose, "#00FF66", "GOAL")]:
            if pose:
                px, py = world_to_canvas(pose[0], pose[1])
                canvas.create_oval(px-6, py-6, px+6, py+6, fill=color, outline="")
                canvas.create_text(px+10, py-10, text=label, fill=color, anchor="nw")
                canvas.create_line(px, py, px + 26*math.cos(-pose[2]), py + 26*math.sin(-pose[2]), fill=color, width=2, arrow=tk.LAST)
                
        if place_mode:
            canvas.create_text(8, 8, text=f"MODE: place {place_mode.upper()} (click+drag yaw)", fill="#FFD24D", anchor="nw")

def on_mouse_down(event):
    global mouse_down, placing_wall, drag_anchor_xy, place_mode, start_pose, goal_pose
    xw, yw = canvas_to_world(event.x, event.y)
    
    if place_mode in ("start", "goal"):
        if place_mode == "goal" and goal_pose:
            start_pose = goal_pose
        drag_anchor_xy = (xw, yw)
        if place_mode == "start": start_pose = (xw, yw, 0.0)
        else: goal_pose = (xw, yw, 0.0)
        draw()
        return

    mouse_down = True
    r, c = event.y // CELL_SIZE, event.x // CELL_SIZE
    if 0 <= r < GRID_H and 0 <= c < GRID_W:
        with lock:
            placing_wall = (grid[r][c] == EMPTY)
            grid[r][c] = WALL if placing_wall else EMPTY
        draw()

def on_mouse_move(event):
    global start_pose, goal_pose
    if drag_anchor_xy and place_mode:
        xw, yw = canvas_to_world(event.x, event.y)
        yaw = math.atan2(yw - drag_anchor_xy[1], xw - drag_anchor_xy[0])
        if place_mode == "start": start_pose = (drag_anchor_xy[0], drag_anchor_xy[1], yaw)
        else: goal_pose = (drag_anchor_xy[0], drag_anchor_xy[1], yaw)
        draw()
        return

    if mouse_down:
        r, c = event.y // CELL_SIZE, event.x // CELL_SIZE
        if 0 <= r < GRID_H and 0 <= c < GRID_W:
            with lock: grid[r][c] = WALL if placing_wall else EMPTY
            draw()

def on_mouse_up(event):
    global mouse_down, drag_anchor_xy, place_mode
    if place_mode == "start" and start_pose and ros_node:
        ros_node.publish_start(*start_pose)
    elif place_mode == "goal" and goal_pose and ros_node:
        ros_node.publish_goal(*goal_pose)
    
    mouse_down = False
    drag_anchor_xy = None
    place_mode = None
    draw()

def on_key(event):
    global place_mode
    key = event.keysym.lower()
    if key == "s": place_mode = "start"
    elif key == "g": place_mode = "goal"
    elif key == "c": 
        with lock:
            for r in range(GRID_H):
                for c in range(GRID_W): grid[r][c] = EMPTY
    draw()

canvas.bind("<Button-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)
root.bind("<Key>", on_key)
draw()

def main():
    global ros_node
    rclpy.init()
    ros_node = GuiNode()
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()
    root.mainloop()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
