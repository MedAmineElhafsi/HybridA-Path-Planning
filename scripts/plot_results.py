#!/usr/bin/env python3
"""Live real-time thesis dashboard — updates while the car is moving.

Layout (single window, 3×3 grid):
  ┌──────────────┬──────────────┬──────────────┐
  │              │              │  e_lat (m)   │
  │  Plot 1      │  Plot 4      ├──────────────┤
  │  Path        │  Tracking    │  e_yaw (°)   │
  │  Planning    │  (live)      ├──────────────┤
  │              │              │  e_v  (m/s)  │
  ├──────────────┼──────────────┤              │
  │  Plot 2      │  Plot 3      │  Plot 5      │
  │  Curvature   │  Velocity    │  (errors)    │
  └──────────────┴──────────────┴──────────────┘

Usage:
    ros2 run hybrid_astar_cpp plot_results.py
"""

import math
import threading

import matplotlib
matplotlib.use("TkAgg")           # interactive GUI backend

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from std_msgs.msg import Float64MultiArray


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def menger_curvature(p0, p1, p2):
    ax, ay = p1[0]-p0[0], p1[1]-p0[1]
    bx, by = p2[0]-p1[0], p2[1]-p1[1]
    cx, cy = p2[0]-p0[0], p2[1]-p0[1]
    cross  = ax*by - ay*bx
    a, b, c = math.hypot(ax,ay), math.hypot(bx,by), math.hypot(cx,cy)
    denom = a*b*c
    return 2.0*abs(cross)/denom if denom > 1e-12 else 0.0

def arc_lengths(pts):
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + math.hypot(pts[i][0]-pts[i-1][0],
                                     pts[i][1]-pts[i-1][1]))
    return s

def curvatures_from_path(pts):
    n = len(pts)
    k = [0.0]*n
    for i in range(1, n-1):
        k[i] = menger_curvature(pts[i-1], pts[i], pts[i+1])
    if n >= 3:
        k[0]   = menger_curvature(pts[0],   pts[1],   pts[2])
        k[n-1] = menger_curvature(pts[n-3], pts[n-2], pts[n-1])
    return k

def path_to_xy(msg):
    return [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

def grid_to_image(msg):
    w, h = msg.info.width, msg.info.height
    data = np.array(msg.data, dtype=np.int16).reshape((h, w))
    img = np.full((h, w), 200, dtype=np.uint8)
    img[data == 0] = 255
    img[data > 50] = 0
    res = msg.info.resolution
    ox  = msg.info.origin.position.x
    oy  = msg.info.origin.position.y
    extent = [ox, ox+w*res, oy, oy+h*res]
    return img, extent


# ---------------------------------------------------------------------------
# ROS 2 data collector node
# ---------------------------------------------------------------------------
class DataCollector(Node):
    ERROR_WINDOW = 60.0   # seconds of error history to show

    def __init__(self):
        super().__init__("live_plotter")
        self.lock = threading.Lock()

        # Latest snapshots
        self.grid_msg        = None
        self.raw_path_msg    = None
        self.smooth_path_msg = None
        self.velocity_data   = []
        self.curvature_data  = []

        # Accumulated per run (reset on new path)
        self.odom_xy   = []          # [(x,y), ...]
        self.err_t     = []          # time (s)
        self.err_lat   = []          # lateral error (m)
        self.err_yaw   = []          # heading error (deg)
        self.err_v     = []          # speed error (m/s)
        self._t0       = None

        # Subscribers
        self.create_subscription(OccupancyGrid,     "astar_grid",
                                 self._cb_grid,     10)
        self.create_subscription(Path,              "astar_path_raw",
                                 self._cb_raw,      10)
        self.create_subscription(Path,              "astar_path",
                                 self._cb_smooth,   10)
        self.create_subscription(Float64MultiArray, "astar_velocity_profile",
                                 self._cb_vel,      10)
        self.create_subscription(Float64MultiArray, "astar_curvature",
                                 self._cb_curv,     10)
        self.create_subscription(Odometry,          "/odom",
                                 self._cb_odom,     20)
        self.create_subscription(Float64MultiArray, "mpc_tracking_error",
                                 self._cb_error,    20)

    def _cb_grid(self, msg):
        with self.lock: self.grid_msg = msg

    def _cb_raw(self, msg):
        with self.lock:
            self.raw_path_msg = msg
            self.odom_xy = []
            self.err_t = []; self.err_lat = []
            self.err_yaw = []; self.err_v = []
            self._t0 = None

    def _cb_smooth(self, msg):
        with self.lock: self.smooth_path_msg = msg

    def _cb_vel(self, msg):
        with self.lock: self.velocity_data = list(msg.data)

    def _cb_curv(self, msg):
        with self.lock: self.curvature_data = list(msg.data)

    def _cb_odom(self, msg):
        with self.lock:
            self.odom_xy.append((msg.pose.pose.position.x,
                                 msg.pose.pose.position.y))

    def _cb_error(self, msg):
        if len(msg.data) < 3: return
        now = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            if self._t0 is None: self._t0 = now
            t = now - self._t0
            self.err_t.append(t)
            self.err_lat.append(msg.data[0])
            self.err_yaw.append(math.degrees(msg.data[1]))
            self.err_v.append(msg.data[2])
            # Rolling window
            cutoff = t - self.ERROR_WINDOW
            while self.err_t and self.err_t[0] < cutoff:
                self.err_t.pop(0); self.err_lat.pop(0)
                self.err_yaw.pop(0); self.err_v.pop(0)

    def snapshot(self):
        """Thread-safe copy of all data."""
        with self.lock:
            return dict(
                grid        = self.grid_msg,
                raw         = self.raw_path_msg,
                smooth      = self.smooth_path_msg,
                vel         = list(self.velocity_data),
                curv        = list(self.curvature_data),
                odom        = list(self.odom_xy),
                err_t       = list(self.err_t),
                err_lat     = list(self.err_lat),
                err_yaw     = list(self.err_yaw),
                err_v       = list(self.err_v),
            )


# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------
BLUE   = "#1f77b4"
RED    = "#d62728"
GRAY   = "#999999"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"

class LiveDashboard:
    def __init__(self, node: DataCollector):
        self.node = node

        self.fig = plt.figure(figsize=(19, 10))
        self.fig.patch.set_facecolor("#1e1e2e")
        plt.suptitle("Hybrid A* + MPC — Live Dashboard",
                     color="white", fontsize=14, fontweight="bold")

        gs = gridspec.GridSpec(2, 2,
                               left=0.07, right=0.97,
                               top=0.93,  bottom=0.07,
                               hspace=0.40, wspace=0.32)

        # Plot 1 — path planning (both rows, col 0)
        self.ax1 = self.fig.add_subplot(gs[0:2, 0])
        # Plot 2 — curvature (row 0, col 1)
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        # Plot 3 — velocity (row 1, col 1)
        self.ax3 = self.fig.add_subplot(gs[1, 1])

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor("#2b2b3b")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555566")
            ax.tick_params(colors="white", labelsize=7)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        self.ani = FuncAnimation(self.fig, self._update,
                                 interval=500, blit=False)

    def _draw_grid_background(self, ax, grid_msg):
        if grid_msg is None: return
        img, extent = grid_to_image(grid_msg)
        ax.imshow(img, cmap="gray", origin="lower",
                  extent=extent, vmin=0, vmax=255, alpha=0.55, zorder=0)

    # ---- Plot 1: Path Planning ------------------------------------------
    def _update_plot1(self, d):
        ax = self.ax1
        ax.cla()
        ax.set_facecolor("#2b2b3b")
        ax.set_title("1 — Path Planning", fontsize=9, color="white")
        ax.set_xlabel("X (m)", fontsize=7); ax.set_ylabel("Y (m)", fontsize=7)

        self._draw_grid_background(ax, d["grid"])

        if d["raw"] and d["raw"].poses:
            pts = path_to_xy(d["raw"])
            rx, ry = zip(*pts)
            ax.plot(rx, ry, color=GRAY, linestyle="--",
                    linewidth=1.0, alpha=0.6, label="Raw A*", zorder=2)

        if d["smooth"] and d["smooth"].poses:
            pts = path_to_xy(d["smooth"])
            sx, sy = zip(*pts)
            ax.plot(sx, sy, color=BLUE, linewidth=1.8,
                    label="B-Spline", zorder=3)
            ax.plot(sx[0],  sy[0],  "o", color=GREEN,  markersize=6, zorder=5)
            ax.plot(sx[-1], sy[-1], "*", color=RED,    markersize=9, zorder=5)

        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#3b3b4b", labelcolor="white", framealpha=0.7)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    # ---- Plot 2: Curvature ----------------------------------------------
    def _update_plot2(self, d):
        ax = self.ax2
        ax.cla()
        ax.set_facecolor("#2b2b3b")
        ax.set_title("2 — Curvature κ(s)", fontsize=9, color="white")
        ax.set_xlabel("Arc length s (m)", fontsize=7)
        ax.set_ylabel("κ (rad/m)", fontsize=7)

        if d["raw"] and d["raw"].poses:
            pts = path_to_xy(d["raw"])
            s_raw = arc_lengths(pts)
            k_raw = curvatures_from_path(pts)
            ax.plot(s_raw, k_raw, color=GRAY, linestyle="--",
                    linewidth=1.0, alpha=0.7, label="Raw")

        if d["smooth"] and d["smooth"].poses:
            pts = path_to_xy(d["smooth"])
            s   = arc_lengths(pts)
            k   = curvatures_from_path(pts)
            if k:
                ax.plot(s, k, color=BLUE, linewidth=1.6, label="Smooth")
                ax.fill_between(s, k, alpha=0.12, color=BLUE)

        ax.legend(fontsize=6, facecolor="#3b3b4b",
                  labelcolor="white", framealpha=0.7)
        ax.grid(True, alpha=0.15, color="white")
        ax.set_ylim(bottom=0)
        ax.tick_params(colors="white", labelsize=6)

    # ---- Plot 3: Velocity Profile ----------------------------------------
    def _update_plot3(self, d):
        ax = self.ax3
        ax.cla()
        ax.set_facecolor("#2b2b3b")
        ax.set_title("3 — Velocity Profile v(s)", fontsize=9, color="white")
        ax.set_xlabel("Arc length s (m)", fontsize=7)
        ax.set_ylabel("v (m/s)", fontsize=7)

        if d["smooth"] and d["smooth"].poses and d["vel"]:
            pts = path_to_xy(d["smooth"])
            n   = min(len(pts), len(d["vel"]))
            s   = arc_lengths(pts[:n])
            v   = d["vel"][:n]
            ax.plot(s, v, color=RED, linewidth=1.8, label="v(s)")
            ax.fill_between(s, v, alpha=0.15, color=RED)
            # Mark cusps
            for i in range(1, n):
                if v[i] < 0.05 and 0 < i < n-1:
                    ax.axvline(s[i], color=ORANGE, linestyle=":",
                               linewidth=0.9, alpha=0.6)
            if v:
                v_max = max(v)
                ax.set_ylim(0, v_max * 1.15)
                ax.axhline(v_max, color=ORANGE, linestyle="--",
                           linewidth=0.8, alpha=0.5,
                           label=f"v_max={v_max:.1f}")
        ax.legend(fontsize=6, facecolor="#3b3b4b",
                  labelcolor="white", framealpha=0.7)
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    # ---- Plot 4: Tracking -----------------------------------------------
    def _update_plot4(self, d):
        ax = self.ax4
        ax.cla()
        ax.set_facecolor("#2b2b3b")
        ax.set_title("4 — Closed-Loop Tracking", fontsize=9, color="white")
        ax.set_xlabel("X (m)", fontsize=7); ax.set_ylabel("Y (m)", fontsize=7)

        self._draw_grid_background(ax, d["grid"])

        if d["smooth"] and d["smooth"].poses:
            pts = path_to_xy(d["smooth"])
            sx, sy = zip(*pts)
            ax.plot(sx, sy, color=BLUE, linewidth=1.8,
                    label="Reference", zorder=3)
            ax.plot(sx[0],  sy[0],  "o", color=GREEN, markersize=6, zorder=5)
            ax.plot(sx[-1], sy[-1], "*", color=RED,   markersize=9, zorder=5)

        if d["odom"]:
            ox, oy = zip(*d["odom"])
            ax.plot(ox, oy, color=RED, linewidth=1.5,
                    linestyle="-", alpha=0.85,
                    label="Actual (MPC)", zorder=4)
            # Current vehicle position
            ax.plot(ox[-1], oy[-1], "D", color=ORANGE,
                    markersize=7, zorder=6, label="Vehicle now")

        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#3b3b4b", labelcolor="white", framealpha=0.7)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    # ---- Plot 5: Tracking Errors ----------------------------------------
    def _update_plot5(self, d):
        specs = [
            (self.ax5a, d["err_lat"], "e_lat (m)",   BLUE,   0.30),
            (self.ax5b, d["err_yaw"], "e_yaw (°)",   ORANGE, 5.0),
            (self.ax5c, d["err_v"],   "e_v (m/s)",   GREEN,  0.50),
        ]
        t = d["err_t"]

        for i, (ax, data, ylabel, color, thresh) in enumerate(specs):
            ax.cla()
            ax.set_facecolor("#2b2b3b")
            ax.set_ylabel(ylabel, fontsize=7, color="white")
            ax.tick_params(colors="white", labelsize=6)
            ax.grid(True, alpha=0.15, color="white")

            if i == 0:
                ax.set_title("5 — Tracking Errors", fontsize=9, color="white")
            if i == 2:
                ax.set_xlabel("Time t (s)", fontsize=7, color="white")

            if t and data:
                ax.plot(t, data, color=color, linewidth=1.3)
                ax.fill_between(t, data, alpha=0.15, color=color)
                ax.axhline( thresh, color=RED, linestyle="--",
                            linewidth=0.8, alpha=0.5)
                ax.axhline(-thresh, color=RED, linestyle="--",
                            linewidth=0.8, alpha=0.5)
                ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
                rms = math.sqrt(sum(e**2 for e in data) / len(data))
                ax.text(0.97, 0.82, f"RMS={rms:.3f}",
                        transform=ax.transAxes, ha="right",
                        fontsize=7, color=color,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#1e1e2e", alpha=0.7))
            else:
                ax.text(0.5, 0.5, "Waiting for MPC...",
                        transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color="#666677")

            for spine in ax.spines.values():
                spine.set_edgecolor("#555566")

    # ---- Main update function -------------------------------------------
    def _update(self, frame):
        d = self.node.snapshot()
        self._update_plot1(d)
        self._update_plot2(d)
        self._update_plot3(d)
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = DataCollector()

    # ROS 2 spin in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    dashboard = LiveDashboard(node)
    plt.show()   # blocks here — closes when window is closed

    rclpy.shutdown()

if __name__ == "__main__":
    main()
