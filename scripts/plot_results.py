#!/usr/bin/env python3
"""Live real-time thesis dashboard — updates while the car is moving.

Plots the professor-style reference trajectory assembled by the planner:
path geometry, kappa(s), v_ref(t), steering delta(t), and acceleration a(t).

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

REFERENCE_FIELDS = ("x", "y", "yaw", "v_ref", "t", "kappa", "delta", "a")
STOP_SPEED_EPS = 0.05


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def arc_lengths(pts):
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + math.hypot(pts[i][0]-pts[i-1][0],
                                     pts[i][1]-pts[i-1][1]))
    return s

def curvature_topic_axis(pts, curv):
    if not curv:
        return [], []

    s_path = arc_lengths(pts)
    if len(curv) == len(s_path):
        return s_path, curv

    total = s_path[-1] if s_path else 0.0
    if len(curv) == 1:
        return [0.0], curv
    return list(np.linspace(0.0, total, len(curv))), curv

def trim_pair(x, y):
    n = min(len(x), len(y))
    return list(x[:n]), list(y[:n])

def stop_break_indices(v, threshold=STOP_SPEED_EPS):
    """Interior zero-speed groups indicate a cusp or segment boundary."""
    n = len(v)
    if n < 5:
        return []

    breaks = []
    start = None
    for i in range(1, n-1):
        stopped = abs(v[i]) <= threshold
        if stopped and start is None:
            start = i
        elif not stopped and start is not None:
            end = i - 1
            if start > 1 and end < n - 2:
                breaks.append((start + end) // 2)
            start = None

    if start is not None:
        end = n - 2
        if start > 1 and end < n - 2:
            breaks.append((start + end) // 2)
    return breaks

def plot_segmented(ax, x, y, color, label, linewidth=1.5,
                   fill_alpha=0.0, break_indices=None):
    x, y = trim_pair(x, y)
    if not x:
        return

    breaks = sorted({i for i in (break_indices or []) if 0 <= i < len(x)-1})
    first = True
    start = 0
    for break_i in breaks + [len(x)-1]:
        end = break_i + 1
        if end > start:
            seg_x = x[start:end]
            seg_y = y[start:end]
            ax.plot(seg_x, seg_y, color=color, linewidth=linewidth,
                    label=label if first else None)
            if fill_alpha > 0.0:
                ax.fill_between(seg_x, seg_y, alpha=fill_alpha, color=color)
            first = False
        start = end

def mark_segment_breaks(ax, x, break_indices):
    x = list(x)
    first = True
    for i in break_indices:
        if 0 <= i < len(x):
            ax.axvline(x[i], color=ORANGE, linestyle=":",
                       linewidth=0.9, alpha=0.65,
                       label="stop / cusp" if first else None)
            first = False

def set_padded_ylim(ax, values, min_span=0.05):
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return
    lo = min(min(finite), 0.0)
    hi = max(max(finite), 0.0)
    span = max(hi - lo, min_span)
    pad = 0.15 * span
    ax.set_ylim(lo - pad, hi + pad)

def show_legend(ax, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=6, loc=loc, facecolor="#3b3b4b",
                  labelcolor="white", framealpha=0.7)

def reference_from_msg(msg):
    cols = len(REFERENCE_FIELDS)
    if msg.layout.dim and len(msg.layout.dim) >= 2 and msg.layout.dim[1].size:
        cols = int(msg.layout.dim[1].size)
    if cols < len(REFERENCE_FIELDS) or len(msg.data) < cols:
        return {name: [] for name in REFERENCE_FIELDS}

    rows = len(msg.data) // cols
    ref = {name: [] for name in REFERENCE_FIELDS}
    for r in range(rows):
        base = r * cols
        for c, name in enumerate(REFERENCE_FIELDS):
            ref[name].append(float(msg.data[base + c]))
    return ref

def reference_xy(ref):
    if ref and ref.get("x") and ref.get("y"):
        return list(zip(ref["x"], ref["y"]))
    return []

def path_axis_for_series(d, series_len):
    ref_pts = reference_xy(d["ref"])
    if ref_pts and len(ref_pts) == series_len:
        return arc_lengths(ref_pts)

    if d["smooth"] and d["smooth"].poses:
        pts = path_to_xy(d["smooth"])
        n = min(len(pts), series_len)
        return arc_lengths(pts[:n])

    if series_len <= 1:
        return [0.0] * series_len
    return list(np.arange(series_len, dtype=float))

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
        self.steering_data   = []
        self.acceleration_data = []
        self.reference_data  = {name: [] for name in REFERENCE_FIELDS}

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
        self.create_subscription(Float64MultiArray, "astar_steering",
                                 self._cb_steer,    10)
        self.create_subscription(Float64MultiArray, "astar_acceleration",
                                 self._cb_accel,    10)
        self.create_subscription(Float64MultiArray, "astar_reference_trajectory",
                                 self._cb_reference, 10)
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

    def _cb_steer(self, msg):
        with self.lock: self.steering_data = list(msg.data)

    def _cb_accel(self, msg):
        with self.lock: self.acceleration_data = list(msg.data)

    def _cb_reference(self, msg):
        with self.lock: self.reference_data = reference_from_msg(msg)

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
                steer       = list(self.steering_data),
                accel       = list(self.acceleration_data),
                ref         = {k: list(v) for k, v in self.reference_data.items()},
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

        self.fig = plt.figure(figsize=(19, 11))
        self.fig.patch.set_facecolor("#1e1e2e")
        plt.suptitle("Reference Trajectory Generation",
                     color="white", fontsize=14, fontweight="bold")

        gs = gridspec.GridSpec(3, 2,
                               left=0.07, right=0.97,
                               top=0.93,  bottom=0.07,
                               hspace=0.45, wspace=0.30)

        # Plot 1 — path planning/reference geometry
        self.ax1 = self.fig.add_subplot(gs[0:2, 0])
        # Plot 2 — curvature (row 0, col 1)
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        # Plot 3 — velocity (row 1, col 1)
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        # Plot 4 — steering (row 2, col 0)
        self.ax4 = self.fig.add_subplot(gs[2, 0])
        # Plot 5 — acceleration (row 2, col 1)
        self.ax5 = self.fig.add_subplot(gs[2, 1])

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
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

        ref_pts = reference_xy(d["ref"])
        display_pts = ref_pts
        display_label = "Reference trajectory"
        if not display_pts and d["smooth"] and d["smooth"].poses:
            display_pts = path_to_xy(d["smooth"])
            display_label = "B-spline smoothed path"

        if display_pts:
            sx, sy = zip(*display_pts)
            ax.plot(sx, sy, color=BLUE, linewidth=1.8,
                    label=display_label, zorder=3)
            ax.plot(sx[0],  sy[0],  "o", color=GREEN, markersize=6,
                    label="Start", zorder=5)
            ax.plot(sx[-1], sy[-1], "*", color=RED, markersize=9,
                    label="Goal", zorder=5)

        show_legend(ax, loc="upper left")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    # ---- Plot 2: Curvature ----------------------------------------------
    def _update_plot2(self, d):
        ax = self.ax2
        ax.cla()
        ax.set_facecolor("#2b2b3b")
        ax.set_title("2 — Curvature κ(s)", fontsize=9, color="white")
        ax.set_xlabel("Arc length s (m)", fontsize=7)
        ax.set_ylabel("κ (1/m)", fontsize=7)

        ref_pts = reference_xy(d["ref"])
        if ref_pts and d["ref"].get("kappa"):
            # Professor-facing plot: use backend κ from the assembled
            # reference trajectory.  The script only computes s for the x-axis.
            n = min(len(ref_pts), len(d["ref"]["kappa"]))
            s = arc_lengths(ref_pts[:n])
            k = d["ref"]["kappa"][:n]
            plot_segmented(ax, s, k, BLUE, "Reference κ(s)",
                           linewidth=1.8, fill_alpha=0.12)
            set_padded_ylim(ax, k, min_span=0.05)
        elif d["smooth"] and d["smooth"].poses:
            pts = path_to_xy(d["smooth"])
            s, k = curvature_topic_axis(pts, d["curv"])
            if k:
                plot_segmented(ax, s, k, BLUE, "/astar_curvature",
                               linewidth=1.8, fill_alpha=0.12)
                set_padded_ylim(ax, k, min_span=0.05)
        elif d["curv"]:
            s = list(np.arange(len(d["curv"]), dtype=float))
            plot_segmented(ax, s, d["curv"], BLUE, "/astar_curvature",
                           linewidth=1.8, fill_alpha=0.12)
            set_padded_ylim(ax, d["curv"], min_span=0.05)

        ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
        show_legend(ax)
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    # ---- Plot 3: Velocity Profile ----------------------------------------
    def _update_plot3(self, d):
        ax = self.ax3
        ax.cla()
        ax.set_facecolor("#2b2b3b")
        ax.set_title("3 — Velocity Profile", fontsize=9, color="white")
        ax.set_ylabel("v_ref (m/s)", fontsize=7)

        x = []
        v = []
        xlabel = "Arc length s (m)"
        label = "v_ref(s)"
        if d["ref"].get("v_ref"):
            v = d["ref"]["v_ref"]
            if d["ref"].get("t") and len(d["ref"]["t"]) == len(v):
                x = d["ref"]["t"]
                xlabel = "Time t (s)"
                label = "v_ref(t)"
            else:
                x = path_axis_for_series(d, len(v))
        elif d["smooth"] and d["smooth"].poses and d["vel"]:
            v = d["vel"]
            x = path_axis_for_series(d, len(v))

        ax.set_xlabel(xlabel, fontsize=7)
        if x and v:
            n = min(len(x), len(v))
            x = x[:n]
            v = v[:n]
            breaks = stop_break_indices(v) if xlabel.startswith("Time") else []
            plot_segmented(ax, x, v, RED, label, linewidth=1.8,
                           fill_alpha=0.15, break_indices=breaks)
            mark_segment_breaks(ax, x, breaks)
            if v:
                v_max = max(v)
                ax.set_ylim(0, max(v_max * 1.15, 0.5))
                ax.axhline(v_max, color=ORANGE, linestyle="--",
                           linewidth=0.8, alpha=0.5,
                           label=f"v_max={v_max:.1f}")
        show_legend(ax)
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    # ---- Plot 4: Steering ------------------------------------------------
    def _update_plot4(self, d):
        ax = self.ax4
        ax.cla()
        ax.set_facecolor("#2b2b3b")
        ax.set_title("4 — Steering δ", fontsize=9, color="white")
        ax.set_ylabel("δ_ref (rad)", fontsize=7)

        x = []
        delta = []
        xlabel = "Arc length s (m)"
        label = "δ_ref(s)"
        break_indices = []
        if d["ref"].get("delta"):
            delta = d["ref"]["delta"]
            if d["ref"].get("t") and len(d["ref"]["t"]) == len(delta):
                x = d["ref"]["t"]
                xlabel = "Time t (s)"
                label = "δ_ref(t)"
                if len(d["ref"].get("v_ref", [])) == len(delta):
                    break_indices = stop_break_indices(d["ref"]["v_ref"])
            else:
                x = path_axis_for_series(d, len(delta))
        elif d["steer"]:
            delta = d["steer"]
            x = path_axis_for_series(d, len(delta))
            label = "/astar_steering"

        ax.set_xlabel(xlabel, fontsize=7)
        if x and delta:
            n = min(len(x), len(delta))
            x = x[:n]
            delta = delta[:n]
            plot_segmented(ax, x, delta, ORANGE, label,
                           linewidth=1.6, fill_alpha=0.12,
                           break_indices=break_indices)
            mark_segment_breaks(ax, x, break_indices)
            set_padded_ylim(ax, delta, min_span=0.08)

        ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
        show_legend(ax)
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    # ---- Plot 5: Acceleration -------------------------------------------
    def _update_plot5(self, d):
        ax = self.ax5
        ax.cla()
        ax.set_facecolor("#2b2b3b")
        ax.set_title("5 — Acceleration a", fontsize=9, color="white")
        ax.set_ylabel("a_ref (m/s²)", fontsize=7)

        x = []
        accel = []
        xlabel = "Arc length s (m)"
        label = "a_ref(s)"
        break_indices = []
        if d["ref"].get("a"):
            accel = d["ref"]["a"]
            if d["ref"].get("t") and len(d["ref"]["t"]) == len(accel):
                x = d["ref"]["t"]
                xlabel = "Time t (s)"
                label = "a_ref(t)"
                if len(d["ref"].get("v_ref", [])) == len(accel):
                    break_indices = stop_break_indices(d["ref"]["v_ref"])
            else:
                x = path_axis_for_series(d, len(accel))
        elif d["accel"]:
            accel = d["accel"]
            x = path_axis_for_series(d, len(accel))
            label = "/astar_acceleration"

        ax.set_xlabel(xlabel, fontsize=7)
        if x and accel:
            n = min(len(x), len(accel))
            x = x[:n]
            accel = accel[:n]
            plot_segmented(ax, x, accel, GREEN, label,
                           linewidth=1.6, fill_alpha=0.12,
                           break_indices=break_indices)
            mark_segment_breaks(ax, x, break_indices)
            set_padded_ylim(ax, accel, min_span=0.2)

        ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
        show_legend(ax)
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    # ---- Main update function -------------------------------------------
    def _update(self, frame):
        d = self.node.snapshot()
        self._update_plot1(d)
        self._update_plot2(d)
        self._update_plot3(d)
        self._update_plot4(d)
        self._update_plot5(d)
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
