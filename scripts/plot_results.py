#!/usr/bin/env python3
"""Live real-time thesis dashboard — updates while the car is moving.

Plots the professor-style reference trajectory assembled by the planner:
path geometry, kappa(s), v_ref(t), steering delta(t), and acceleration a(t).

Usage:
    ros2 run hybrid_astar_cpp plot_results.py
"""

import math
import threading
from bisect import bisect_left

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
ACTUAL_FIELDS = ("t", "x", "y", "yaw", "vx", "vy", "r")
STOP_SPEED_EPS = 0.05
DEFAULT_WHEELBASE = 2.7
MIN_TRACKING_DURATION_S = 5.0
MIN_ODOM_SAMPLES = 50
MIN_ACTUAL_TRAVEL_M = 0.25
NEAR_REFERENCE_END_M = 1.0
NEAR_ZERO_VX_EPS = 0.05
ZERO_CMD_A_EPS = 0.02
ZERO_CMD_DELTA_EPS = 0.01
MIN_ACTUAL_POINTS_FOR_PATH = 5


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

def raw_path_xy(d):
    if d["raw"] and d["raw"].poses:
        return path_to_xy(d["raw"])
    return []

def reference_display_xy(d):
    ref_pts = reference_xy(d["ref"])
    if ref_pts:
        return ref_pts
    if d["smooth"] and d["smooth"].poses:
        return path_to_xy(d["smooth"])
    return []

def actual_odom_states(d):
    if d.get("actual_source") == "/odom":
        return d["actual"]
    return []

def marker_stride(n, target=60):
    return max(1, int(math.ceil(max(1, n) / float(target))))

def nearest_point_index(pt, pts):
    if not pts:
        return 0
    px, py = pt
    return min(range(len(pts)),
               key=lambda i: (pts[i][0] - px) ** 2 + (pts[i][1] - py) ** 2)

def path_difference_focus(raw_pts, ref_pts):
    if not raw_pts or not ref_pts:
        return None

    best_ref = ref_pts[len(ref_pts) // 2]
    best_raw = raw_pts[nearest_point_index(best_ref, raw_pts)]
    best_d = -1.0
    for pt in ref_pts:
        idx = nearest_point_index(pt, raw_pts)
        d = math.hypot(pt[0] - raw_pts[idx][0], pt[1] - raw_pts[idx][1])
        if d > best_d:
            best_d = d
            best_ref = pt
            best_raw = raw_pts[idx]

    cx = 0.5 * (best_raw[0] + best_ref[0])
    cy = 0.5 * (best_raw[1] + best_ref[1])
    all_x = [p[0] for p in raw_pts] + [p[0] for p in ref_pts]
    all_y = [p[1] for p in raw_pts] + [p[1] for p in ref_pts]
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0)
    half_width = max(1.0, min(5.0, 0.18 * span), 4.0 * best_d)
    return cx, cy, half_width, best_d, best_raw, best_ref

def raw_reference_deviation(raw_pts, ref_pts):
    """Nearest-point distance from each reference point to the raw path."""
    if not raw_pts or not ref_pts:
        return [], [], None, None

    s = arc_lengths(ref_pts)
    err = []
    for ref_pt in ref_pts:
        idx = nearest_point_index(ref_pt, raw_pts)
        raw_pt = raw_pts[idx]
        err.append(math.hypot(ref_pt[0] - raw_pt[0],
                              ref_pt[1] - raw_pt[1]))

    return s, err, max(err) if err else None, float(np.mean(err)) if err else None

def draw_path_comparison(ax, raw_pts, ref_pts, actual_states,
                         include_labels=True, show_current=True,
                         dense_markers=False):
    if ref_pts:
        sx, sy = zip(*ref_pts)
        ax.plot(
            sx, sy,
            color=BLUE,
            linestyle="-",
            linewidth=3.2,
            alpha=0.82,
            solid_capstyle="round",
            label="Curvature-aware reference trajectory" if include_labels else None,
            zorder=3,
        )

    if raw_pts:
        rx, ry = zip(*raw_pts)
        stride = marker_stride(len(raw_pts), target=120 if dense_markers else 45)
        ax.plot(
            rx, ry,
            color="#303030",
            linestyle=(0, (5, 3)),
            linewidth=1.5,
            marker="s",
            markersize=4.0 if dense_markers else 3.2,
            markerfacecolor="#f2f2f2",
            markeredgecolor="#303030",
            markeredgewidth=0.8,
            markevery=stride,
            alpha=0.98,
            label="Raw Hybrid A* path" if include_labels else None,
            zorder=6,
        )

    if actual_states:
        stride = marker_stride(len(actual_states), target=140 if dense_markers else 55)
        ax.plot(
            [s[1] for s in actual_states],
            [s[2] for s in actual_states],
            color=ORANGE,
            linestyle="-",
            linewidth=2.7,
            marker="o",
            markersize=4.0 if dense_markers else 3.4,
            markerfacecolor="#1e1e2e",
            markeredgewidth=0.8,
            markevery=stride,
            alpha=0.95,
            label="Actual dynamic vehicle trajectory" if include_labels else None,
            zorder=7,
        )
        if show_current:
            ax.plot(
                actual_states[0][1],
                actual_states[0][2],
                "^",
                color=ORANGE,
                markersize=7,
                label="Actual start" if include_labels else None,
                zorder=8,
            )
            ax.plot(
                actual_states[-1][1],
                actual_states[-1][2],
                "o",
                color=RED,
                markersize=8,
                label="Current vehicle pose" if include_labels else None,
                zorder=9,
            )

def max_abs(values):
    return max((abs(v) for v in values), default=0.0)

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

def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)

def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

def valid_reference_time_axis(t, n_expected):
    if len(t) < n_expected or n_expected < 2:
        return False
    t = t[:n_expected]
    if not all(math.isfinite(v) for v in t):
        return False
    if t[-1] - t[0] <= 1e-6:
        return False
    return all(t[i] <= t[i + 1] + 1e-9 for i in range(len(t) - 1))

def nearest_time_index(ref_t, t_query):
    i = bisect_left(ref_t, t_query)
    if i <= 0:
        return 0
    if i >= len(ref_t):
        return len(ref_t) - 1
    before = i - 1
    return before if abs(t_query - ref_t[before]) <= abs(ref_t[i] - t_query) else i

def reference_signature(ref):
    x = ref.get("x", [])
    y = ref.get("y", [])
    yaw = ref.get("yaw", [])
    if not x or not y:
        return None
    return (
        len(x),
        round(x[0], 3), round(y[0], 3),
        round(x[-1], 3), round(y[-1], 3),
        round(yaw[0], 3) if yaw else 0.0,
        round(yaw[-1], 3) if yaw else 0.0,
    )

def build_tracking_series(ref, actual_states, wheelbase=DEFAULT_WHEELBASE,
                          matching_mode="position"):
    """Match every actual state against the reference and compute errors."""
    required = ("x", "y", "yaw", "v_ref")
    n_ref = min((len(ref.get(name, [])) for name in required), default=0)
    if n_ref == 0 or not actual_states:
        return {"available": False, "match_mode": "none"}

    ref_x = np.array(ref["x"][:n_ref], dtype=float)
    ref_y = np.array(ref["y"][:n_ref], dtype=float)
    ref_yaw = np.array(ref["yaw"][:n_ref], dtype=float)
    ref_v = np.array(ref["v_ref"][:n_ref], dtype=float)

    has_kappa = len(ref.get("kappa", [])) >= n_ref
    has_delta = len(ref.get("delta", [])) >= n_ref
    ref_kappa = np.array(ref.get("kappa", [])[:n_ref], dtype=float) if has_kappa else None
    ref_delta = np.array(ref.get("delta", [])[:n_ref], dtype=float) if has_delta else None

    ref_t = list(ref.get("t", [])[:n_ref])
    requested_mode = str(matching_mode or "position").strip().lower()
    if requested_mode not in ("position", "time", "auto"):
        requested_mode = "position"
    use_time = requested_mode == "time" and valid_reference_time_axis(ref_t, n_ref)
    match_mode = "time" if use_time else "position"

    out = {
        "available": True,
        "match_mode": match_mode,
        "t": [],
        "x": [],
        "y": [],
        "yaw": [],
        "vx": [],
        "vy": [],
        "r": [],
        "x_ref": [],
        "y_ref": [],
        "yaw_ref": [],
        "v_ref": [],
        "r_ref": [],
        "e_x": [],
        "e_y": [],
        "e_pos": [],
        "e_lat": [],
        "e_yaw": [],
        "e_yaw_deg": [],
        "e_v": [],
    }

    for state in actual_states:
        if len(state) < len(ACTUAL_FIELDS):
            continue
        t, x, y, yaw, vx, vy, r = state[:len(ACTUAL_FIELDS)]
        values = (t, x, y, yaw, vx, vy, r)
        if not all(math.isfinite(float(v)) for v in values):
            continue

        if use_time:
            idx = nearest_time_index(ref_t, t)
        else:
            idx = int(np.argmin((ref_x - x) ** 2 + (ref_y - y) ** 2))

        dx = x - ref_x[idx]
        dy = y - ref_y[idx]
        e_yaw = wrap_to_pi(yaw - ref_yaw[idx])
        e_lat = -math.sin(ref_yaw[idx]) * dx + math.cos(ref_yaw[idx]) * dy
        v_ref = ref_v[idx]
        if has_kappa:
            r_ref = v_ref * ref_kappa[idx]
        elif has_delta:
            r_ref = v_ref * math.tan(ref_delta[idx]) / max(1e-3, wheelbase)
        else:
            r_ref = 0.0

        out["t"].append(t)
        out["x"].append(x)
        out["y"].append(y)
        out["yaw"].append(yaw)
        out["vx"].append(vx)
        out["vy"].append(vy)
        out["r"].append(r)
        out["x_ref"].append(ref_x[idx])
        out["y_ref"].append(ref_y[idx])
        out["yaw_ref"].append(ref_yaw[idx])
        out["v_ref"].append(v_ref)
        out["r_ref"].append(r_ref)
        out["e_x"].append(dx)
        out["e_y"].append(dy)
        out["e_pos"].append(math.hypot(dx, dy))
        out["e_lat"].append(e_lat)
        out["e_yaw"].append(e_yaw)
        out["e_yaw_deg"].append(math.degrees(e_yaw))
        out["e_v"].append(vx - v_ref)

    if not out["t"]:
        return {"available": False, "match_mode": match_mode}
    return out

def actual_duration(actual_states):
    if len(actual_states) < 2:
        return 0.0
    return max(0.0, actual_states[-1][0] - actual_states[0][0])

def actual_path_length(actual_states):
    if len(actual_states) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(actual_states)):
        dist += math.hypot(actual_states[i][1] - actual_states[i - 1][1],
                           actual_states[i][2] - actual_states[i - 1][2])
    return dist

def distance_to_reference_end(ref, actual_states):
    if not actual_states or not ref.get("x") or not ref.get("y"):
        return None
    return math.hypot(actual_states[-1][1] - ref["x"][-1],
                      actual_states[-1][2] - ref["y"][-1])

def recent_command_is_zero(cmd, window_s=1.0):
    t = cmd.get("t", [])
    a = cmd.get("a", [])
    delta = cmd.get("delta", [])
    if not t or not a or not delta:
        return False
    t_latest = t[-1]
    recent = [
        i for i, ti in enumerate(t)
        if ti >= t_latest - window_s and i < len(a) and i < len(delta)
    ]
    if not recent:
        recent = [len(t) - 1]
    return all(abs(a[i]) <= ZERO_CMD_A_EPS and
               abs(delta[i]) <= ZERO_CMD_DELTA_EPS for i in recent)

def command_series_is_zero(cmd):
    a = cmd.get("a", [])
    delta = cmd.get("delta", [])
    n = min(len(a), len(delta))
    if n == 0:
        return False
    return all(abs(a[i]) <= ZERO_CMD_A_EPS and
               abs(delta[i]) <= ZERO_CMD_DELTA_EPS for i in range(n))

def build_tracking_diagnostics(d):
    actual = actual_odom_states(d)
    ref = d["ref"]
    cmd = d["cmd"]
    min_duration = d["tracking_min_duration_s"]
    min_samples = d["tracking_min_odom_samples"]
    min_travel = d["tracking_min_travel_m"]
    near_end_m = d["tracking_near_end_m"]

    sample_count = len(actual)
    duration = actual_duration(actual)
    travel = actual_path_length(actual)
    end_dist = distance_to_reference_end(ref, actual)
    near_end = end_dist is not None and end_dist <= near_end_m
    has_reference = bool(reference_xy(ref))
    max_abs_vx = max((abs(s[4]) for s in actual), default=0.0)
    cmd_all_zero = command_series_is_zero(cmd)
    cmd_recent_zero = recent_command_is_zero(cmd)

    warnings = []
    if not has_reference:
        warnings.append("Waiting for /astar_reference_trajectory.")
    if sample_count == 0:
        warnings.append("No valid actual dynamic trajectory from /odom yet.")
    elif sample_count < min_samples:
        warnings.append(
            "Actual trajectory too short for tracking error evaluation "
            f"({sample_count}/{min_samples} samples).")
    if sample_count > 0 and duration < min_duration and not near_end:
        warnings.append(
            f"Actual trajectory duration is {duration:.1f}s; waiting for {min_duration:.1f}s or near goal.")
    if sample_count > 0 and duration >= 1.0 and max_abs_vx <= NEAR_ZERO_VX_EPS:
        warnings.append(
            f"Actual vx is nearly zero: max |vx|={max_abs_vx:.2f} m/s.")
    if sample_count > 0 and duration >= 1.0 and travel < min_travel:
        warnings.append("/odom received but actual vehicle did not move enough.")
    if not cmd.get("t"):
        warnings.append("Waiting for /mpc_cmd.")
    elif cmd_all_zero:
        warnings.append("Tracking invalid: MPC command is zero.")
        warnings.append("MPC command is zero, vehicle is not moving.")
    elif cmd_recent_zero:
        warnings.append("Recent /mpc_cmd is zero.")

    tracking_invalid = (
        (cmd_all_zero or cmd_recent_zero) and
        sample_count > 0 and
        travel < min_travel and
        not near_end
    )
    tracking_ready = (
        has_reference and
        sample_count >= min_samples and
        (duration >= min_duration or near_end) and
        (travel >= min_travel or near_end) and
        not tracking_invalid
    )
    return {
        "tracking_ready": tracking_ready,
        "tracking_invalid": tracking_invalid,
        "cmd_all_zero": cmd_all_zero,
        "cmd_recent_zero": cmd_recent_zero,
        "warnings": warnings,
        "sample_count": sample_count,
        "min_samples": min_samples,
        "min_travel": min_travel,
        "duration": duration,
        "travel": travel,
        "near_end": near_end,
        "end_dist": end_dist,
        "max_abs_vx": max_abs_vx,
        "has_reference": has_reference,
    }

def stationary_pose_error_snapshot(ref, actual_states):
    ref_pts = reference_xy(ref)
    if not ref_pts or not actual_states:
        return None
    state = actual_states[-1]
    x = state[1]
    y = state[2]
    idx = nearest_point_index((x, y), ref_pts)
    x_ref, y_ref = ref_pts[idx]
    dx = x - x_ref
    dy = y - y_ref
    yaw_values = ref.get("yaw", [])
    yaw_ref = yaw_values[idx] if idx < len(yaw_values) else 0.0
    return {
        "index": idx,
        "x": x,
        "y": y,
        "x_ref": x_ref,
        "y_ref": y_ref,
        "e_pos": math.hypot(dx, dy),
        "e_lat": -math.sin(yaw_ref) * dx + math.cos(yaw_ref) * dy,
    }


# ---------------------------------------------------------------------------
# ROS 2 data collector node
# ---------------------------------------------------------------------------
class DataCollector(Node):
    ERROR_WINDOW = 60.0   # seconds of error history to show

    def __init__(self):
        super().__init__("live_plotter")
        self.lock = threading.Lock()
        self.declare_parameter("reference_matching_mode", "position")
        self.declare_parameter("tracking_min_duration_s", MIN_TRACKING_DURATION_S)
        self.declare_parameter("tracking_min_odom_samples", MIN_ODOM_SAMPLES)
        self.declare_parameter("tracking_min_travel_m", MIN_ACTUAL_TRAVEL_M)
        self.declare_parameter("tracking_near_end_m", NEAR_REFERENCE_END_M)
        self.declare_parameter("show_debug_view", False)

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
        self.actual_states = []      # [(t,x,y,yaw,vx,vy,r), ...]
        self.actual_source = ""
        self.slip_t = []
        self.alpha_f = []
        self.alpha_r = []
        self.force_t = []
        self.Fyf = []
        self.Fyr = []
        self.cmd_t = []
        self.cmd_a = []
        self.cmd_delta = []
        self.err_t = []              # /mpc_tracking_error time (s)
        self.err_lat = []            # lateral error (m)
        self.err_yaw = []            # heading error (deg)
        self.err_v = []              # speed error (m/s)
        self._actual_t0 = None
        self._fallback_t0 = None
        self._mpc_err_t0 = None
        self._odom_seen = False
        self._warned_waiting_actual = False
        self._reference_signature = None
        self._raw_ref_debug_signature = None

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
        self.create_subscription(Float64MultiArray, "/dynamic_vehicle_state",
                                 self._cb_dynamic_state, 20)
        self.create_subscription(Float64MultiArray, "/dynamic_slip_angles",
                                 self._cb_slip, 20)
        self.create_subscription(Float64MultiArray, "/dynamic_tire_forces",
                                 self._cb_forces, 20)
        self.create_subscription(Float64MultiArray, "/mpc_cmd",
                                 self._cb_mpc_cmd, 20)
        self.create_subscription(Float64MultiArray, "mpc_tracking_error",
                                 self._cb_error,    20)

    def _cb_grid(self, msg):
        with self.lock: self.grid_msg = msg

    def _reset_run_tracking_locked(self):
        self.actual_states = []
        self.actual_source = ""
        self.slip_t = []
        self.alpha_f = []
        self.alpha_r = []
        self.force_t = []
        self.Fyf = []
        self.Fyr = []
        self.cmd_t = []
        self.cmd_a = []
        self.cmd_delta = []
        self.err_t = []
        self.err_lat = []
        self.err_yaw = []
        self.err_v = []
        self._actual_t0 = None
        self._fallback_t0 = None
        self._mpc_err_t0 = None
        self._odom_seen = False
        self._warned_waiting_actual = False

    def _cb_raw(self, msg):
        with self.lock:
            self.raw_path_msg = msg
            self._debug_print_raw_vs_ref_locked()

    def _cb_smooth(self, msg):
        with self.lock: self.smooth_path_msg = msg

    def _debug_print_raw_vs_ref_locked(self):
        """Print raw vs reference path stats whenever a new pair arrives.

        Confirms that the two arrays come from different topics and exposes
        their actual numerical separation. Signature-gated so it fires at most
        once per (raw, reference) update — not on every redraw."""
        raw_msg = self.raw_path_msg
        ref = self.reference_data
        if not raw_msg or not raw_msg.poses:
            return
        if not ref or not ref.get("x") or not ref.get("y"):
            return

        raw_pts = [(p.pose.position.x, p.pose.position.y) for p in raw_msg.poses]
        ref_pts = list(zip(ref["x"], ref["y"]))

        sig = (len(raw_pts), len(ref_pts),
               raw_pts[0], raw_pts[-1], ref_pts[0], ref_pts[-1])
        if sig == self._raw_ref_debug_signature:
            return
        self._raw_ref_debug_signature = sig

        def fmt5(pts, label):
            head = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in pts[:5])
            tail = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in pts[-5:])
            return f"  {label} first5: {head}\n  {label} last5:  {tail}"

        # Symmetric nearest-neighbour distance (ref -> raw)
        dists = []
        for rx, ry in ref_pts:
            best = float("inf")
            for px, py in raw_pts:
                d = (rx - px) * (rx - px) + (ry - py) * (ry - py)
                if d < best:
                    best = d
            dists.append(math.sqrt(best))
        max_d = max(dists) if dists else float("nan")
        mean_d = (sum(dists) / len(dists)) if dists else float("nan")

        log = self.get_logger()
        log.info("[raw-vs-ref debug] new path pair received:")
        log.info(f"  raw points (/astar_path_raw)             = {len(raw_pts)}")
        log.info(f"  reference points (/astar_reference_trajectory) = {len(ref_pts)}")
        log.info(f"  max nearest-neighbour distance  = {max_d:.4f} m")
        log.info(f"  mean nearest-neighbour distance = {mean_d:.4f} m")
        log.info(fmt5(raw_pts, "raw"))
        log.info(fmt5(ref_pts, "ref"))

    def _cb_vel(self, msg):
        with self.lock: self.velocity_data = list(msg.data)

    def _cb_curv(self, msg):
        with self.lock: self.curvature_data = list(msg.data)

    def _cb_steer(self, msg):
        with self.lock: self.steering_data = list(msg.data)

    def _cb_accel(self, msg):
        with self.lock: self.acceleration_data = list(msg.data)

    def _cb_reference(self, msg):
        ref = reference_from_msg(msg)
        sig = reference_signature(ref)
        with self.lock:
            self.reference_data = ref
            if sig != self._reference_signature:
                self._reference_signature = sig
                self._reset_run_tracking_locked()
            self._debug_print_raw_vs_ref_locked()

    def _append_actual_state_locked(self, t_abs, x, y, yaw, vx, vy, r):
        if self._actual_t0 is None:
            self._actual_t0 = t_abs
        t = max(0.0, t_abs - self._actual_t0)
        self.actual_states.append((t, x, y, wrap_to_pi(yaw), vx, vy, r))

    def _relative_signal_time_locked(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._actual_t0 is not None:
            return max(0.0, now - self._actual_t0)
        if self._fallback_t0 is None:
            self._fallback_t0 = now
        return now - self._fallback_t0

    def _cb_odom(self, msg):
        t_abs = stamp_to_seconds(msg.header.stamp)
        if t_abs <= 0.0:
            t_abs = self.get_clock().now().nanoseconds * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        r = msg.twist.twist.angular.z
        with self.lock:
            if not self._odom_seen:
                if self.actual_source == "/dynamic_vehicle_state":
                    self.actual_states = []
                    self._actual_t0 = None
                self._odom_seen = True
            self.actual_source = "/odom"
            self._append_actual_state_locked(t_abs, x, y, yaw, vx, vy, r)

    def _cb_dynamic_state(self, msg):
        if len(msg.data) < 6:
            return
        t_abs = self.get_clock().now().nanoseconds * 1e-9
        x, y, yaw, vx, vy, r = (float(v) for v in msg.data[:6])
        with self.lock:
            if self._odom_seen:
                return
            self.actual_source = "/dynamic_vehicle_state"
            self._append_actual_state_locked(t_abs, x, y, yaw, vx, vy, r)

    def _cb_slip(self, msg):
        if len(msg.data) < 2:
            return
        with self.lock:
            self.slip_t.append(self._relative_signal_time_locked())
            self.alpha_f.append(float(msg.data[0]))
            self.alpha_r.append(float(msg.data[1]))

    def _cb_forces(self, msg):
        if len(msg.data) < 2:
            return
        with self.lock:
            self.force_t.append(self._relative_signal_time_locked())
            self.Fyf.append(float(msg.data[0]))
            self.Fyr.append(float(msg.data[1]))

    def _cb_mpc_cmd(self, msg):
        if len(msg.data) < 2:
            return
        with self.lock:
            self.cmd_t.append(self._relative_signal_time_locked())
            self.cmd_a.append(float(msg.data[0]))
            self.cmd_delta.append(float(msg.data[1]))

    def _cb_error(self, msg):
        if len(msg.data) < 3: return
        now = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            if self._mpc_err_t0 is None:
                self._mpc_err_t0 = now
            t = now - self._mpc_err_t0
            self.err_t.append(t)
            self.err_lat.append(msg.data[0])
            self.err_yaw.append(math.degrees(msg.data[1]))
            self.err_v.append(msg.data[2])

    def warn_waiting_for_actual_once(self):
        should_warn = False
        with self.lock:
            if not self.actual_states and not self._warned_waiting_actual:
                self._warned_waiting_actual = True
                should_warn = True
        if should_warn:
            self.get_logger().warn(
                "Waiting for /odom to plot actual dynamic vehicle trajectory.")

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
                actual      = list(self.actual_states),
                actual_source = self.actual_source,
                odom_seen   = self._odom_seen,
                mpc_err_t   = list(self.err_t),
                mpc_err_lat = list(self.err_lat),
                mpc_err_yaw = list(self.err_yaw),
                mpc_err_v   = list(self.err_v),
                slip        = dict(t=list(self.slip_t),
                                   alpha_f=list(self.alpha_f),
                                   alpha_r=list(self.alpha_r)),
                forces      = dict(t=list(self.force_t),
                                   Fyf=list(self.Fyf),
                                   Fyr=list(self.Fyr)),
                cmd         = dict(t=list(self.cmd_t),
                                   a=list(self.cmd_a),
                                   delta=list(self.cmd_delta)),
                matching_mode = str(
                    self.get_parameter("reference_matching_mode").value),
                tracking_min_duration_s = float(
                    self.get_parameter("tracking_min_duration_s").value),
                tracking_min_odom_samples = int(
                    self.get_parameter("tracking_min_odom_samples").value),
                tracking_min_travel_m = float(
                    self.get_parameter("tracking_min_travel_m").value),
                tracking_near_end_m = float(
                    self.get_parameter("tracking_near_end_m").value),
                show_debug_view = bool(
                    self.get_parameter("show_debug_view").value),
            )


# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------
BLUE   = "#1f77b4"
RED    = "#d62728"
GRAY   = "#999999"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
PURPLE = "#9467bd"
CYAN   = "#17becf"

class LiveDashboard:
    def __init__(self, node: DataCollector):
        self.node = node

        self.fig = plt.figure(figsize=(19, 11))
        self.fig.patch.set_facecolor("#1e1e2e")
        self.fig.suptitle("Part A - Reference Trajectory Generation",
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
            self._style_axis(ax)

        self.show_debug_view = bool(
            self.node.get_parameter("show_debug_view").value)

        self.fig_tracking = plt.figure(figsize=(18, 11))
        self.fig_tracking.patch.set_facecolor("#1e1e2e")
        self.fig_tracking.suptitle("Path and Tracking Error Evaluation",
                                   color="white", fontsize=14,
                                   fontweight="bold")

        gs_b = gridspec.GridSpec(3, 2,
                                 left=0.055, right=0.98,
                                 top=0.91, bottom=0.07,
                                 hspace=0.45, wspace=0.28,
                                 width_ratios=[1.35, 1.0])
        self.ax_prof_path = self.fig_tracking.add_subplot(gs_b[0:2, 0])
        self.ax_prof_zoom = self.fig_tracking.add_subplot(gs_b[0, 1])
        self.ax_prof_raw_ref = self.fig_tracking.add_subplot(gs_b[1, 1])
        self.ax_prof_curv = self.fig_tracking.add_subplot(gs_b[2, 0])
        self.ax_prof_error = self.fig_tracking.add_subplot(gs_b[2, 1])

        for ax in [
            self.ax_prof_path, self.ax_prof_zoom, self.ax_prof_raw_ref,
            self.ax_prof_curv, self.ax_prof_error,
        ]:
            self._style_axis(ax)

        self._last_counts_log_time = None
        self._last_counts_logged = None

        self.fig_debug = None
        if self.show_debug_view:
            self.fig_debug = plt.figure(figsize=(20, 14))
            self.fig_debug.patch.set_facecolor("#1e1e2e")
            self.fig_debug.suptitle("Dynamic Debug View",
                                    color="white", fontsize=14,
                                    fontweight="bold")
            gs_debug = gridspec.GridSpec(4, 3,
                                         left=0.055, right=0.98,
                                         top=0.93, bottom=0.06,
                                         hspace=0.55, wspace=0.30)
            self.ax_track_path = self.fig_debug.add_subplot(gs_debug[0:2, 0])
            self.ax_lat_err = self.fig_debug.add_subplot(gs_debug[0, 1])
            self.ax_yaw_err = self.fig_debug.add_subplot(gs_debug[1, 1])
            self.ax_vel_cmp = self.fig_debug.add_subplot(gs_debug[0, 2])
            self.ax_vel_err = self.fig_debug.add_subplot(gs_debug[1, 2])
            self.ax_yaw_rate = self.fig_debug.add_subplot(gs_debug[2, 0])
            self.ax_vy = self.fig_debug.add_subplot(gs_debug[2, 1])
            self.ax_slip = self.fig_debug.add_subplot(gs_debug[2, 2])
            self.ax_forces = self.fig_debug.add_subplot(gs_debug[3, 0:2])
            self.ax_cmd = self.fig_debug.add_subplot(gs_debug[3, 2])
            self.ax_cmd_delta = self.ax_cmd.twinx()
            for ax in [
                self.ax_track_path, self.ax_lat_err, self.ax_yaw_err,
                self.ax_vel_cmp, self.ax_vel_err, self.ax_yaw_rate,
                self.ax_vy, self.ax_slip, self.ax_forces, self.ax_cmd,
                self.ax_cmd_delta,
            ]:
                self._style_axis(ax)

        self.ani = FuncAnimation(self.fig, self._update,
                                 interval=500, blit=False)

    def _style_axis(self, ax):
        ax.set_facecolor("#2b2b3b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555566")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    def _annotate_waiting(self, ax, text):
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", color="#dddddd", fontsize=8,
                bbox=dict(facecolor="#3b3b4b", edgecolor="#666677",
                          alpha=0.85, boxstyle="round,pad=0.35"))

    def _annotate_warnings(self, ax, warnings, max_lines=5,
                           loc="upper left"):
        if not warnings:
            return
        shown = list(warnings[:max_lines])
        if len(warnings) > max_lines:
            shown.append(f"... {len(warnings) - max_lines} more")
        if loc == "upper right":
            x = 0.98
            ha = "right"
        else:
            x = 0.02
            ha = "left"
        ax.text(x, 0.98, "\n".join(shown), transform=ax.transAxes,
                ha=ha, va="top", color="#fff2b3", fontsize=7,
                bbox=dict(facecolor="#4b3b2b", edgecolor="#8a7446",
                          alpha=0.88, boxstyle="round,pad=0.35"))

    def _annotate_tracking_hold(self, ax, diagnostics):
        if diagnostics.get("tracking_invalid"):
            msg = (
                "Tracking invalid: MPC command is zero\n"
                f"/odom samples: {diagnostics['sample_count']}, "
                f"duration: {diagnostics['duration']:.1f}s, "
                f"travel: {diagnostics['travel']:.2f}m"
            )
            self._annotate_waiting(ax, msg)
            return

        msg = (
            "Waiting for meaningful tracking data\n"
            f"/odom samples: {diagnostics['sample_count']}, "
            f"duration: {diagnostics['duration']:.1f}s, "
            f"travel: {diagnostics['travel']:.2f}m"
        )
        self._annotate_waiting(ax, msg)

    def _draw_grid_background(self, ax, grid_msg):
        if grid_msg is None: return
        img, extent = grid_to_image(grid_msg)
        ax.imshow(img, cmap="gray", origin="lower",
                  extent=extent, vmin=0, vmax=255, alpha=0.55, zorder=0)

    # ---- Plot 1: Path Planning ------------------------------------------
    def _update_plot1(self, d):
        ax = self.ax1
        ax.cla()
        self._style_axis(ax)
        ax.set_title("A1-A2 - Raw Path and Reference Trajectory",
                     fontsize=9, color="white")
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
        self._style_axis(ax)
        ax.set_title("A3 - Curvature kappa(s)", fontsize=9, color="white")
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
        self._style_axis(ax)
        ax.set_title("A4 - Velocity Profile", fontsize=9, color="white")
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
        self._style_axis(ax)
        ax.set_title("A5 - Steering delta_ref", fontsize=9, color="white")
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
        self._style_axis(ax)
        ax.set_title("A6 - Acceleration a_ref", fontsize=9, color="white")
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

    # ---- Professor View ---------------------------------------------------
    def _professor_actual_warning(self, diagnostics):
        if diagnostics["sample_count"] == 0:
            return "No valid actual dynamic trajectory from /odom yet."
        if diagnostics.get("tracking_invalid"):
            return "Tracking invalid: MPC command is zero."
        if ((diagnostics.get("cmd_all_zero") or
             diagnostics.get("cmd_recent_zero")) and
                diagnostics["travel"] < diagnostics.get("min_travel", MIN_ACTUAL_TRAVEL_M)):
            return "Tracking invalid: MPC command is zero."
        if diagnostics["sample_count"] < diagnostics.get("min_samples", MIN_ODOM_SAMPLES):
            return "Actual trajectory too short for tracking error evaluation."
        if diagnostics["travel"] < diagnostics.get("min_travel", MIN_ACTUAL_TRAVEL_M):
            return "/odom received but actual vehicle did not move enough."
        if not diagnostics.get("has_reference", False):
            return "Waiting for curvature-aware reference trajectory"
        if not diagnostics["tracking_ready"]:
            return "Actual trajectory too short for meaningful error evaluation"
        return ""

    def _professor_counts(self, d):
        raw_count = len(raw_path_xy(d))
        ref_count = len(reference_display_xy(d))
        actual_count = len(actual_odom_states(d))
        return raw_count, ref_count, actual_count

    def _log_professor_counts(self, d):
        counts = self._professor_counts(d)
        now = self.node.get_clock().now().nanoseconds * 1e-9
        should_log = self._last_counts_logged is None
        if self._last_counts_logged != counts:
            if self._last_counts_log_time is None:
                should_log = True
            elif now - self._last_counts_log_time >= 2.0:
                should_log = True

        if should_log:
            raw_count, ref_count, actual_count = counts
            actual = actual_odom_states(d)
            _, _, max_gap, mean_gap = raw_reference_deviation(
                raw_path_xy(d), reference_display_xy(d))
            gap_text = "n/a"
            if max_gap is not None and mean_gap is not None:
                gap_text = f"max={max_gap:.3f} m, mean={mean_gap:.3f} m"
            self.node.get_logger().info(
                f"Raw path points: {raw_count} | "
                f"Reference points: {ref_count} | "
                f"Actual trajectory points: {actual_count} | "
                f"Actual travel: {actual_path_length(actual):.3f} m | "
                f"Max |vx|: {max_abs([s[4] for s in actual]):.3f} m/s | "
                f"Raw-reference gap: {gap_text}")
            self._last_counts_logged = counts
            self._last_counts_log_time = now

    def _update_professor_path(self, d, track, diagnostics):
        ax = self.ax_prof_path
        ax.cla()
        self._style_axis(ax)
        ax.set_title("Path Comparison", fontsize=10, color="white")
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)

        self._draw_grid_background(ax, d["grid"])

        raw_pts = raw_path_xy(d)
        ref_pts = reference_display_xy(d)
        actual = actual_odom_states(d)
        actual_travel = actual_path_length(actual)
        actual_max_vx = max_abs([s[4] for s in actual])
        _, _, raw_ref_max_gap, raw_ref_mean_gap = raw_reference_deviation(
            raw_pts, ref_pts)
        draw_path_comparison(ax, raw_pts, ref_pts, actual,
                             include_labels=True, show_current=True)

        if not actual:
            self.node.warn_waiting_for_actual_once()
        elif actual_travel < MIN_ACTUAL_TRAVEL_M:
            ax.plot(actual[-1][1], actual[-1][2],
                    marker="o", markersize=15, markerfacecolor="none",
                    markeredgecolor=ORANGE, markeredgewidth=2.0,
                    zorder=10)
            ax.text(actual[-1][1], actual[-1][2],
                    f"  /odom received\n  travel={actual_travel:.3f} m",
                    color="#111111", fontsize=8, ha="left", va="center",
                    bbox=dict(facecolor="#fff4cc", edgecolor=ORANGE,
                              alpha=0.90, boxstyle="round,pad=0.3"),
                    zorder=11)

        start_goal_pts = ref_pts
        if not start_goal_pts:
            start_goal_pts = raw_pts
        if start_goal_pts:
            ax.plot(start_goal_pts[0][0], start_goal_pts[0][1], "o",
                    color=GREEN, markersize=7, label="Start", zorder=6)
            ax.plot(start_goal_pts[-1][0], start_goal_pts[-1][1], "*",
                    color=RED, markersize=10, label="Goal", zorder=6)

        warning = self._professor_actual_warning(diagnostics)
        if warning:
            self._annotate_waiting(ax, warning)
        elif track.get("available"):
            ax.text(0.02, 0.02, "Reference matching: nearest point",
                    transform=ax.transAxes, color="#dddddd", fontsize=7,
                    ha="left", va="bottom")

        if raw_ref_max_gap is not None and raw_ref_mean_gap is not None:
            if raw_ref_max_gap < 0.15:
                gap_note = (
                    "Raw/reference paths overlap because the smoothing "
                    "displacement is small.\n"
                    f"Max gap: {raw_ref_max_gap:.3f} m, "
                    f"mean gap: {raw_ref_mean_gap:.3f} m"
                )
            else:
                gap_note = (
                    "Raw-reference geometric displacement\n"
                    f"Max gap: {raw_ref_max_gap:.3f} m, "
                    f"mean gap: {raw_ref_mean_gap:.3f} m"
                )
            ax.text(0.02, 0.09, gap_note,
                    transform=ax.transAxes, color="#111111", fontsize=8,
                    ha="left", va="bottom",
                    bbox=dict(facecolor="#eeeeee", edgecolor="#555555",
                              alpha=0.78, boxstyle="round,pad=0.3"))
        self._annotate_warnings(ax, diagnostics["warnings"], max_lines=4,
                                loc="upper right")

        raw_count, ref_count, actual_count = self._professor_counts(d)
        gap_line = "Raw-ref gap: n/a"
        if raw_ref_max_gap is not None and raw_ref_mean_gap is not None:
            gap_line = (
                f"Raw-ref gap: max {raw_ref_max_gap:.3f} m, "
                f"mean {raw_ref_mean_gap:.3f} m")
        ax.text(0.98, 0.02,
                f"Raw: {raw_count} pts\n"
                f"Reference: {ref_count} pts\n"
                f"Actual /odom: {actual_count} pts\n"
                f"Travel: {actual_travel:.3f} m\n"
                f"max |vx|: {actual_max_vx:.3f} m/s\n"
                f"{gap_line}",
                transform=ax.transAxes, color="#111111", fontsize=8,
                ha="right", va="bottom",
                bbox=dict(facecolor="#eeeeee", edgecolor="#555555",
                          alpha=0.78, boxstyle="round,pad=0.3"))

        show_legend(ax, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=7)

    def _update_professor_zoom(self, d, diagnostics):
        ax = self.ax_prof_zoom
        ax.cla()
        self._style_axis(ax)
        ax.set_title("Zoom on Trajectory Difference", fontsize=10, color="white")
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)

        raw_pts = raw_path_xy(d)
        ref_pts = reference_display_xy(d)
        actual = actual_odom_states(d)
        draw_path_comparison(ax, raw_pts, ref_pts, actual,
                             include_labels=False, show_current=False,
                             dense_markers=True)

        focus = path_difference_focus(raw_pts, ref_pts)
        if focus:
            cx, cy, half_width, max_diff, best_raw, best_ref = focus
            ax.set_xlim(cx - half_width, cx + half_width)
            ax.set_ylim(cy - half_width, cy + half_width)
            ax.plot([best_raw[0], best_ref[0]], [best_raw[1], best_ref[1]],
                    color=RED, linestyle=":", linewidth=1.4, zorder=10)
            ax.plot(best_raw[0], best_raw[1], "s", color="#303030",
                    markerfacecolor="#f2f2f2", markersize=6, zorder=11)
            ax.plot(best_ref[0], best_ref[1], "o", color=BLUE,
                    markerfacecolor=BLUE, markersize=6, zorder=11)
            ax.text(0.02, 0.02, f"Max raw-reference gap: {max_diff:.3f} m",
                    transform=ax.transAxes, color="#dddddd", fontsize=7,
                    ha="left", va="bottom")
        else:
            self._annotate_waiting(
                ax, "Waiting for raw and reference paths to zoom.")

        actual_warnings = [
            w for w in diagnostics["warnings"]
            if ("actual" in w.lower() or "/odom" in w.lower() or
                "tracking invalid" in w.lower() or "/mpc_cmd" in w.lower() or
                "mpc command" in w.lower())
        ]
        if actual_warnings:
            self._annotate_warnings(ax, actual_warnings, max_lines=2,
                                    loc="upper right")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=7)

    def _update_professor_raw_ref_deviation(self, d):
        ax = self.ax_prof_raw_ref
        ax.cla()
        self._style_axis(ax)
        ax.set_title("Raw Path vs Reference Path Deviation",
                     fontsize=10, color="white")
        ax.set_xlabel("Reference arc length s (m)", fontsize=8)
        ax.set_ylabel("Nearest distance (m)", fontsize=8)

        raw_pts = raw_path_xy(d)
        ref_pts = reference_display_xy(d)
        s, err, max_gap, mean_gap = raw_reference_deviation(raw_pts, ref_pts)
        if s and err:
            ax.plot(s, err, color=PURPLE, linewidth=1.8,
                    label="e_raw_ref(s)")
            ax.fill_between(s, err, color=PURPLE, alpha=0.12)
            ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
            ax.axhline(mean_gap, color=ORANGE, linestyle="--",
                       linewidth=1.0, alpha=0.85,
                       label=f"mean={mean_gap:.3f} m")
            set_padded_ylim(ax, err, min_span=0.03)
            ax.text(0.98, 0.95,
                    f"max gap: {max_gap:.3f} m\n"
                    f"mean gap: {mean_gap:.3f} m",
                    transform=ax.transAxes, color="#111111", fontsize=8,
                    ha="right", va="top",
                    bbox=dict(facecolor="#eeeeee", edgecolor="#555555",
                              alpha=0.82, boxstyle="round,pad=0.3"))
            show_legend(ax, loc="upper left")
        else:
            self._annotate_waiting(
                ax, "Waiting for raw path and reference trajectory.")

        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=7)

    def _update_professor_curvature(self, d):
        ax = self.ax_prof_curv
        ax.cla()
        self._style_axis(ax)
        ax.set_title("Reference Curvature", fontsize=10, color="white")
        ax.set_xlabel("Arc length s (m)", fontsize=8)
        ax.set_ylabel("kappa (1/m)", fontsize=8)

        ref_pts = reference_xy(d["ref"])
        if ref_pts and d["ref"].get("kappa"):
            n = min(len(ref_pts), len(d["ref"]["kappa"]))
            s = arc_lengths(ref_pts[:n])
            k = d["ref"]["kappa"][:n]
            plot_segmented(ax, s, k, BLUE, "kappa(s)",
                           linewidth=1.8, fill_alpha=0.12)
            set_padded_ylim(ax, k, min_span=0.05)
        elif d["smooth"] and d["smooth"].poses:
            pts = path_to_xy(d["smooth"])
            s, k = curvature_topic_axis(pts, d["curv"])
            if k:
                plot_segmented(ax, s, k, BLUE, "kappa(s)",
                               linewidth=1.8, fill_alpha=0.12)
                set_padded_ylim(ax, k, min_span=0.05)
        elif d["curv"]:
            s = list(np.arange(len(d["curv"]), dtype=float))
            plot_segmented(ax, s, d["curv"], BLUE, "kappa(s)",
                           linewidth=1.8, fill_alpha=0.12)
            set_padded_ylim(ax, d["curv"], min_span=0.05)
        else:
            self._annotate_waiting(ax, "Waiting for reference curvature.")

        ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
        show_legend(ax)
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=7)

    def _update_professor_error(self, d, track, diagnostics):
        ax = self.ax_prof_error
        ax.cla()
        self._style_axis(ax)
        ax.set_title("Tracking Error", fontsize=10, color="white")
        ax.set_xlabel("Time t (s)", fontsize=8)
        ax.set_ylabel("Error (m)", fontsize=8)
        ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)

        if diagnostics["tracking_ready"] and track.get("available"):
            ax.plot(track["t"], track["e_lat"], color=BLUE, linewidth=1.8,
                    label="Lateral error")
            ax.plot(track["t"], track["e_pos"], color=RED, linewidth=1.6,
                    label="Position error")
            set_padded_ylim(ax, track["e_lat"] + track["e_pos"],
                            min_span=0.1)
            show_legend(ax)
        elif diagnostics.get("tracking_invalid"):
            snapshot = stationary_pose_error_snapshot(
                d["ref"], actual_odom_states(d))
            if snapshot:
                labels = ["Static lateral error", "Static position error"]
                values = [snapshot["e_lat"], snapshot["e_pos"]]
                x = np.arange(len(labels))
                bars = ax.bar(x, values, color=[BLUE, RED], alpha=0.86)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, color="white", fontsize=7)
                ax.set_xlabel(
                    "Stationary /odom pose matched to nearest reference point",
                    fontsize=8)
                set_padded_ylim(ax, values, min_span=0.5)
                for bar, value in zip(bars, values):
                    y = bar.get_height()
                    va = "bottom" if y >= 0.0 else "top"
                    offset = 0.03 if y >= 0.0 else -0.03
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        y + offset,
                        f"{value:.3f} m",
                        ha="center",
                        va=va,
                        color="white",
                        fontsize=7)
                ax.text(
                    0.02, 0.95,
                    "Tracking invalid: MPC command is zero.\n"
                    "Static pose-to-reference error shown,\n"
                    "not tracking performance.\n"
                    f"Nearest reference index: {snapshot['index']}",
                    transform=ax.transAxes, color="#fff2b3", fontsize=7,
                    ha="left", va="top",
                    bbox=dict(facecolor="#4b3b2b", edgecolor="#8a7446",
                              alpha=0.88, boxstyle="round,pad=0.35"))
                show_legend(ax)
            else:
                self._annotate_waiting(
                    ax, self._professor_actual_warning(diagnostics))
            self._annotate_warnings(ax, diagnostics["warnings"], max_lines=4,
                                    loc="upper right")
        else:
            self._annotate_waiting(
                ax, self._professor_actual_warning(diagnostics))
            self._annotate_warnings(ax, diagnostics["warnings"], max_lines=4,
                                    loc="upper right")

        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=7)

    # ---- Part B helpers ---------------------------------------------------
    def _plot_mpc_error_overlay(self, ax, d, key, label):
        if d["mpc_err_t"] and d.get(key):
            n = min(len(d["mpc_err_t"]), len(d[key]))
            ax.plot(d["mpc_err_t"][:n], d[key][:n], color=GRAY,
                    linestyle=":", linewidth=1.0, alpha=0.75,
                    label=label)

    def _set_tracking_axis_common(self, ax, title, ylabel=None):
        ax.cla()
        self._style_axis(ax)
        ax.set_title(title, fontsize=9, color="white")
        ax.set_xlabel("Time t (s)", fontsize=7)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=7)
        ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    def _update_tracking_path(self, d, track, diagnostics):
        ax = self.ax_track_path
        ax.cla()
        self._style_axis(ax)
        ax.set_title("B1 - Reference vs Actual Dynamic Vehicle Trajectory",
                     fontsize=9, color="white")
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)

        self._draw_grid_background(ax, d["grid"])

        ref_pts = reference_xy(d["ref"])
        if ref_pts:
            rx, ry = zip(*ref_pts)
            ax.plot(rx, ry, color=BLUE, linewidth=2.0,
                    label="Reference trajectory", zorder=3)
        elif d["smooth"] and d["smooth"].poses:
            pts = path_to_xy(d["smooth"])
            rx, ry = zip(*pts)
            ax.plot(rx, ry, color=BLUE, linewidth=1.8,
                    label="Smoothed path", zorder=3)

        actual = d["actual"]
        if actual:
            ax.plot([s[1] for s in actual], [s[2] for s in actual],
                    color=RED, linewidth=1.8,
                    label="Actual dynamic vehicle trajectory", zorder=4)
            ax.plot(actual[-1][1], actual[-1][2], "o", color=ORANGE,
                    markersize=5, label="Current vehicle pose", zorder=5)
        else:
            self.node.warn_waiting_for_actual_once()
            self._annotate_waiting(
                ax, "Waiting for /odom to plot actual dynamic vehicle trajectory.")

        if track.get("available"):
            mode = "time" if track["match_mode"] == "time" else "nearest position"
            ax.text(0.02, 0.02, f"Reference matching: {mode}",
                    transform=ax.transAxes, color="#dddddd", fontsize=7,
                    ha="left", va="bottom")
        elif actual:
            mode = str(d.get("matching_mode", "position")).lower()
            if mode != "time":
                mode = "nearest position"
            ax.text(0.02, 0.02,
                    f"Recording actual trajectory: {diagnostics['duration']:.1f}s, "
                    f"{diagnostics['travel']:.2f}m. Matching: {mode}",
                    transform=ax.transAxes, color="#dddddd", fontsize=7,
                    ha="left", va="bottom")
            if not diagnostics["tracking_ready"]:
                self._annotate_warnings(ax, diagnostics["warnings"])

        show_legend(ax, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white", labelsize=6)

    def _update_lateral_error(self, d, track, diagnostics):
        ax = self.ax_lat_err
        self._set_tracking_axis_common(
            ax, "B2 - Lateral / Cross-Track Error", "e_lat (m)")
        if diagnostics["tracking_ready"] and track.get("available"):
            ax.plot(track["t"], track["e_lat"], color=BLUE, linewidth=1.6,
                    label="Computed from actual vs reference")
            self._plot_mpc_error_overlay(ax, d, "mpc_err_lat",
                                         "/mpc_tracking_error")
            set_padded_ylim(ax, track["e_lat"] + d["mpc_err_lat"],
                            min_span=0.1)
            show_legend(ax)
        else:
            self._annotate_tracking_hold(ax, diagnostics)
            self._annotate_warnings(ax, diagnostics["warnings"])

    def _update_heading_error(self, d, track, diagnostics):
        ax = self.ax_yaw_err
        self._set_tracking_axis_common(
            ax, "B3 - Heading / Yaw Error", "e_yaw (deg)")
        if diagnostics["tracking_ready"] and track.get("available"):
            ax.plot(track["t"], track["e_yaw_deg"], color=PURPLE,
                    linewidth=1.6, label="wrap_to_pi(yaw - yaw_ref)")
            self._plot_mpc_error_overlay(ax, d, "mpc_err_yaw",
                                         "/mpc_tracking_error")
            set_padded_ylim(ax, track["e_yaw_deg"] + d["mpc_err_yaw"],
                            min_span=1.0)
            show_legend(ax)
        else:
            self._annotate_tracking_hold(ax, diagnostics)
            self._annotate_warnings(ax, diagnostics["warnings"])

    def _update_velocity_comparison(self, d, track, diagnostics):
        ax = self.ax_vel_cmp
        self._set_tracking_axis_common(
            ax, "B4 - Velocity Tracking", "Velocity (m/s)")
        if diagnostics["tracking_ready"] and track.get("available"):
            ax.plot(track["t"], track["v_ref"], color=BLUE, linewidth=1.7,
                    label="Reference v_ref")
            ax.plot(track["t"], track["vx"], color=RED, linewidth=1.4,
                    label="Actual vx")
            vals = track["v_ref"] + track["vx"]
            set_padded_ylim(ax, vals, min_span=0.5)
            show_legend(ax)
        elif d["actual"]:
            t = [s[0] for s in d["actual"]]
            vx = [s[4] for s in d["actual"]]
            ax.plot(t, vx, color=RED, linewidth=1.4, label="Actual vx")
            set_padded_ylim(ax, vx, min_span=0.5)
            show_legend(ax)
            self._annotate_warnings(ax, diagnostics["warnings"])
        else:
            self._annotate_tracking_hold(ax, diagnostics)
            self._annotate_warnings(ax, diagnostics["warnings"])

    def _update_velocity_error(self, d, track, diagnostics):
        ax = self.ax_vel_err
        self._set_tracking_axis_common(
            ax, "B5 - Velocity Error", "e_v = vx - v_ref (m/s)")
        if diagnostics["tracking_ready"] and track.get("available"):
            ax.plot(track["t"], track["e_v"], color=GREEN, linewidth=1.6,
                    label="Computed velocity error")
            self._plot_mpc_error_overlay(ax, d, "mpc_err_v",
                                         "/mpc_tracking_error")
            set_padded_ylim(ax, track["e_v"] + d["mpc_err_v"],
                            min_span=0.2)
            show_legend(ax)
        else:
            self._annotate_tracking_hold(ax, diagnostics)
            self._annotate_warnings(ax, diagnostics["warnings"])

    def _update_yaw_rate_comparison(self, d, track, diagnostics):
        ax = self.ax_yaw_rate
        self._set_tracking_axis_common(
            ax, "B6 - Yaw-Rate Tracking", "Yaw rate r (rad/s)")
        if diagnostics["tracking_ready"] and track.get("available"):
            ax.plot(track["t"], track["r_ref"], color=BLUE, linewidth=1.7,
                    label="Reference r_ref")
            ax.plot(track["t"], track["r"], color=RED, linewidth=1.4,
                    label="Actual r")
            set_padded_ylim(ax, track["r_ref"] + track["r"], min_span=0.05)
            show_legend(ax)
        elif d["actual"]:
            t = [s[0] for s in d["actual"]]
            r = [s[6] for s in d["actual"]]
            ax.plot(t, r, color=RED, linewidth=1.4, label="Actual r")
            set_padded_ylim(ax, r, min_span=0.05)
            show_legend(ax)
            self._annotate_warnings(ax, diagnostics["warnings"])
        else:
            self._annotate_tracking_hold(ax, diagnostics)
            self._annotate_warnings(ax, diagnostics["warnings"])

    def _update_lateral_velocity(self, d, track):
        ax = self.ax_vy
        self._set_tracking_axis_common(
            ax, "B7 - Lateral Velocity", "vy (m/s)")
        if d["actual"]:
            t = [s[0] for s in d["actual"]]
            vy = [s[5] for s in d["actual"]]
            ax.plot(t, vy, color=CYAN, linewidth=1.5, label="Actual vy")
            set_padded_ylim(ax, vy, min_span=0.1)
            show_legend(ax)
        else:
            self._annotate_waiting(ax, "Waiting for /odom.")

    def _update_slip_angles(self, d):
        ax = self.ax_slip
        self._set_tracking_axis_common(
            ax, "B8 - Tire Slip Angles", "Slip angle (rad)")
        slip = d["slip"]
        if slip["t"]:
            n = min(len(slip["t"]), len(slip["alpha_f"]), len(slip["alpha_r"]))
            ax.plot(slip["t"][:n], slip["alpha_f"][:n], color=ORANGE,
                    linewidth=1.5, label="alpha_f")
            ax.plot(slip["t"][:n], slip["alpha_r"][:n], color=PURPLE,
                    linewidth=1.5, label="alpha_r")
            set_padded_ylim(ax, slip["alpha_f"][:n] + slip["alpha_r"][:n],
                            min_span=0.05)
            show_legend(ax)
        else:
            self._annotate_waiting(ax, "Waiting for /dynamic_slip_angles.")

    def _update_tire_forces(self, d):
        ax = self.ax_forces
        self._set_tracking_axis_common(
            ax, "B9 - Dynamic Tire Forces", "Lateral force (N)")
        forces = d["forces"]
        if forces["t"]:
            n = min(len(forces["t"]), len(forces["Fyf"]), len(forces["Fyr"]))
            ax.plot(forces["t"][:n], forces["Fyf"][:n], color=ORANGE,
                    linewidth=1.5, label="Fyf")
            ax.plot(forces["t"][:n], forces["Fyr"][:n], color=PURPLE,
                    linewidth=1.5, label="Fyr")
            set_padded_ylim(ax, forces["Fyf"][:n] + forces["Fyr"][:n],
                            min_span=500.0)
            show_legend(ax)
        else:
            self._annotate_waiting(ax, "Waiting for /dynamic_tire_forces.")

    def _update_mpc_cmd(self, d, diagnostics):
        ax = self.ax_cmd
        ax_delta = self.ax_cmd_delta
        ax.cla()
        ax_delta.cla()
        self._style_axis(ax)
        self._style_axis(ax_delta)
        ax.set_title("B10 - MPC Commands", fontsize=9, color="white")
        ax.set_xlabel("Time t (s)", fontsize=7)
        ax.set_ylabel("a_cmd (m/s²)", fontsize=7, color=GREEN)
        ax_delta.set_ylabel("delta_cmd (rad)", fontsize=7, color=ORANGE)
        ax.tick_params(axis="y", colors=GREEN, labelsize=6)
        ax_delta.tick_params(axis="y", colors=ORANGE, labelsize=6)
        ax.grid(True, alpha=0.15, color="white")

        cmd = d["cmd"]
        if cmd["t"]:
            n = min(len(cmd["t"]), len(cmd["a"]), len(cmd["delta"]))
            line_a, = ax.plot(cmd["t"][:n], cmd["a"][:n], color=GREEN,
                              linewidth=1.5, label="a_cmd")
            line_delta, = ax_delta.plot(cmd["t"][:n], cmd["delta"][:n],
                                        color=ORANGE, linewidth=1.5,
                                        label="delta_cmd")
            ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.35)
            ax_delta.axhline(0.0, color="white", linewidth=0.6, alpha=0.20)
            set_padded_ylim(ax, cmd["a"][:n], min_span=0.2)
            set_padded_ylim(ax_delta, cmd["delta"][:n], min_span=0.05)
            ax.legend([line_a, line_delta], ["a_cmd", "delta_cmd"],
                      fontsize=6, loc="best", facecolor="#3b3b4b",
                      labelcolor="white", framealpha=0.7)
            if recent_command_is_zero(cmd):
                self._annotate_warnings(
                    ax, ["/mpc_cmd is zero over the recent command window."],
                    max_lines=2)
        else:
            self._annotate_waiting(ax, "Waiting for /mpc_cmd.")
            self._annotate_warnings(ax, diagnostics["warnings"], max_lines=3)

    # ---- Main update function -------------------------------------------
    def _update(self, frame):
        d = self.node.snapshot()
        diagnostics = build_tracking_diagnostics(d)
        if diagnostics["tracking_ready"]:
            track = build_tracking_series(
                d["ref"], actual_odom_states(d), matching_mode=d["matching_mode"])
        else:
            track = {"available": False, "match_mode": d["matching_mode"]}
        self._log_professor_counts(d)
        self._update_plot1(d)
        self._update_plot2(d)
        self._update_plot3(d)
        self._update_plot4(d)
        self._update_plot5(d)
        self._update_professor_path(d, track, diagnostics)
        self._update_professor_zoom(d, diagnostics)
        self._update_professor_raw_ref_deviation(d)
        self._update_professor_curvature(d)
        self._update_professor_error(d, track, diagnostics)

        if self.show_debug_view:
            self._update_tracking_path(d, track, diagnostics)
            self._update_lateral_error(d, track, diagnostics)
            self._update_heading_error(d, track, diagnostics)
            self._update_velocity_comparison(d, track, diagnostics)
            self._update_velocity_error(d, track, diagnostics)
            self._update_yaw_rate_comparison(d, track, diagnostics)
            self._update_lateral_velocity(d, track)
            self._update_slip_angles(d)
            self._update_tire_forces(d)
            self._update_mpc_cmd(d, diagnostics)
            self.fig_debug.canvas.draw_idle()

        self.fig.canvas.draw_idle()
        self.fig_tracking.canvas.draw_idle()


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
