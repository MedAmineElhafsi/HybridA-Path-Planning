#!/usr/bin/env python3
"""Dynamic bicycle simulator for closed-loop MPC testing.

The simulator integrates a six-state dynamic bicycle plant:

    x_dot   = vx*cos(yaw) - vy*sin(yaw)
    y_dot   = vx*sin(yaw) + vy*cos(yaw)
    yaw_dot = r

    alpha_f = delta - atan2(vy + lf*r, max(abs(vx), vx_eps))
    alpha_r =        - atan2(vy - lr*r, max(abs(vx), vx_eps))

    Fyf = Cf*alpha_f
    Fyr = Cr*alpha_r

    vx_dot = a + vy*r
    vy_dot = (Fyf + Fyr)/m - vx*r
    r_dot  = (lf*Fyf - lr*Fyr)/Iz

using RK4 at a fixed integration rate. Inputs [a, delta] are read from
/mpc_cmd. The pose and dynamic body-frame velocity are published on /odom and
as a TF broadcast.

At very low longitudinal speed the tire model is blended with a stable
low-speed regularisation so the vehicle can start from rest and steer without
large slip-angle transients.
"""

import math
import threading
from typing import Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster


State = Tuple[float, float, float, float, float, float]
Deriv = Tuple[float, float, float, float, float, float]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0
    t = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def wrap_angle(yaw: float) -> float:
    return math.atan2(math.sin(yaw), math.cos(yaw))


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def yaw_from_quat(q: Quaternion) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class DynamicBicycleSimulator(Node):
    def __init__(self) -> None:
        super().__init__("dynamic_bicycle_simulator")

        self.declare_parameter("mass", 1500.0)
        self.declare_parameter("yaw_inertia", 2250.0)
        self.declare_parameter("lf", 1.2)
        self.declare_parameter("lr", 1.5)
        self.declare_parameter("Cf", 60000.0)
        self.declare_parameter("Cr", 60000.0)
        self.declare_parameter("vx_eps", 0.5)
        self.declare_parameter("slip_angle_limit", 0.5)
        self.declare_parameter("tire_force_limit", 8000.0)
        self.declare_parameter("a_min", -5.0)
        self.declare_parameter("a_max", 3.0)
        self.declare_parameter("delta_min", -0.6)
        self.declare_parameter("delta_max", 0.6)
        self.declare_parameter("vx_min", -5.0)
        self.declare_parameter("vx_max", 10.0)
        self.declare_parameter("vy_limit", 4.0)
        self.declare_parameter("yaw_rate_limit", 2.0)
        self.declare_parameter("integrate_rate_hz", 100.0)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("cmd_topic", "/mpc_cmd")
        self.declare_parameter("odom_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.m = max(1.0, float(self.get_parameter("mass").value))
        self.Iz = max(1.0, float(self.get_parameter("yaw_inertia").value))
        self.lf = max(1e-3, float(self.get_parameter("lf").value))
        self.lr = max(1e-3, float(self.get_parameter("lr").value))
        self.Cf = float(self.get_parameter("Cf").value)
        self.Cr = float(self.get_parameter("Cr").value)
        self.vx_eps = max(1e-3, float(self.get_parameter("vx_eps").value))
        self.slip_limit = abs(float(self.get_parameter("slip_angle_limit").value))
        self.force_limit = abs(float(self.get_parameter("tire_force_limit").value))
        self.a_min = float(self.get_parameter("a_min").value)
        self.a_max = float(self.get_parameter("a_max").value)
        self.delta_min = float(self.get_parameter("delta_min").value)
        self.delta_max = float(self.get_parameter("delta_max").value)
        self.vx_min = float(self.get_parameter("vx_min").value)
        self.vx_max = float(self.get_parameter("vx_max").value)
        self.vy_limit = abs(float(self.get_parameter("vy_limit").value))
        self.r_limit = abs(float(self.get_parameter("yaw_rate_limit").value))
        self.int_rate = max(1.0, float(self.get_parameter("integrate_rate_hz").value))
        self.pub_rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.wheelbase = self.lf + self.lr
        self.low_speed_tau = 0.35

        self.state_lock = threading.Lock()
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.r = 0.0
        self.a_cmd = 0.0
        self.delta_cmd = 0.0
        self.last_alpha_f = 0.0
        self.last_alpha_r = 0.0
        self.last_Fyf = 0.0
        self.last_Fyr = 0.0
        self.has_pose = False

        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.state_pub = self.create_publisher(Float64MultiArray, "/dynamic_vehicle_state", 20)
        self.force_pub = self.create_publisher(Float64MultiArray, "/dynamic_tire_forces", 20)
        self.slip_pub = self.create_publisher(Float64MultiArray, "/dynamic_slip_angles", 20)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                 self.initial_pose_cb, 10)
        self.create_subscription(Float64MultiArray, cmd_topic,
                                 self.cmd_cb, 20)
        self.create_subscription(Path, "/astar_path_raw",
                                 self.path_cb, 1)

        self.create_timer(1.0 / self.int_rate, self.integrate_step)
        self.create_timer(1.0 / self.pub_rate, self.publish_state)

        self.get_logger().info(
            "Dynamic bicycle simulator up: "
            f"m={self.m:.0f}kg Iz={self.Iz:.0f}kg*m^2 lf={self.lf:.2f}m "
            f"lr={self.lr:.2f}m Cf={self.Cf:.0f}N/rad Cr={self.Cr:.0f}N/rad "
            f"integrate={self.int_rate:.0f}Hz publish={self.pub_rate:.0f}Hz")

    # ---- Callbacks ----
    def initial_pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        with self.state_lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            self.yaw = yaw_from_quat(msg.pose.pose.orientation)
            self.vx = 0.0
            self.vy = 0.0
            self.r = 0.0
            self.a_cmd = 0.0
            self.delta_cmd = 0.0
            self.has_pose = True
        self.get_logger().info(
            f"Reset dynamic bicycle pose to ({self.x:.2f}, {self.y:.2f}, "
            f"{self.yaw:.2f} rad)")

    def path_cb(self, msg: Path) -> None:
        """Seed pose from path start if /initialpose was missed at launch."""
        with self.state_lock:
            if self.has_pose or not msg.poses:
                return
            p = msg.poses[0].pose
            self.x = p.position.x
            self.y = p.position.y
            self.yaw = yaw_from_quat(p.orientation)
            self.vx = 0.0
            self.vy = 0.0
            self.r = 0.0
            self.has_pose = True
        self.get_logger().info(
            f"Seeded dynamic bicycle pose from path start ({self.x:.2f}, {self.y:.2f})")

    def cmd_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            return
        a = clamp(float(msg.data[0]), self.a_min, self.a_max)
        delta = clamp(float(msg.data[1]), self.delta_min, self.delta_max)
        with self.state_lock:
            self.a_cmd = a
            self.delta_cmd = delta

    # ---- Dynamic bicycle model ----
    def _tire_response(self, vx: float, vy: float, r: float, delta: float):
        denom = max(abs(vx), self.vx_eps)
        alpha_f = delta - math.atan2(vy + self.lf * r, denom)
        alpha_r = -math.atan2(vy - self.lr * r, denom)
        alpha_f = clamp(alpha_f, -self.slip_limit, self.slip_limit)
        alpha_r = clamp(alpha_r, -self.slip_limit, self.slip_limit)

        Fyf = clamp(self.Cf * alpha_f, -self.force_limit, self.force_limit)
        Fyr = clamp(self.Cr * alpha_r, -self.force_limit, self.force_limit)
        return alpha_f, alpha_r, Fyf, Fyr

    def _f(self, state: State, a: float, delta: float) -> Deriv:
        x, y, yaw, vx, vy, r = state
        del x, y

        alpha_f, alpha_r, Fyf, Fyr = self._tire_response(vx, vy, r, delta)
        del alpha_f, alpha_r

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        dyn = (
            vx * cos_yaw - vy * sin_yaw,
            vx * sin_yaw + vy * cos_yaw,
            r,
            a + vy * r,
            (Fyf + Fyr) / self.m - vx * r,
            (self.lf * Fyf - self.lr * Fyr) / self.Iz,
        )

        target_r = vx * math.tan(delta) / max(1e-3, self.wheelbase)
        low = (
            vx * cos_yaw,
            vx * sin_yaw,
            r,
            a,
            -vy / self.low_speed_tau,
            (target_r - r) / self.low_speed_tau,
        )

        w = smoothstep(0.0, self.vx_eps, abs(vx))
        return tuple((1.0 - w) * lo + w * hi for lo, hi in zip(low, dyn))

    def integrate_step(self) -> None:
        dt = 1.0 / self.int_rate
        with self.state_lock:
            if not self.has_pose:
                return
            state = (self.x, self.y, self.yaw, self.vx, self.vy, self.r)
            a = self.a_cmd
            delta = self.delta_cmd

        k1 = self._f(state, a, delta)
        k2_state = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
        k2 = self._f(k2_state, a, delta)
        k3_state = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
        k3 = self._f(k3_state, a, delta)
        k4_state = tuple(s + dt * k for s, k in zip(state, k3))
        k4 = self._f(k4_state, a, delta)

        next_state = tuple(
            s + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            for s, a1, a2, a3, a4 in zip(state, k1, k2, k3, k4)
        )

        nx, ny, nyaw, nvx, nvy, nr = next_state
        nyaw = wrap_angle(nyaw)
        nvx = clamp(nvx, self.vx_min, self.vx_max)
        nvy = clamp(nvy, -self.vy_limit, self.vy_limit)
        nr = clamp(nr, -self.r_limit, self.r_limit)
        alpha_f, alpha_r, Fyf, Fyr = self._tire_response(nvx, nvy, nr, delta)

        with self.state_lock:
            self.x = nx
            self.y = ny
            self.yaw = nyaw
            self.vx = nvx
            self.vy = nvy
            self.r = nr
            self.last_alpha_f = alpha_f
            self.last_alpha_r = alpha_r
            self.last_Fyf = Fyf
            self.last_Fyr = Fyr

    # ---- Publishing ----
    def publish_state(self) -> None:
        with self.state_lock:
            if not self.has_pose:
                return
            x = self.x
            y = self.y
            yaw = self.yaw
            vx = self.vx
            vy = self.vy
            r = self.r
            Fyf = self.last_Fyf
            Fyr = self.last_Fyr
            alpha_f = self.last_alpha_f
            alpha_r = self.last_alpha_r

        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = yaw_to_quat(yaw)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = r
        self.odom_pub.publish(odom)

        vehicle_state = Float64MultiArray()
        vehicle_state.data = [x, y, yaw, vx, vy, r]
        self.state_pub.publish(vehicle_state)

        tire_forces = Float64MultiArray()
        tire_forces.data = [Fyf, Fyr]
        self.force_pub.publish(tire_forces)

        slip_angles = Float64MultiArray()
        slip_angles.data = [alpha_f, alpha_r]
        self.slip_pub.publish(slip_angles)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = yaw_to_quat(yaw)
        self.tf_broadcaster.sendTransform(tf_msg)


def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(DynamicBicycleSimulator())
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
