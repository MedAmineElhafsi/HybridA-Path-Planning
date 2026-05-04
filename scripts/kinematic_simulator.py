#!/usr/bin/env python3
"""Ackermann kinematic simulator for closed-loop MPC testing.

Integrates the continuous Ackermann bicycle model:

    ẋ = v·cos(θ)
    ẏ = v·sin(θ)
    θ̇ = (v/L)·tan(δ)
    v̇ = a

using an RK4 step at fixed rate.  Inputs [a, δ] are read from /mpc_cmd.
The pose and velocity are published on /odom and as a TF broadcast.

Also seeds the initial pose from /initialpose (RViz "2D Pose Estimate")
so the same GUI flow used by the planner also spawns the simulator.
"""

import math
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def yaw_from_quat(q: Quaternion) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class KinematicSimulator(Node):
    def __init__(self) -> None:
        super().__init__("kinematic_simulator")

        self.declare_parameter("wheelbase", 2.7)
        self.declare_parameter("integrate_rate_hz", 100.0)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("odom_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("cmd_topic", "/mpc_cmd")
        self.declare_parameter("a_min", -5.0)
        self.declare_parameter("a_max", 3.0)
        self.declare_parameter("delta_min", -0.6)
        self.declare_parameter("delta_max", 0.6)
        self.declare_parameter("v_max", 10.0)
        self.declare_parameter("v_min", -5.0)

        self.L = float(self.get_parameter("wheelbase").value)
        self.int_rate = float(self.get_parameter("integrate_rate_hz").value)
        self.pub_rate = float(self.get_parameter("publish_rate_hz").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.a_min = float(self.get_parameter("a_min").value)
        self.a_max = float(self.get_parameter("a_max").value)
        self.delta_min = float(self.get_parameter("delta_min").value)
        self.delta_max = float(self.get_parameter("delta_max").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.v_min = float(self.get_parameter("v_min").value)

        # State
        self.state_lock = threading.Lock()
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.a_cmd = 0.0
        self.delta_cmd = 0.0
        self.has_pose = False

        # ROS I/O
        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                 self.initial_pose_cb, 10)
        self.create_subscription(Float64MultiArray, cmd_topic,
                                 self.cmd_cb, 20)
        # Fallback: seed pose from path start when /initialpose was missed
        self.create_subscription(Path, "/astar_path_raw",
                                 self.path_cb, 1)

        self.create_timer(1.0 / self.int_rate, self.integrate_step)
        self.create_timer(1.0 / self.pub_rate, self.publish_state)

        self.get_logger().info(
            f"Kinematic simulator up: L={self.L} m, integrate={self.int_rate:.0f}Hz, "
            f"publish={self.pub_rate:.0f}Hz")

    # ---- Callbacks ----
    def initial_pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        with self.state_lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            self.yaw = yaw_from_quat(msg.pose.pose.orientation)
            self.v = 0.0
            self.has_pose = True
        self.get_logger().info(
            f"Reset pose to ({self.x:.2f}, {self.y:.2f}, {self.yaw:.2f} rad)")

    def path_cb(self, msg: Path) -> None:
        """Seed pose from path start if /initialpose was missed at launch."""
        with self.state_lock:
            if self.has_pose or not msg.poses:
                return
            p = msg.poses[0].pose
            self.x   = p.position.x
            self.y   = p.position.y
            self.yaw = yaw_from_quat(p.orientation)
            self.v   = 0.0
            self.has_pose = True
        self.get_logger().info(
            f"Seeded pose from path start ({self.x:.2f}, {self.y:.2f})")

    def cmd_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            return
        a = max(self.a_min, min(self.a_max, float(msg.data[0])))
        d = max(self.delta_min, min(self.delta_max, float(msg.data[1])))
        with self.state_lock:
            self.a_cmd = a
            self.delta_cmd = d

    # ---- Dynamics ----
    def _f(self, x, y, yaw, v, a, delta):
        """Ackermann kinematic continuous derivatives."""
        return (v * math.cos(yaw),
                v * math.sin(yaw),
                (v / self.L) * math.tan(delta),
                a)

    def integrate_step(self) -> None:
        dt = 1.0 / self.int_rate
        with self.state_lock:
            if not self.has_pose:
                return
            x, y, yaw, v = self.x, self.y, self.yaw, self.v
            a, d = self.a_cmd, self.delta_cmd

        # RK4 on (x, y, yaw, v)
        k1 = self._f(x, y, yaw, v, a, d)
        k2 = self._f(x + 0.5*dt*k1[0], y + 0.5*dt*k1[1],
                     yaw + 0.5*dt*k1[2], v + 0.5*dt*k1[3], a, d)
        k3 = self._f(x + 0.5*dt*k2[0], y + 0.5*dt*k2[1],
                     yaw + 0.5*dt*k2[2], v + 0.5*dt*k2[3], a, d)
        k4 = self._f(x + dt*k3[0], y + dt*k3[1],
                     yaw + dt*k3[2], v + dt*k3[3], a, d)

        nx = x + dt/6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        ny = y + dt/6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        nyaw = yaw + dt/6.0 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        nv = v + dt/6.0 * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

        # Clamp velocity (safety)
        nv = max(self.v_min, min(self.v_max, nv))
        nyaw = math.atan2(math.sin(nyaw), math.cos(nyaw))

        with self.state_lock:
            self.x, self.y, self.yaw, self.v = nx, ny, nyaw, nv

    # ---- Publishing ----
    def publish_state(self) -> None:
        with self.state_lock:
            if not self.has_pose:
                return
            x, y, yaw, v = self.x, self.y, self.yaw, self.v
            d = self.delta_cmd

        stamp = self.get_clock().now().to_msg()

        # /odom
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = yaw_to_quat(yaw)
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = v * math.tan(d) / self.L
        self.odom_pub.publish(odom)

        # TF
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
        rclpy.spin(KinematicSimulator())
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
