#!/usr/bin/env python3

import bisect
import math

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def interpolate_angle(a: float, b: float, t: float) -> float:
    delta = math.atan2(math.sin(b - a), math.cos(b - a))
    return a + t * delta


class PathVehicleVisualizer(Node):
    def __init__(self):
        super().__init__("path_vehicle_visualizer")

        self.declare_parameter("path_topic", "/astar_path")
        self.declare_parameter("marker_topic", "/astar_vehicle")
        self.declare_parameter("animation_speed", 1.2)
        self.declare_parameter("update_rate_hz", 30.0)
        self.declare_parameter("vehicle_length", 0.9)
        self.declare_parameter("vehicle_width", 0.45)
        self.declare_parameter("vehicle_height", 0.12)
        self.declare_parameter("cabin_length", 0.28)
        self.declare_parameter("cabin_width", 0.30)
        self.declare_parameter("cabin_height", 0.10)
        self.declare_parameter("cabin_forward_offset", 0.12)
        self.declare_parameter("headlight_spacing", 0.22)
        self.declare_parameter("headlight_radius", 0.05)
        self.declare_parameter("light_beam_length", 1.3)
        self.declare_parameter("light_beam_width", 0.9)

        path_topic = self.get_parameter("path_topic").value
        marker_topic = self.get_parameter("marker_topic").value
        update_rate_hz = max(1.0, float(self.get_parameter("update_rate_hz").value))

        self.vehicle_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.path_sub = self.create_subscription(Path, path_topic, self.path_callback, 10)
        self.vel_sub  = self.create_subscription(
            Float64MultiArray, '/astar_velocity_profile',
            self.velocity_callback, 10)
        self.timer = self.create_timer(1.0 / update_rate_hz, self.timer_callback)

        self.path_points = []
        self.path_distances = []
        self.path_times = []          # cumulative time at each waypoint (velocity-profile driven)
        self.path_frame = "map"
        self.path_start_time = None
        self.last_path_signature = None
        self._latest_velocities: list = []   # most recent velocity profile (raw)

    def velocity_callback(self, msg: Float64MultiArray):
        self._latest_velocities = list(msg.data)

    def path_callback(self, msg: Path):
        self.path_frame = msg.header.frame_id or "map"
        if not msg.poses:
            self.path_points = []
            self.path_distances = []
            self.path_times = []
            self.path_start_time = None
            self.last_path_signature = None
            self.publish_delete_all()
            return

        points = []
        cumulative = [0.0]
        total = 0.0
        signature = []
        for i, pose_stamped in enumerate(msg.poses):
            pose = pose_stamped.pose
            x = float(pose.position.x)
            y = float(pose.position.y)
            yaw = yaw_from_quaternion(pose.orientation)
            points.append((x, y, yaw))
            signature.append((round(x, 3), round(y, 3), round(yaw, 3)))
            if i > 0:
                dx = x - points[i - 1][0]
                dy = y - points[i - 1][1]
                total += math.hypot(dx, dy)
                cumulative.append(total)

        signature = tuple(signature)
        if signature == self.last_path_signature:
            return

        self.path_points    = points
        self.path_distances = cumulative
        self.path_start_time = self.get_clock().now()
        self.last_path_signature = signature

        # Build cumulative time array from velocity profile.
        # For each segment [i, i+1]: dt = ds / v_avg  (trapezoidal rule).
        # If velocities are unavailable or mismatched, fall back to constant speed.
        vels = self._latest_velocities
        if len(vels) == len(points):
            times = [0.0]
            for i in range(len(points) - 1):
                ds    = cumulative[i + 1] - cumulative[i]
                v_avg = 0.5 * (abs(vels[i]) + abs(vels[i + 1]))
                dt    = ds / v_avg if v_avg > 1e-3 else ds / 0.05
                times.append(times[-1] + dt)
            self.path_times = times
        else:
            self.path_times = []

    def timer_callback(self):
        if not self.path_points or self.path_start_time is None:
            return

        elapsed = (self.get_clock().now() - self.path_start_time).nanoseconds / 1e9

        if self.path_times and len(self.path_times) == len(self.path_points):
            # Velocity-profile driven: sample pose by elapsed wall-clock time.
            t = min(elapsed, self.path_times[-1])
            x, y, yaw = self.sample_pose_by_time(t)
        else:
            # Fallback: constant animation speed (no velocity profile yet).
            speed = max(0.05, float(self.get_parameter("animation_speed").value))
            target_distance = min(self.path_distances[-1], elapsed * speed)
            x, y, yaw = self.sample_pose(target_distance)

        marker_array = self.build_vehicle_markers(x, y, yaw)
        self.vehicle_pub.publish(marker_array)

    def sample_pose(self, distance: float):
        if len(self.path_points) == 1 or self.path_distances[-1] <= 1e-6:
            return self.path_points[-1]

        idx = bisect.bisect_right(self.path_distances, distance) - 1
        idx = max(0, min(idx, len(self.path_points) - 2))

        d0 = self.path_distances[idx]
        d1 = self.path_distances[idx + 1]
        p0 = self.path_points[idx]
        p1 = self.path_points[idx + 1]

        if d1 <= d0 + 1e-9:
            return p1

        ratio = (distance - d0) / (d1 - d0)
        x = p0[0] + ratio * (p1[0] - p0[0])
        y = p0[1] + ratio * (p1[1] - p0[1])
        yaw = interpolate_angle(p0[2], p1[2], ratio)
        return x, y, yaw

    def sample_pose_by_time(self, t: float):
        """Interpolate pose using cumulative time array (velocity-profile driven)."""
        if len(self.path_points) == 1 or self.path_times[-1] <= 1e-6:
            return self.path_points[-1]

        idx = bisect.bisect_right(self.path_times, t) - 1
        idx = max(0, min(idx, len(self.path_points) - 2))

        t0 = self.path_times[idx]
        t1 = self.path_times[idx + 1]
        p0 = self.path_points[idx]
        p1 = self.path_points[idx + 1]

        if t1 <= t0 + 1e-9:
            return p1

        ratio = (t - t0) / (t1 - t0)
        x   = p0[0] + ratio * (p1[0] - p0[0])
        y   = p0[1] + ratio * (p1[1] - p0[1])
        yaw = interpolate_angle(p0[2], p1[2], ratio)
        return x, y, yaw

    def build_vehicle_markers(self, x: float, y: float, yaw: float) -> MarkerArray:
        length = float(self.get_parameter("vehicle_length").value)
        width = float(self.get_parameter("vehicle_width").value)
        height = float(self.get_parameter("vehicle_height").value)
        cabin_length = float(self.get_parameter("cabin_length").value)
        cabin_width = float(self.get_parameter("cabin_width").value)
        cabin_height = float(self.get_parameter("cabin_height").value)
        cabin_forward_offset = float(self.get_parameter("cabin_forward_offset").value)
        headlight_spacing = float(self.get_parameter("headlight_spacing").value)
        headlight_radius = float(self.get_parameter("headlight_radius").value)
        light_beam_length = float(self.get_parameter("light_beam_length").value)
        light_beam_width = float(self.get_parameter("light_beam_width").value)

        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        lateral_x = -math.sin(yaw)
        lateral_y = math.cos(yaw)

        front_center_x = x + forward_x * (length * 0.5)
        front_center_y = y + forward_y * (length * 0.5)

        left_light = Point(
            x=front_center_x + lateral_x * (headlight_spacing * 0.5),
            y=front_center_y + lateral_y * (headlight_spacing * 0.5),
            z=height * 0.75,
        )
        right_light = Point(
            x=front_center_x - lateral_x * (headlight_spacing * 0.5),
            y=front_center_y - lateral_y * (headlight_spacing * 0.5),
            z=height * 0.75,
        )

        beam_tip_center_x = front_center_x + forward_x * light_beam_length
        beam_tip_center_y = front_center_y + forward_y * light_beam_length
        beam_left = Point(
            x=beam_tip_center_x + lateral_x * (light_beam_width * 0.5),
            y=beam_tip_center_y + lateral_y * (light_beam_width * 0.5),
            z=0.02,
        )
        beam_right = Point(
            x=beam_tip_center_x - lateral_x * (light_beam_width * 0.5),
            y=beam_tip_center_y - lateral_y * (light_beam_width * 0.5),
            z=0.02,
        )

        header_frame = self.path_frame or "map"
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        body = Marker()
        body.header.frame_id = header_frame
        body.header.stamp = stamp
        body.ns = "animated_vehicle"
        body.id = 0
        body.type = Marker.CUBE
        body.action = Marker.ADD
        body.pose.position.x = x
        body.pose.position.y = y
        body.pose.position.z = height * 0.5
        body.pose.orientation.z = math.sin(yaw * 0.5)
        body.pose.orientation.w = math.cos(yaw * 0.5)
        body.scale.x = length
        body.scale.y = width
        body.scale.z = height
        body.color.r = 0.09
        body.color.g = 0.45
        body.color.b = 0.92
        body.color.a = 0.95
        markers.markers.append(body)

        cabin = Marker()
        cabin.header.frame_id = header_frame
        cabin.header.stamp = stamp
        cabin.ns = "animated_vehicle"
        cabin.id = 1
        cabin.type = Marker.CUBE
        cabin.action = Marker.ADD
        cabin.pose.position.x = x + forward_x * cabin_forward_offset
        cabin.pose.position.y = y + forward_y * cabin_forward_offset
        cabin.pose.position.z = height + (cabin_height * 0.5)
        cabin.pose.orientation = body.pose.orientation
        cabin.scale.x = cabin_length
        cabin.scale.y = cabin_width
        cabin.scale.z = cabin_height
        cabin.color.r = 0.85
        cabin.color.g = 0.92
        cabin.color.b = 1.0
        cabin.color.a = 0.95
        markers.markers.append(cabin)

        for marker_id, point in ((2, left_light), (3, right_light)):
            headlight = Marker()
            headlight.header.frame_id = header_frame
            headlight.header.stamp = stamp
            headlight.ns = "animated_vehicle"
            headlight.id = marker_id
            headlight.type = Marker.SPHERE
            headlight.action = Marker.ADD
            headlight.pose.position = point
            headlight.pose.orientation.w = 1.0
            headlight.scale.x = headlight_radius * 2.0
            headlight.scale.y = headlight_radius * 2.0
            headlight.scale.z = headlight_radius * 2.0
            headlight.color.r = 1.0
            headlight.color.g = 0.97
            headlight.color.b = 0.76
            headlight.color.a = 0.95
            markers.markers.append(headlight)

        beam = Marker()
        beam.header.frame_id = header_frame
        beam.header.stamp = stamp
        beam.ns = "animated_vehicle"
        beam.id = 4
        beam.type = Marker.TRIANGLE_LIST
        beam.action = Marker.ADD
        beam.pose.orientation.w = 1.0
        beam.scale.x = 1.0
        beam.scale.y = 1.0
        beam.scale.z = 1.0
        beam.color.r = 1.0
        beam.color.g = 0.95
        beam.color.b = 0.55
        beam.color.a = 0.20
        beam.points = [
            left_light,
            right_light,
            beam_left,
            right_light,
            beam_right,
            beam_left,
        ]
        markers.markers.append(beam)

        return markers

    def publish_delete_all(self):
        marker = Marker()
        marker.header.frame_id = self.path_frame or "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        array = MarkerArray()
        array.markers.append(marker)
        self.vehicle_pub.publish(array)


def main():
    rclpy.init()
    node = PathVehicleVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.publish_delete_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
