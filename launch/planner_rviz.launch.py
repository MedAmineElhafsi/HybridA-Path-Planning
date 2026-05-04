from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    package_share = get_package_share_directory("hybrid_astar_cpp")
    default_rviz   = os.path.join(package_share, "rviz",   "planner.rviz")
    default_config = os.path.join(package_share, "config", "planner.yaml")

    planner_node = Node(
        package="hybrid_astar_cpp",
        executable="planner_node",
        name="planner_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    vehicle_visualizer = Node(
        package="hybrid_astar_cpp",
        executable="path_vehicle_visualizer.py",
        name="path_vehicle_visualizer",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            DeclareLaunchArgument("config",      default_value=default_config),
            planner_node,
            vehicle_visualizer,
            rviz_node,
        ]
    )
