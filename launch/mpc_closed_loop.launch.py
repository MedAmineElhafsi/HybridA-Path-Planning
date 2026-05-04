"""Closed-loop launch — planner + MPC + kinematic simulator + RViz.

Pipeline:
    GUI scene    ────►  /astar_grid
    RViz Pose    ────►  /initialpose  ────► (a) planner_node  (b) simulator
    RViz Goal    ────►  /goal_pose    ────►     planner_node
    planner_node ────►  /astar_path, /astar_velocity_profile,
                         /astar_steering, /astar_acceleration
    mpc_node     ────►  /mpc_cmd                     (a, δ commands)
    simulator    ────►  /odom + TF (closes the loop)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("hybrid_astar_cpp")
    default_rviz = os.path.join(share, "rviz", "planner.rviz")
    default_cfg  = os.path.join(share, "config", "planner.yaml")

    planner = Node(
        package="hybrid_astar_cpp", executable="planner_node",
        name="planner_node", output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    mpc = Node(
        package="hybrid_astar_cpp", executable="mpc_node",
        name="mpc_controller_node", output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    sim = Node(
        package="hybrid_astar_cpp", executable="kinematic_simulator.py",
        name="kinematic_simulator", output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    visualizer = Node(
        package="hybrid_astar_cpp", executable="path_vehicle_visualizer.py",
        name="path_vehicle_visualizer", output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),
        DeclareLaunchArgument("config",      default_value=default_cfg),
        planner,
        mpc,
        sim,
        visualizer,
        rviz,
    ])
