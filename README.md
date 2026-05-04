# HybridA-Path-Planning

A ROS 2 (Jazzy) package for car-like vehicle path planning and closed-loop tracking.
The planner is a Hybrid A\* search with Dubins / Reeds–Shepp analytic shortcuts,
followed by a gradient-descent smoother. A separate MPC node tracks the smoothed
path, and a kinematic simulator closes the loop in RViz.

## Features

- **Hybrid A\* planner** (`src/hybrid_astar.cpp`)
  - 3D state lattice (x, y, yaw) with motion primitives
  - Dubins and Reeds–Shepp analytic expansions for goal shortcuts
  - Heuristic = `max(euclidean, BFS-on-obstacles)` for a tight admissible bound
  - Node pool + index-based parents (no per-node heap allocation)
- **Collision checking** (`src/grid_collision.cpp`)
  - Pre-computed footprint LUT over discretised yaw bins
  - BFS / Dijkstra distance maps cached and reused across plans
- **Path smoother** (`src/smoother.cpp`)
  - 4-term gradient: data fidelity, smoothness, obstacle clearance, Voronoi/medial-axis
  - Default 200 iterations (Habrador-style, tuned for real-time)
- **Velocity & steering profiles** — `velocity_profile.cpp`, `inverse_kinematics.cpp`
- **MPC controller** (`src/mpc_controller.cpp`, `src/mpc_node.cpp`)
  - Tracks the planner output and publishes `/mpc_cmd` (acceleration, steering)
- **Visualization & tooling**
  - RViz config in `rviz/planner.rviz`
  - Python GUI for scene editing (`scripts/gui_node.py`)
  - Vehicle / path visualizer, kinematic simulator, plotting helpers

## Repository layout

```
include/hybrid_astar_cpp/    # headers (planner, smoother, MPC, curves, ...)
src/                         # C++ implementations + node entry points
scripts/                     # Python helpers (GUI, simulator, visualizer)
launch/                      # planner_rviz, mpc_closed_loop
config/                      # planner.yaml — all ROS 2 parameters
rviz/                        # RViz layout
```

## Build

Drop the package into a ROS 2 Jazzy workspace and build with colcon:

```bash
cd ~/ros2_jazzy
colcon build --packages-select hybrid_astar_cpp
source install/setup.bash
```

Dependencies (declared in `package.xml`): `rclcpp`, `rclpy`, `nav_msgs`,
`geometry_msgs`, `visualization_msgs`, `tf2`, `tf2_geometry_msgs`, `tf2_ros`,
`eigen`, `rviz2`.

## Run

**Planner only (RViz scene + planner):**
```bash
ros2 launch hybrid_astar_cpp planner_rviz.launch.py
```

**Closed-loop (planner + MPC + kinematic simulator):**
```bash
ros2 launch hybrid_astar_cpp mpc_closed_loop.launch.py
```

In RViz: use the *2D Pose Estimate* tool to set the start, and *2D Goal Pose*
to set the goal. The GUI node (`scripts/gui_node.py`) lets you edit the
occupancy grid published on `/astar_grid`.

## Topics (closed-loop pipeline)

| Direction | Topic | Description |
|---|---|---|
| in  | `/astar_grid`     | occupancy grid scene |
| in  | `/initialpose`    | start pose (RViz) |
| in  | `/goal_pose`      | goal pose (RViz) |
| out | `/astar_path`     | smoothed path |
| out | `/astar_velocity_profile`, `/astar_steering`, `/astar_acceleration` | profiles |
| out | `/mpc_cmd`        | MPC control commands |
| out | `/odom` + TF      | from kinematic simulator |

## Configuration

All planner / smoother / MPC parameters live in `config/planner.yaml` and are
exposed as ROS 2 parameters (collision footprint, motion primitives, smoother
weights, MPC horizon, etc.).
