cmake_minimum_required(VERSION 3.8)
project(hybrid_astar_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find all required ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# 2. Tell CMake where to find your header files
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

# 3. Create the executable and link all the .cpp files together
add_executable(planner_node
  src/main.cpp
  src/planner_node.cpp
  src/grid_collision.cpp
  src/curves.cpp
  src/hybrid_astar.cpp
  src/smoother.cpp
  src/velocity_profile.cpp
  src/inverse_kinematics.cpp
)

# 4. Link the ROS 2 dependencies to your executable
ament_target_dependencies(planner_node
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
)

# MPC controller executable
add_executable(mpc_node
  src/mpc_main.cpp
  src/mpc_node.cpp
  src/mpc_controller.cpp
)
ament_target_dependencies(mpc_node
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
)

# 5. Install the executables so `ros2 run` can find them
install(TARGETS
  planner_node
  mpc_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install the include directory (optional but good practice for C++ packages)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY
  launch
  rviz
  config
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
install(PROGRAMS
  scripts/gui_node.py
  scripts/path_vehicle_visualizer.py
  scripts/kinematic_simulator.py
  scripts/plot_results.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
