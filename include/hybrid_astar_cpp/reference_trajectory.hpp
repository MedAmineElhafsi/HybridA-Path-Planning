#ifndef HYBRID_ASTAR_CPP_REFERENCE_TRAJECTORY_HPP_
#define HYBRID_ASTAR_CPP_REFERENCE_TRAJECTORY_HPP_

#include <cstddef>
#include <vector>

#include <std_msgs/msg/float64_multi_array.hpp>

#include "hybrid_astar_cpp/curves.hpp"
#include "hybrid_astar_cpp/inverse_kinematics.hpp"

namespace ReferenceTrajectory {

constexpr std::size_t kFieldCount = 8;

// One row of the professor-style reference trajectory:
// <x, y, yaw/phi, v_ref, t, curvature kappa, steering delta, acceleration a>.
struct Point {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double v_ref = 0.0;
    double t = 0.0;
    double kappa = 0.0;
    double delta = 0.0;
    double acceleration = 0.0;
};

// Cumulative time parameterisation from the index-aligned reference velocity.
// A small speed floor keeps timestamps finite at start/goal and cusp stops.
std::vector<double> computeTimeAxis(const std::vector<Pose2D>& path,
                                    double min_time_speed = 0.1);

std::vector<Point> assemble(const std::vector<Pose2D>& path,
                            const std::vector<WaypointKinematics>& kinematics,
                            double min_time_speed = 0.1);

// Float64MultiArray layout:
//   dim[0] = points, dim[1] = "x,y,yaw,v_ref,t,kappa,delta,a"
//   data rows are stored in row-major order with kFieldCount values per point.
std_msgs::msg::Float64MultiArray toMessage(const std::vector<Point>& trajectory);

}  // namespace ReferenceTrajectory

#endif  // HYBRID_ASTAR_CPP_REFERENCE_TRAJECTORY_HPP_
