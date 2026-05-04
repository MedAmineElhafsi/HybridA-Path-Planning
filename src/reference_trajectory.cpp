#include "hybrid_astar_cpp/reference_trajectory.hpp"

#include <algorithm>
#include <cmath>

namespace {

double distance2D(const Pose2D& a, const Pose2D& b) {
    return std::hypot(b.x - a.x, b.y - a.y);
}

}  // namespace

namespace ReferenceTrajectory {

std::vector<double> computeTimeAxis(const std::vector<Pose2D>& path,
                                    double min_time_speed) {
    std::vector<double> time(path.size(), 0.0);
    if (path.size() < 2) return time;

    const double speed_floor = std::max(1e-3, min_time_speed);
    for (size_t i = 1; i < path.size(); ++i) {
        const double ds = distance2D(path[i - 1], path[i]);
        const double v_prev = std::abs(path[i - 1].velocity);
        const double v_curr = std::abs(path[i].velocity);
        const double v_ref = std::max(speed_floor, 0.5 * (v_prev + v_curr));
        time[i] = time[i - 1] + ds / v_ref;
    }

    return time;
}

std::vector<Point> assemble(const std::vector<Pose2D>& path,
                            const std::vector<WaypointKinematics>& kinematics,
                            double min_time_speed) {
    const std::vector<double> time = computeTimeAxis(path, min_time_speed);
    std::vector<Point> trajectory;
    trajectory.reserve(path.size());

    for (size_t i = 0; i < path.size(); ++i) {
        Point p;
        p.x = path[i].x;
        p.y = path[i].y;
        p.yaw = path[i].yaw;
        p.v_ref = path[i].velocity;
        p.t = (i < time.size()) ? time[i] : 0.0;

        if (i < kinematics.size()) {
            p.kappa = kinematics[i].curvature;
            p.delta = kinematics[i].delta_avg;
            p.acceleration = kinematics[i].acceleration;
        }

        trajectory.push_back(p);
    }

    return trajectory;
}

std_msgs::msg::Float64MultiArray toMessage(const std::vector<Point>& trajectory) {
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(2);
    msg.layout.dim[0].label = "points";
    msg.layout.dim[0].size = trajectory.size();
    msg.layout.dim[0].stride = trajectory.size() * kFieldCount;
    msg.layout.dim[1].label = "x,y,yaw,v_ref,t,kappa,delta,a";
    msg.layout.dim[1].size = kFieldCount;
    msg.layout.dim[1].stride = kFieldCount;
    msg.layout.data_offset = 0;

    msg.data.reserve(trajectory.size() * kFieldCount);
    for (const auto& p : trajectory) {
        msg.data.push_back(p.x);
        msg.data.push_back(p.y);
        msg.data.push_back(p.yaw);
        msg.data.push_back(p.v_ref);
        msg.data.push_back(p.t);
        msg.data.push_back(p.kappa);
        msg.data.push_back(p.delta);
        msg.data.push_back(p.acceleration);
    }

    return msg;
}

}  // namespace ReferenceTrajectory
