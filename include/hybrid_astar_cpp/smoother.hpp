#ifndef HYBRID_ASTAR_CPP_SMOOTHER_HPP_
#define HYBRID_ASTAR_CPP_SMOOTHER_HPP_

#include <vector>
#include <memory>
#include "hybrid_astar_cpp/curves.hpp"
#include "hybrid_astar_cpp/grid_collision.hpp"

// Cubic B-Spline path smoother.
//
// Fits a uniform cubic B-spline through the waypoints and blends each interior
// point toward the spline by `blend` (0 = no change, 1 = pure spline).
// The result is C2-continuous (smooth curvature), which is required for MPC.
//
// Locked points (start, goal proximity zone, cusp guards) are never moved.
// Each candidate section is collision-checked before being accepted.
class PathSmoother {
public:
    PathSmoother(double goal_lock_distance = 2.5,
                 int    cusp_guard_points  = 3);

    void smoothPath(std::vector<Pose2D>& path,
                    std::shared_ptr<GridCollision> collision_checker,
                    double blend = 0.65,
                    double max_point_shift = 0.0);

private:
    double goal_lock_distance_;
    int    cusp_guard_points_;
};

#endif  // HYBRID_ASTAR_CPP_SMOOTHER_HPP_
