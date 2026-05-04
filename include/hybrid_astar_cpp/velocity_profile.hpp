#ifndef HYBRID_ASTAR_CPP_VELOCITY_PROFILE_HPP_
#define HYBRID_ASTAR_CPP_VELOCITY_PROFILE_HPP_

#include <vector>
#include "hybrid_astar_cpp/curves.hpp"

// Computes a curvature-aware, S-curve-smoothed velocity profile for a planned
// path.
//
// Algorithm (three passes):
//   1. Curvature pass  — limits speed at each waypoint by lateral-acceleration budget.
//   2. Forward pass    — limits acceleration from the previous waypoint.
//   3. Backward pass   — limits deceleration toward the next waypoint.
//   4. Jerk pass       — rounds acceleration/deceleration ramps into an
//                        S-curve-like profile.
//
// Curvature for the lateral-acceleration pass uses the same professor-style
// derivative kappa(s) pipeline as /astar_curvature: B-spline path, uniform
// arc-length spline fitting, analytic spline derivatives, Savitzky-Golay
// filtering, and bounded curvature rate.
//
// Direction changes (cusps) are treated as hard zero-velocity waypoints so the
// vehicle always comes to a full stop before reversing.  Start and goal are
// also pinned to zero.
//
// The resulting Pose2D::velocity is always non-negative; the sign convention is
// carried by Pose2D::direction so MPC/path-follower can read both independently.
class VelocityProfiler {
public:
    struct Params {
        double v_max          = 1.5;   // max forward speed (m/s)
        double v_max_reverse  = 0.5;   // max reverse speed (m/s)
        double a_max          = 1.0;   // max longitudinal acceleration (m/s²)
        double d_max          = 1.5;   // max longitudinal deceleration (m/s²)
        double j_max          = 1.2;   // max longitudinal jerk for S-curve smoothing (m/s³)
        double a_lat_max      = 1.5;   // max lateral acceleration (m/s²) — governs curvature limit
        double v_min_curv     = 0.1;   // floor speed used when curvature is near-zero (m/s)
    };

    // Fill Pose2D::velocity for every waypoint in-place.
    // safe_curvature_eps: curvature below this is treated as straight (avoids divide-by-zero).
    static void compute(std::vector<Pose2D>& path,
                        const Params& params,
                        double safe_curvature_eps = 1e-4,
                        double curvature_resample_ds = 0.05,
                        int curvature_filter_window = 5,
                        double curvature_rate_limit = 0.06);
};

#endif  // HYBRID_ASTAR_CPP_VELOCITY_PROFILE_HPP_
