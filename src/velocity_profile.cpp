#include "hybrid_astar_cpp/velocity_profile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// Arc-length between consecutive waypoints (Euclidean chord length).
double arcLength(const Pose2D& a, const Pose2D& b) {
    return std::hypot(b.x - a.x, b.y - a.y);
}

// Menger curvature at the middle point of three consecutive waypoints.
// Returns 0 when the three points are collinear or too close together.
double mengerCurvature(const Pose2D& p0, const Pose2D& p1, const Pose2D& p2) {
    const double ax = p1.x - p0.x,  ay = p1.y - p0.y;
    const double bx = p2.x - p1.x,  by = p2.y - p1.y;
    const double cx = p2.x - p0.x,  cy = p2.y - p0.y;

    const double cross = ax * by - ay * bx;
    const double a = std::hypot(ax, ay);
    const double b = std::hypot(bx, by);
    const double c = std::hypot(cx, cy);

    const double denom = a * b * c;
    if (denom < 1e-12) return 0.0;
    return 2.0 * std::abs(cross) / denom;
}

// Speed cap imposed by lateral-acceleration budget at a given curvature.
double curvatureSpeedLimit(double curvature, double a_lat_max,
                           double v_min, double v_max) {
    if (curvature < 1e-9) return v_max;
    return std::clamp(std::sqrt(a_lat_max / curvature), v_min, v_max);
}

// True when direction changes between waypoint i-1 and i (a cusp).
bool isCusp(const std::vector<Pose2D>& path, size_t i) {
    return i > 0 && path[i].direction != path[i - 1].direction;
}

}  // namespace

void VelocityProfiler::compute(std::vector<Pose2D>& path,
                                const Params& params,
                                double safe_curvature_eps) {
    const size_t n = path.size();
    if (n == 0) return;
    if (n == 1) { path[0].velocity = 0.0; return; }

    // -----------------------------------------------------------------------
    // Pass 1 — curvature ceiling
    //   v_ceil[i] = max speed allowed at waypoint i by lateral-accel budget.
    //   Start, end, and cusp waypoints are hard zeros.
    // -----------------------------------------------------------------------
    std::vector<double> v_ceil(n, 0.0);

    for (size_t i = 1; i + 1 < n; ++i) {
        if (isCusp(path, i)) {
            v_ceil[i] = 0.0;
            continue;
        }
        const double v_max_i = (path[i].direction < 0)
            ? params.v_max_reverse : params.v_max;

        const double k = mengerCurvature(path[i - 1], path[i], path[i + 1]);
        v_ceil[i] = (k < safe_curvature_eps)
            ? v_max_i
            : curvatureSpeedLimit(k, params.a_lat_max, params.v_min_curv, v_max_i);
    }
    // v_ceil[0] = 0 and v_ceil[n-1] = 0 from initialisation — vehicle starts
    // and stops at rest.

    // -----------------------------------------------------------------------
    // Pass 2 — forward pass (acceleration)
    //   v[i] = min(v_ceil[i],  sqrt(v[i-1]^2 + 2*a_max*ds))
    //
    //   At a cusp the vehicle must stop: v[cusp] = 0, and the segment after
    //   the cusp restarts acceleration from zero because v[cusp] = 0.
    // -----------------------------------------------------------------------
    std::vector<double> v(n, 0.0);  // v[0] = 0 (starts at rest)

    for (size_t i = 1; i < n; ++i) {
        if (isCusp(path, i)) {
            v[i] = 0.0;   // full stop at every direction change
            continue;
        }
        const double ds      = arcLength(path[i - 1], path[i]);
        const double v_accel = std::sqrt(
            std::max(0.0, v[i - 1] * v[i - 1] + 2.0 * params.a_max * ds));
        v[i] = std::min(v_ceil[i], v_accel);
    }

    // -----------------------------------------------------------------------
    // Pass 3 — backward pass (deceleration)
    //   v[i] = min(v[i],  sqrt(v[i+1]^2 + 2*d_max*ds))
    //
    //   No special cusp handling needed here: cusps and the final endpoint
    //   already have v = 0 from pass 2, so the sqrt formula naturally
    //   propagates the deceleration ramp backward through every preceding
    //   waypoint — exactly like a car braking to a stop.
    // -----------------------------------------------------------------------
    // v[n-1] = 0 from pass 2 (v_ceil[n-1] = 0).
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        const size_t ui = static_cast<size_t>(i);
        const double ds      = arcLength(path[ui], path[ui + 1]);
        const double v_decel = std::sqrt(
            std::max(0.0, v[ui + 1] * v[ui + 1] + 2.0 * params.d_max * ds));
        v[ui] = std::min(v[ui], v_decel);
    }

    // Write back — always non-negative (sign carried by direction field).
    for (size_t i = 0; i < n; ++i) {
        path[i].velocity = std::max(0.0, v[i]);
    }
}
