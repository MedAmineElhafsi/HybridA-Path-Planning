#include "hybrid_astar_cpp/inverse_kinematics.hpp"

#include <cmath>
#include <algorithm>

namespace {

// ---------------------------------------------------------------------------
// Signed Menger curvature at the middle point of three consecutive waypoints.
//
// Sign:  positive = left turn   (consistent with vehicle frame: x-forward, y-left)
//        negative = right turn
//
// Returns 0 if any segment is degenerate (near-zero length).
// ---------------------------------------------------------------------------
double signedMengerCurvature(double ax, double ay,
                              double bx, double by,
                              double cx, double cy)
{
    const double d1x = bx - ax,  d1y = by - ay;
    const double d2x = cx - bx,  d2y = cy - by;
    const double d3x = cx - ax,  d3y = cy - ay;

    const double len1 = std::hypot(d1x, d1y);
    const double len2 = std::hypot(d2x, d2y);
    const double len3 = std::hypot(d3x, d3y);

    constexpr double kEps = 1e-9;
    if (len1 < kEps || len2 < kEps || len3 < kEps) return 0.0;

    // z-component of d1 × d2 — positive = counter-clockwise = left turn
    const double cross = d1x * d2y - d1y * d2x;
    return 2.0 * cross / (len1 * len2 * len3);
}

// ---------------------------------------------------------------------------
// Ackermann per-wheel steering angles from signed curvature κ.
//
// The instantaneous centre of rotation (ICR) is at distance R = 1/κ from the
// rear-axle centre (positive = to the left of the vehicle).
//
// Ackermann condition:
//   δ_left  = atan(L / (R - w/2))
//   δ_right = atan(L / (R + w/2))
//
// Because atan is defined for all real arguments, both expressions are
// well-posed for all R except R = ±w/2 (ICR directly under a wheel — not
// reachable for properly constrained min_turning_radius ≥ w/2 + ε).
//
// When κ → 0 both angles → 0 continuously (straight line).
// ---------------------------------------------------------------------------
void ackermannWheelAngles(double kappa, double wheelbase, double track_width,
                           double& delta_avg,
                           double& delta_left,
                           double& delta_right)
{
    constexpr double kMinR = 1e-6;   // guard against κ = ±∞

    if (std::abs(kappa) < 1e-9) {
        // Straight line — all angles are zero
        delta_avg   = 0.0;
        delta_left  = 0.0;
        delta_right = 0.0;
        return;
    }

    const double R = 1.0 / kappa;   // signed turning radius

    // Bicycle-equivalent steering angle
    delta_avg = std::atan(wheelbase / R);

    // Half track width
    const double hw = 0.5 * track_width;

    // Guard: don't let R_left or R_right collapse to zero
    const double R_left  = R - hw;
    const double R_right = R + hw;

    delta_left  = (std::abs(R_left)  > kMinR) ? std::atan(wheelbase / R_left)  : std::copysign(M_PI_2, R_left);
    delta_right = (std::abs(R_right) > kMinR) ? std::atan(wheelbase / R_right) : std::copysign(M_PI_2, R_right);
}

// Estimate travel time between waypoints i-1 and i from their velocities.
double segmentDt(double v_prev, double v_curr, double ds)
{
    constexpr double kMinSpeed = 1e-4;
    const double v_avg = 0.5 * (std::abs(v_prev) + std::abs(v_curr));
    return ds / std::max(v_avg, kMinSpeed);
}

}  // namespace

// ---------------------------------------------------------------------------

std::vector<WaypointKinematics> InverseKinematics::compute(
    const std::vector<Pose2D>& path,
    double wheelbase,
    double track_width)
{
    const size_t n = path.size();
    std::vector<WaypointKinematics> result(n);

    if (n <= 1) return result;

    // -----------------------------------------------------------------------
    // Pass 1: curvature + Ackermann wheel angles
    // -----------------------------------------------------------------------
    for (size_t i = 0; i < n; ++i) {
        double kappa = 0.0;

        if (i == 0) {
            if (n >= 3) {
                kappa = signedMengerCurvature(
                    path[0].x, path[0].y,
                    path[1].x, path[1].y,
                    path[2].x, path[2].y);
            }
        } else if (i == n - 1) {
            if (n >= 3) {
                kappa = signedMengerCurvature(
                    path[n-3].x, path[n-3].y,
                    path[n-2].x, path[n-2].y,
                    path[n-1].x, path[n-1].y);
            }
        } else {
            // Zero curvature at cusps — vehicle is stopped and steering is undefined
            const bool cusp_before = (path[i].direction != path[i-1].direction);
            const bool cusp_after  = (path[i+1].direction != path[i].direction);
            if (cusp_before || cusp_after) {
                kappa = 0.0;
            } else {
                kappa = signedMengerCurvature(
                    path[i-1].x, path[i-1].y,
                    path[i  ].x, path[i  ].y,
                    path[i+1].x, path[i+1].y);
            }
        }

        result[i].curvature = kappa;
        ackermannWheelAngles(kappa, wheelbase, track_width,
                             result[i].delta_avg,
                             result[i].delta_left,
                             result[i].delta_right);
    }

    // -----------------------------------------------------------------------
    // Pass 2: longitudinal acceleration — central finite difference Δv/Δt
    // -----------------------------------------------------------------------
    std::vector<double> dt(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        const double ds = std::hypot(path[i].x - path[i-1].x,
                                     path[i].y - path[i-1].y);
        dt[i] = segmentDt(path[i-1].velocity, path[i].velocity, ds);
    }

    for (size_t i = 0; i < n; ++i) {
        const bool is_endpoint = (i == 0 || i == n - 1);
        const bool is_cusp =
            (i > 0     && path[i].direction != path[i-1].direction) ||
            (i + 1 < n && path[i].direction != path[i+1].direction);

        if (is_endpoint || is_cusp) {
            result[i].acceleration = 0.0;
            continue;
        }

        const double dt_total = dt[i] + dt[i+1];
        if (dt_total > 1e-9) {
            result[i].acceleration = (path[i+1].velocity - path[i-1].velocity) / dt_total;
        }
    }

    return result;
}
