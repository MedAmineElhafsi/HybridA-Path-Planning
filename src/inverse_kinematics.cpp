#include "hybrid_astar_cpp/inverse_kinematics.hpp"
#include "hybrid_astar_cpp/curvature.hpp"

#include <cmath>
#include <algorithm>

namespace {

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
    constexpr double kMinSpeed = 0.1;
    const double v_avg = 0.5 * (std::abs(v_prev) + std::abs(v_curr));
    return ds / std::max(v_avg, kMinSpeed);
}

bool accelerationLocked(const std::vector<Pose2D>& path, size_t i) {
    return i == 0 || i + 1 == path.size() ||
           (i > 0 && path[i].direction != path[i - 1].direction) ||
           (i + 1 < path.size() && path[i].direction != path[i + 1].direction);
}

int normalizedWindow(int window) {
    if (window < 1) return 1;
    return (window % 2 == 0) ? window + 1 : window;
}

void applySteeringRateLimit(std::vector<WaypointKinematics>& result,
                            const std::vector<Pose2D>& path,
                            const std::vector<double>& dt,
                            double wheelbase,
                            double track_width,
                            double max_steering_rate) {
    if (result.size() < 2 || max_steering_rate <= 0.0) return;

    for (size_t i = 1; i < result.size(); ++i) {
        if (path[i].direction != path[i - 1].direction) continue;
        const double max_step = max_steering_rate * dt[i];
        result[i].delta_avg = std::clamp(
            result[i].delta_avg,
            result[i - 1].delta_avg - max_step,
            result[i - 1].delta_avg + max_step);
    }

    for (size_t i = result.size() - 1; i > 0; --i) {
        if (path[i].direction != path[i - 1].direction) continue;
        const double max_step = max_steering_rate * dt[i];
        result[i - 1].delta_avg = std::clamp(
            result[i - 1].delta_avg,
            result[i].delta_avg - max_step,
            result[i].delta_avg + max_step);
    }

    for (auto& wk : result) {
        const double steering_kappa = std::tan(wk.delta_avg) / std::max(1e-6, wheelbase);
        double limited_delta = 0.0;
        ackermannWheelAngles(steering_kappa, wheelbase, track_width,
                             limited_delta, wk.delta_left, wk.delta_right);
        wk.delta_avg = limited_delta;
    }
}

void smoothAccelerations(std::vector<WaypointKinematics>& result,
                         const std::vector<Pose2D>& path,
                         const std::vector<double>& dt,
                         int filter_window,
                         double jerk_limit) {
    const size_t n = result.size();
    if (n < 3) return;

    std::vector<double> raw(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        raw[i] = result[i].acceleration;
    }

    std::vector<double> filtered = raw;
    const int window = normalizedWindow(filter_window);
    if (window > 1) {
        const int half_window = window / 2;
        for (size_t i = 0; i < n; ++i) {
            if (accelerationLocked(path, i)) {
                filtered[i] = 0.0;
                continue;
            }

            const size_t lo = i > static_cast<size_t>(half_window)
                ? i - static_cast<size_t>(half_window) : 0;
            const size_t hi = std::min(n - 1, i + static_cast<size_t>(half_window));

            double sum = 0.0;
            size_t count = 0;
            for (size_t j = lo; j <= hi; ++j) {
                if (accelerationLocked(path, j) ||
                    path[j].direction != path[i].direction) {
                    continue;
                }
                sum += raw[j];
                ++count;
            }
            filtered[i] = (count > 0) ? sum / static_cast<double>(count) : 0.0;
        }
    }

    if (jerk_limit > 0.0) {
        for (size_t i = 1; i < n; ++i) {
            if (accelerationLocked(path, i) ||
                path[i].direction != path[i - 1].direction) {
                filtered[i] = 0.0;
                continue;
            }
            const double max_step = jerk_limit * dt[i];
            filtered[i] = std::clamp(
                filtered[i], filtered[i - 1] - max_step, filtered[i - 1] + max_step);
        }

        for (size_t i = n - 1; i > 0; --i) {
            if (accelerationLocked(path, i - 1) ||
                path[i].direction != path[i - 1].direction) {
                filtered[i - 1] = 0.0;
                continue;
            }
            const double max_step = jerk_limit * dt[i];
            filtered[i - 1] = std::clamp(
                filtered[i - 1], filtered[i] - max_step, filtered[i] + max_step);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        result[i].acceleration = accelerationLocked(path, i) ? 0.0 : filtered[i];
    }
}

}  // namespace

// ---------------------------------------------------------------------------

std::vector<WaypointKinematics> InverseKinematics::compute(
    const std::vector<Pose2D>& path,
    double wheelbase,
    double track_width,
    double curvature_resample_ds,
    int curvature_filter_window,
    double curvature_rate_limit,
    double steering_rate_limit,
    int acceleration_filter_window,
    double acceleration_jerk_limit)
{
    const size_t n = path.size();
    std::vector<WaypointKinematics> result(n);

    if (n <= 1) return result;

    // -----------------------------------------------------------------------
    // Pass 1: curvature + Ackermann wheel angles.
    //
    // Curvature follows the professor's kappa(s) sketch:
    //   smoothed path -> uniform arc-length resampling -> central-difference
    //   derivative curvature -> light moving-average filtering.
    //
    // The helper splits at forward/reverse direction changes, so derivatives
    // and filtering never cross cusps.  Positive curvature is a left turn,
    // negative curvature is a right turn.
    // -----------------------------------------------------------------------
    std::vector<double> dt(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        const double ds = std::hypot(path[i].x - path[i-1].x,
                                     path[i].y - path[i-1].y);
        dt[i] = segmentDt(path[i-1].velocity, path[i].velocity, ds);
    }

    const std::vector<double> curvature = PathCurvature::computeAtPathPoints(
        path, curvature_resample_ds, curvature_filter_window, curvature_rate_limit);

    for (size_t i = 0; i < n; ++i) {
        const double kappa = (i < curvature.size()) ? curvature[i] : 0.0;

        result[i].curvature = kappa;
        ackermannWheelAngles(kappa, wheelbase, track_width,
                             result[i].delta_avg,
                             result[i].delta_left,
                             result[i].delta_right);
    }

    // Bound steering slew after delta = atan(L * kappa).  This keeps the
    // feedforward steering sent to MPC physically plausible instead of jumping
    // instantly at curvature transitions.
    applySteeringRateLimit(
        result, path, dt, wheelbase, track_width, steering_rate_limit);

    // -----------------------------------------------------------------------
    // Pass 2: longitudinal acceleration — central finite difference Δv/Δt
    // -----------------------------------------------------------------------
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

    smoothAccelerations(
        result, path, dt, acceleration_filter_window, acceleration_jerk_limit);

    return result;
}
