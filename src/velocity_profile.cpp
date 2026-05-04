#include "hybrid_astar_cpp/velocity_profile.hpp"
#include "hybrid_astar_cpp/curvature.hpp"

#include <algorithm>
#include <cmath>

namespace {

// Arc-length between consecutive waypoints (Euclidean chord length).
double arcLength(const Pose2D& a, const Pose2D& b) {
    return std::hypot(b.x - a.x, b.y - a.y);
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

double segmentDt(double v_prev, double v_curr, double ds) {
    constexpr double kMinTimeSpeed = 0.1;
    const double v_avg = 0.5 * (std::abs(v_prev) + std::abs(v_curr));
    return ds / std::max(kMinTimeSpeed, v_avg);
}

std::vector<double> arcLengths(const std::vector<Pose2D>& path) {
    std::vector<double> s(path.size(), 0.0);
    for (size_t i = 1; i < path.size(); ++i) {
        s[i] = s[i - 1] + arcLength(path[i - 1], path[i]);
    }
    return s;
}

bool segmentBoundaryAfter(const std::vector<Pose2D>& path, size_t i) {
    return i + 1 < path.size() && path[i + 1].direction != path[i].direction;
}

std::vector<double> zeroPhaseArcLengthLowPass(
    const std::vector<Pose2D>& path,
    const std::vector<double>& s,
    const std::vector<double>& input,
    double smoothing_length) {
    const size_t n = input.size();
    if (n < 3 || path.size() != n || s.size() != n || smoothing_length <= 1e-9) {
        return input;
    }

    std::vector<double> output = input;
    size_t run_start = 0;
    while (run_start < n) {
        size_t run_end = run_start;
        while (run_end + 1 < n &&
               path[run_end + 1].direction == path[run_start].direction) {
            ++run_end;
        }

        if (run_end <= run_start + 1) {
            run_start = run_end + 1;
            continue;
        }

        std::vector<double> forward(run_end - run_start + 1, 0.0);
        forward.front() = input[run_start];
        for (size_t i = run_start + 1; i <= run_end; ++i) {
            const double ds = std::max(0.0, s[i] - s[i - 1]);
            const double alpha = ds / (smoothing_length + ds);
            const size_t local_i = i - run_start;
            forward[local_i] = forward[local_i - 1] +
                alpha * (input[i] - forward[local_i - 1]);
        }

        std::vector<double> backward(forward.size(), 0.0);
        backward.back() = input[run_end];
        for (size_t i = run_end; i > run_start; --i) {
            const double ds = std::max(0.0, s[i] - s[i - 1]);
            const double alpha = ds / (smoothing_length + ds);
            const size_t local_i = i - run_start;
            backward[local_i - 1] = backward[local_i] +
                alpha * (forward[local_i - 1] - backward[local_i]);
        }

        for (size_t i = run_start; i <= run_end; ++i) {
            output[i] = backward[i - run_start];
        }

        run_start = run_end + 1;
    }

    return output;
}

std::vector<double> smoothCurvatureForSpeedCeiling(
    const std::vector<Pose2D>& path,
    const std::vector<double>& curvature,
    double smoothing_length) {
    std::vector<double> abs_curvature(curvature.size(), 0.0);
    for (size_t i = 0; i < curvature.size(); ++i) {
        abs_curvature[i] = std::abs(curvature[i]);
    }

    const std::vector<double> s = arcLengths(path);
    const std::vector<double> filtered = zeroPhaseArcLengthLowPass(
        path, s, abs_curvature, smoothing_length);

    // Conservative smoothing: never reduce the local curvature used for the
    // lateral-acceleration cap.  This broadens speed reductions around curves
    // instead of letting v_ref react to one waypoint at a time.
    std::vector<double> safe_curvature = abs_curvature;
    const size_t n = std::min(abs_curvature.size(), filtered.size());
    for (size_t i = 0; i < n; ++i) {
        safe_curvature[i] = std::max(abs_curvature[i], filtered[i]);
    }
    return safe_curvature;
}

std::vector<double> smoothSpeedCeilingForPreview(
    const std::vector<Pose2D>& path,
    const std::vector<double>& raw_ceiling,
    double smoothing_length) {
    const std::vector<double> s = arcLengths(path);
    const std::vector<double> filtered = zeroPhaseArcLengthLowPass(
        path, s, raw_ceiling, smoothing_length);

    std::vector<double> ceiling = raw_ceiling;
    const size_t n = std::min(raw_ceiling.size(), filtered.size());
    for (size_t i = 0; i < n; ++i) {
        if (i == 0 || i + 1 == n || isCusp(path, i) || segmentBoundaryAfter(path, i)) {
            ceiling[i] = 0.0;
            continue;
        }
        // Keep the original pointwise ceiling as a hard safety cap.
        ceiling[i] = std::clamp(filtered[i], 0.0, raw_ceiling[i]);
    }
    return ceiling;
}

void enforceAccelerationDecelerationLimits(
    const std::vector<Pose2D>& path,
    const VelocityProfiler::Params& params,
    const std::vector<double>& v_ceil,
    std::vector<double>& v) {
    const size_t n = path.size();
    if (n == 0 || v.size() != n || v_ceil.size() != n) return;

    v.front() = 0.0;
    for (size_t i = 1; i < n; ++i) {
        if (isCusp(path, i)) {
            v[i] = 0.0;
            continue;
        }

        const double ds = arcLength(path[i - 1], path[i]);
        const double v_allowed = std::sqrt(
            std::max(0.0, v[i - 1] * v[i - 1] + 2.0 * params.a_max * ds));
        v[i] = std::min({std::max(0.0, v[i]), v_ceil[i], v_allowed});
    }

    v.back() = 0.0;
    for (int idx = static_cast<int>(n) - 2; idx >= 0; --idx) {
        const size_t i = static_cast<size_t>(idx);
        if (segmentBoundaryAfter(path, i)) {
            v[i + 1] = 0.0;
        }

        const double ds = arcLength(path[i], path[i + 1]);
        const double v_allowed = std::sqrt(
            std::max(0.0, v[i + 1] * v[i + 1] + 2.0 * params.d_max * ds));
        v[i] = std::min({std::max(0.0, v[i]), v_ceil[i], v_allowed});
    }
}

void applyVelocityRounding(const std::vector<Pose2D>& path,
                           const std::vector<double>& v_ceil,
                           std::vector<double>& v,
                           double smoothing_length) {
    if (path.size() < 3 || path.size() != v.size() || v_ceil.size() != v.size()) {
        return;
    }

    const std::vector<double> s = arcLengths(path);
    std::vector<double> rounded = zeroPhaseArcLengthLowPass(
        path, s, v, smoothing_length);

    for (size_t i = 0; i < v.size(); ++i) {
        if (i == 0 || i + 1 == v.size() || isCusp(path, i) ||
            segmentBoundaryAfter(path, i)) {
            v[i] = 0.0;
            continue;
        }
        v[i] = std::clamp(rounded[i], 0.0, v_ceil[i]);
    }
}

void applyJerkLimitedSmoothing(const std::vector<Pose2D>& path,
                               const VelocityProfiler::Params& params,
                               const std::vector<double>& v_ceil,
                               std::vector<double>& v) {
    const size_t n = path.size();
    if (n < 3 || params.j_max <= 0.0) return;

    // S-curve style pass: acceleration/deceleration are allowed to grow only
    // by j_max * dt, so the velocity profile no longer has rectangular
    // acceleration steps.  The final values are still clipped by curvature,
    // acceleration, deceleration and cusp stop constraints.
    constexpr int kPasses = 3;
    for (int pass = 0; pass < kPasses; ++pass) {
        double accel = 0.0;
        v.front() = 0.0;

        for (size_t i = 1; i < n; ++i) {
            if (isCusp(path, i)) {
                v[i] = 0.0;
                accel = 0.0;
                continue;
            }

            const double ds = arcLength(path[i - 1], path[i]);
            if (ds < 1e-9) continue;

            const double dt = segmentDt(v[i - 1], v[i], ds);
            accel = std::min(params.a_max, accel + params.j_max * dt);
            const double v_allowed = std::sqrt(
                std::max(0.0, v[i - 1] * v[i - 1] + 2.0 * accel * ds));
            v[i] = std::min({v[i], v_allowed, v_ceil[i]});

            const double actual_accel =
                (v[i] * v[i] - v[i - 1] * v[i - 1]) / (2.0 * ds);
            accel = std::clamp(actual_accel, 0.0, params.a_max);
        }

        double decel = 0.0;
        v.back() = 0.0;

        for (int idx = static_cast<int>(n) - 2; idx >= 0; --idx) {
            const size_t i = static_cast<size_t>(idx);
            if (isCusp(path, i)) {
                v[i] = 0.0;
                decel = 0.0;
                continue;
            }

            const bool boundary_after = segmentBoundaryAfter(path, i);
            if (boundary_after) {
                v[i + 1] = 0.0;
                decel = 0.0;
            }

            const double ds = arcLength(path[i], path[i + 1]);
            if (ds < 1e-9) continue;

            const double dt = segmentDt(v[i], v[i + 1], ds);
            decel = std::min(params.d_max, decel + params.j_max * dt);
            const double v_allowed = std::sqrt(
                std::max(0.0, v[i + 1] * v[i + 1] + 2.0 * decel * ds));
            v[i] = std::min({v[i], v_allowed, v_ceil[i]});

            const double actual_decel =
                (v[i] * v[i] - v[i + 1] * v[i + 1]) / (2.0 * ds);
            decel = std::clamp(actual_decel, 0.0, params.d_max);
        }
    }
}

}  // namespace

void VelocityProfiler::compute(std::vector<Pose2D>& path,
                                const Params& params,
                                double safe_curvature_eps,
                                double curvature_resample_ds,
                                int curvature_filter_window,
                                double curvature_rate_limit) {
    const size_t n = path.size();
    if (n == 0) return;
    if (n == 1) { path[0].velocity = 0.0; return; }

    // -----------------------------------------------------------------------
    // Pass 1 — curvature ceiling
    //   v_ceil[i] = max speed allowed at waypoint i by lateral-accel budget.
    //   Start, end, and cusp waypoints are hard zeros.
    // -----------------------------------------------------------------------
    std::vector<double> v_ceil_raw(n, 0.0);
    const std::vector<double> curvature = PathCurvature::computeAtPathPoints(
        path, curvature_resample_ds, curvature_filter_window, curvature_rate_limit);
    const double preview_length = std::clamp(params.v_max * 0.45, 0.8, 2.0);
    const std::vector<double> speed_curvature =
        smoothCurvatureForSpeedCeiling(path, curvature, preview_length);

    for (size_t i = 1; i + 1 < n; ++i) {
        if (isCusp(path, i)) {
            v_ceil_raw[i] = 0.0;
            continue;
        }
        const double v_max_i = (path[i].direction < 0)
            ? params.v_max_reverse : params.v_max;

        const double k = (i < speed_curvature.size()) ? speed_curvature[i] : 0.0;
        v_ceil_raw[i] = (k < safe_curvature_eps)
            ? v_max_i
            : curvatureSpeedLimit(k, params.a_lat_max, params.v_min_curv, v_max_i);
    }
    const std::vector<double> v_ceil = smoothSpeedCeilingForPreview(
        path, v_ceil_raw, preview_length);
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

    applyJerkLimitedSmoothing(path, params, v_ceil, v);
    applyVelocityRounding(path, v_ceil, v, std::clamp(preview_length * 0.5, 0.5, 1.0));
    enforceAccelerationDecelerationLimits(path, params, v_ceil, v);
    applyJerkLimitedSmoothing(path, params, v_ceil, v);

    // Write back — always non-negative (sign carried by direction field).
    for (size_t i = 0; i < n; ++i) {
        path[i].velocity = std::max(0.0, v[i]);
    }
}
