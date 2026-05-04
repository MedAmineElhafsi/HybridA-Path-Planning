#include "hybrid_astar_cpp/curvature.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

constexpr double kEps = 1e-9;

double distance2D(const Pose2D& a, const Pose2D& b) {
    return std::hypot(b.x - a.x, b.y - a.y);
}

std::vector<double> arcLengths(const std::vector<Pose2D>& path) {
    std::vector<double> s(path.size(), 0.0);
    for (size_t i = 1; i < path.size(); ++i) {
        s[i] = s[i - 1] + distance2D(path[i - 1], path[i]);
    }
    return s;
}

int normalizedWindow(int window) {
    if (window < 1) return 1;
    return (window % 2 == 0) ? window + 1 : window;
}

Pose2D interpolatePose(const Pose2D& a, const Pose2D& b, double t) {
    Pose2D out = a;
    out.x = a.x + t * (b.x - a.x);
    out.y = a.y + t * (b.y - a.y);

    const double dyaw = AnalyticCurves::wrapAngle(b.yaw - a.yaw);
    out.yaw = AnalyticCurves::wrapAngle(a.yaw + t * dyaw);
    out.velocity = a.velocity + t * (b.velocity - a.velocity);
    out.direction = a.direction;
    return out;
}

Pose2D interpolateByArcLength(const std::vector<Pose2D>& path,
                              const std::vector<double>& s,
                              size_t start,
                              size_t end,
                              double target_s) {
    if (target_s <= s[start] + kEps) return path[start];
    if (target_s >= s[end] - kEps) return path[end];

    auto hi_it = std::lower_bound(s.begin() + static_cast<long>(start + 1),
                                  s.begin() + static_cast<long>(end + 1),
                                  target_s);
    size_t hi = static_cast<size_t>(std::distance(s.begin(), hi_it));
    if (hi > end) return path[end];

    size_t lo = hi - 1;
    while (hi <= end && s[hi] - s[lo] < kEps) {
        ++hi;
    }
    if (hi > end) return path[end];

    const double segment_len = s[hi] - s[lo];
    const double t = std::clamp((target_s - s[lo]) / segment_len, 0.0, 1.0);
    return interpolatePose(path[lo], path[hi], t);
}

bool hasDirectionChangeNear(const std::vector<Pose2D>& path, size_t i) {
    return (i > 0 && path[i].direction != path[i - 1].direction) ||
           (i + 1 < path.size() && path[i + 1].direction != path[i].direction);
}

struct SplineValue {
    double first = 0.0;
    double second = 0.0;
};

class NaturalCubicSpline {
public:
    bool fit(const std::vector<double>& x, const std::vector<double>& y) {
        x_.clear();
        y_.clear();
        second_.clear();

        if (x.size() != y.size() || x.size() < 3) return false;

        x_.reserve(x.size());
        y_.reserve(y.size());
        for (size_t i = 0; i < x.size(); ++i) {
            if (!x_.empty() && x[i] - x_.back() < kEps) continue;
            x_.push_back(x[i]);
            y_.push_back(y[i]);
        }

        const size_t n = x_.size();
        if (n < 3) return false;

        std::vector<double> lower(n, 0.0);
        std::vector<double> diag(n, 1.0);
        std::vector<double> upper(n, 0.0);
        std::vector<double> rhs(n, 0.0);

        for (size_t i = 1; i + 1 < n; ++i) {
            const double h0 = x_[i] - x_[i - 1];
            const double h1 = x_[i + 1] - x_[i];
            if (h0 < kEps || h1 < kEps) return false;

            lower[i] = h0;
            diag[i] = 2.0 * (h0 + h1);
            upper[i] = h1;
            rhs[i] = 6.0 * ((y_[i + 1] - y_[i]) / h1 -
                            (y_[i] - y_[i - 1]) / h0);
        }

        for (size_t i = 1; i < n; ++i) {
            const double factor = lower[i] / diag[i - 1];
            diag[i] -= factor * upper[i - 1];
            rhs[i] -= factor * rhs[i - 1];
        }

        second_.assign(n, 0.0);
        second_[n - 1] = rhs[n - 1] / diag[n - 1];
        for (size_t i = n - 1; i > 0; --i) {
            second_[i - 1] = (rhs[i - 1] - upper[i - 1] * second_[i]) / diag[i - 1];
        }

        second_.front() = 0.0;
        second_.back() = 0.0;
        return true;
    }

    SplineValue derivatives(double query) const {
        SplineValue out;
        if (x_.size() < 2) return out;

        if (query <= x_.front()) query = x_.front();
        if (query >= x_.back()) query = x_.back();

        auto hi_it = std::upper_bound(x_.begin(), x_.end(), query);
        size_t hi = static_cast<size_t>(std::distance(x_.begin(), hi_it));
        if (hi == 0) hi = 1;
        if (hi >= x_.size()) hi = x_.size() - 1;
        const size_t lo = hi - 1;

        const double h = x_[hi] - x_[lo];
        if (h < kEps) return out;

        const double t = query - x_[lo];
        const double b = (y_[hi] - y_[lo]) / h -
                         h * (2.0 * second_[lo] + second_[hi]) / 6.0;
        const double c = 0.5 * second_[lo];
        const double d = (second_[hi] - second_[lo]) / (6.0 * h);

        out.first = b + 2.0 * c * t + 3.0 * d * t * t;
        out.second = second_[lo] + (second_[hi] - second_[lo]) * t / h;
        return out;
    }

private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> second_;
};

void updateInteriorYaws(std::vector<Pose2D>& path) {
    size_t seg_start = 0;
    while (seg_start < path.size()) {
        size_t seg_end = seg_start;
        while (seg_end + 1 < path.size() &&
               path[seg_end + 1].direction == path[seg_start].direction) {
            ++seg_end;
        }

        for (size_t i = seg_start + 1; i < seg_end; ++i) {
            const double heading = std::atan2(path[i + 1].y - path[i - 1].y,
                                              path[i + 1].x - path[i - 1].x);
            path[i].yaw = (path[i].direction < 0)
                ? AnalyticCurves::wrapAngle(heading + M_PI)
                : heading;
        }

        seg_start = seg_end + 1;
    }
}

std::vector<double> savitzkyGolayWeights(int window) {
    window = normalizedWindow(window);
    if (window >= 9) {
        return {-21.0/231.0, 14.0/231.0, 39.0/231.0, 54.0/231.0,
                59.0/231.0, 54.0/231.0, 39.0/231.0, 14.0/231.0,
                -21.0/231.0};
    }
    if (window >= 7) {
        return {-2.0/21.0, 3.0/21.0, 6.0/21.0, 7.0/21.0,
                6.0/21.0, 3.0/21.0, -2.0/21.0};
    }
    if (window >= 5) {
        return {-3.0/35.0, 12.0/35.0, 17.0/35.0, 12.0/35.0, -3.0/35.0};
    }
    return {1.0};
}

void applySavitzkyGolay(const std::vector<Pose2D>& samples,
                        std::vector<double>& kappa,
                        int filter_window) {
    if (samples.size() < 5 || kappa.size() != samples.size()) return;

    const std::vector<double> weights = savitzkyGolayWeights(filter_window);
    if (weights.size() <= 1) return;

    const int half = static_cast<int>(weights.size() / 2);
    std::vector<double> filtered = kappa;

    size_t run_start = 0;
    while (run_start < samples.size()) {
        size_t run_end = run_start;
        while (run_end + 1 < samples.size() &&
               samples[run_end + 1].direction == samples[run_start].direction) {
            ++run_end;
        }

        for (size_t i = run_start; i <= run_end; ++i) {
            if (i == run_start || i == run_end) {
                filtered[i] = 0.0;
                continue;
            }

            double weighted_sum = 0.0;
            double weight_sum = 0.0;
            for (int offset = -half; offset <= half; ++offset) {
                const int j = static_cast<int>(i) + offset;
                if (j <= static_cast<int>(run_start) ||
                    j >= static_cast<int>(run_end)) {
                    continue;
                }
                const double w = weights[static_cast<size_t>(offset + half)];
                weighted_sum += w * kappa[static_cast<size_t>(j)];
                weight_sum += w;
            }
            filtered[i] = (std::abs(weight_sum) > kEps)
                ? weighted_sum / weight_sum : kappa[i];
        }

        run_start = run_end + 1;
    }

    kappa = std::move(filtered);
}

double interpolateCurvature(const PathCurvature::CurvatureProfile& profile,
                            double target_s) {
    if (profile.s.empty() || profile.kappa.empty()) return 0.0;
    if (target_s <= profile.s.front() + kEps) return profile.kappa.front();
    if (target_s >= profile.s.back() - kEps) return profile.kappa.back();

    auto hi_it = std::lower_bound(profile.s.begin(), profile.s.end(), target_s);
    size_t hi = static_cast<size_t>(std::distance(profile.s.begin(), hi_it));
    if (hi == 0) return profile.kappa.front();
    if (hi >= profile.s.size()) return profile.kappa.back();

    size_t lo = hi - 1;
    while (hi < profile.s.size() && profile.s[hi] - profile.s[lo] < kEps) {
        ++hi;
    }
    if (hi >= profile.s.size()) return profile.kappa[lo];

    const double segment_len = profile.s[hi] - profile.s[lo];
    const double t = std::clamp((target_s - profile.s[lo]) / segment_len, 0.0, 1.0);
    return profile.kappa[lo] + t * (profile.kappa[hi] - profile.kappa[lo]);
}

void limitCurvatureRateByArcLength(const std::vector<Pose2D>& samples,
                                   const std::vector<double>& s,
                                   std::vector<double>& kappa,
                                   double max_curvature_rate) {
    if (samples.size() < 2 || samples.size() != s.size() ||
        samples.size() != kappa.size() || max_curvature_rate <= 0.0) {
        return;
    }

    auto clamp_to_neighbor = [&](double value, double neighbor, double ds) {
        const double max_step = max_curvature_rate * std::max(0.0, ds);
        return std::clamp(value, neighbor - max_step, neighbor + max_step);
    };

    size_t run_start = 0;
    while (run_start < samples.size()) {
        size_t run_end = run_start;
        while (run_end + 1 < samples.size() &&
               samples[run_end + 1].direction == samples[run_start].direction) {
            ++run_end;
        }

        kappa[run_start] = 0.0;
        if (run_end > run_start) {
            kappa[run_end] = 0.0;
        }

        for (size_t i = run_start + 1; i <= run_end; ++i) {
            kappa[i] = clamp_to_neighbor(kappa[i], kappa[i - 1], s[i] - s[i - 1]);
        }

        for (size_t i = run_end; i > run_start; --i) {
            kappa[i - 1] = clamp_to_neighbor(kappa[i - 1], kappa[i], s[i] - s[i - 1]);
        }

        run_start = run_end + 1;
    }
}

}  // namespace

namespace PathCurvature {

std::vector<Pose2D> resampleUniformArcLength(const std::vector<Pose2D>& path,
                                             double target_ds) {
    std::vector<Pose2D> resampled;
    if (path.empty()) return resampled;

    const double ds = std::max(target_ds, 0.01);

    size_t seg_start = 0;
    while (seg_start < path.size()) {
        size_t seg_end = seg_start;
        while (seg_end + 1 < path.size() &&
               path[seg_end + 1].direction == path[seg_start].direction) {
            ++seg_end;
        }

        if (seg_end == seg_start) {
            resampled.push_back(path[seg_start]);
            seg_start = seg_end + 1;
            continue;
        }

        std::vector<Pose2D> section;
        section.reserve(seg_end - seg_start + 1);
        for (size_t i = seg_start; i <= seg_end; ++i) {
            section.push_back(path[i]);
        }

        std::vector<double> local_s(section.size(), 0.0);
        for (size_t i = 1; i < section.size(); ++i) {
            local_s[i] = local_s[i - 1] + distance2D(section[i - 1], section[i]);
        }

        const double length = local_s.back();
        if (length < kEps) {
            resampled.push_back(path[seg_start]);
            resampled.push_back(path[seg_end]);
            seg_start = seg_end + 1;
            continue;
        }

        const int intervals = std::max(1, static_cast<int>(std::ceil(length / ds)));
        const double spacing = length / intervals;

        for (int step = 0; step <= intervals; ++step) {
            if (step == 0) {
                resampled.push_back(path[seg_start]);
                continue;
            }
            if (step == intervals) {
                resampled.push_back(path[seg_end]);
                continue;
            }

            const double target_s = step * spacing;
            Pose2D sample = interpolateByArcLength(
                section, local_s, 0, local_s.size() - 1, target_s);
            sample.direction = path[seg_start].direction;
            resampled.push_back(sample);
        }

        seg_start = seg_end + 1;
    }

    updateInteriorYaws(resampled);
    return resampled;
}

CurvatureProfile computeProfile(const std::vector<Pose2D>& path,
                                double target_ds,
                                int filter_window,
                                double max_curvature_rate) {
    CurvatureProfile profile;
    std::vector<Pose2D> samples;
    if (path.size() < 3) return profile;

    const std::vector<double> global_s = arcLengths(path);
    const double ds = std::max(0.005, target_ds);

    // Fit x(s) and y(s) cubic splines to each same-direction section of the
    // already B-spline-smoothed path.  Curvature is then computed from the
    // spline derivatives, not from finite differences on sparse chords.  This
    // gives C0-continuous kappa(s) transitions suitable for feedforward
    // steering.
    size_t seg_start = 0;
    while (seg_start < path.size()) {
        size_t seg_end = seg_start;
        while (seg_end + 1 < path.size() &&
               path[seg_end + 1].direction == path[seg_start].direction) {
            ++seg_end;
        }

        const double seg_length = global_s[seg_end] - global_s[seg_start];
        if (seg_end - seg_start + 1 < 3 || seg_length < kEps) {
            profile.s.push_back(global_s[seg_start]);
            profile.kappa.push_back(0.0);
            samples.push_back(path[seg_start]);
            seg_start = seg_end + 1;
            continue;
        }

        std::vector<double> local_s;
        std::vector<double> xs;
        std::vector<double> ys;
        local_s.reserve(seg_end - seg_start + 1);
        xs.reserve(seg_end - seg_start + 1);
        ys.reserve(seg_end - seg_start + 1);

        for (size_t i = seg_start; i <= seg_end; ++i) {
            local_s.push_back(global_s[i] - global_s[seg_start]);
            xs.push_back(path[i].x);
            ys.push_back(path[i].y);
        }

        NaturalCubicSpline spline_x;
        NaturalCubicSpline spline_y;
        if (!spline_x.fit(local_s, xs) || !spline_y.fit(local_s, ys)) {
            profile.s.push_back(global_s[seg_start]);
            profile.kappa.push_back(0.0);
            samples.push_back(path[seg_start]);
            seg_start = seg_end + 1;
            continue;
        }

        const int intervals = std::max(1, static_cast<int>(std::ceil(seg_length / ds)));
        const double spacing = seg_length / intervals;
        for (int step = 0; step <= intervals; ++step) {
            const double q = (step == intervals) ? seg_length : step * spacing;
            const SplineValue dx = spline_x.derivatives(q);
            const SplineValue dy = spline_y.derivatives(q);

            const double speed_sq = dx.first * dx.first + dy.first * dy.first;
            const double denom = std::pow(speed_sq, 1.5);
            double kappa = 0.0;
            if (denom > kEps) {
                kappa = (dx.first * dy.second - dy.first * dx.second) / denom;
            }

            Pose2D sample = path[seg_start];
            sample.direction = path[seg_start].direction;
            samples.push_back(sample);
            profile.s.push_back(global_s[seg_start] + q);
            profile.kappa.push_back(kappa);
        }

        seg_start = seg_end + 1;
    }

    applySavitzkyGolay(samples, profile.kappa, filter_window);
    limitCurvatureRateByArcLength(
        samples, profile.s, profile.kappa, max_curvature_rate);

    return profile;
}

std::vector<double> computeAtPathPoints(const std::vector<Pose2D>& path,
                                        double target_ds,
                                        int filter_window,
                                        double max_curvature_rate) {
    std::vector<double> kappa(path.size(), 0.0);
    if (path.size() < 3) return kappa;

    const CurvatureProfile profile = computeProfile(
        path, target_ds, filter_window, max_curvature_rate);
    if (profile.s.empty()) return kappa;

    const std::vector<double> path_s = arcLengths(path);
    for (size_t i = 0; i < path.size(); ++i) {
        if (i == 0 || i + 1 == path.size() || hasDirectionChangeNear(path, i)) {
            kappa[i] = 0.0;
            continue;
        }
        kappa[i] = interpolateCurvature(profile, path_s[i]);
    }

    return kappa;
}

}  // namespace PathCurvature
