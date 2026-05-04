#include "hybrid_astar_cpp/smoother.hpp"

#include <algorithm>
#include <cmath>

namespace {

struct Point2D { double x, y; };

double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

// Build a boolean mask of waypoints that must not be moved:
//   - first and last point
//   - any point within goal_lock_distance of the end
//   - cusp_guard_points either side of every direction change
std::vector<bool> buildLockedPoints(const std::vector<Pose2D>& path,
                                    double goal_lock_distance,
                                    int cusp_guard_points) {
    std::vector<bool> locked(path.size(), false);
    if (path.empty()) return locked;

    // Distance-to-end for each point
    std::vector<double> dist_to_end(path.size(), 0.0);
    for (size_t i = path.size() - 1; i > 0; --i)
        dist_to_end[i - 1] = dist_to_end[i] +
            std::hypot(path[i].x - path[i-1].x, path[i].y - path[i-1].y);

    locked.front() = true;
    locked.back()  = true;
    for (size_t i = 0; i < path.size(); ++i)
        if (dist_to_end[i] <= goal_lock_distance)
            locked[i] = true;

    // Guard points around cusps
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i].direction == path[i-1].direction) continue;
        const int lo = std::max(0,  static_cast<int>(i) - cusp_guard_points);
        const int hi = std::min(static_cast<int>(path.size()) - 1,
                                static_cast<int>(i) + cusp_guard_points);
        for (int j = lo; j <= hi; ++j) locked[j] = true;
    }
    return locked;
}

// Evaluate a uniform cubic B-spline at parameter u ∈ [0,1].
// The control polygon is padded at both ends so the curve passes through
// the first and last control points.
Point2D evaluateUniformCubicBSpline(const std::vector<Pose2D>& controls,
                                    double normalized_u) {
    if (controls.empty())  return {0.0, 0.0};
    if (controls.size() == 1) return {controls.front().x, controls.front().y};

    // Clamp-padding: repeat first/last point twice each
    std::vector<Point2D> p;
    p.reserve(controls.size() + 4);
    for (int i = 0; i < 2; ++i) p.push_back({controls.front().x, controls.front().y});
    for (const auto& c : controls) p.push_back({c.x, c.y});
    for (int i = 0; i < 2; ++i) p.push_back({controls.back().x,  controls.back().y});

    const int spans = static_cast<int>(p.size()) - 3;
    double su = clamp01(normalized_u) * spans;
    int    sp = static_cast<int>(std::floor(su));
    double t  = su - sp;
    if (sp >= spans) { sp = spans - 1; t = 1.0; }

    const double t2 = t*t, t3 = t2*t;
    const double b0 = (1 - 3*t + 3*t2 -   t3) / 6.0;
    const double b1 = (4      - 6*t2 + 3*t3) / 6.0;
    const double b2 = (1 + 3*t + 3*t2 - 3*t3) / 6.0;
    const double b3 =                      t3  / 6.0;

    return {
        b0*p[sp].x + b1*p[sp+1].x + b2*p[sp+2].x + b3*p[sp+3].x,
        b0*p[sp].y + b1*p[sp+1].y + b2*p[sp+2].y + b3*p[sp+3].y
    };
}

// Recompute interior yaw values from the direction of travel.
void updateInteriorYaws(std::vector<Pose2D>& sec) {
    for (size_t i = 1; i + 1 < sec.size(); ++i) {
        const double heading = std::atan2(sec[i+1].y - sec[i].y,
                                          sec[i+1].x - sec[i].x);
        sec[i].yaw = (sec[i].direction < 0)
            ? AnalyticCurves::wrapAngle(heading + M_PI) : heading;
    }
}

// Blend a single monotone section toward its B-spline.
// max_point_shift <= 0 disables the displacement clamp.
std::vector<Pose2D> blendSection(const std::vector<Pose2D>& section,
                                 double blend,
                                 double max_point_shift) {
    std::vector<Pose2D> out = section;
    if (section.size() < 4) return out;

    // Chord-length parameterisation
    std::vector<double> cum(section.size(), 0.0);
    for (size_t i = 1; i < section.size(); ++i)
        cum[i] = cum[i-1] + std::hypot(section[i].x - section[i-1].x,
                                        section[i].y - section[i-1].y);
    const double L = cum.back();

    const double b = clamp01(blend);
    for (size_t i = 1; i + 1 < section.size(); ++i) {
        const double u = (L > 1e-9) ? cum[i] / L
                                    : static_cast<double>(i) / (section.size() - 1);
        const Point2D sp = evaluateUniformCubicBSpline(section, u);
        double dx = b * (sp.x - section[i].x);
        double dy = b * (sp.y - section[i].y);
        if (max_point_shift > 0.0) {
            const double shift = std::hypot(dx, dy);
            if (shift > max_point_shift) {
                const double scale = max_point_shift / shift;
                dx *= scale;
                dy *= scale;
            }
        }
        out[i].x = section[i].x + dx;
        out[i].y = section[i].y + dy;
    }
    out.front() = section.front();
    out.back()  = section.back();
    updateInteriorYaws(out);
    return out;
}

// Dense collision check along a section (5 cm sub-steps between waypoints).
bool isSectionCollisionFree(const std::vector<Pose2D>& sec,
                             const std::shared_ptr<GridCollision>& cc) {
    if (!cc) return true;
    constexpr double kStep = 0.05;
    for (const auto& pose : sec)
        if (!cc->isCollisionFree(pose.x, pose.y, pose.yaw)) return false;
    for (size_t i = 1; i < sec.size(); ++i) {
        const double d = std::hypot(sec[i].x - sec[i-1].x, sec[i].y - sec[i-1].y);
        const int steps = std::max(1, static_cast<int>(std::ceil(d / kStep)));
        for (int s = 1; s < steps; ++s) {
            const double t = static_cast<double>(s) / steps;
            const double x   = sec[i-1].x + t*(sec[i].x - sec[i-1].x);
            const double y   = sec[i-1].y + t*(sec[i].y - sec[i-1].y);
            const double dyaw = AnalyticCurves::wrapAngle(sec[i].yaw - sec[i-1].yaw);
            const double yaw  = AnalyticCurves::wrapAngle(sec[i-1].yaw + t*dyaw);
            if (!cc->isCollisionFree(x, y, yaw)) return false;
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------

PathSmoother::PathSmoother(double goal_lock_distance, int cusp_guard_points)
    : goal_lock_distance_(std::max(0.0, goal_lock_distance)),
      cusp_guard_points_(std::max(0, cusp_guard_points)) {}

void PathSmoother::smoothPath(std::vector<Pose2D>& path,
                               std::shared_ptr<GridCollision> collision_checker,
                               double blend,
                               double max_point_shift) {
    if (path.size() < 4 || blend <= 0.0) return;

    const std::vector<bool> locked =
        buildLockedPoints(path, goal_lock_distance_, cusp_guard_points_);

    std::vector<Pose2D> result = path;

    // Process each monotone segment (constant direction) independently
    size_t seg_start = 0;
    while (seg_start < path.size()) {
        // Find end of this segment
        size_t seg_end = seg_start;
        while (seg_end + 1 < path.size() &&
               path[seg_end + 1].direction == path[seg_start].direction)
            ++seg_end;

        // Walk runs of consecutive unlocked interior points
        size_t i = seg_start + 1;
        while (i < seg_end) {
            // Skip locked points
            while (i < seg_end && locked[i]) ++i;
            if (i >= seg_end) break;

            const size_t run_start = i;
            while (i < seg_end && !locked[i]) ++i;
            const size_t run_end = i - 1;

            // Include one locked anchor on each side
            const size_t sec_start = run_start - 1;
            const size_t sec_end   = run_end   + 1;
            if (sec_end - sec_start + 1 < 4) continue;

            std::vector<Pose2D> section;
            section.reserve(sec_end - sec_start + 1);
            for (size_t j = sec_start; j <= sec_end; ++j)
                section.push_back(path[j]);

            std::vector<Pose2D> candidate = blendSection(section, blend, max_point_shift);
            if (!isSectionCollisionFree(candidate, collision_checker)) continue;

            for (size_t j = run_start; j <= run_end; ++j)
                result[j] = candidate[j - sec_start];
        }

        seg_start = seg_end + 1;
    }

    // Final yaw pass for all unlocked interior points
    for (size_t i = 1; i + 1 < result.size(); ++i) {
        if (locked[i] ||
            result[i].direction != result[i-1].direction ||
            result[i].direction != result[i+1].direction) continue;
        const double h = std::atan2(result[i+1].y - result[i].y,
                                    result[i+1].x - result[i].x);
        result[i].yaw = (result[i].direction < 0)
            ? AnalyticCurves::wrapAngle(h + M_PI) : h;
    }

    path = result;
}
