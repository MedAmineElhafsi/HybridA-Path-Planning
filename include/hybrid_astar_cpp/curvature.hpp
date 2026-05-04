#ifndef HYBRID_ASTAR_CPP_CURVATURE_HPP_
#define HYBRID_ASTAR_CPP_CURVATURE_HPP_

#include <vector>

#include "hybrid_astar_cpp/curves.hpp"

namespace PathCurvature {

struct CurvatureProfile {
    std::vector<double> s;      // cumulative arc length of the resampled path (m)
    std::vector<double> kappa;  // signed curvature kappa(s), positive = left turn
};

// Resample each forward/reverse section independently with approximately
// constant arc-length spacing.  Direction changes are kept as section
// boundaries so curvature is never differentiated across cusps.
std::vector<Pose2D> resampleUniformArcLength(const std::vector<Pose2D>& path,
                                             double target_ds);

// Professor's curvature pipeline:
// B-spline smoothed path -> cubic spline fit x(s), y(s) -> analytic spline
// derivatives -> derivative curvature kappa(s) -> Savitzky-Golay filtering ->
// curvature-rate limiting -> short zero-phase arc-length smoothing.
CurvatureProfile computeProfile(const std::vector<Pose2D>& path,
                                double target_ds = 0.05,
                                int filter_window = 5,
                                double max_curvature_rate = 0.06);

// Same curvature estimate interpolated back to the original path waypoint
// indices, so ROS arrays remain aligned with nav_msgs/Path poses.
std::vector<double> computeAtPathPoints(const std::vector<Pose2D>& path,
                                        double target_ds = 0.05,
                                        int filter_window = 5,
                                        double max_curvature_rate = 0.06);

}  // namespace PathCurvature

#endif  // HYBRID_ASTAR_CPP_CURVATURE_HPP_
