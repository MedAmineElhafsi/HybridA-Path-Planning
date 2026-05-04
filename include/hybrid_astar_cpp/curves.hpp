#ifndef HYBRID_ASTAR_CPP_CURVES_HPP_
#define HYBRID_ASTAR_CPP_CURVES_HPP_

#include <vector>
#include <cmath>
#include <string>
#include <tuple>

// Simple struct to represent a 3D state
struct Pose2D {
    double x;
    double y;
    double yaw;
    int direction = 1;    // 1 = forward, -1 = reverse
    double velocity = 0.0; // target speed at this waypoint (m/s), signed by direction
};

// Represents a piece of the analytic path
struct PathSegment {
    char type;     // 'L', 'R', 'S'
    double length; // length of the segment (normalized)
    int gear;      // 1 for forward, -1 for reverse
};

class AnalyticCurves {
public:
    // Shared angle helpers used by the planner and analytic path routines.
    static double mod2pi(double theta);
    static double wrapAngle(double a);

    // Computes and samples the shortest Forward-Only (Dubins) path
    static bool getDubinsPath(const Pose2D& start, const Pose2D& goal, 
                              double rho, double step_size, 
                              std::vector<Pose2D>& path);

    // Computes and samples the shortest Forward/Reverse (Reeds-Shepp) path
    static bool getReedsSheppPath(const Pose2D& start, const Pose2D& goal, 
                                  double rho, double step_size, 
                                  std::vector<Pose2D>& path, 
                                  std::vector<int>& gears);

private:
    // Dubins Words
    static bool dubinsLSL(double alpha, double beta, double d, double& t, double& p, double& q);
    static bool dubinsRSR(double alpha, double beta, double d, double& t, double& p, double& q);
    static bool dubinsLSR(double alpha, double beta, double d, double& t, double& p, double& q);
    static bool dubinsRSL(double alpha, double beta, double d, double& t, double& p, double& q);
    static bool dubinsRLR(double alpha, double beta, double d, double& t, double& p, double& q);
    static bool dubinsLRL(double alpha, double beta, double d, double& t, double& p, double& q);

    static std::vector<Pose2D> samplePath(const Pose2D& start, const std::vector<PathSegment>& segments, double rho, double step_size);
};

#endif // HYBRID_ASTAR_CPP_CURVES_HPP_
