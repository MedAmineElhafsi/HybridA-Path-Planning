#include "hybrid_astar_cpp/curves.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

// ============================================================================
// MATH UTILITIES
// ============================================================================
double AnalyticCurves::mod2pi(double theta) {
    double res = std::fmod(theta, 2.0 * M_PI);
    if (res < 0) res += 2.0 * M_PI;
    return res;
}

double AnalyticCurves::wrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

namespace {
    inline void polar(double x, double y, double& r, double& theta) {
        r = std::hypot(x, y);
        theta = std::atan2(y, x);
    }

    inline void rs_timeflip(std::vector<PathSegment>& path) {
        for (auto& seg : path) seg.gear = -seg.gear;
    }

    inline void rs_reflect(std::vector<PathSegment>& path) {
        for (auto& seg : path) {
            if (seg.type == 'L') seg.type = 'R';
            else if (seg.type == 'R') seg.type = 'L';
        }
    }
}

// ============================================================================
// DUBINS MATH (Forward Only)
// ============================================================================
bool AnalyticCurves::dubinsLSL(double alpha, double beta, double d, double& t, double& p, double& q) {
    double tmp = 2.0 + d*d - 2.0*std::cos(alpha - beta) + 2.0*d*(std::sin(alpha) - std::sin(beta));
    if (tmp < 0) return false;
    p = std::sqrt(tmp);
    t = mod2pi(std::atan2((std::cos(beta) - std::cos(alpha)), (d + std::sin(alpha) - std::sin(beta))) - alpha);
    q = mod2pi(beta - std::atan2((std::cos(beta) - std::cos(alpha)), (d + std::sin(alpha) - std::sin(beta))));
    return true;
}

bool AnalyticCurves::dubinsRSR(double alpha, double beta, double d, double& t, double& p, double& q) {
    double tmp = 2.0 + d*d - 2.0*std::cos(alpha - beta) + 2.0*d*(std::sin(beta) - std::sin(alpha));
    if (tmp < 0) return false;
    p = std::sqrt(tmp);
    t = mod2pi(alpha - std::atan2((std::cos(beta) - std::cos(alpha)), (d - std::sin(alpha) + std::sin(beta))));
    q = mod2pi(-beta + std::atan2((std::cos(beta) - std::cos(alpha)), (d - std::sin(alpha) + std::sin(beta))));
    return true;
}

bool AnalyticCurves::dubinsLSR(double alpha, double beta, double d, double& t, double& p, double& q) {
    double tmp = -2.0 + d*d + 2.0*std::cos(alpha - beta) + 2.0*d*(std::sin(alpha) + std::sin(beta));
    if (tmp < 0) return false;
    p = std::sqrt(tmp);
    t = mod2pi(std::atan2((-std::cos(beta) - std::cos(alpha)), (d + std::sin(alpha) + std::sin(beta))) - alpha);
    q = mod2pi(t - beta);
    return true;
}

bool AnalyticCurves::dubinsRSL(double alpha, double beta, double d, double& t, double& p, double& q) {
    double tmp = -2.0 + d*d + 2.0*std::cos(alpha - beta) - 2.0*d*(std::sin(alpha) + std::sin(beta));
    if (tmp < 0) return false;
    p = std::sqrt(tmp);
    t = mod2pi(alpha - std::atan2((std::cos(beta) + std::cos(alpha)), (d - std::sin(alpha) - std::sin(beta))));
    q = mod2pi(beta - t);
    return true;
}

bool AnalyticCurves::dubinsRLR(double alpha, double beta, double d, double& t, double& p, double& q) {
    double tmp = (6.0 - d*d + 2.0*std::cos(alpha - beta) + 2.0*d*(std::sin(alpha) - std::sin(beta))) / 8.0;
    if (std::abs(tmp) > 1.0) return false;
    p = mod2pi(2.0 * M_PI - std::acos(tmp));
    t = mod2pi(alpha - std::atan2((std::cos(alpha) - std::cos(beta)), (d - std::sin(alpha) + std::sin(beta))) + p / 2.0);
    q = mod2pi(alpha - beta - t + p);
    return true;
}

bool AnalyticCurves::dubinsLRL(double alpha, double beta, double d, double& t, double& p, double& q) {
    double tmp = (6.0 - d*d + 2.0*std::cos(alpha - beta) + 2.0*d*(std::sin(beta) - std::sin(alpha))) / 8.0;
    if (std::abs(tmp) > 1.0) return false;
    p = mod2pi(2.0 * M_PI - std::acos(tmp));
    t = mod2pi(-alpha - std::atan2((std::cos(alpha) - std::cos(beta)), (d + std::sin(alpha) - std::sin(beta))) + p / 2.0);
    q = mod2pi(beta - alpha - t + p);
    return true;
}

// ============================================================================
// REEDS-SHEPP MATH (Forward + Reverse)
// ============================================================================
namespace {
    inline bool rs_path1(double x, double y, double phi, std::vector<PathSegment>& path) {
        double u, t;
        polar(x - std::sin(phi), y - 1.0 + std::cos(phi), u, t);
        double v = AnalyticCurves::mod2pi(phi - t);
        path = {{'L', t, 1}, {'S', u, 1}, {'L', v, 1}};
        return true;
    }

    inline bool rs_path2(double x, double y, double phi, std::vector<PathSegment>& path) {
        double rho, t1;
        polar(x + std::sin(phi), y - 1.0 - std::cos(phi), rho, t1);
        if (rho * rho < 4.0) return false;
        double u = std::sqrt(rho * rho - 4.0);
        double t = AnalyticCurves::mod2pi(t1 + std::atan2(2.0, u));
        double v = AnalyticCurves::mod2pi(t - phi);
        path = {{'L', t, 1}, {'S', u, 1}, {'R', v, 1}};
        return true;
    }

    inline bool rs_path3(double x, double y, double phi, std::vector<PathSegment>& path) {
        double rho, theta;
        polar(x - std::sin(phi), y - 1.0 + std::cos(phi), rho, theta);
        if (rho > 4.0) return false;
        double A = std::acos(rho / 4.0);
        double t = AnalyticCurves::mod2pi(theta + M_PI / 2.0 + A);
        double u = AnalyticCurves::mod2pi(M_PI - 2.0 * A);
        double v = AnalyticCurves::mod2pi(phi - t + u); // FIXED
        path = {{'L', t, 1}, {'R', u, -1}, {'L', v, 1}};
        return true;
    }

    inline bool rs_path4(double x, double y, double phi, std::vector<PathSegment>& path) {
        double rho, theta;
        polar(x - std::sin(phi), y - 1.0 + std::cos(phi), rho, theta);
        if (rho > 4.0) return false;
        double A = std::acos(rho / 4.0);
        double t = AnalyticCurves::mod2pi(theta + M_PI / 2.0 + A);
        double u = AnalyticCurves::mod2pi(M_PI - 2.0 * A);
        double v = AnalyticCurves::mod2pi(t - u - phi); // FIXED
        path = {{'L', t, 1}, {'R', u, -1}, {'L', v, -1}};
        return true;
    }

    inline bool rs_path5(double x, double y, double phi, std::vector<PathSegment>& path) {
        double rho, theta;
        polar(x - std::sin(phi), y - 1.0 + std::cos(phi), rho, theta);
        if (rho > 4.0) return false;
        double u = std::acos(1.0 - rho * rho / 8.0);
        double A = std::asin(2.0 * std::sin(u) / rho);
        double t = AnalyticCurves::mod2pi(theta + M_PI / 2.0 - A);
        double v = AnalyticCurves::mod2pi(t - u - phi);
        path = {{'L', t, 1}, {'R', u, 1}, {'L', v, -1}};
        return true;
    }

    inline bool rs_path6(double x, double y, double phi, std::vector<PathSegment>& path) {
        double rho, theta;
        polar(x + std::sin(phi), y - 1.0 - std::cos(phi), rho, theta);
        if (rho > 4.0) return false;
        double t, u, v;
        if (rho <= 2.0) {
            double A = std::acos((rho + 2.0) / 4.0);
            t = AnalyticCurves::mod2pi(theta + M_PI / 2.0 + A);
            u = AnalyticCurves::mod2pi(A);
            v = AnalyticCurves::mod2pi(phi - t + 2.0 * u);
        } else {
            double A = std::acos((rho - 2.0) / 4.0);
            t = AnalyticCurves::mod2pi(theta + M_PI / 2.0 - A);
            u = AnalyticCurves::mod2pi(M_PI - A);
            v = AnalyticCurves::mod2pi(phi - t + 2.0 * u);
        }
        path = {{'L', t, 1}, {'R', u, 1}, {'L', u, -1}, {'R', v, -1}};
        return true;
    }
}

// ============================================================================
// SAMPLING & GENERATION
// ============================================================================
std::vector<Pose2D> AnalyticCurves::samplePath(const Pose2D& start, const std::vector<PathSegment>& segments, double rho, double step_size) {
    std::vector<Pose2D> pts;
    double x = start.x;
    double y = start.y;
    double yaw = start.yaw;

    for (const auto& seg : segments) {
        double remaining = std::abs(seg.length) * rho; 
        int sgn = seg.gear;

        while (remaining > 1e-6) {
            double step = std::min(step_size, remaining) * sgn;
            remaining -= std::abs(step);

            if (seg.type == 'S') {
                x += step * std::cos(yaw);
                y += step * std::sin(yaw);
            } else {
                double k = (seg.type == 'L') ? (1.0 / rho) : (-1.0 / rho);
                double yaw0 = yaw;
                double yaw1 = wrapAngle(yaw0 + k * step);
                x += (std::sin(yaw1) - std::sin(yaw0)) / k;
                y += (-std::cos(yaw1) + std::cos(yaw0)) / k;
                yaw = yaw1;
            }
            pts.push_back({x, y, yaw, sgn});
        }
    }
    return pts;
}

bool AnalyticCurves::getDubinsPath(const Pose2D& start, const Pose2D& goal, double rho, double step_size, std::vector<Pose2D>& path) {
    double dx = goal.x - start.x;
    double dy = goal.y - start.y;
    double d = std::hypot(dx, dy) / rho;
    
    if (d < 1e-6) return false;

    double theta = std::atan2(dy, dx);
    double alpha = mod2pi(theta - start.yaw); 
    double beta = mod2pi(goal.yaw - theta);

    double best_len = std::numeric_limits<double>::infinity();
    std::vector<PathSegment> best_segments;

    double t, p, q;
    
    if (dubinsLSL(alpha, beta, d, t, p, q) && (t+p+q < best_len)) {
        best_len = t+p+q; best_segments = {{'L', t, 1}, {'S', p, 1}, {'L', q, 1}};
    }
    if (dubinsRSR(alpha, beta, d, t, p, q) && (t+p+q < best_len)) {
        best_len = t+p+q; best_segments = {{'R', t, 1}, {'S', p, 1}, {'R', q, 1}};
    }
    if (dubinsLSR(alpha, beta, d, t, p, q) && (t+p+q < best_len)) {
        best_len = t+p+q; best_segments = {{'L', t, 1}, {'S', p, 1}, {'R', q, 1}};
    }
    if (dubinsRSL(alpha, beta, d, t, p, q) && (t+p+q < best_len)) {
        best_len = t+p+q; best_segments = {{'R', t, 1}, {'S', p, 1}, {'L', q, 1}};
    }
    if (dubinsRLR(alpha, beta, d, t, p, q) && (t+p+q < best_len)) {
        best_len = t+p+q; best_segments = {{'R', t, 1}, {'L', p, 1}, {'R', q, 1}};
    }
    if (dubinsLRL(alpha, beta, d, t, p, q) && (t+p+q < best_len)) {
        best_len = t+p+q; best_segments = {{'L', t, 1}, {'R', p, 1}, {'L', q, 1}};
    }

    if (best_segments.empty()) return false;

    path = samplePath(start, best_segments, rho, step_size);
    return true;
}

bool AnalyticCurves::getReedsSheppPath(const Pose2D& start, const Pose2D& goal, double rho, double step_size, std::vector<Pose2D>& path, std::vector<int>& gears) {
    double dx = goal.x - start.x;
    double dy = goal.y - start.y;
    double c = std::cos(start.yaw);
    double s = std::sin(start.yaw);
    
    double x = (dx * c + dy * s) / rho;
    double y = (-dx * s + dy * c) / rho;
    double phi = wrapAngle(goal.yaw - start.yaw);

    std::vector<std::vector<PathSegment>> all_paths;
    
    auto add_paths = [&](double x_in, double y_in, double phi_in) {
        std::vector<PathSegment> p;
        if (rs_path1(x_in, y_in, phi_in, p)) all_paths.push_back(p);
        if (rs_path2(x_in, y_in, phi_in, p)) all_paths.push_back(p);
        if (rs_path3(x_in, y_in, phi_in, p)) all_paths.push_back(p);
        if (rs_path4(x_in, y_in, phi_in, p)) all_paths.push_back(p);
        if (rs_path5(x_in, y_in, phi_in, p)) all_paths.push_back(p);
        if (rs_path6(x_in, y_in, phi_in, p)) all_paths.push_back(p);
    };

    // Base
    add_paths(x, y, phi);
    // Timeflip
    size_t cur_size = all_paths.size();
    add_paths(-x, y, -phi);
    for (size_t i = cur_size; i < all_paths.size(); ++i) rs_timeflip(all_paths[i]);
    // Reflect
    cur_size = all_paths.size();
    add_paths(x, -y, -phi);
    for (size_t i = cur_size; i < all_paths.size(); ++i) rs_reflect(all_paths[i]);
    // Timeflip + Reflect
    cur_size = all_paths.size();
    add_paths(-x, -y, phi);
    for (size_t i = cur_size; i < all_paths.size(); ++i) {
        rs_timeflip(all_paths[i]);
        rs_reflect(all_paths[i]);
    }

    if (all_paths.empty()) return false;

    double best_len = std::numeric_limits<double>::infinity();
    std::vector<PathSegment> best_segments;

    for (const auto& p : all_paths) {
        double len = 0;
        for (const auto& seg : p) len += std::abs(seg.length);
        if (len > 0.01 && len < best_len) {
            best_len = len;
            best_segments = p;
        }
    }

    if (best_segments.empty()) return false;

    path = samplePath(start, best_segments, rho, step_size);
    
    gears.clear();
    for (const auto& seg : best_segments) {
        int n_steps = std::max(1, (int)std::ceil((std::abs(seg.length) * rho) / step_size));
        for (int i = 0; i < n_steps; ++i) gears.push_back(seg.gear);
    }
    while(gears.size() < path.size()) gears.push_back(gears.back());
    if (gears.size() > path.size()) gears.resize(path.size());

    return true;
}
