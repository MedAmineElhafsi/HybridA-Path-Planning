#ifndef HYBRID_ASTAR_CPP_INVERSE_KINEMATICS_HPP_
#define HYBRID_ASTAR_CPP_INVERSE_KINEMATICS_HPP_

#include <vector>
#include "hybrid_astar_cpp/curves.hpp"

// Per-waypoint kinematic quantities derived from the planned path using the
// Ackermann kinematic model.
//
// The Ackermann steering geometry requires that all four wheels rotate about a
// common instantaneous centre of rotation (ICR).  This gives each front wheel a
// different steering angle: the inner wheel must turn sharper than the outer wheel.
//
// Reference point: rear-axle centre.
//
// Sign conventions:
//   curvature       > 0  →  left turn   (ICR to the left of the vehicle)
//   delta_avg       > 0  →  steer left  (bicycle equivalent angle)
//   delta_left      > 0  →  left wheel steered left
//   delta_right     > 0  →  right wheel steered left
//   acceleration    > 0  →  speeding up along the travel direction
//                   < 0  →  slowing down (deceleration)
//
// Ackermann geometry:
//   R       = 1 / κ            (signed turning radius at rear-axle centre)
//   δ_avg   = atan(L / R)      (bicycle equivalent — equal to atan(L · κ))
//   δ_left  = atan(L / (R - w/2))   (inner wheel on left turn)
//   δ_right = atan(L / (R + w/2))   (outer wheel on left turn)
//
// When κ → 0 (straight line): δ_avg = δ_left = δ_right = 0.
struct WaypointKinematics {
    double curvature       = 0.0;   // κ       (rad/m)   signed Menger curvature
    double delta_avg       = 0.0;   // δ_avg   (rad)     bicycle-equivalent front steering
    double delta_left      = 0.0;   // δ_left  (rad)     left  front-wheel steering angle
    double delta_right     = 0.0;   // δ_right (rad)     right front-wheel steering angle
    double acceleration    = 0.0;   // a       (m/s²)    dv/dt along the path
};

// Computes Ackermann inverse kinematics for every waypoint in a smoothed,
// velocity-profiled path.
//
// Parameters:
//   path       — planned + smoothed + velocity-profiled path (Pose2D::velocity filled)
//   wheelbase  — distance between front and rear axles (m)
//   track_width — distance between left and right wheel contact patches (m)
class InverseKinematics {
public:
    static std::vector<WaypointKinematics> compute(
        const std::vector<Pose2D>& path,
        double wheelbase,
        double track_width);
};

#endif  // HYBRID_ASTAR_CPP_INVERSE_KINEMATICS_HPP_
