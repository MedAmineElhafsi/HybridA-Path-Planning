#ifndef HYBRID_ASTAR_CPP_MPC_NODE_HPP_
#define HYBRID_ASTAR_CPP_MPC_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <mutex>
#include <memory>
#include <vector>
#include <limits>
#include <string>

#include "hybrid_astar_cpp/mpc_controller.hpp"

// ROS 2 wrapper around MpcController.
//
// Subscribes to:
//     /astar_path              (reference geometry)
//     /astar_velocity_profile  (v_ref per waypoint)
//     /astar_steering          (δ_ref per waypoint — feedforward)
//     /astar_acceleration      (a_ref per waypoint — feedforward)
//     /odom                    (current state x,y,θ,vx,vy,r)
//
// Publishes:
//     /mpc_cmd                 (Float64MultiArray [a, δ])
//     /mpc_predicted_path      (Path — MPC open-loop prediction for RViz)
//     /mpc_tracking_error      (Float64MultiArray [e_y, e_yaw, e_vx])
//
// Reference construction:
//     The published path, velocity and IK arrays are stored internally.
//     Reverse motion is inferred from path geometry: if the displacement to
//     the next knot is opposite the vehicle yaw, MPC tracks signed negative
//     body-frame longitudinal velocity while the dashboard can keep plotting
//     speed magnitude.  The planner's v_ref becomes vx_ref, vy_ref is zero,
//     and r_ref is generated from vx_ref and the steering feedforward.
//     A per-waypoint arc-length and cumulative time are pre-computed from
//     the velocity profile (t_{i+1} = t_i + ds_i / ((v_i+v_{i+1})/2)).  At
//     every control tick the node finds the closest waypoint to the current
//     vehicle pose, then samples the reference at times t_0 + k·dt for
//     k = 0..N by linear interpolation between waypoints.
//
// In the professor's block sketch this node is the vehicle model/controller
// block after smoothing, velocity profiling and inverse kinematics.  It uses
// MPC in place of a simpler PID lateral/longitudinal controller.
class MpcNode : public rclcpp::Node {
public:
    MpcNode();

private:
    // -- Callbacks ----------------------------------------------------------
    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg);
    void velocityCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void steeringCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void accelerationCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void referenceTrajectoryCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

    // -- Main control tick --------------------------------------------------
    void controlTick();

    // -- Reference horizon construction ------------------------------------
    //  Returns false if the reference is not fully populated or the vehicle
    //  is too far from the path.
    bool buildReferenceHorizon(
        const MpcController::State& current,
        std::vector<MpcController::State>&   ref_x,
        std::vector<MpcController::Control>& ref_u);

    // Recompute cumulative-time parameterisation of the path.
    void rebuildTimeAxis();

    // Fill dynamic reference fields not published directly by the planner.
    void updateReferenceYawRates();

    // Publish or log zero-command reasons using the exact diagnostic strings
    // expected by the closed-loop debug workflow.
    void logZeroCommandReason(const char* reason);
    void publishZeroCommand(const char* reason);

    // Repair late/missing feedforward arrays once the path and velocity are
    // available.  Must be called with ref_mutex_ held.
    void normalizeReferenceProfilesLocked();

    // -- ROS I/O ------------------------------------------------------------
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr vel_sub_, steer_sub_, accel_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr reference_traj_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr predicted_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr error_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr reference_pub_;

    rclcpp::TimerBase::SharedPtr control_timer_;

    // -- Cached reference trajectory ---------------------------------------
    std::mutex ref_mutex_;
    std::vector<MpcController::State>   path_states_;   // (x, y, θ, vx, vy, r)
    std::vector<int>                    path_direction_sign_; // +1 forward, -1 reverse
    std::vector<double>                 path_a_ref_;    // a_ref per waypoint
    std::vector<double>                 path_delta_ref_;// δ_ref per waypoint
    std::vector<double>                 path_time_;     // cumulative time (s)
    size_t                              velocity_profile_size_ = 0;
    size_t                              steering_profile_size_ = 0;
    size_t                              acceleration_profile_size_ = 0;
    bool                                has_path_ = false;
    bool                                has_vel_  = false;
    bool                                has_steer_= false;
    bool                                has_accel_= false;
    bool                                has_reference_trajectory_ = false;
    std::string                         path_frame_id_;
    bool                                mpc_ready_logged_ = false;
    bool                                steering_fallback_warned_ = false;
    bool                                acceleration_fallback_warned_ = false;

    // -- Vehicle state ------------------------------------------------------
    std::mutex state_mutex_;
    MpcController::State                current_state_;
    bool                                has_odom_ = false;

    // -- Controller --------------------------------------------------------
    std::unique_ptr<MpcController>      mpc_;
    double                              control_dt_ = 0.05;   // control rate period (s)
    double                              lookahead_capture_radius_ = 5.0;
    double                              launch_assist_accel_ = 0.8;
    double                              launch_assist_speed_threshold_ = 0.05;
    double                              launch_assist_ref_speed_threshold_ = 0.20;
    double                              wheelbase_ = 2.7;
    double                              goal_latch_min_travel_ = 0.2;
    bool                                path_follow_start_valid_ = false;
    double                              path_follow_start_x_ = 0.0;
    double                              path_follow_start_y_ = 0.0;
    double                              path_follow_travel_ = 0.0;
    double                              last_nearest_ref_dist_ =
                                            std::numeric_limits<double>::quiet_NaN();
    int                                 last_nearest_ref_index_ = -1;
    MpcController::State                last_nearest_ref_state_;
    bool                                last_nearest_ref_valid_ = false;
    std::string                         last_reference_failure_reason_;
    std::string                         last_zero_reason_;
    rclcpp::Time                        last_zero_log_time_;

    // -- Goal-reached state ------------------------------------------------
    // Set when the vehicle is within goal_reach_radius_ of the path endpoint
    // AND slower than goal_reach_speed_.  Reset when a new path arrives.
    bool   goal_reached_       = false;
    double goal_reach_radius_  = 0.5;   // m   — overridden by ROS param
    double goal_reach_speed_   = 0.3;   // m/s — overridden by ROS param
};

#endif  // HYBRID_ASTAR_CPP_MPC_NODE_HPP_
