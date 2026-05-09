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

#include "hybrid_astar_cpp/mpc_controller.hpp"

// ROS 2 wrapper around MpcController.
//
// Subscribes to:
//     /astar_path              (reference geometry)
//     /astar_velocity_profile  (v_ref per waypoint)
//     /astar_steering          (δ_ref per waypoint — feedforward)
//     /astar_acceleration      (a_ref per waypoint — feedforward)
//     /odom                    (current state x,y,θ,v)
//
// Publishes:
//     /mpc_cmd                 (Float64MultiArray [a, δ])
//     /mpc_predicted_path      (Path — MPC open-loop prediction for RViz)
//     /mpc_tracking_error      (Float64MultiArray [e_y, e_yaw, e_v])
//
// Reference construction:
//     The published path, velocity and IK arrays are stored internally.
//     Reverse motion is inferred from path geometry: if the displacement to
//     the next knot is opposite the vehicle yaw, MPC tracks signed negative
//     body-frame velocity while the dashboard can keep plotting speed magnitude.
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

    // -- ROS I/O ------------------------------------------------------------
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr vel_sub_, steer_sub_, accel_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr predicted_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr error_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr reference_pub_;

    rclcpp::TimerBase::SharedPtr control_timer_;

    // -- Cached reference trajectory ---------------------------------------
    std::mutex ref_mutex_;
    std::vector<MpcController::State>   path_states_;   // (x, y, θ, v)
    std::vector<int>                    path_direction_sign_; // +1 forward, -1 reverse
    std::vector<double>                 path_a_ref_;    // a_ref per waypoint
    std::vector<double>                 path_delta_ref_;// δ_ref per waypoint
    std::vector<double>                 path_time_;     // cumulative time (s)
    bool                                has_path_ = false;
    bool                                has_vel_  = false;
    bool                                has_steer_= false;
    bool                                has_accel_= false;
    std::string                         path_frame_id_;

    // -- Vehicle state ------------------------------------------------------
    std::mutex state_mutex_;
    MpcController::State                current_state_;
    bool                                has_odom_ = false;

    // -- Controller --------------------------------------------------------
    std::unique_ptr<MpcController>      mpc_;
    double                              control_dt_ = 0.05;   // control rate period (s)
    double                              lookahead_capture_radius_ = 25.0;

    // -- Goal-reached state ------------------------------------------------
    // Set when the vehicle is within goal_reach_radius_ of the path endpoint
    // AND slower than goal_reach_speed_.  Reset when a new path arrives.
    bool   goal_reached_       = false;
    double goal_reach_radius_  = 0.5;   // m   — overridden by ROS param
    double goal_reach_speed_   = 0.3;   // m/s — overridden by ROS param
};

#endif  // HYBRID_ASTAR_CPP_MPC_NODE_HPP_
