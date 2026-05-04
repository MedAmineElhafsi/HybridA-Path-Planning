#include "hybrid_astar_cpp/planner_node.hpp"
#include "hybrid_astar_cpp/curvature.hpp"
#include "hybrid_astar_cpp/reference_trajectory.hpp"

#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <utility>

namespace {
double wrapAngle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}


Pose2D pose2DFromPose(const geometry_msgs::msg::Pose& pose) {
    return Pose2D{
        pose.position.x,
        pose.position.y,
        tf2::getYaw(pose.orientation),
    };
}

geometry_msgs::msg::Pose poseFromPose2D(const Pose2D& pose2d) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = pose2d.x;
    pose.position.y = pose2d.y;
    pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, pose2d.yaw);
    pose.orientation = tf2::toMsg(q);
    return pose;
}

void appendPoseMarker(visualization_msgs::msg::MarkerArray& marker_array,
                      const std_msgs::msg::Header& header,
                      const geometry_msgs::msg::Pose& pose,
                      int marker_id,
                      float r,
                      float g,
                      float b,
                      const std::string& label) {
    visualization_msgs::msg::Marker arrow;
    arrow.header = header;
    arrow.ns = "planner_pose_arrows";
    arrow.id = marker_id;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.pose = pose;
    arrow.pose.position.z = 0.12;
    arrow.scale.x = 0.55;
    arrow.scale.y = 0.10;
    arrow.scale.z = 0.10;
    arrow.color.r = r;
    arrow.color.g = g;
    arrow.color.b = b;
    arrow.color.a = 0.95;
    marker_array.markers.push_back(arrow);

    visualization_msgs::msg::Marker text;
    text.header = header;
    text.ns = "planner_pose_labels";
    text.id = marker_id;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose = pose;
    text.pose.position.z = 0.48;
    text.scale.z = 0.22;
    text.color.r = r;
    text.color.g = g;
    text.color.b = b;
    text.color.a = 0.95;
    text.text = label;
    marker_array.markers.push_back(text);
}

void appendDeleteMarker(visualization_msgs::msg::MarkerArray& marker_array,
                        const std_msgs::msg::Header& header,
                        const std::string& ns,
                        int marker_id) {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = marker_id;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    marker_array.markers.push_back(marker);
}

double pathStepLength(const Pose2D& a, const Pose2D& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

bool hasDirectionChangeInTail(const std::vector<Pose2D>& path, double tail_distance) {
    if (path.size() < 2 || tail_distance <= 0.0) {
        return false;
    }

    double accumulated_distance = 0.0;
    for (size_t i = path.size() - 1; i > 0; --i) {
        if (path[i].direction != path[i - 1].direction) {
            return true;
        }
        accumulated_distance += pathStepLength(path[i], path[i - 1]);
        if (accumulated_distance >= tail_distance) {
            break;
        }
    }
    return false;
}

bool isComfortablySafePose(const Pose2D& pose,
                           const std::shared_ptr<GridCollision>& strict_checker,
                           const std::shared_ptr<GridCollision>& clearance_checker,
                           double min_center_clearance) {
    if (!strict_checker || !clearance_checker) {
        return false;
    }

    if (!strict_checker->isCollisionFree(pose.x, pose.y, pose.yaw)) {
        return false;
    }

    return clearance_checker->getObstacleDistance(pose.x, pose.y) >= min_center_clearance;
}

bool findSafeGoalPose(const Pose2D& requested_goal,
                      const std::shared_ptr<GridCollision>& strict_checker,
                      const std::shared_ptr<GridCollision>& clearance_checker,
                      double max_pullback_distance,
                      double pullback_step,
                      double min_center_clearance,
                      Pose2D& safe_goal) {
    if (isComfortablySafePose(
            requested_goal, strict_checker, clearance_checker, min_center_clearance)) {
        safe_goal = requested_goal;
        return true;
    }

    if (max_pullback_distance <= 0.0 || pullback_step <= 0.0) {
        return false;
    }

    for (double distance = pullback_step;
         distance <= max_pullback_distance + 1e-6;
         distance += pullback_step) {
        Pose2D candidate = requested_goal;
        candidate.x -= distance * std::cos(requested_goal.yaw);
        candidate.y -= distance * std::sin(requested_goal.yaw);
        if (isComfortablySafePose(
                candidate, strict_checker, clearance_checker, min_center_clearance)) {
            safe_goal = candidate;
            return true;
        }
    }

    return false;
}

bool findSafeStartPose(const Pose2D& requested_start,
                       const std::shared_ptr<GridCollision>& strict_checker,
                       const std::shared_ptr<GridCollision>& clearance_checker,
                       double max_adjust_distance,
                       double adjust_step,
                       double min_center_clearance,
                       Pose2D& safe_start) {
    if (isComfortablySafePose(
            requested_start, strict_checker, clearance_checker, min_center_clearance)) {
        safe_start = requested_start;
        return true;
    }

    if (max_adjust_distance <= 0.0 || adjust_step <= 0.0) {
        return false;
    }

    bool found = false;
    double best_distance = std::numeric_limits<double>::infinity();
    Pose2D best_pose = requested_start;

    for (double distance = adjust_step;
         distance <= max_adjust_distance + 1e-6;
         distance += adjust_step) {
        for (double sign : {1.0, -1.0}) {
            Pose2D candidate = requested_start;
            candidate.x += sign * distance * std::cos(requested_start.yaw);
            candidate.y += sign * distance * std::sin(requested_start.yaw);
            if (!isComfortablySafePose(
                    candidate, strict_checker, clearance_checker, min_center_clearance)) {
                continue;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_pose = candidate;
                found = true;
            }
        }
    }

    if (found) {
        safe_start = best_pose;
    }
    return found;
}

Pose2D chooseSafeHandoffPose(const std::vector<Pose2D>& path,
                             const std::shared_ptr<GridCollision>& strict_checker,
                             const std::shared_ptr<GridCollision>& clearance_checker,
                             double tail_distance,
                             double min_backtrack_distance,
                             double min_center_clearance) {
    if (path.empty()) {
        return Pose2D{};
    }

    const bool tail_has_direction_change = hasDirectionChangeInTail(path, tail_distance);
    if (!tail_has_direction_change &&
        isComfortablySafePose(
            path.back(), strict_checker, clearance_checker, min_center_clearance)) {
        return path.back();
    }

    const double required_backtrack = tail_has_direction_change
        ? std::max(tail_distance, min_backtrack_distance)
        : min_backtrack_distance;

    double accumulated_distance = 0.0;
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        if (i < static_cast<int>(path.size()) - 1) {
            accumulated_distance += pathStepLength(path[i], path[i + 1]);
        }

        if (accumulated_distance + 1e-6 < required_backtrack) {
            continue;
        }

        if (isComfortablySafePose(
                path[i], strict_checker, clearance_checker, min_center_clearance)) {
            return path[i];
        }
    }

    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        if (isComfortablySafePose(
                path[i], strict_checker, clearance_checker, min_center_clearance)) {
            return path[i];
        }
    }

    return path.back();
}
}  // namespace

PlannerNode::PlannerNode() : Node("hybrid_astar_cpp_node") {
    // 1. Declare Parameters
    this->declare_parameter("map_frame", "map");
    this->declare_parameter("xy_resolution", 0.25);
    this->declare_parameter("min_turning_radius", 1.0);
    this->declare_parameter("car_length", 0.9);
    this->declare_parameter("car_width", 0.45);
    this->declare_parameter("car_track_width", 0.38);  // left-to-right wheel spacing (m)
    this->declare_parameter("plan_margin", 0.04);
    this->declare_parameter("hard_obstacle_margin", 0.0);
    this->declare_parameter("footprint_display_margin", 0.0);
    this->declare_parameter("footprint_spacing", 0.4);
    this->declare_parameter("planner_step_size", 0.2);
    this->declare_parameter("steer_samples", 7);
    this->declare_parameter("planner_wheelbase", 0.5);
    this->declare_parameter("soft_obstacle_margin", 0.05);
    this->declare_parameter("obstacle_clearance_weight", 0.75);
    this->declare_parameter("clearance_relaxation_radius", 1.0);
    this->declare_parameter("use_live_start_from_odom", false);
    this->declare_parameter("live_start_topic", "/odom");
    this->declare_parameter("odom_replan_min_translation", 0.20);
    this->declare_parameter("odom_replan_min_yaw", 0.18);
    this->declare_parameter("odom_replan_min_interval_s", 0.6);

    // Planner threshold parameters (previously hard-coded magic numbers)
    this->declare_parameter("goal_pos_thresh",       0.50);
    this->declare_parameter("goal_yaw_thresh",       0.20);
    this->declare_parameter("analytic_every_n",      10);
    this->declare_parameter("analytic_radius",       8.0);
    this->declare_parameter("analytic_endpoint_tol", 0.50);
    this->declare_parameter("final_snap_pos",        0.35);
    this->declare_parameter("final_snap_yaw",        0.35);
    this->declare_parameter("detour_ratio",          3.0);
    this->declare_parameter("penalty_reverse",       0.5);
    this->declare_parameter("penalty_steer",         0.005);
    this->declare_parameter("penalty_steer_change",  0.04);
    this->declare_parameter("penalty_direction_change", 0.5);

    // Path smoother parameters (cubic B-spline only)
    this->declare_parameter("smoother_bspline_blend",      0.65);
    this->declare_parameter("smoother_max_point_shift",    0.0);
    this->declare_parameter("smoother_goal_lock_distance", 2.5);
    this->declare_parameter("smoother_cusp_guard_points",  3);
    this->declare_parameter("trajectory_resample_ds",      0.05);
    this->declare_parameter("curvature_resample_ds",       0.025);
    this->declare_parameter("curvature_filter_window",     9);
    this->declare_parameter("curvature_rate_limit",        0.06);
    this->declare_parameter("steering_rate_limit",         0.35);
    this->declare_parameter("acceleration_filter_window",  7);
    this->declare_parameter("terminal_safety_extra_margin", 0.05);
    this->declare_parameter("terminal_goal_pullback_step", 0.10);
    this->declare_parameter("terminal_goal_pullback_max_distance", 1.5);
    this->declare_parameter("chain_start_min_backtrack", 1.0);
    this->declare_parameter("chain_start_clearance_padding", 0.05);

    // Velocity profile parameters
    this->declare_parameter("vel_max",          1.5);   // max forward speed (m/s)
    this->declare_parameter("vel_max_reverse",  0.5);   // max reverse speed (m/s)
    this->declare_parameter("vel_accel_max",    1.0);   // max acceleration (m/s²)
    this->declare_parameter("vel_decel_max",    1.5);   // max deceleration (m/s²)
    this->declare_parameter("vel_jerk_max",     2.0);   // max jerk for S-curve smoothing (m/s³)
    this->declare_parameter("vel_lat_accel_max",1.5);   // lateral accel budget (m/s²)
    this->declare_parameter("vel_min_curv",     0.1);   // min speed on tight curves (m/s)
    
    map_frame_ = this->get_parameter("map_frame").as_string();
    xy_res_ = this->get_parameter("xy_resolution").as_double();
    min_turning_radius_ = this->get_parameter("min_turning_radius").as_double();
    use_live_start_from_odom_ = this->get_parameter("use_live_start_from_odom").as_bool();
    odom_replan_min_translation_ = std::max(0.0, this->get_parameter("odom_replan_min_translation").as_double());
    odom_replan_min_yaw_ = std::max(0.0, this->get_parameter("odom_replan_min_yaw").as_double());
    odom_replan_min_interval_s_ = std::max(0.0, this->get_parameter("odom_replan_min_interval_s").as_double());

    // 2. Setup Publishers
    path_pub_           = this->create_publisher<nav_msgs::msg::Path>("astar_path", 10);
    raw_path_pub_       = this->create_publisher<nav_msgs::msg::Path>("astar_path_raw", 10);
    velocity_pub_       = this->create_publisher<std_msgs::msg::Float64MultiArray>("astar_velocity_profile", 10);
    curvature_pub_      = this->create_publisher<std_msgs::msg::Float64MultiArray>("astar_curvature", 10);
    steering_avg_pub_   = this->create_publisher<std_msgs::msg::Float64MultiArray>("astar_steering", 10);
    steering_left_pub_  = this->create_publisher<std_msgs::msg::Float64MultiArray>("astar_steering_left", 10);
    steering_right_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("astar_steering_right", 10);
    acceleration_pub_   = this->create_publisher<std_msgs::msg::Float64MultiArray>("astar_acceleration", 10);
    reference_trajectory_pub_ =
        this->create_publisher<std_msgs::msg::Float64MultiArray>("astar_reference_trajectory", 10);
    footprint_pub_      = this->create_publisher<visualization_msgs::msg::MarkerArray>("astar_footprints", 10);

    // 3. Setup Subscribers
    grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "astar_grid", 10, std::bind(&PlannerNode::gridCallback, this, std::placeholders::_1));
        
    start_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 10, std::bind(&PlannerNode::initialPoseCallback, this, std::placeholders::_1));

    if (use_live_start_from_odom_) {
        const std::string odom_topic = this->get_parameter("live_start_topic").as_string();
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, 10, std::bind(&PlannerNode::odomCallback, this, std::placeholders::_1));
    }
        
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10, std::bind(&PlannerNode::goalPoseCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "C++ Hybrid A* Backend Initialized. Waiting for Grid, Start, and Goal...");
}

void PlannerNode::gridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    bool should_replan = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_grid_ = *msg;
        has_grid_ = true;
        should_replan = has_start_ && has_goal_;
    }

    if (should_replan) {
        planPath();
    }
}

void PlannerNode::initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    bool should_replan = false;
    geometry_msgs::msg::Pose start_pose_snapshot;
    std::string frame_id;
    int footprints_to_delete = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        start_pose_ = msg->pose.pose;
        has_start_ = true;
        chain_from_last_goal_ = false;
        has_last_plan_end_ = false;
        has_goal_ = false;
        cached_dist_map_goal_x_ = std::numeric_limits<double>::quiet_NaN();
        cached_dist_map_goal_y_ = std::numeric_limits<double>::quiet_NaN();
        last_replan_start_pose_ = start_pose_;
        has_last_replan_start_ = true;
        last_replan_time_ = this->now();
        start_pose_snapshot = start_pose_;
        frame_id = current_grid_.header.frame_id.empty() ? map_frame_ : current_grid_.header.frame_id;
        footprints_to_delete = last_footprint_count_;
        last_footprint_count_ = 0;
        should_replan = false;
    }
    RCLCPP_INFO(this->get_logger(), "Start pose received.");
    publishResetVisualization(start_pose_snapshot, frame_id, footprints_to_delete);
    if (should_replan) {
        planPath();
    }
}

void PlannerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    if (!use_live_start_from_odom_) {
        return;
    }

    bool should_replan = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        start_pose_ = msg->pose.pose;
        has_start_ = true;

        // Only evaluate replanning conditions once all inputs are available.
        // Avoid early returns inside the lock so the control flow is explicit.
        if (has_grid_ && has_goal_) {
            const rclcpp::Time replan_stamp = msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0
                ? this->now()
                : rclcpp::Time(msg->header.stamp);

            const double dx = start_pose_.position.x - last_replan_start_pose_.position.x;
            const double dy = start_pose_.position.y - last_replan_start_pose_.position.y;
            const double translation_delta = std::hypot(dx, dy);
            const double yaw_delta = std::abs(wrapAngle(
                tf2::getYaw(start_pose_.orientation) - tf2::getYaw(last_replan_start_pose_.orientation)));
            const double interval_delta = has_last_replan_start_
                ? (replan_stamp - last_replan_time_).seconds()
                : odom_replan_min_interval_s_;

            if (!has_last_replan_start_ || (
                interval_delta >= odom_replan_min_interval_s_ &&
                (translation_delta >= odom_replan_min_translation_ || yaw_delta >= odom_replan_min_yaw_))) {
                should_replan = true;
                last_replan_start_pose_ = start_pose_;
                has_last_replan_start_ = true;
                last_replan_time_ = replan_stamp;
            }
        }
    }

    if (should_replan) {
        planPath();
    }
}

void PlannerNode::goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    bool should_replan = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!use_live_start_from_odom_ && chain_from_last_goal_ && has_goal_) {
            start_pose_ = goal_pose_;
            has_start_ = true;
            last_replan_start_pose_ = start_pose_;
            has_last_replan_start_ = true;
            last_replan_time_ = this->now();
            RCLCPP_INFO(this->get_logger(), "Chaining next plan from previous goal pose.");
        }
        goal_pose_ = msg->pose;
        has_goal_ = true;
        chain_from_last_goal_ = true;
        should_replan = has_grid_ && has_start_;
    }
    RCLCPP_INFO(this->get_logger(), "Goal pose received.");
    if (should_replan) {
        planPath();
    }
}

void PlannerNode::planPath() {
    nav_msgs::msg::OccupancyGrid grid_snapshot;
    geometry_msgs::msg::Pose start_pose_snapshot;
    geometry_msgs::msg::Pose goal_pose_snapshot;

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!has_grid_ || !has_start_ || !has_goal_) {
            return;
        }
        grid_snapshot = current_grid_;
        start_pose_snapshot = start_pose_;
        goal_pose_snapshot = goal_pose_;
    }

    RCLCPP_INFO(this->get_logger(), "Starting C++ Hybrid A* planning...");
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Read parameters
    double c_len       = this->get_parameter("car_length").as_double();
    double c_wid       = this->get_parameter("car_width").as_double();
    double c_mar       = this->get_parameter("plan_margin").as_double();
    double hard_obstacle_margin      = std::max(0.0, this->get_parameter("hard_obstacle_margin").as_double());
    double footprint_display_margin  = std::max(0.0, this->get_parameter("footprint_display_margin").as_double());
    double footprint_spacing         = std::max(0.05, this->get_parameter("footprint_spacing").as_double());
    double planner_step_size         = std::max(0.05, this->get_parameter("planner_step_size").as_double());
    int steer_samples                = std::max(3, static_cast<int>(this->get_parameter("steer_samples").as_int()));
    double planner_wheelbase         = std::max(0.05, this->get_parameter("planner_wheelbase").as_double());
    double soft_obstacle_margin      = std::max(0.0, this->get_parameter("soft_obstacle_margin").as_double());
    double obstacle_clearance_weight = std::max(0.0, this->get_parameter("obstacle_clearance_weight").as_double());
    double clearance_relaxation_radius = std::max(0.0, this->get_parameter("clearance_relaxation_radius").as_double());
    double terminal_safety_extra_margin = std::max(
        0.0, this->get_parameter("terminal_safety_extra_margin").as_double());
    double terminal_goal_pullback_step = std::max(
        0.01, this->get_parameter("terminal_goal_pullback_step").as_double());
    double terminal_goal_pullback_max_distance = std::max(
        0.0, this->get_parameter("terminal_goal_pullback_max_distance").as_double());
    const double smoother_bspline_blend = std::clamp(
        this->get_parameter("smoother_bspline_blend").as_double(), 0.0, 1.0);
    const double smoother_max_point_shift = std::max(
        0.0, this->get_parameter("smoother_max_point_shift").as_double());
    double smoother_goal_lock_distance = std::max(
        0.0, this->get_parameter("smoother_goal_lock_distance").as_double());
    int smoother_cusp_guard_points = std::max(
        0, static_cast<int>(this->get_parameter("smoother_cusp_guard_points").as_int()));
    const double trajectory_resample_ds = std::max(
        0.01, this->get_parameter("trajectory_resample_ds").as_double());
    const double curvature_resample_ds = std::max(
        0.01, this->get_parameter("curvature_resample_ds").as_double());
    const int curvature_filter_window = std::max(
        1, static_cast<int>(this->get_parameter("curvature_filter_window").as_int()));
    const double curvature_rate_limit = std::max(
        0.0, this->get_parameter("curvature_rate_limit").as_double());
    const double steering_rate_limit = std::max(
        0.0, this->get_parameter("steering_rate_limit").as_double());
    const int acceleration_filter_window = std::max(
        1, static_cast<int>(this->get_parameter("acceleration_filter_window").as_int()));
    const double vel_jerk_max = std::max(
        0.01, this->get_parameter("vel_jerk_max").as_double());
    double chain_start_min_backtrack = std::max(
        0.0, this->get_parameter("chain_start_min_backtrack").as_double());
    double chain_start_clearance_padding = std::max(
        0.0, this->get_parameter("chain_start_clearance_padding").as_double());
    min_turning_radius_ = std::max(0.05, this->get_parameter("min_turning_radius").as_double());

    double planner_max_steer  = std::atan(planner_wheelbase / min_turning_radius_);
    double collision_margin   = c_mar + hard_obstacle_margin;
    double desired_clearance  = (c_wid * 0.5) + collision_margin + soft_obstacle_margin;

    // 2. Rebuild the collision checker and planner only when geometry params change.
    //    updateGrid() is always called to pick up new obstacle data; the LUT inside
    //    GridCollision is only recomputed when the grid resolution actually changes.
    const bool rebuild = !collision_checker_ || !planner_ ||
        c_len                    != cached_c_len_            ||
        c_wid                    != cached_c_wid_            ||
        c_mar                    != cached_c_mar_            ||
        hard_obstacle_margin     != cached_hard_margin_      ||
        planner_step_size        != cached_step_size_        ||
        steer_samples            != cached_steer_samples_    ||
        planner_wheelbase        != cached_wheelbase_        ||
        soft_obstacle_margin     != cached_soft_margin_      ||
        obstacle_clearance_weight!= cached_clearance_weight_ ||
        clearance_relaxation_radius != cached_relax_radius_;

    if (rebuild) {
        collision_checker_ = std::make_shared<GridCollision>(72, c_len, c_wid, collision_margin);
        planner_ = std::make_unique<HybridAStar>(
            planner_step_size, planner_max_steer, steer_samples, planner_wheelbase, xy_res_, 72,
            desired_clearance, obstacle_clearance_weight, clearance_relaxation_radius);

        // Apply tunable thresholds from ROS 2 parameters
        planner_->goal_pos_thresh       = this->get_parameter("goal_pos_thresh").as_double();
        planner_->goal_yaw_thresh       = this->get_parameter("goal_yaw_thresh").as_double();
        planner_->analytic_every_n      = this->get_parameter("analytic_every_n").as_int();
        planner_->analytic_radius       = this->get_parameter("analytic_radius").as_double();
        planner_->analytic_endpoint_tol = this->get_parameter("analytic_endpoint_tol").as_double();
        planner_->final_snap_pos        = this->get_parameter("final_snap_pos").as_double();
        planner_->final_snap_yaw        = this->get_parameter("final_snap_yaw").as_double();
        planner_->detour_ratio          = this->get_parameter("detour_ratio").as_double();
        planner_->penalty_reverse_      = std::max(0.0, this->get_parameter("penalty_reverse").as_double());
        planner_->penalty_steer_        = std::max(0.0, this->get_parameter("penalty_steer").as_double());
        planner_->penalty_steer_change_ = std::max(0.0, this->get_parameter("penalty_steer_change").as_double());
        planner_->penalty_direction_change_ =
            std::max(0.0, this->get_parameter("penalty_direction_change").as_double());

        // Invalidate the cached distance map so it is recomputed with the new checker
        cached_dist_map_goal_x_ = std::numeric_limits<double>::quiet_NaN();
        cached_dist_map_goal_y_ = std::numeric_limits<double>::quiet_NaN();

        cached_c_len_            = c_len;
        cached_c_wid_            = c_wid;
        cached_c_mar_            = c_mar;
        cached_hard_margin_      = hard_obstacle_margin;
        cached_step_size_        = planner_step_size;
        cached_steer_samples_    = steer_samples;
        cached_wheelbase_        = planner_wheelbase;
        cached_soft_margin_      = soft_obstacle_margin;
        cached_clearance_weight_ = obstacle_clearance_weight;
        cached_relax_radius_     = clearance_relaxation_radius;

        RCLCPP_INFO(this->get_logger(), "Rebuilt collision checker and planner.");
    }

    // Always refresh the obstacle grid (fast when resolution is unchanged)
    collision_checker_->updateGrid(grid_snapshot);

    std::shared_ptr<GridCollision> terminal_checker = collision_checker_;
    if (terminal_safety_extra_margin > 1e-6) {
        terminal_checker = std::make_shared<GridCollision>(
            72, c_len, c_wid, collision_margin + terminal_safety_extra_margin);
        terminal_checker->updateGrid(grid_snapshot);
    }

    // 3. Extract Poses
    const Pose2D requested_start_pose2d = pose2DFromPose(start_pose_snapshot);
    Pose2D planning_start_pose2d = requested_start_pose2d;
    geometry_msgs::msg::Pose planning_start_pose_snapshot = start_pose_snapshot;
    const Pose2D requested_goal_pose2d = pose2DFromPose(goal_pose_snapshot);
    Pose2D planning_goal_pose2d = requested_goal_pose2d;
    geometry_msgs::msg::Pose planning_goal_pose_snapshot = goal_pose_snapshot;
    const double terminal_center_clearance =
        (c_wid * 0.5) + collision_margin + terminal_safety_extra_margin +
        chain_start_clearance_padding;
    bool start_adjusted = false;
    const bool safe_start_found = findSafeStartPose(
        requested_start_pose2d,
        terminal_checker,
        collision_checker_,
        terminal_goal_pullback_max_distance,
        terminal_goal_pullback_step,
        terminal_center_clearance,
        planning_start_pose2d);
    if (!safe_start_found) {
        RCLCPP_WARN(
            this->get_logger(),
            "Start pose is too tight or colliding. Move the estimate farther from obstacles "
            "or align it toward free space.");
    } else if (std::hypot(
                   planning_start_pose2d.x - requested_start_pose2d.x,
                   planning_start_pose2d.y - requested_start_pose2d.y) > 1e-3) {
        start_adjusted = true;
        planning_start_pose_snapshot = poseFromPose2D(planning_start_pose2d);
        RCLCPP_WARN(
            this->get_logger(),
            "Requested start was too tight near obstacles. Planning from a safer start at "
            "(%.2f, %.2f, %.2f rad).",
            planning_start_pose2d.x,
            planning_start_pose2d.y,
            planning_start_pose2d.yaw);
    }
    bool goal_adjusted = false;
    const bool safe_goal_found = findSafeGoalPose(
        requested_goal_pose2d,
        terminal_checker,
        collision_checker_,
        terminal_goal_pullback_max_distance,
        terminal_goal_pullback_step,
        terminal_center_clearance,
        planning_goal_pose2d);
    if (!safe_goal_found) {
        RCLCPP_WARN(
            this->get_logger(),
            "Goal pose is too tight for a safe terminal maneuver. Move it farther from obstacles "
            "or point it toward free space.");
    } else if (std::hypot(
                   planning_goal_pose2d.x - requested_goal_pose2d.x,
                   planning_goal_pose2d.y - requested_goal_pose2d.y) > 1e-3) {
        goal_adjusted = true;
        planning_goal_pose_snapshot = poseFromPose2D(planning_goal_pose2d);
        RCLCPP_WARN(
            this->get_logger(),
            "Requested goal was too tight near obstacles. Planning to a safer pulled-back goal at "
            "(%.2f, %.2f, %.2f rad).",
            planning_goal_pose2d.x,
            planning_goal_pose2d.y,
            planning_goal_pose2d.yaw);
    }

    // Recompute the 2D BFS heuristic only when the goal has moved by more than half a cell.
    // This is the most expensive part of setup for large maps.
    if (safe_goal_found) {
        const double goal_delta = std::hypot(
            planning_goal_pose2d.x - cached_dist_map_goal_x_,
            planning_goal_pose2d.y - cached_dist_map_goal_y_);
        if (!std::isfinite(cached_dist_map_goal_x_) || goal_delta > xy_res_ * 0.5) {
            collision_checker_->computeDistanceMap(planning_goal_pose2d.x, planning_goal_pose2d.y);
            cached_dist_map_goal_x_ = planning_goal_pose2d.x;
            cached_dist_map_goal_y_ = planning_goal_pose2d.y;
        }
    }

    // 4. Run the Search
    std::vector<Pose2D> final_path;
    bool success = false;
    if (safe_start_found && safe_goal_found) {
        success = planner_->plan(
            planning_start_pose2d,
            planning_goal_pose2d,
            collision_checker_,
            terminal_checker,
            final_path);
    }

    // 5. Publish raw path (before smoothing) for analysis and plotting.
    if (success && !final_path.empty()) {
        const std::string fid = grid_snapshot.header.frame_id.empty()
                                ? map_frame_ : grid_snapshot.header.frame_id;
        nav_msgs::msg::Path raw_msg;
        raw_msg.header.stamp    = this->now();
        raw_msg.header.frame_id = fid;
        for (const auto& p : final_path) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = raw_msg.header;
            ps.pose.position.x = p.x;
            ps.pose.position.y = p.y;
            tf2::Quaternion q; q.setRPY(0, 0, p.yaw);
            ps.pose.orientation = tf2::toMsg(q);
            raw_msg.poses.push_back(ps);
        }
        raw_path_pub_->publish(raw_msg);
    }

    // 5b. Smooth the path with cubic B-spline, then resample it uniformly by
    // arc length.  This is the trajectory-generation front half in the
    // professor sketch:
    //   Hybrid A* path -> trajectory smoothing -> uniform reference knots.
    if (success && !final_path.empty()) {
        PathSmoother smoother(smoother_goal_lock_distance, smoother_cusp_guard_points);
        smoother.smoothPath(
            final_path,
            collision_checker_,
            smoother_bspline_blend,
            smoother_max_point_shift);

        std::vector<Pose2D> resampled_path =
            PathCurvature::resampleUniformArcLength(final_path, trajectory_resample_ds);
        if (!resampled_path.empty()) {
            final_path = std::move(resampled_path);
        }
    }

    // 6. Compute velocity profile on the resampled reference knots.
    if (success && !final_path.empty()) {
        VelocityProfiler::Params vp;
        vp.v_max         = std::max(0.01, this->get_parameter("vel_max").as_double());
        vp.v_max_reverse = std::max(0.01, this->get_parameter("vel_max_reverse").as_double());
        vp.a_max         = std::max(0.01, this->get_parameter("vel_accel_max").as_double());
        vp.d_max         = std::max(0.01, this->get_parameter("vel_decel_max").as_double());
        vp.j_max         = vel_jerk_max;
        vp.a_lat_max     = std::max(0.01, this->get_parameter("vel_lat_accel_max").as_double());
        vp.v_min_curv    = std::max(0.01, this->get_parameter("vel_min_curv").as_double());
        VelocityProfiler::compute(
            final_path,
            vp,
            1e-4,
            curvature_resample_ds,
            curvature_filter_window,
            curvature_rate_limit);
    }

    // 7. Compute inverse kinematics on the same knots:
    //    derivative kappa(s), Ackermann steering delta, and acceleration.
    const double car_track_width = std::max(0.01, this->get_parameter("car_track_width").as_double());
    std::vector<WaypointKinematics> ik_result;
    if (success && !final_path.empty()) {
        ik_result = InverseKinematics::compute(
            final_path,
            planner_wheelbase,
            car_track_width,
            curvature_resample_ds,
            curvature_filter_window,
            curvature_rate_limit,
            steering_rate_limit,
            acceleration_filter_window,
            vel_jerk_max);
    }

    // 8. Assemble the explicit professor-style reference trajectory:
    //    <x, y, yaw/phi, v_ref, t, kappa, delta, a>.
    //    The MPC consumes the legacy aligned topics below; conceptually it is
    //    the vehicle model/controller block that follows this reference.
    std::vector<ReferenceTrajectory::Point> reference_trajectory;
    if (success && !final_path.empty()) {
        reference_trajectory = ReferenceTrajectory::assemble(final_path, ik_result);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 6. Construct the ROS 2 Messages
    const std::string frame_id = grid_snapshot.header.frame_id.empty() ? map_frame_ : grid_snapshot.header.frame_id;
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = frame_id;

    visualization_msgs::msg::MarkerArray marker_array;
    appendPoseMarker(marker_array, path_msg.header, start_pose_snapshot, 1000, 0.10f, 0.82f, 1.0f, "START");
    if (start_adjusted) {
        appendPoseMarker(
            marker_array, path_msg.header, planning_start_pose_snapshot, 1003, 1.0f, 0.65f, 0.10f, "SAFE START");
    } else {
        appendDeleteMarker(marker_array, path_msg.header, "planner_pose_arrows", 1003);
        appendDeleteMarker(marker_array, path_msg.header, "planner_pose_labels", 1003);
    }
    appendPoseMarker(marker_array, path_msg.header, goal_pose_snapshot, 1001, 0.12f, 1.0f, 0.45f, "GOAL");
    if (goal_adjusted) {
        appendPoseMarker(
            marker_array, path_msg.header, planning_goal_pose_snapshot, 1002, 1.0f, 0.65f, 0.10f, "SAFE GOAL");
    } else {
        appendDeleteMarker(marker_array, path_msg.header, "planner_pose_arrows", 1002);
        appendDeleteMarker(marker_array, path_msg.header, "planner_pose_labels", 1002);
    }

    if (success && !final_path.empty()) {
        RCLCPP_INFO(this->get_logger(), "Path found! Length: %zu nodes. Time: %ld ms", final_path.size(), duration.count());
        
        int id_counter = 0;
        double distance_since_last_footprint = 0.0;
        for (size_t i = 0; i < final_path.size(); ++i) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = path_msg.header;
            ps.pose.position.x = final_path[i].x;
            ps.pose.position.y = final_path[i].y;
            ps.pose.position.z = 0.0;
            
            tf2::Quaternion q;
            q.setRPY(0, 0, final_path[i].yaw);
            ps.pose.orientation = tf2::toMsg(q);
            path_msg.poses.push_back(ps);

            if (i > 0) {
                distance_since_last_footprint += std::hypot(
                    final_path[i].x - final_path[i - 1].x,
                    final_path[i].y - final_path[i - 1].y);
            }

            const bool is_last_pose = (i == final_path.size() - 1);
            const bool should_publish_footprint =
                (i == 0) || (distance_since_last_footprint >= footprint_spacing) || is_last_pose;

            // Build footprint boxes at a fixed traveled-distance spacing so
            // dense analytic samples do not collapse into a solid ribbon in RViz.
            if (should_publish_footprint) {
                visualization_msgs::msg::Marker m;
                m.header = path_msg.header;
                m.ns = "car_footprints";
                m.id = id_counter++;
                m.type = visualization_msgs::msg::Marker::CUBE;
                m.action = visualization_msgs::msg::Marker::ADD;
                m.pose = ps.pose;
                m.pose.position.z = 0.05;
                m.scale.x = c_len + 2.0 * footprint_display_margin;
                m.scale.y = c_wid + 2.0 * footprint_display_margin;
                m.scale.z = 0.05;

                // Use the direction field populated by the planner (1=forward, -1=reverse).
                // This is reliable even for densely-sampled analytic suffix points where
                // the displacement between consecutive samples is only 0.05 m.
                bool is_reversing = (final_path[i].direction < 0);
                if (is_reversing) {
                    m.color.r = 1.0; m.color.g = 0.35; m.color.b = 0.2; m.color.a = 0.40; 
                } else {
                    m.color.r = 0.1; m.color.g = 0.7; m.color.b = 1.0; m.color.a = 0.30; 
                }
                marker_array.markers.push_back(m);
                distance_since_last_footprint = 0.0;
            }
        }
        
        // Delete markers from previous plan that are beyond the current count.
        // Using the tracked previous count avoids both under-deletion (stale markers)
        // and over-deletion (wasted DELETE messages).
        for (int j = id_counter; j < last_footprint_count_; ++j) {
            visualization_msgs::msg::Marker m_del;
            m_del.header = path_msg.header;
            m_del.ns = "car_footprints";
            m_del.id = j;
            m_del.action = visualization_msgs::msg::Marker::DELETE;
            marker_array.markers.push_back(m_del);
        }
        last_footprint_count_ = id_counter;

        const Pose2D handoff_pose2d = chooseSafeHandoffPose(
            final_path,
            terminal_checker,
            collision_checker_,
            smoother_goal_lock_distance,
            chain_start_min_backtrack,
            terminal_center_clearance);
        const geometry_msgs::msg::Pose next_start_pose = poseFromPose2D(handoff_pose2d);

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_plan_end_pose_ = next_start_pose;
            has_last_plan_end_ = true;
        }
    } else {
        RCLCPP_WARN(this->get_logger(), "Planner failed to find a valid path. Time: %ld ms", duration.count());
        for (int j = 0; j < last_footprint_count_; ++j) {
            visualization_msgs::msg::Marker m_del;
            m_del.header = path_msg.header;
            m_del.ns = "car_footprints";
            m_del.id = j;
            m_del.action = visualization_msgs::msg::Marker::DELETE;
            marker_array.markers.push_back(m_del);
        }
        last_footprint_count_ = 0;
    }

    // Publish velocity profile — one entry per path pose, in the same order.
    // Subscribers (e.g. MPC controller) match indices with path_msg.poses[i].
    std_msgs::msg::Float64MultiArray vel_msg;
    vel_msg.data.reserve(final_path.size());
    for (const auto& pose : final_path) {
        vel_msg.data.push_back(pose.velocity);
    }
    velocity_pub_->publish(vel_msg);

    // Publish Ackermann inverse kinematics.
    // /astar_curvature uses the professor's derivative kappa(s) formula on the
    // B-spline smoothed path after uniform arc-length resampling and filtering.
    // Every array is index-aligned with path_msg.poses and astar_velocity_profile.
    std_msgs::msg::Float64MultiArray curv_msg, steer_avg_msg, steer_left_msg, steer_right_msg, accel_msg;
    curv_msg.data.reserve(ik_result.size());
    steer_avg_msg.data.reserve(ik_result.size());
    steer_left_msg.data.reserve(ik_result.size());
    steer_right_msg.data.reserve(ik_result.size());
    accel_msg.data.reserve(ik_result.size());
    for (const auto& wk : ik_result) {
        curv_msg.data.push_back(wk.curvature);
        steer_avg_msg.data.push_back(wk.delta_avg);
        steer_left_msg.data.push_back(wk.delta_left);
        steer_right_msg.data.push_back(wk.delta_right);
        accel_msg.data.push_back(wk.acceleration);
    }
    curvature_pub_->publish(curv_msg);
    steering_avg_pub_->publish(steer_avg_msg);
    steering_left_pub_->publish(steer_left_msg);
    steering_right_pub_->publish(steer_right_msg);
    acceleration_pub_->publish(accel_msg);

    // Clear, row-major reference trajectory topic:
    // [x, y, yaw, v_ref, t, kappa, delta, a] for every /astar_path pose.
    reference_trajectory_pub_->publish(
        ReferenceTrajectory::toMessage(reference_trajectory));

    path_pub_->publish(path_msg);
    footprint_pub_->publish(marker_array);
}

void PlannerNode::publishResetVisualization(
    const geometry_msgs::msg::Pose& start_pose,
    const std::string& frame_id,
    int footprints_to_delete) {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = frame_id.empty() ? map_frame_ : frame_id;
    path_pub_->publish(path_msg);
    reference_trajectory_pub_->publish(ReferenceTrajectory::toMessage({}));

    visualization_msgs::msg::MarkerArray marker_array;
    appendPoseMarker(marker_array, path_msg.header, start_pose, 1000, 0.10f, 0.82f, 1.0f, "START");
    appendDeleteMarker(marker_array, path_msg.header, "planner_pose_arrows", 1003);
    appendDeleteMarker(marker_array, path_msg.header, "planner_pose_labels", 1003);
    appendDeleteMarker(marker_array, path_msg.header, "planner_pose_arrows", 1001);
    appendDeleteMarker(marker_array, path_msg.header, "planner_pose_labels", 1001);
    appendDeleteMarker(marker_array, path_msg.header, "planner_pose_arrows", 1002);
    appendDeleteMarker(marker_array, path_msg.header, "planner_pose_labels", 1002);

    for (int j = 0; j < footprints_to_delete; ++j) {
        appendDeleteMarker(marker_array, path_msg.header, "car_footprints", j);
    }

    footprint_pub_->publish(marker_array);
}
