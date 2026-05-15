#include "hybrid_astar_cpp/mpc_node.hpp"

#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {
inline double wrapAngle(double a) { return std::atan2(std::sin(a), std::cos(a)); }

inline bool nearZero(double value, double eps = 1e-4) {
    return std::abs(value) <= eps;
}

std::vector<double> resampleProfileByIndex(const std::vector<double>& values,
                                           size_t target_size) {
    if (target_size == 0) return {};
    if (values.empty()) return {};
    if (values.size() == target_size) return values;
    if (values.size() == 1 || target_size == 1) {
        return std::vector<double>(target_size, values.front());
    }

    std::vector<double> out(target_size, 0.0);
    const double scale =
        static_cast<double>(values.size() - 1) / static_cast<double>(target_size - 1);
    for (size_t i = 0; i < target_size; ++i) {
        const double src = static_cast<double>(i) * scale;
        const size_t lo = static_cast<size_t>(std::floor(src));
        const size_t hi = std::min(lo + 1, values.size() - 1);
        const double alpha = src - static_cast<double>(lo);
        out[i] = values[lo] + alpha * (values[hi] - values[lo]);
    }
    return out;
}

int inferDirectionSign(const MpcController::State& pose,
                       const MpcController::State& next) {
    const double dx = next.x - pose.x;
    const double dy = next.y - pose.y;
    if (std::hypot(dx, dy) < 1e-6) return 1;

    const double forward_projection =
        std::cos(pose.yaw) * dx + std::sin(pose.yaw) * dy;
    return (forward_projection < -1e-4) ? -1 : 1;
}
}  // namespace

MpcNode::MpcNode() : Node("mpc_controller_node") {
    // -------- Parameters --------
    this->declare_parameter("control_rate_hz", 20.0);
    this->declare_parameter("mpc_horizon_N",   20);
    this->declare_parameter("mpc_dt",          0.1);
    this->declare_parameter("wheelbase",       2.7);
    this->declare_parameter("mass",            1500.0);
    this->declare_parameter("yaw_inertia",     2250.0);
    this->declare_parameter("lf",              1.2);
    this->declare_parameter("lr",              1.5);
    this->declare_parameter("Cf",              60000.0);
    this->declare_parameter("Cr",              60000.0);
    this->declare_parameter("vx_eps",          0.5);
    this->declare_parameter("slip_angle_limit", 0.5);
    this->declare_parameter("tire_force_limit", 8000.0);

    this->declare_parameter("q_x",      5.0);
    this->declare_parameter("q_y",      5.0);
    this->declare_parameter("q_yaw",    3.0);
    this->declare_parameter("q_v",      1.0);
    this->declare_parameter("q_vx",     this->get_parameter("q_v").as_double());
    this->declare_parameter("q_vy",     0.5);
    this->declare_parameter("q_r",      0.5);

    this->declare_parameter("r_a",      0.1);
    this->declare_parameter("r_delta",  1.0);

    this->declare_parameter("p_x",      50.0);
    this->declare_parameter("p_y",      50.0);
    this->declare_parameter("p_yaw",    10.0);
    this->declare_parameter("p_v",      5.0);
    this->declare_parameter("p_vx",     this->get_parameter("p_v").as_double());
    this->declare_parameter("p_vy",     2.0);
    this->declare_parameter("p_r",      2.0);

    this->declare_parameter("a_min",     -3.0);
    this->declare_parameter("a_max",      2.0);
    this->declare_parameter("delta_min", -0.6);
    this->declare_parameter("delta_max",  0.6);

    this->declare_parameter("fista_max_iter", 200);
    this->declare_parameter("fista_tol",      1e-4);

    this->declare_parameter("odom_topic",          "/odom");
    this->declare_parameter("cmd_topic",           "/mpc_cmd");
    this->declare_parameter("goal_reach_radius",   0.5);
    this->declare_parameter("goal_reach_speed",    0.3);
    this->declare_parameter("lookahead_capture_radius", 10.0);
    this->declare_parameter("launch_assist_accel", 0.8);
    this->declare_parameter("launch_assist_speed_threshold", 0.05);
    this->declare_parameter("launch_assist_ref_speed_threshold", 0.20);
    this->declare_parameter("goal_latch_min_travel", 0.2);

    // -------- Controller --------
    MpcController::Params p;
    p.horizon_N = this->get_parameter("mpc_horizon_N").as_int();
    p.dt        = this->get_parameter("mpc_dt").as_double();
    p.wheelbase = this->get_parameter("wheelbase").as_double();
    p.mass      = this->get_parameter("mass").as_double();
    p.yaw_inertia = this->get_parameter("yaw_inertia").as_double();
    p.lf        = this->get_parameter("lf").as_double();
    p.lr        = this->get_parameter("lr").as_double();
    p.Cf        = this->get_parameter("Cf").as_double();
    p.Cr        = this->get_parameter("Cr").as_double();
    p.vx_eps    = this->get_parameter("vx_eps").as_double();
    p.slip_angle_limit = this->get_parameter("slip_angle_limit").as_double();
    p.tire_force_limit = this->get_parameter("tire_force_limit").as_double();
    p.q_x       = this->get_parameter("q_x").as_double();
    p.q_y       = this->get_parameter("q_y").as_double();
    p.q_yaw     = this->get_parameter("q_yaw").as_double();
    p.q_vx      = this->get_parameter("q_vx").as_double();
    p.q_vy      = this->get_parameter("q_vy").as_double();
    p.q_r       = this->get_parameter("q_r").as_double();
    p.r_a       = this->get_parameter("r_a").as_double();
    p.r_delta   = this->get_parameter("r_delta").as_double();
    p.p_x       = this->get_parameter("p_x").as_double();
    p.p_y       = this->get_parameter("p_y").as_double();
    p.p_yaw     = this->get_parameter("p_yaw").as_double();
    p.p_vx      = this->get_parameter("p_vx").as_double();
    p.p_vy      = this->get_parameter("p_vy").as_double();
    p.p_r       = this->get_parameter("p_r").as_double();
    p.a_min     = this->get_parameter("a_min").as_double();
    p.a_max     = this->get_parameter("a_max").as_double();
    p.delta_min = this->get_parameter("delta_min").as_double();
    p.delta_max = this->get_parameter("delta_max").as_double();
    p.max_iter  = this->get_parameter("fista_max_iter").as_int();
    p.tol       = this->get_parameter("fista_tol").as_double();
    mpc_ = std::make_unique<MpcController>(p);
    wheelbase_ = std::max(1e-3, p.wheelbase);
    goal_reach_radius_ = this->get_parameter("goal_reach_radius").as_double();
    goal_reach_speed_  = this->get_parameter("goal_reach_speed").as_double();
    lookahead_capture_radius_ = std::max(
        0.0, this->get_parameter("lookahead_capture_radius").as_double());
    launch_assist_accel_ = std::max(
        0.0, this->get_parameter("launch_assist_accel").as_double());
    launch_assist_speed_threshold_ = std::max(
        0.0, this->get_parameter("launch_assist_speed_threshold").as_double());
    launch_assist_ref_speed_threshold_ = std::max(
        0.0, this->get_parameter("launch_assist_ref_speed_threshold").as_double());
    goal_latch_min_travel_ = std::max(
        0.0, this->get_parameter("goal_latch_min_travel").as_double());

    // -------- Subscriptions --------
    const std::string odom_topic = this->get_parameter("odom_topic").as_string();
    const std::string cmd_topic  = this->get_parameter("cmd_topic").as_string();

    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "astar_path", 10,
        std::bind(&MpcNode::pathCallback, this, std::placeholders::_1));
    vel_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "astar_velocity_profile", 10,
        std::bind(&MpcNode::velocityCallback, this, std::placeholders::_1));
    steer_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "astar_steering", 10,
        std::bind(&MpcNode::steeringCallback, this, std::placeholders::_1));
    accel_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "astar_acceleration", 10,
        std::bind(&MpcNode::accelerationCallback, this, std::placeholders::_1));
    reference_traj_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "astar_reference_trajectory", 10,
        std::bind(&MpcNode::referenceTrajectoryCallback, this, std::placeholders::_1));
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, 10,
        std::bind(&MpcNode::odomCallback, this, std::placeholders::_1));

    // -------- Publishers --------
    cmd_pub_       = this->create_publisher<std_msgs::msg::Float64MultiArray>(cmd_topic, 10);
    predicted_pub_ = this->create_publisher<nav_msgs::msg::Path>("mpc_predicted_path", 10);
    error_pub_     = this->create_publisher<std_msgs::msg::Float64MultiArray>("mpc_tracking_error", 10);
    reference_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("mpc_reference_window", 10);

    // -------- Control timer --------
    const double rate_hz = std::max(1.0, this->get_parameter("control_rate_hz").as_double());
    control_dt_ = 1.0 / rate_hz;
    control_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(control_dt_),
        std::bind(&MpcNode::controlTick, this));

    RCLCPP_INFO(this->get_logger(),
        "Dynamic bicycle MPC node up: N=%d dt=%.3f rate=%.1fHz", p.horizon_N, p.dt, rate_hz);
}

// -------------------------------------------------------------------------
// Callbacks
// -------------------------------------------------------------------------
void MpcNode::pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    if (has_reference_trajectory_) {
        return;
    }
    path_states_.clear();
    path_states_.reserve(msg->poses.size());
    path_direction_sign_.clear();
    path_direction_sign_.reserve(msg->poses.size());
    for (const auto& ps : msg->poses) {
        MpcController::State s;
        s.x   = ps.pose.position.x;
        s.y   = ps.pose.position.y;
        s.yaw = tf2::getYaw(ps.pose.orientation);
        s.vx  = 0.0;     // filled by velocityCallback
        s.vy  = 0.0;
        s.r   = 0.0;     // updated after steeringCallback
        path_states_.push_back(s);
    }
    path_direction_sign_.assign(path_states_.size(), 1);
    for (size_t i = 0; i + 1 < path_states_.size(); ++i) {
        path_direction_sign_[i] =
            inferDirectionSign(path_states_[i], path_states_[i + 1]);
    }
    if (path_direction_sign_.size() >= 2) {
        path_direction_sign_.back() = path_direction_sign_[path_direction_sign_.size() - 2];
    }
    path_frame_id_ = msg->header.frame_id;
    has_path_     = !path_states_.empty();
    has_vel_      = false;
    has_steer_    = false;
    has_accel_    = false;
    velocity_profile_size_ = 0;
    steering_profile_size_ = 0;
    acceleration_profile_size_ = 0;
    path_delta_ref_.clear();
    path_a_ref_.clear();
    goal_reached_ = false;   // new path resets goal-reached latch
    mpc_ready_logged_ = false;
    steering_fallback_warned_ = false;
    acceleration_fallback_warned_ = false;
    path_follow_start_valid_ = false;
    path_follow_travel_ = 0.0;
    last_nearest_ref_valid_ = false;
    last_zero_reason_.clear();
    rebuildTimeAxis();
}

void MpcNode::referenceTrajectoryCallback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    constexpr size_t kFieldsPerPoint = 8;
    if (msg->data.empty() || (msg->data.size() % kFieldsPerPoint) != 0) {
        RCLCPP_WARN(
            this->get_logger(),
            "MPC reference trajectory ignored: expected a non-empty array "
            "with size multiple of 8, got %zu values.",
            msg->data.size());
        return;
    }
    const size_t n = msg->data.size() / kFieldsPerPoint;
    if (n < 2) {
        RCLCPP_WARN(
            this->get_logger(),
            "MPC reference trajectory ignored: need at least 2 points, got %zu.",
            n);
        return;
    }

    std::lock_guard<std::mutex> lock(ref_mutex_);

    path_states_.clear();
    path_states_.reserve(n);
    path_direction_sign_.clear();
    path_direction_sign_.reserve(n);
    path_delta_ref_.clear();
    path_delta_ref_.reserve(n);
    path_a_ref_.clear();
    path_a_ref_.reserve(n);
    path_time_.clear();
    path_time_.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const size_t offset = kFieldsPerPoint * i;
        const double x = msg->data[offset + 0];
        const double y = msg->data[offset + 1];
        const double yaw = msg->data[offset + 2];
        const double v_ref = msg->data[offset + 3];
        const double t = msg->data[offset + 4];
        const double kappa = msg->data[offset + 5];
        const double delta = msg->data[offset + 6];
        const double accel = msg->data[offset + 7];

        MpcController::State s;
        s.x = x;
        s.y = y;
        s.yaw = yaw;
        s.vx = v_ref;
        s.vy = 0.0;
        s.r = v_ref * kappa;
        path_states_.push_back(s);

        path_direction_sign_.push_back(v_ref >= 0.0 ? 1 : -1);
        path_delta_ref_.push_back(delta);
        path_a_ref_.push_back(accel);
        path_time_.push_back(t);
    }

    velocity_profile_size_ = n;
    steering_profile_size_ = n;
    acceleration_profile_size_ = n;
    has_path_ = true;
    has_vel_ = true;
    has_steer_ = true;
    has_accel_ = true;
    has_reference_trajectory_ = true;
    goal_reached_ = false;
    mpc_ready_logged_ = false;
    steering_fallback_warned_ = false;
    acceleration_fallback_warned_ = false;
    path_follow_start_valid_ = false;
    path_follow_travel_ = 0.0;
    last_nearest_ref_valid_ = false;
    last_reference_failure_reason_.clear();
    last_zero_reason_.clear();

    const double duration =
        path_time_.empty() ? 0.0 : (path_time_.back() - path_time_.front());
    RCLCPP_INFO(
        this->get_logger(),
        "MPC reference trajectory received: %zu points, duration %.3f s",
        n, duration);
}

void MpcNode::velocityCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    if (has_reference_trajectory_) {
        return;
    }
    velocity_profile_size_ = msg->data.size();
    if (!has_path_) return;
    std::vector<double> velocity = msg->data;
    if (!velocity.empty() && velocity.size() != path_states_.size()) {
        RCLCPP_WARN(
            this->get_logger(),
            "MPC warning: /astar_velocity_profile length mismatch: "
            "velocity=%zu, path=%zu; resampling velocity profile to path size.",
            velocity.size(), path_states_.size());
        velocity = resampleProfileByIndex(velocity, path_states_.size());
    }
    const size_t n = std::min(velocity.size(), path_states_.size());
    for (size_t i = 0; i < n; ++i) {
        const int dir = (i < path_direction_sign_.size()) ? path_direction_sign_[i] : 1;
        path_states_[i].vx = static_cast<double>(dir) * std::abs(velocity[i]);
        path_states_[i].vy = 0.0;
    }
    updateReferenceYawRates();
    has_vel_ = has_path_ && velocity.size() == path_states_.size();
    rebuildTimeAxis();
}

void MpcNode::steeringCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    if (has_reference_trajectory_) {
        return;
    }
    steering_profile_size_ = msg->data.size();
    std::vector<double> steering = msg->data;
    if (has_path_ && !steering.empty() && steering.size() != path_states_.size()) {
        RCLCPP_WARN(
            this->get_logger(),
            "MPC warning: /astar_steering length mismatch: steering=%zu, "
            "path=%zu; resampling steering feedforward to path size.",
            steering.size(), path_states_.size());
        steering = resampleProfileByIndex(steering, path_states_.size());
    }
    path_delta_ref_ = steering;
    const size_t n = std::min(path_delta_ref_.size(), path_direction_sign_.size());
    for (size_t i = 0; i < n; ++i) {
        // For reverse, vx is negative, so feedforward steering flips sign to
        // track the same geometric curvature.
        path_delta_ref_[i] *= static_cast<double>(path_direction_sign_[i]);
    }
    updateReferenceYawRates();
    has_steer_ = has_path_ && path_delta_ref_.size() == path_states_.size();
}

void MpcNode::accelerationCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    if (has_reference_trajectory_) {
        return;
    }
    acceleration_profile_size_ = msg->data.size();
    std::vector<double> acceleration = msg->data;
    if (has_path_ && !acceleration.empty() && acceleration.size() != path_states_.size()) {
        RCLCPP_WARN(
            this->get_logger(),
            "MPC warning: /astar_acceleration length mismatch: acceleration=%zu, "
            "path=%zu; resampling acceleration feedforward to path size.",
            acceleration.size(), path_states_.size());
        acceleration = resampleProfileByIndex(acceleration, path_states_.size());
    }
    path_a_ref_ = acceleration;
    const size_t n = std::min(path_a_ref_.size(), path_direction_sign_.size());
    for (size_t i = 0; i < n; ++i) {
        // /astar_acceleration is along the path-speed magnitude.  MPC tracks
        // signed body-frame velocity, so reverse acceleration is signed too.
        path_a_ref_[i] *= static_cast<double>(path_direction_sign_[i]);
    }
    has_accel_ = has_path_ && path_a_ref_.size() == path_states_.size();
}

void MpcNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_.x   = msg->pose.pose.position.x;
    current_state_.y   = msg->pose.pose.position.y;
    current_state_.yaw = tf2::getYaw(msg->pose.pose.orientation);
    current_state_.vx  = msg->twist.twist.linear.x;
    current_state_.vy  = msg->twist.twist.linear.y;
    current_state_.r   = msg->twist.twist.angular.z;
    has_odom_ = true;
}

void MpcNode::logZeroCommandReason(const char* reason) {
    // Log immediately whenever the reason changes, otherwise throttle so the
    // same reason is re-emitted at most once every 2 s.
    if (last_zero_reason_ != reason) {
        RCLCPP_WARN(this->get_logger(), "%s", reason);
        last_zero_reason_ = reason;
        last_zero_log_time_ = this->now();
        return;
    }
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000, "%s", reason);
}

void MpcNode::publishZeroCommand(const char* reason) {
    logZeroCommandReason(reason);
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data = {0.0, 0.0};
    cmd_pub_->publish(cmd);
}

void MpcNode::normalizeReferenceProfilesLocked() {
    if (!has_path_ || path_states_.empty() || !has_vel_) {
        return;
    }

    const size_t n = path_states_.size();
    if (!path_delta_ref_.empty() && path_delta_ref_.size() != n) {
        RCLCPP_WARN(
            this->get_logger(),
            "MPC warning: stored steering feedforward length mismatch: "
            "steering=%zu, path=%zu; resampling steering feedforward to path size.",
            path_delta_ref_.size(), n);
        path_delta_ref_ = resampleProfileByIndex(path_delta_ref_, n);
        has_steer_ = path_delta_ref_.size() == n;
        updateReferenceYawRates();
    }

    if (!path_a_ref_.empty() && path_a_ref_.size() != n) {
        RCLCPP_WARN(
            this->get_logger(),
            "MPC warning: stored acceleration feedforward length mismatch: "
            "acceleration=%zu, path=%zu; resampling acceleration feedforward to path size.",
            path_a_ref_.size(), n);
        path_a_ref_ = resampleProfileByIndex(path_a_ref_, n);
        has_accel_ = path_a_ref_.size() == n;
    }

    if (!has_steer_ && steering_profile_size_ == 0) {
        path_delta_ref_.assign(n, 0.0);
        has_steer_ = true;
        updateReferenceYawRates();
        if (!steering_fallback_warned_) {
            RCLCPP_WARN(
                this->get_logger(),
                "MPC warning: steering feedforward missing, using zeros.");
            steering_fallback_warned_ = true;
        }
    }

    if (!has_accel_ && acceleration_profile_size_ == 0) {
        path_a_ref_.assign(n, 0.0);
        has_accel_ = true;
        if (!acceleration_fallback_warned_) {
            RCLCPP_WARN(
                this->get_logger(),
                "MPC warning: acceleration feedforward missing, using zeros.");
            acceleration_fallback_warned_ = true;
        }
    }
}

void MpcNode::updateReferenceYawRates() {
    if (path_states_.empty() || path_delta_ref_.empty()) {
        for (auto& s : path_states_) {
            s.vy = 0.0;
            s.r = 0.0;
        }
        return;
    }

    for (size_t i = 0; i < path_states_.size(); ++i) {
        const size_t idx = std::min(i, path_delta_ref_.size() - 1);
        path_states_[i].vy = 0.0;
        path_states_[i].r = path_states_[i].vx * std::tan(path_delta_ref_[idx]) / wheelbase_;
    }
}

// -------------------------------------------------------------------------
// Time-axis parameterisation of the path
//    t_{i+1} = t_i + ds_i / v_avg_i           (v_avg guarded ≥ 0.1 m/s)
// -------------------------------------------------------------------------
void MpcNode::rebuildTimeAxis() {
    const size_t n = path_states_.size();
    path_time_.assign(n, 0.0);
    if (n < 2) return;
    for (size_t i = 1; i < n; ++i) {
        const double ds = std::hypot(path_states_[i].x - path_states_[i-1].x,
                                     path_states_[i].y - path_states_[i-1].y);
        const double v1 = std::abs(path_states_[i-1].vx);
        const double v2 = std::abs(path_states_[i].vx);
        const double v_avg = std::max(0.1, 0.5 * (v1 + v2));
        path_time_[i] = path_time_[i-1] + ds / v_avg;
    }
}

// -------------------------------------------------------------------------
// Reference horizon — samples at t₀ + k·dt for k = 0..N using linear interp.
// -------------------------------------------------------------------------
bool MpcNode::buildReferenceHorizon(
    const MpcController::State& current,
    std::vector<MpcController::State>&   ref_x,
    std::vector<MpcController::Control>& ref_u)
{
    const int N  = mpc_->params().horizon_N;
    const double dt = mpc_->params().dt;
    last_reference_failure_reason_.clear();
    last_nearest_ref_valid_ = false;

    auto log_failure = [&](const char* detail) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "buildReferenceHorizon failed: %s. current=(x=%.2f, y=%.2f, "
            "yaw=%.2f rad, vx=%.2f m/s), nearest_ref=(x=%.2f, y=%.2f, "
            "yaw=%.2f rad, vx=%.2f m/s, valid=%s), nearest_dist=%.2f m, "
            "lookahead_capture_radius=%.2f m, path size=%zu, "
            "path_time size=%zu, velocity=%zu, steering=%zu, acceleration=%zu",
            detail, current.x, current.y, current.yaw, current.vx,
            last_nearest_ref_state_.x, last_nearest_ref_state_.y,
            last_nearest_ref_state_.yaw, last_nearest_ref_state_.vx,
            last_nearest_ref_valid_ ? "yes" : "no",
            last_nearest_ref_dist_, lookahead_capture_radius_,
            path_states_.size(), path_time_.size(), velocity_profile_size_,
            steering_profile_size_, acceleration_profile_size_);
    };

    last_nearest_ref_dist_ = std::numeric_limits<double>::quiet_NaN();
    last_nearest_ref_index_ = -1;
    last_nearest_ref_state_ = MpcController::State{};
    if (!path_states_.empty()) {
        int    i_cl  = 0;
        double d2_cl = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < path_states_.size(); ++i) {
            const double dx = path_states_[i].x - current.x;
            const double dy = path_states_[i].y - current.y;
            const double d2 = dx*dx + dy*dy;
            if (d2 < d2_cl) { d2_cl = d2; i_cl = static_cast<int>(i); }
        }
        last_nearest_ref_dist_ = std::sqrt(d2_cl);
        last_nearest_ref_index_ = i_cl;
        last_nearest_ref_state_ = path_states_[static_cast<size_t>(i_cl)];
        last_nearest_ref_valid_ = true;
    }

    if (path_states_.size() < 2) {
        last_reference_failure_reason_ =
            "reference path has fewer than 2 points";
        log_failure(last_reference_failure_reason_.c_str());
        return false;
    }
    if (path_time_.size() != path_states_.size()) {
        last_reference_failure_reason_ =
            "path_time size does not match reference path size";
        log_failure(last_reference_failure_reason_.c_str());
        return false;
    }
    if (path_delta_ref_.size() != path_states_.size()) {
        last_reference_failure_reason_ =
            "steering profile size does not match reference path size";
        log_failure(last_reference_failure_reason_.c_str());
        return false;
    }
    if (path_a_ref_.size() != path_states_.size()) {
        last_reference_failure_reason_ =
            "acceleration profile size does not match reference path size";
        log_failure(last_reference_failure_reason_.c_str());
        return false;
    }

    const int i_cl = last_nearest_ref_index_;
    if (last_nearest_ref_dist_ > lookahead_capture_radius_) {
        last_reference_failure_reason_ =
            "nearest reference point is outside lookahead_capture_radius";
        log_failure(last_reference_failure_reason_.c_str());
        return false;
    }

    ref_x.assign(N + 1, MpcController::State{});
    ref_u.assign(N,     MpcController::Control{});

    const double t_start = path_time_[i_cl];

    // Running pointer into path (monotonic)
    int i = i_cl;
    const int n = static_cast<int>(path_states_.size());

    for (int k = 0; k <= N; ++k) {
        const double t_q = t_start + k * dt;

        // Advance i until t_q is between path_time_[i] and path_time_[i+1]
        while (i < n - 1 && path_time_[i + 1] < t_q) ++i;

        if (i >= n - 1) {
            // Beyond end of path — freeze on last waypoint (stopped, δ=0, a=0)
            ref_x[k] = path_states_.back();
            ref_x[k].vx = 0.0;
            ref_x[k].vy = 0.0;
            ref_x[k].r = 0.0;
            if (k < N) { ref_u[k].a = 0.0; ref_u[k].delta = 0.0; }
            continue;
        }

        const double t0 = path_time_[i];
        const double t1 = path_time_[i + 1];
        const double dt_seg = std::max(1e-6, t1 - t0);
        const double alpha = std::clamp((t_q - t0) / dt_seg, 0.0, 1.0);

        const auto& s0 = path_states_[i];
        const auto& s1 = path_states_[i + 1];
        ref_x[k].x   = s0.x + alpha * (s1.x - s0.x);
        ref_x[k].y   = s0.y + alpha * (s1.y - s0.y);
        ref_x[k].yaw = s0.yaw + alpha * wrapAngle(s1.yaw - s0.yaw);
        ref_x[k].vx  = s0.vx + alpha * (s1.vx - s0.vx);
        ref_x[k].vy  = 0.0;
        ref_x[k].r   = s0.r + alpha * (s1.r - s0.r);

        // Reference controls — step-constant per segment (feedforward from IK)
        if (k < N) {
            const size_t idx = std::min<size_t>(path_a_ref_.size() - 1, i);
            ref_u[k].a     = path_a_ref_[idx];
            ref_u[k].delta = path_delta_ref_[std::min<size_t>(path_delta_ref_.size() - 1, i)];
        }
    }
    return true;
}

// -------------------------------------------------------------------------
// Control tick — runs at control_rate_hz
// -------------------------------------------------------------------------
void MpcNode::controlTick() {
    MpcController::State x0;
    bool odom_ready = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        x0 = current_state_;
        odom_ready = has_odom_;
    }

    {
        std::lock_guard<std::mutex> lock(ref_mutex_);
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "MPC readiness: has_odom=%s has_path=%s has_vel=%s "
            "has_steer=%s has_accel=%s has_reference_trajectory=%s "
            "path_states=%zu path_a_ref=%zu path_delta_ref=%zu path_time=%zu",
            odom_ready ? "true" : "false",
            has_path_ ? "true" : "false",
            has_vel_ ? "true" : "false",
            has_steer_ ? "true" : "false",
            has_accel_ ? "true" : "false",
            has_reference_trajectory_ ? "true" : "false",
            path_states_.size(), path_a_ref_.size(),
            path_delta_ref_.size(), path_time_.size());
    }

    if (!odom_ready) {
        publishZeroCommand("MPC zero: waiting for odom");
        return;
    }

    // Goal-reached: once latched, brake to a stop and then hold zero until a new path arrives.
    double goal_dist = std::numeric_limits<double>::quiet_NaN();
    {
        std::lock_guard<std::mutex> lock(ref_mutex_);
        if (has_path_) {
            const auto& goal = path_states_.back();
            goal_dist = std::hypot(x0.x - goal.x, x0.y - goal.y);

            if (!path_follow_start_valid_) {
                path_follow_start_x_ = x0.x;
                path_follow_start_y_ = x0.y;
                path_follow_start_valid_ = true;
                path_follow_travel_ = 0.0;
            } else {
                path_follow_travel_ = std::max(
                    path_follow_travel_,
                    std::hypot(x0.x - path_follow_start_x_,
                               x0.y - path_follow_start_y_));
            }
        }
        if (has_path_ && !goal_reached_) {
            const bool goal_pose_and_speed_match =
                goal_dist < goal_reach_radius_ &&
                std::abs(x0.vx) < goal_reach_speed_;
            if (goal_pose_and_speed_match &&
                path_follow_travel_ >= goal_latch_min_travel_) {
                goal_reached_ = true;
                RCLCPP_WARN(
                    this->get_logger(),
                    "MPC marked goal already reached: distance_to_goal=%.3f m, "
                    "current_vx=%.3f m/s, goal_reach_radius=%.3f m, "
                    "goal_reach_speed=%.3f m/s, path_follow_travel=%.3f m, "
                    "path=%zu",
                    goal_dist, x0.vx, goal_reach_radius_, goal_reach_speed_,
                    path_follow_travel_, path_states_.size());
            } else if (goal_pose_and_speed_match) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "MPC goal latch suppressed at startup: distance_to_goal=%.3f m, "
                    "current_vx=%.3f m/s, goal_reach_radius=%.3f m, "
                    "goal_reach_speed=%.3f m/s, path_follow_travel=%.3f m, "
                    "required_travel=%.3f m",
                    goal_dist, x0.vx, goal_reach_radius_, goal_reach_speed_,
                    path_follow_travel_, goal_latch_min_travel_);
            }
        }
    }
    if (goal_reached_) {
        // Apply gentle braking until fully stopped, then hold zero.
        const double brake = (std::abs(x0.vx) > 0.05) ? -std::copysign(1.0, x0.vx) : 0.0;
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "MPC goal already reached; publishing hold/brake /mpc_cmd. "
            "command=(a=%.3f, delta=0.000), "
            "distance_to_goal=%.3f m, current_vx=%.3f m/s, "
            "goal_reach_radius=%.3f m, goal_reach_speed=%.3f m/s",
            brake, goal_dist, x0.vx, goal_reach_radius_, goal_reach_speed_);
        if (nearZero(brake)) {
            publishZeroCommand("MPC zero: goal already reached");
            return;
        }
        std_msgs::msg::Float64MultiArray cmd;
        cmd.data = {brake, 0.0};
        cmd_pub_->publish(cmd);
        return;
    }

    std::vector<MpcController::State>   ref_x;
    std::vector<MpcController::Control> ref_u;
    bool ready = false;
    std::string not_ready_reason;
    const char* zero_reason = "MPC zero: buildReferenceHorizon failed";
    double nearest_ref_dist = std::numeric_limits<double>::quiet_NaN();
    int nearest_ref_index = -1;
    size_t path_points = 0;
    {
        std::lock_guard<std::mutex> lock(ref_mutex_);
        normalizeReferenceProfilesLocked();
        path_points = path_states_.size();
        if (has_path_) {
            double best_d2 = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < path_states_.size(); ++i) {
                const double dx = path_states_[i].x - x0.x;
                const double dy = path_states_[i].y - x0.y;
                const double d2 = dx * dx + dy * dy;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    nearest_ref_index = static_cast<int>(i);
                }
            }
            nearest_ref_dist = std::sqrt(best_d2);
        }

        const bool reference_inputs_ready =
            has_path_ && has_vel_ && has_steer_ && has_accel_;
        if (!reference_inputs_ready) {
            not_ready_reason =
                "waiting for /astar_reference_trajectory (or full fallback "
                "set /astar_path + /astar_velocity_profile + /astar_steering "
                "+ /astar_acceleration); has_path=" +
                std::string(has_path_ ? "true" : "false") +
                ", has_vel=" + (has_vel_ ? "true" : "false") +
                ", has_steer=" + (has_steer_ ? "true" : "false") +
                ", has_accel=" + (has_accel_ ? "true" : "false") +
                ", has_reference_trajectory=" +
                (has_reference_trajectory_ ? "true" : "false");
            zero_reason = "MPC zero: waiting for reference trajectory";
        } else if (path_states_.size() < 2) {
            not_ready_reason =
                "reference has fewer than 2 points (size=" +
                std::to_string(path_states_.size()) + ")";
            zero_reason = "MPC zero: reference too short";
        } else {
            ready = buildReferenceHorizon(x0, ref_x, ref_u);
            nearest_ref_dist = last_nearest_ref_dist_;
            nearest_ref_index = last_nearest_ref_index_;
            if (!ready) {
                not_ready_reason = last_reference_failure_reason_.empty()
                    ? "buildReferenceHorizon returned false"
                    : last_reference_failure_reason_;
                zero_reason = "MPC zero: buildReferenceHorizon failed";
            }
        }

        if (ready && !mpc_ready_logged_) {
            const auto& ref0 = ref_x.front();
            const auto& u0 = ref_u.front();
            RCLCPP_INFO(
                this->get_logger(),
                "MPC ready: path=%zu, velocity=OK, steering=OK, "
                "acceleration=OK, odom=OK",
                path_states_.size());
            RCLCPP_INFO(
                this->get_logger(),
                "MPC ready detail: profile_sizes={velocity:%zu, steering:%zu, "
                "acceleration:%zu}, nearest_ref_dist=%.3f m, "
                "current=(%.2f, %.2f, yaw=%.2f, vx=%.2f), "
                "ref0=(%.2f, %.2f, yaw=%.2f, vx=%.2f), "
                "u_ref0=(a=%.3f, delta=%.3f)",
                velocity_profile_size_, steering_profile_size_,
                acceleration_profile_size_,
                nearest_ref_dist, x0.x, x0.y, x0.yaw, x0.vx,
                ref0.x, ref0.y, ref0.yaw, ref0.vx, u0.a, u0.delta);
            mpc_ready_logged_ = true;
        }
    }

    if (!ready) {
        MpcController::State nearest_ref;
        bool nearest_valid = false;
        bool has_path_snapshot = false;
        bool has_vel_snapshot = false;
        bool has_steer_snapshot = false;
        bool has_accel_snapshot = false;
        bool has_reference_trajectory_snapshot = false;
        size_t path_states_size = 0;
        size_t path_delta_size = 0;
        size_t path_a_size = 0;
        size_t path_time_size = 0;
        size_t velocity_size = 0;
        size_t steering_size = 0;
        size_t acceleration_size = 0;
        {
            std::lock_guard<std::mutex> lock(ref_mutex_);
            nearest_ref = last_nearest_ref_state_;
            nearest_valid = last_nearest_ref_valid_;
            has_path_snapshot = has_path_;
            has_vel_snapshot = has_vel_;
            has_steer_snapshot = has_steer_;
            has_accel_snapshot = has_accel_;
            has_reference_trajectory_snapshot = has_reference_trajectory_;
            path_states_size = path_states_.size();
            path_delta_size = path_delta_ref_.size();
            path_a_size = path_a_ref_.size();
            path_time_size = path_time_.size();
            velocity_size = velocity_profile_size_;
            steering_size = steering_profile_size_;
            acceleration_size = acceleration_profile_size_;
        }
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "MPC publishing zero /mpc_cmd: %s. current=(%.2f, %.2f, "
            "yaw=%.2f rad, vx=%.2f m/s), nearest_ref_dist=%.2f m "
            "(index=%d/%zu), nearest_ref=(%.2f, %.2f, yaw=%.2f rad, "
            "vx=%.2f m/s, valid=%s), lookahead_capture_radius=%.2f m, "
            "ready_flags={has_odom:%s, has_path:%s, has_vel:%s, "
            "has_steer:%s, has_accel:%s, has_reference_trajectory:%s}, "
            "reference_sizes={path_states:%zu, path_delta_ref:%zu, "
            "path_a_ref:%zu, path_time:%zu}, "
            "profile_sizes={velocity:%zu, steering:%zu, acceleration:%zu}",
            not_ready_reason.c_str(), x0.x, x0.y, x0.yaw, x0.vx,
            nearest_ref_dist, nearest_ref_index, path_points,
            nearest_ref.x, nearest_ref.y, nearest_ref.yaw, nearest_ref.vx,
            nearest_valid ? "yes" : "no", lookahead_capture_radius_,
            odom_ready ? "true" : "false",
            has_path_snapshot ? "true" : "false",
            has_vel_snapshot ? "true" : "false",
            has_steer_snapshot ? "true" : "false",
            has_accel_snapshot ? "true" : "false",
            has_reference_trajectory_snapshot ? "true" : "false",
            path_states_size, path_delta_size, path_a_size, path_time_size,
            velocity_size, steering_size, acceleration_size);

        // No usable reference horizon -> publish zero command.
        publishZeroCommand(zero_reason);
        return;
    }

    // -- Solve MPC --
    auto result = mpc_->solve(x0, ref_x, ref_u);
    if (!result.success) {
        const auto ref0 = ref_x.empty() ? MpcController::State{} : ref_x.front();
        const auto u0 = ref_u.empty() ? MpcController::Control{} : ref_u.front();
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "MPC solver failed; publishing zero /mpc_cmd. success=false, "
            "reason=%s, current=(x=%.2f, y=%.2f, yaw=%.2f rad, "
            "vx=%.2f m/s, vy=%.2f m/s, r=%.2f rad/s), "
            "first_ref=(x=%.2f, y=%.2f, yaw=%.2f rad, vx=%.2f m/s, "
            "vy=%.2f m/s, r=%.2f rad/s), first_ref_input=(a=%.3f, "
            "delta=%.3f), horizon_states=%zu, horizon_controls=%zu, "
            "iterations=%d, solve_time=%.3f ms",
            result.failure_reason.empty() ? "unknown" : result.failure_reason.c_str(),
            x0.x, x0.y, x0.yaw, x0.vx, x0.vy, x0.r,
            ref0.x, ref0.y, ref0.yaw, ref0.vx, ref0.vy, ref0.r,
            u0.a, u0.delta, ref_x.size(), ref_u.size(),
            result.iterations, result.solve_time_ms);
        publishZeroCommand("MPC zero: solver failed");
        return;
    }

    if (std::abs(x0.vx) < 0.05 &&
        ref_x.size() > 1 &&
        !ref_u.empty() &&
        ref_x[1].vx > 0.1 &&
        std::abs(result.u.a) < 0.05) {
        result.u.a = 0.4;
        result.u.delta = ref_u[0].delta;
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "MPC startup assist: applying initial acceleration");
    }

    // -- Publish command [a, δ] --
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data = {result.u.a, result.u.delta};

    if (nearZero(cmd.data[0]) && nearZero(cmd.data[1])) {
        logZeroCommandReason("MPC zero: solver failed");
        const auto& ref0 = ref_x.front();
        const auto& refN = ref_x.back();
        const auto& u0 = ref_u.front();
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "MPC publishing near-zero /mpc_cmd after successful solve. "
            "solver_command=(a=%.6f, delta=%.6f), published_command=(a=%.6f, "
            "delta=%.6f), current=(x=%.2f, y=%.2f, "
            "yaw=%.2f rad, vx=%.3f m/s), ref0=(x=%.2f, y=%.2f, "
            "yaw=%.2f rad, vx=%.3f m/s), refN=(x=%.2f, y=%.2f, "
            "yaw=%.2f rad, vx=%.3f m/s), u_ref0=(a=%.3f, delta=%.3f), "
            "iterations=%d, cost=%.6f, solve_time=%.3f ms",
            result.u.a, result.u.delta, cmd.data[0], cmd.data[1],
            x0.x, x0.y, x0.yaw, x0.vx,
            ref0.x, ref0.y, ref0.yaw, ref0.vx,
            refN.x, refN.y, refN.yaw, refN.vx,
            u0.a, u0.delta, result.iterations, result.cost,
            result.solve_time_ms);
    }
    cmd_pub_->publish(cmd);

    // -- Publish predicted path (for RViz) --
    nav_msgs::msg::Path pred;
    pred.header.stamp = this->now();
    pred.header.frame_id = path_frame_id_.empty() ? "map" : path_frame_id_;
    pred.poses.reserve(result.predicted_x.size());
    for (const auto& s : result.predicted_x) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = pred.header;
        ps.pose.position.x = s.x;
        ps.pose.position.y = s.y;
        tf2::Quaternion q; q.setRPY(0, 0, s.yaw);
        ps.pose.orientation = tf2::toMsg(q);
        pred.poses.push_back(ps);
    }
    predicted_pub_->publish(pred);

    // -- Publish tracking errors  [e_lat, e_yaw, e_vx] --
    //    Cross-track (lateral) error in the reference frame:
    //    e_lat = −sin(θ_r)·(x−x_r) + cos(θ_r)·(y−y_r)
    const double dxw = x0.x - ref_x[0].x;
    const double dyw = x0.y - ref_x[0].y;
    const double e_lat = -std::sin(ref_x[0].yaw) * dxw + std::cos(ref_x[0].yaw) * dyw;
    const double e_yaw = wrapAngle(x0.yaw - ref_x[0].yaw);
    const double e_v   = x0.vx - ref_x[0].vx;
    std_msgs::msg::Float64MultiArray err;
    err.data = {e_lat, e_yaw, e_v};
    error_pub_->publish(err);
}
