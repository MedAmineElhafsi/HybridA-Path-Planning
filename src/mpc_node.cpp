#include "hybrid_astar_cpp/mpc_node.hpp"

#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
inline double wrapAngle(double a) { return std::atan2(std::sin(a), std::cos(a)); }

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
    path_delta_ref_.clear();
    path_a_ref_.clear();
    goal_reached_ = false;   // new path resets goal-reached latch
    rebuildTimeAxis();
}

void MpcNode::velocityCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    if (!has_path_) return;
    const size_t n = std::min(msg->data.size(), path_states_.size());
    for (size_t i = 0; i < n; ++i) {
        const int dir = (i < path_direction_sign_.size()) ? path_direction_sign_[i] : 1;
        path_states_[i].vx = static_cast<double>(dir) * std::abs(msg->data[i]);
        path_states_[i].vy = 0.0;
    }
    updateReferenceYawRates();
    has_vel_ = (n == path_states_.size());
    rebuildTimeAxis();
}

void MpcNode::steeringCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    path_delta_ref_.assign(msg->data.begin(), msg->data.end());
    const size_t n = std::min(path_delta_ref_.size(), path_direction_sign_.size());
    for (size_t i = 0; i < n; ++i) {
        // For reverse, vx is negative, so feedforward steering flips sign to
        // track the same geometric curvature.
        path_delta_ref_[i] *= static_cast<double>(path_direction_sign_[i]);
    }
    updateReferenceYawRates();
    has_steer_ = !path_delta_ref_.empty();
}

void MpcNode::accelerationCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    path_a_ref_.assign(msg->data.begin(), msg->data.end());
    const size_t n = std::min(path_a_ref_.size(), path_direction_sign_.size());
    for (size_t i = 0; i < n; ++i) {
        // /astar_acceleration is along the path-speed magnitude.  MPC tracks
        // signed body-frame velocity, so reverse acceleration is signed too.
        path_a_ref_[i] *= static_cast<double>(path_direction_sign_[i]);
    }
    has_accel_ = !path_a_ref_.empty();
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

    // 1. Closest waypoint (Euclidean)
    int    i_cl  = 0;
    double d2_cl = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < path_states_.size(); ++i) {
        const double dx = path_states_[i].x - current.x;
        const double dy = path_states_[i].y - current.y;
        const double d2 = dx*dx + dy*dy;
        if (d2 < d2_cl) { d2_cl = d2; i_cl = static_cast<int>(i); }
    }
    if (std::sqrt(d2_cl) > lookahead_capture_radius_) return false;

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

    if (!odom_ready) return;

    // Goal-reached: once latched, publish zero command until a new path arrives.
    {
        std::lock_guard<std::mutex> lock(ref_mutex_);
        if (has_path_ && !goal_reached_) {
            const auto& goal = path_states_.back();
            const double dist = std::hypot(x0.x - goal.x, x0.y - goal.y);
            if (dist < goal_reach_radius_ && std::abs(x0.vx) < goal_reach_speed_)
                goal_reached_ = true;
        }
    }
    if (goal_reached_) {
        std_msgs::msg::Float64MultiArray cmd;
        // Apply gentle braking until fully stopped, then hold zero.
        const double brake = (std::abs(x0.vx) > 0.05) ? -std::copysign(1.0, x0.vx) : 0.0;
        cmd.data = {brake, 0.0};
        cmd_pub_->publish(cmd);
        return;
    }

    std::vector<MpcController::State>   ref_x;
    std::vector<MpcController::Control> ref_u;
    bool ready = false;
    {
        std::lock_guard<std::mutex> lock(ref_mutex_);
        if (has_path_ && has_vel_ && has_steer_ && has_accel_ &&
            path_states_.size() >= 2) {
            ready = buildReferenceHorizon(x0, ref_x, ref_u);
        }
    }

    if (!ready) {
        // No reference → publish zero command
        std_msgs::msg::Float64MultiArray cmd;
        cmd.data = {0.0, 0.0};
        cmd_pub_->publish(cmd);
        return;
    }

    // -- Solve MPC --
    auto result = mpc_->solve(x0, ref_x, ref_u);
    if (!result.success) {
        std_msgs::msg::Float64MultiArray cmd;
        cmd.data = {0.0, 0.0};
        cmd_pub_->publish(cmd);
        return;
    }

    // -- Publish command [a, δ] --
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data = {result.u.a, result.u.delta};
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
