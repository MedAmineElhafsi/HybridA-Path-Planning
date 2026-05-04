#ifndef HYBRID_ASTAR_CPP_PLANNER_NODE_HPP_
#define HYBRID_ASTAR_CPP_PLANNER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rclcpp/time.hpp>
#include <mutex>
#include <string>
#include <memory>
#include <limits>
#include "hybrid_astar_cpp/grid_collision.hpp"
#include "hybrid_astar_cpp/hybrid_astar.hpp"
#include "hybrid_astar_cpp/smoother.hpp"
#include "hybrid_astar_cpp/velocity_profile.hpp"
#include "hybrid_astar_cpp/inverse_kinematics.hpp"

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode();

private:
    void gridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void planPath();
    void publishResetVisualization(
        const geometry_msgs::msg::Pose& start_pose,
        const std::string& frame_id,
        int footprints_to_delete);

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr start_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_path_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr velocity_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr curvature_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr steering_avg_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr steering_left_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr steering_right_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr acceleration_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr reference_trajectory_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr footprint_pub_;

    std::mutex state_mutex_;
    nav_msgs::msg::OccupancyGrid current_grid_;
    geometry_msgs::msg::Pose start_pose_;
    geometry_msgs::msg::Pose goal_pose_;
    geometry_msgs::msg::Pose last_plan_end_pose_;
    geometry_msgs::msg::Pose last_replan_start_pose_;
    bool has_grid_ = false;
    bool has_start_ = false;
    bool has_goal_ = false;
    bool chain_from_last_goal_ = false;
    bool has_last_plan_end_ = false;
    bool use_live_start_from_odom_ = false;
    bool has_last_replan_start_ = false;
    double odom_replan_min_translation_ = 0.0;
    double odom_replan_min_yaw_ = 0.0;
    double odom_replan_min_interval_s_ = 0.0;
    rclcpp::Time last_replan_time_{0, 0, RCL_ROS_TIME};
    
    double xy_res_;
    double min_turning_radius_;
    std::string map_frame_;

    // -----------------------------------------------------------------------
    // Cached planning objects (rebuilt only when parameters change)
    // -----------------------------------------------------------------------
    std::shared_ptr<GridCollision> collision_checker_;
    std::unique_ptr<HybridAStar>   planner_;

    // Parameter snapshot used to detect changes that require a rebuild
    double cached_c_len_           = -1.0;
    double cached_c_wid_           = -1.0;
    double cached_c_mar_           = -1.0;
    double cached_hard_margin_     = -1.0;
    double cached_step_size_       = -1.0;
    int    cached_steer_samples_   = -1;
    double cached_wheelbase_       = -1.0;
    double cached_soft_margin_     = -1.0;
    double cached_clearance_weight_= -1.0;
    double cached_relax_radius_    = -1.0;

    // Goal position for which the heuristic distance map was last computed
    double cached_dist_map_goal_x_ = std::numeric_limits<double>::quiet_NaN();
    double cached_dist_map_goal_y_ = std::numeric_limits<double>::quiet_NaN();

    // Number of footprint markers published in the last plan (for correct deletion)
    int last_footprint_count_ = 0;
};

#endif // HYBRID_ASTAR_CPP_PLANNER_NODE_HPP_
