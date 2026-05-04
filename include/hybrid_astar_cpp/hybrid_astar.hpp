#ifndef HYBRID_ASTAR_CPP_HYBRID_ASTAR_HPP_
#define HYBRID_ASTAR_CPP_HYBRID_ASTAR_HPP_

#include <vector>
#include <memory>
#include <queue>
#include <unordered_map>
#include "hybrid_astar_cpp/grid_collision.hpp"
#include "hybrid_astar_cpp/curves.hpp"

// 3D Grid Key for the Closed List
struct StateKey {
    int x_idx;
    int y_idx;
    int yaw_idx;

    bool operator==(const StateKey& other) const {
        return x_idx == other.x_idx && y_idx == other.y_idx && yaw_idx == other.yaw_idx;
    }
};

// Custom Hash for the 3D StateKey to make unordered_map lightning fast
struct StateKeyHash {
    std::size_t operator()(const StateKey& k) const {
        return ((std::hash<int>()(k.x_idx) ^ (std::hash<int>()(k.y_idx) << 1)) >> 1) ^ (std::hash<int>()(k.yaw_idx) << 1);
    }
};

// Represents a single node in the A* search tree.
// parent_idx is an index into the flat node pool (HybridAStar::plan uses a
// std::vector<Node3D> to avoid per-node heap allocation).  -1 means root.
struct Node3D {
    double x, y, yaw;
    double g_cost;
    double f_cost;
    double steer;
    int    direction;
    int    parent_idx;
};

class HybridAStar {
public:
    HybridAStar(double step_size, double max_steer, int steer_samples, double wheelbase, double xy_res,
                int yaw_bins, double clearance_distance, double clearance_weight,
                double clearance_relaxation_radius);

    // Main Planning Function
    bool plan(const Pose2D& start, const Pose2D& goal, 
              std::shared_ptr<GridCollision> collision_checker,
              std::shared_ptr<GridCollision> terminal_checker,
              std::vector<Pose2D>& out_path);

    // Tunable planning thresholds – expose as public so the caller (PlannerNode)
    // can map ROS 2 parameters onto them without changing the constructor signature.
    double goal_pos_thresh          = 0.50;  // m  – grid node accepted as "at goal"
    double goal_yaw_thresh          = 0.20;  // rad
    int    analytic_every_n         = 10;    // attempt analytic shot every N expansions
    double analytic_radius          = 8.0;   // m  – only attempt when within this range
    double analytic_endpoint_tol    = 0.50;  // m  – reject analytic path if endpoint too far
    double final_snap_pos           = 0.35;  // m  – snap last pose to exact goal
    double final_snap_yaw           = 0.35;  // rad
    double detour_ratio             = 3.0;   // reject analytic paths longer than ratio * dxy

    // Motion penalties – higher values steer the planner toward forward-only,
    // smooth trajectories.  Applied to both grid expansion and analytic shots.
    double penalty_reverse_       = 0.5;    // fraction of step_size added per reverse metre
    double penalty_steer_         = 0.005;  // fraction of step_size added per radian of steer
    double penalty_steer_change_  = 0.04;   // fraction added per radian of steer change
    double penalty_direction_change_ = 0.5; // fixed cost per forward/reverse switch

private:
    StateKey poseToKey(double x, double y, double yaw) const;
    double normalizeYaw(double yaw) const;

    double step_size_;
    double max_steer_;
    int steer_samples_;
    double wheelbase_;
    double xy_res_;
    int yaw_bins_;
    double clearance_distance_;
    double clearance_weight_;
    double clearance_relaxation_radius_;
};

#endif // HYBRID_ASTAR_CPP_HYBRID_ASTAR_HPP_
