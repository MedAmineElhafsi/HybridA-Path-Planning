#include "hybrid_astar_cpp/hybrid_astar.hpp"
#include "hybrid_astar_cpp/curves.hpp"
#include <cmath>
#include <queue>
#include <unordered_map>
#include <algorithm>

namespace {
// Ackermann kinematic model — forward integration step.
//
// The Ackermann steering geometry requires that all four wheels rotate about a
// common instantaneous centre of rotation (ICR).  For the motion of the rear-axle
// centre reference point the equations reduce to the classical bicycle kinematic
// model (same differential equations) because both models share the same rear-axle
// centre trajectory.  The difference appears at the wheel level: the front-left
// and front-right wheels have different steering angles (inner turns sharper than
// outer).  Those per-wheel angles are computed in InverseKinematics::compute().
//
// State:   (x, y, θ)  — rear-axle centre in world frame
// Input:   δ_avg      — equivalent single-wheel (bicycle) steering angle, radians
//          distance   — signed arc length (positive = forward)
//
// Equations of motion:
//   κ = tan(δ_avg) / L
//   x(s+ds) = x(s) + (sin(θ+κ·ds) - sin(θ)) / κ     (κ ≠ 0)
//   y(s+ds) = y(s) + (cos(θ)  - cos(θ+κ·ds)) / κ     (κ ≠ 0)
//   θ(s+ds) = θ(s) + κ·ds
Pose2D integrateAckermannStep(const Pose2D& pose, double distance,
                               double steer_avg, double wheelbase) {
    Pose2D next = pose;
    const double curvature = std::tan(steer_avg) / std::max(1e-6, wheelbase);

    if (std::abs(curvature) < 1e-9) {
        // Straight-line motion
        next.x += distance * std::cos(pose.yaw);
        next.y += distance * std::sin(pose.yaw);
    } else {
        const double next_yaw = AnalyticCurves::wrapAngle(pose.yaw + distance * curvature);
        next.x += (std::sin(next_yaw) - std::sin(pose.yaw)) / curvature;
        next.y += (-std::cos(next_yaw) + std::cos(pose.yaw)) / curvature;
        next.yaw = next_yaw;
    }

    next.yaw = AnalyticCurves::wrapAngle(next.yaw);
    return next;
}

bool rolloutMotion(const Pose2D& start, double distance, double steer, double wheelbase,
                   double collision_check_step,
                   const std::shared_ptr<GridCollision>& collision_checker,
                   Pose2D& end_pose) {
    const int rollout_steps = std::max(1, static_cast<int>(
        std::ceil(std::abs(distance) / std::max(1e-6, collision_check_step))));
    const double substep_distance = distance / static_cast<double>(rollout_steps);

    Pose2D pose = start;
    for (int i = 0; i < rollout_steps; ++i) {
        pose = integrateAckermannStep(pose, substep_distance, steer, wheelbase);
        if (!collision_checker->isCollisionFree(pose.x, pose.y, pose.yaw)) {
            return false;
        }
    }

    end_pose = pose;
    return true;
}
}  // namespace

HybridAStar::HybridAStar(double step_size, double max_steer, int steer_samples, double wheelbase, double xy_res,
                         int yaw_bins, double clearance_distance, double clearance_weight,
                         double clearance_relaxation_radius)
    : step_size_(step_size), max_steer_(max_steer), steer_samples_(steer_samples),
      wheelbase_(wheelbase), xy_res_(xy_res), yaw_bins_(yaw_bins),
      clearance_distance_(clearance_distance), clearance_weight_(clearance_weight),
      clearance_relaxation_radius_(clearance_relaxation_radius) {}

namespace {
double goalHeuristic(const std::shared_ptr<GridCollision>& collision_checker,
                     double x, double y, const Pose2D& goal) {
    // Use max(holonomic-with-obstacles, non-holonomic-without-obstacles).
    // Both are individually admissible; their maximum is a tighter admissible
    // lower bound — the same strategy used in nav2_smac_planner and Junior (Stanford).
    const double euclidean = std::hypot(x - goal.x, y - goal.y);
    const double grid_h    = collision_checker->getHeuristicCost(x, y);
    if (std::isfinite(grid_h)) {
        return std::max(euclidean, grid_h);
    }
    return euclidean;
}

double clearancePenalty(const std::shared_ptr<GridCollision>& collision_checker,
                        double x, double y, double desired_clearance, double weight,
                        const Pose2D& start, const Pose2D& goal,
                        double relaxation_radius) {
    if (desired_clearance <= 0.0 || weight <= 0.0) {
        return 0.0;
    }

    const double obstacle_distance = collision_checker->getObstacleDistance(x, y);
    if (!std::isfinite(obstacle_distance) || obstacle_distance >= desired_clearance) {
        return 0.0;
    }

    const double deficit = desired_clearance - obstacle_distance;
    double relaxation_scale = 1.0;
    if (relaxation_radius > 0.0) {
        const double start_dist = std::hypot(x - start.x, y - start.y);
        const double goal_dist = std::hypot(x - goal.x, y - goal.y);
        const double terminal_dist = std::min(start_dist, goal_dist);
        relaxation_scale = std::min(1.0, terminal_dist / relaxation_radius);
    }

    return weight * deficit * deficit * relaxation_scale;
}
}  // namespace

StateKey HybridAStar::poseToKey(double x, double y, double yaw) const {
    StateKey key;
    key.x_idx = static_cast<int>(std::round(x / xy_res_));
    key.y_idx = static_cast<int>(std::round(y / xy_res_));
    key.yaw_idx = static_cast<int>(std::round((AnalyticCurves::wrapAngle(yaw) / (2.0 * M_PI)) * yaw_bins_)) % yaw_bins_;
    if (key.yaw_idx < 0) key.yaw_idx += yaw_bins_;
    return key;
}

bool HybridAStar::plan(const Pose2D& start, const Pose2D& goal,
                       std::shared_ptr<GridCollision> collision_checker,
                       std::shared_ptr<GridCollision> terminal_checker,
                       std::vector<Pose2D>& final_path) {
    if (!collision_checker->isCollisionFree(start.x, start.y, start.yaw) ||
        !collision_checker->isCollisionFree(goal.x, goal.y, goal.yaw)) {
        return false;
    }

    // -----------------------------------------------------------------------
    // Node pool: all Node3D objects live in a flat vector so there are zero
    // per-node heap allocations during search.  Indices into this vector are
    // used wherever the old code used shared_ptr.  Reserve a generous capacity
    // to eliminate reallocation (and therefore reference invalidation) for
    // typical plans; the vector grows automatically for very long paths.
    // -----------------------------------------------------------------------
    std::vector<Node3D> node_pool;
    node_pool.reserve(16384);

    // Map from grid cell key → index in node_pool (the "canonical" node).
    std::unordered_map<StateKey, int, StateKeyHash> all_nodes;
    all_nodes.reserve(8192);

    // Priority queue over node_pool indices, ordered by f_cost (min-heap).
    // Captures node_pool by reference — safe because both live in this scope.
    auto comp = [&node_pool](int a, int b) {
        return node_pool[a].f_cost > node_pool[b].f_cost;
    };
    std::priority_queue<int, std::vector<int>, decltype(comp)> open_set(comp);

    // Seed the search
    const double start_h = goalHeuristic(collision_checker, start.x, start.y, goal);
    node_pool.push_back({start.x, start.y, start.yaw, 0.0, start_h, 0.0, 1, -1});
    const StateKey start_key = poseToKey(start.x, start.y, start.yaw);
    all_nodes[start_key] = 0;
    open_set.push(0);

    int  best_goal_idx  = -1;
    std::vector<Pose2D> analytic_suffix;
    int  expansions     = 0;

    // Pre-build steering angle list (computed once, not per expansion)
    std::vector<double> steer_angles;
    steer_angles.reserve(steer_samples_);
    if (steer_samples_ > 1) {
        const double d_steer = 2.0 * max_steer_ / (steer_samples_ - 1);
        for (int i = 0; i < steer_samples_; ++i)
            steer_angles.push_back(-max_steer_ + i * d_steer);
    } else {
        steer_angles.push_back(0.0);
    }

    const double collision_check_step = std::max(0.05, std::min(xy_res_, step_size_ / 4.0));

    while (!open_set.empty()) {
        const int current_idx = open_set.top();
        open_set.pop();

        // ---- Lazy deletion --------------------------------------------------
        // When a node gets a better g-cost later, a NEW entry is pushed to the
        // queue (the old one isn't removed — std::priority_queue has no erase).
        // Skip stale entries here instead: if the canonical index for this key
        // differs from what we just popped, this entry is outdated.
        {
            const Node3D& cn = node_pool[current_idx];
            const StateKey ck = poseToKey(cn.x, cn.y, cn.yaw);
            const auto it = all_nodes.find(ck);
            if (it == all_nodes.end() || it->second != current_idx) {
                continue;   // superseded — skip without expanding
            }
        }

        // Value-copy the current node to avoid holding a reference across
        // any push_back into node_pool (which may trigger reallocation).
        const Node3D current = node_pool[current_idx];

        // ---- Goal check -----------------------------------------------------
        if (std::hypot(current.x - goal.x, current.y - goal.y) < goal_pos_thresh &&
            std::abs(AnalyticCurves::wrapAngle(current.yaw - goal.yaw)) < goal_yaw_thresh) {
            best_goal_idx = current_idx;
            break;
        }

        expansions++;

        // ---- Analytic expansion (Dubins / Reeds-Shepp "shot") ---------------
        if (expansions % analytic_every_n == 0 &&
            std::hypot(current.x - goal.x, current.y - goal.y) < analytic_radius) {

            const double min_turn_rad = wheelbase_ / std::max(1e-6, std::tan(max_steer_));

            std::vector<Pose2D> dubins_path, rs_path;
            std::vector<int>    rs_gears;

            const bool dubins_valid = AnalyticCurves::getDubinsPath(
                {current.x, current.y, current.yaw}, goal, min_turn_rad, 0.05, dubins_path);
            const bool rs_valid = AnalyticCurves::getReedsSheppPath(
                {current.x, current.y, current.yaw}, goal, min_turn_rad, 0.05, rs_path, rs_gears);

            auto get_path_length = [](const std::vector<Pose2D>& p) {
                double len = 0.0;
                for (size_t i = 1; i < p.size(); ++i)
                    len += std::hypot(p[i].x - p[i-1].x, p[i].y - p[i-1].y);
                return len;
            };

            auto get_clearance_cost = [&](const std::vector<Pose2D>& p) {
                double cost = 0.0;
                for (const auto& pt : p)
                    cost += clearancePenalty(collision_checker, pt.x, pt.y,
                                             clearance_distance_, clearance_weight_,
                                             start, goal, clearance_relaxation_radius_) * 0.05;
                return cost;
            };

            // Reverse-arc penalty for analytic shots: sum the arc length of all
            // reverse segments and multiply by penalty_reverse_.  This keeps the
            // analytic cost consistent with the grid-expansion cost so the planner
            // doesn't prefer RS paths purely because their reverse arcs are "free".
            auto get_reverse_penalty = [&](const std::vector<Pose2D>& p) {
                double rev_len = 0.0;
                for (size_t i = 1; i < p.size(); ++i)
                    if (p[i].direction < 0)
                        rev_len += std::hypot(p[i].x - p[i-1].x, p[i].y - p[i-1].y);
                return penalty_reverse_ * rev_len;
            };

            auto get_direction_change_penalty = [&](const std::vector<int>& gears) {
                if (current.parent_idx == -1 || gears.empty() || penalty_direction_change_ <= 0.0) {
                    return 0.0;
                }

                double penalty = 0.0;
                int previous_dir = current.direction;
                for (int gear : gears) {
                    if (gear != previous_dir) {
                        penalty += penalty_direction_change_;
                        previous_dir = gear;
                    }
                }
                return penalty;
            };

            auto is_endpoint_correct = [this](const std::vector<Pose2D>& p, const Pose2D& g) {
                if (p.empty()) return false;
                return std::hypot(p.back().x - g.x, p.back().y - g.y) < analytic_endpoint_tol;
            };

            const double dxy = std::hypot(current.x - goal.x, current.y - goal.y);
            double best_shot_cost = std::numeric_limits<double>::infinity();
            std::vector<Pose2D> winning_shot;
            bool analytic_used_rs = false;

            // Dubins (forward-only) — no reverse penalty needed
            if (dubins_valid && is_endpoint_correct(dubins_path, goal)) {
                const double len = get_path_length(dubins_path);
                if (!(dxy < 2.0 && len > detour_ratio * std::max(dxy, 0.1))) {
                    bool safe = true;
                    for (const auto& pt : dubins_path)
                        if (!collision_checker->isCollisionFree(pt.x, pt.y, pt.yaw)) { safe = false; break; }
                    const std::vector<int> dubins_gears(dubins_path.size(), 1);
                    const double cost = len
                        + get_clearance_cost(dubins_path)
                        + get_direction_change_penalty(dubins_gears);
                    if (safe && cost < best_shot_cost) {
                        best_shot_cost = cost;
                        winning_shot   = dubins_path;
                        analytic_used_rs = false;
                    }
                }
            }

            // Reeds-Shepp (forward + reverse) — penalise reverse arcs so the
            // planner only accepts an RS path when it genuinely saves cost over
            // the Dubins/grid alternatives.
            if (rs_valid && is_endpoint_correct(rs_path, goal)) {
                const double len = get_path_length(rs_path);
                if (!(dxy < 2.0 && len > detour_ratio * std::max(dxy, 0.1))) {
                    bool safe = true;
                    for (const auto& pt : rs_path)
                        if (!collision_checker->isCollisionFree(pt.x, pt.y, pt.yaw)) { safe = false; break; }
                    const double cost = len
                        + get_clearance_cost(rs_path)
                        + get_reverse_penalty(rs_path)
                        + get_direction_change_penalty(rs_gears);
                    if (safe && cost < best_shot_cost) {
                        best_shot_cost = cost;
                        winning_shot   = rs_path;
                        analytic_used_rs = true;
                    }
                }
            }

            if (!winning_shot.empty()) {
                // Tag each pose with its motion direction for the visualiser
                if (analytic_used_rs) {
                    for (size_t i = 0; i < winning_shot.size() && i < rs_gears.size(); ++i)
                        winning_shot[i].direction = rs_gears[i];
                } else {
                    for (auto& p : winning_shot) p.direction = 1;
                }
                best_goal_idx   = current_idx;
                analytic_suffix = winning_shot;
                break;
            }
        }

        // ---- Grid expansion -------------------------------------------------
        const int dirs[] = {1, -1};
        for (int dir : dirs) {
            for (double steer : steer_angles) {
                Pose2D next_pose;
                if (!rolloutMotion({current.x, current.y, current.yaw},
                                   dir * step_size_, steer, wheelbase_,
                                   collision_check_step, collision_checker, next_pose))
                    continue;

                const StateKey key = poseToKey(next_pose.x, next_pose.y, next_pose.yaw);

                double step_cost = step_size_;
                if (dir == -1)
                    step_cost *= (1.0 + penalty_reverse_);
                step_cost += penalty_steer_ * std::abs(steer);
                if (current.parent_idx != -1) {
                    step_cost += penalty_steer_change_ * std::abs(steer - current.steer);
                    if (current.direction != dir)
                        step_cost += penalty_direction_change_;
                }
                step_cost += clearancePenalty(
                    collision_checker, next_pose.x, next_pose.y,
                    clearance_distance_, clearance_weight_,
                    start, goal, clearance_relaxation_radius_);

                const double new_g = current.g_cost + step_cost;
                const double h     = goalHeuristic(collision_checker, next_pose.x, next_pose.y, goal);

                const auto it = all_nodes.find(key);
                if (it == all_nodes.end() || new_g < node_pool[it->second].g_cost) {
                    const int next_idx = static_cast<int>(node_pool.size());
                    node_pool.push_back({next_pose.x, next_pose.y, next_pose.yaw,
                                         new_g, new_g + h, steer, dir, current_idx});
                    all_nodes[key] = next_idx;
                    open_set.push(next_idx);
                }
            }
        }
    }

    // ---- Path reconstruction ------------------------------------------------
    if (best_goal_idx >= 0) {
        std::vector<Pose2D> path_prefix;
        // Walk parent chain via indices — no pointer chasing, no ref-counting.
        for (int idx = best_goal_idx; idx >= 0; idx = node_pool[idx].parent_idx) {
            const Node3D& n = node_pool[idx];
            Pose2D p{n.x, n.y, n.yaw};
            p.direction = n.direction;
            path_prefix.push_back(p);
        }
        std::reverse(path_prefix.begin(), path_prefix.end());

        final_path = path_prefix;
        final_path.insert(final_path.end(), analytic_suffix.begin(), analytic_suffix.end());

        if (!final_path.empty()) {
            Pose2D& last_pose = final_path.back();
            const double terminal_distance = std::hypot(last_pose.x - goal.x, last_pose.y - goal.y);
            const double terminal_yaw_error = std::abs(AnalyticCurves::wrapAngle(last_pose.yaw - goal.yaw));
            if (terminal_distance <= final_snap_pos && terminal_yaw_error <= final_snap_yaw) {
                const std::shared_ptr<GridCollision>& snap_checker =
                    terminal_checker ? terminal_checker : collision_checker;
                if (snap_checker && snap_checker->isCollisionFree(goal.x, goal.y, goal.yaw)) {
                    const int saved_dir = last_pose.direction;
                    last_pose = goal;
                    last_pose.direction = saved_dir;
                }
            }
        }
        return true;
    }

    return false;
}
