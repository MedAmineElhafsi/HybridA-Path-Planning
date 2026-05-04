#ifndef HYBRID_ASTAR_CPP_GRID_COLLISION_HPP_
#define HYBRID_ASTAR_CPP_GRID_COLLISION_HPP_

#include <vector>
#include <cmath>
#include <queue>
#include <limits>
#include <nav_msgs/msg/occupancy_grid.hpp>

struct Cell {
    int x;
    int y;
};

class GridCollision {
public:
    // Initialize the Lookup Tables
    GridCollision(int yaw_bins, double car_length, double car_width, double margin);

    // Update internal map when a new grid arrives
    void updateGrid(const nav_msgs::msg::OccupancyGrid& grid);
    
    // O(N) integer-only collision check. Extremely fast.
    bool isCollisionFree(double wx, double wy, double yaw) const;

    // Computes the 2D obstacle-aware distance from the goal
    void computeDistanceMap(double goal_wx, double goal_wy);
    
    // Look up the 2D heuristic cost for any point
    double getHeuristicCost(double wx, double wy) const;
    double getObstacleDistance(double wx, double wy) const;

private:
    void precomputeFootprintLUT();
    void computeObstacleDistanceMap();
    int getYawBin(double yaw) const;
    bool worldToGrid(double wx, double wy, int& gx, int& gy) const;
    int getIndex(int gx, int gy) const;
    bool isOccupiedCell(int gx, int gy) const;

    int yaw_bins_;
    double car_length_;
    double car_width_;
    double margin_;

    nav_msgs::msg::OccupancyGrid grid_;
    
    // footprint_lut_[yaw_bin] gives a list of (dx, dy) cell offsets
    std::vector<std::vector<Cell>> footprint_lut_; 
    
    // 1D array for the 2D heuristic distance map
    std::vector<double> distance_map_; 
    std::vector<double> obstacle_distance_map_;
};

#endif // HYBRID_ASTAR_CPP_GRID_COLLISION_HPP_
