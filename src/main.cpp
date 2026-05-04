#include <memory>
#include <rclcpp/rclcpp.hpp>
#include "hybrid_astar_cpp/planner_node.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    // std::make_shared requires <memory>
    // PlannerNode requires the planner_node.hpp header
    auto node = std::make_shared<PlannerNode>();
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}