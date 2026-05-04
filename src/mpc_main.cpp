#include "hybrid_astar_cpp/mpc_node.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MpcNode>());
    rclcpp::shutdown();
    return 0;
}
