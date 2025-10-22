#include <rclcpp_components/register_node_macro.hpp>

#include "rclcpp/rclcpp.hpp"

class DetectionNode : public rclcpp::Node {
 public:
  explicit DetectionNode(const rclcpp::NodeOptions& options)
      : Node("detection_node", options) {
    RCLCPP_INFO(get_logger(), "Hello World");
  }
};

RCLCPP_COMPONENTS_REGISTER_NODE(DetectionNode)