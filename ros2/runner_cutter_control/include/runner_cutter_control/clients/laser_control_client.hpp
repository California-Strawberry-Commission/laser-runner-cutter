#pragma once

#include "laser_control_interfaces/msg/path.hpp"
#include "laser_control_interfaces/msg/state.hpp"
#include "laser_control_interfaces/srv/get_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

class LaserControlClient {
 public:
  explicit LaserControlClient(rclcpp::Node& callerNode,
                              const std::string& clientNodeName,
                              int timeoutSecs = 3);

  bool startDevice();
  bool closeDevice();
  bool setColor(float r, float g, float b, float i);
  bool setPoint(float x, float y);
  bool clearPoint();
  bool play();
  bool stop();
  laser_control_interfaces::msg::State::SharedPtr getState();

 private:
  rclcpp::Node& node_;
  int timeoutSecs_{0};

  std::shared_ptr<rclcpp::AsyncParametersClient> parametersClient_;
  rclcpp::Publisher<laser_control_interfaces::msg::Path>::SharedPtr
      pathPublisher_;
  rclcpp::CallbackGroup::SharedPtr clientCallbackGroup_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr startDeviceClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr closeDeviceClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr playClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopClient_;
  rclcpp::Client<laser_control_interfaces::srv::GetState>::SharedPtr
      getStateClient_;
};