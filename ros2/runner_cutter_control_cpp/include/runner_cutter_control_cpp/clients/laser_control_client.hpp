#pragma once

#include "laser_control_interfaces/msg/state.hpp"
#include "laser_control_interfaces/srv/add_point.hpp"
#include "laser_control_interfaces/srv/get_state.hpp"
#include "laser_control_interfaces/srv/set_color.hpp"
#include "laser_control_interfaces/srv/set_playback_params.hpp"
#include "laser_control_interfaces/srv/set_points.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

class LaserControlClient {
 public:
  LaserControlClient(rclcpp::Node& callerNode,
                     const std::string& clientNodeName, int timeoutSecs = 3);

  bool startDevice();
  bool closeDevice();
  bool setColor(float r, float g, float b, float i);
  bool addPoint(std::pair<float, float> point);
  bool setPoints(const std::vector<std::pair<float, float>>& points);
  bool removePoint();
  bool clearPoints();
  bool setPlaybackParams(int fps, int pps, float transitionDurationMs);
  bool play();
  bool stop();
  laser_control_interfaces::msg::State::SharedPtr getState();

 private:
  rclcpp::Node& node_;
  int timeoutSecs_{0};

  rclcpp::CallbackGroup::SharedPtr clientCallbackGroup_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr startDeviceClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr closeDeviceClient_;
  rclcpp::Client<laser_control_interfaces::srv::SetColor>::SharedPtr
      setColorClient_;
  rclcpp::Client<laser_control_interfaces::srv::AddPoint>::SharedPtr
      addPointClient_;
  rclcpp::Client<laser_control_interfaces::srv::SetPoints>::SharedPtr
      setPointsClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr removePointClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr clearPointsClient_;
  rclcpp::Client<laser_control_interfaces::srv::SetPlaybackParams>::SharedPtr
      setPlaybackParamsClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr playClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopClient_;
  rclcpp::Client<laser_control_interfaces::srv::GetState>::SharedPtr
      getStateClient_;
};