#pragma once

#include <cstdint>
#include <vector>

#include "laser_control_interfaces/msg/path_updates.hpp"
#include "laser_control_interfaces/msg/state.hpp"
#include "laser_control_interfaces/srv/get_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "std_srvs/srv/trigger.hpp"

class LaserControlClient {
 public:
  struct Waypoint {
    uint32_t pathId;
    LaserCoord destination;
    double timestampSec;
  };

  enum class PathStatus {
    ACTIVE,
    DISABLED,
    REMOVED,
  };

  struct PathState {
    uint32_t pathId;
    PathStatus status;
  };

  explicit LaserControlClient(rclcpp::Node& callerNode,
                              const std::string& clientNodeName,
                              int timeoutSecs = 3);
  ~LaserControlClient() = default;

  bool startDevice();
  bool closeDevice();
  bool setColor(const LaserColor& color);
  void setPoint(uint32_t pathId, const LaserCoord& point, bool enabled = true);
  void updatePaths(const std::vector<Waypoint>& pathWaypoints = {},
                   const std::vector<PathState>& pathStates = {});
  bool clearPaths();
  bool play();
  bool stop();
  laser_control_interfaces::msg::State::SharedPtr getState();

 private:
  rclcpp::Node& node_;
  int timeoutSecs_{0};

  std::shared_ptr<rclcpp::AsyncParametersClient> parametersClient_;
  rclcpp::Publisher<laser_control_interfaces::msg::PathUpdates>::SharedPtr
      updatePathPublisher_;
  rclcpp::CallbackGroup::SharedPtr clientCallbackGroup_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr startDeviceClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr closeDeviceClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr clearPathsClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr playClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopClient_;
  rclcpp::Client<laser_control_interfaces::srv::GetState>::SharedPtr
      getStateClient_;
};