#include "runner_cutter_control/clients/laser_control_client.hpp"

#include "common/ros_utils.hpp"
#include "common_interfaces/msg/vector2.hpp"
#include "runner_cutter_control/clients/service_client_utils.hpp"

LaserControlClient::LaserControlClient(rclcpp::Node& callerNode,
                                       const std::string& clientNodeName,
                                       int timeoutSecs)
    : node_{callerNode}, timeoutSecs_{timeoutSecs} {
  std::string servicePrefix{"/" + clientNodeName};
  parametersClient_ = std::make_shared<rclcpp::AsyncParametersClient>(
      &callerNode, clientNodeName);
  updatePathPublisher_ =
      callerNode.create_publisher<laser_control_interfaces::msg::PathUpdates>(
          servicePrefix + "/path_updates", 1);
  clientCallbackGroup_ =
      callerNode.create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  startDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/start_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  closeDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/close_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  clearPathsClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/clear_paths", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  playClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/play", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  stopClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/stop", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  getStateClient_ =
      callerNode.create_client<laser_control_interfaces::srv::GetState>(
          servicePrefix + "/get_state", rmw_qos_profile_services_default,
          clientCallbackGroup_);
}

bool LaserControlClient::startDevice() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto result{client_utils::callService<std_srvs::srv::Trigger>(
      startDeviceClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

bool LaserControlClient::closeDevice() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto result{client_utils::callService<std_srvs::srv::Trigger>(
      closeDeviceClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

bool LaserControlClient::setColor(const LaserColor& color) {
  std::vector<double> colorVec{color.r, color.g, color.b, color.i};
  return client_utils::setParameters(parametersClient_,
                                     {rclcpp::Parameter("color", colorVec)},
                                     timeoutSecs_, node_.get_logger());
}

void LaserControlClient::setPoint(uint32_t pathId, const LaserCoord& point,
                                  bool enabled) {
  updatePaths(
      {Waypoint{pathId, point, /*timestampSec=*/0.0}},
      {PathState{pathId, enabled ? PathStatus::ACTIVE : PathStatus::DISABLED}});
}

void LaserControlClient::updatePaths(const std::vector<Waypoint>& pathWaypoints,
                                     const std::vector<PathState>& pathStates) {
  auto msg{laser_control_interfaces::msg::PathUpdates()};
  for (const auto& waypoint : pathWaypoints) {
    if (waypoint.destination.x < 0.0 || waypoint.destination.x > 1.0 ||
        waypoint.destination.y < 0.0 || waypoint.destination.y > 1.0) {
      continue;
    }

    auto pathWaypoint{laser_control_interfaces::msg::PathWaypoint()};
    pathWaypoint.path_id = waypoint.pathId;
    pathWaypoint.destination.x = waypoint.destination.x;
    pathWaypoint.destination.y = waypoint.destination.y;
    pathWaypoint.timestamp = common::toRosTime(waypoint.timestampSec);
    msg.path_waypoints.push_back(std::move(pathWaypoint));
  }
  for (const auto& pathState : pathStates) {
    auto entry{laser_control_interfaces::msg::PathState()};
    entry.path_id = pathState.pathId;
    switch (pathState.status) {
      case PathStatus::ACTIVE:
        entry.status = laser_control_interfaces::msg::PathState::ACTIVE;
        break;
      case PathStatus::DISABLED:
        entry.status = laser_control_interfaces::msg::PathState::DISABLED;
        break;
      case PathStatus::REMOVED:
        entry.status = laser_control_interfaces::msg::PathState::REMOVED;
        break;
    }
    msg.path_states.push_back(std::move(entry));
  }
  updatePathPublisher_->publish(std::move(msg));
}

bool LaserControlClient::clearPaths() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto result{client_utils::callService<std_srvs::srv::Trigger>(
      clearPathsClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

bool LaserControlClient::play() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto result{client_utils::callService<std_srvs::srv::Trigger>(
      playClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

bool LaserControlClient::stop() {
  auto request{std::make_shared<std_srvs::srv::Trigger::Request>()};
  auto result{client_utils::callService<std_srvs::srv::Trigger>(
      stopClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
}

laser_control_interfaces::msg::State::SharedPtr LaserControlClient::getState() {
  auto request{
      std::make_shared<laser_control_interfaces::srv::GetState::Request>()};
  auto result{
      client_utils::callService<laser_control_interfaces::srv::GetState>(
          getStateClient_, request, timeoutSecs_, node_.get_logger())};
  if (!result) {
    return std::make_shared<laser_control_interfaces::msg::State>();
  }

  return std::make_shared<laser_control_interfaces::msg::State>(result->state);
}
