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
      callerNode.create_publisher<laser_control_interfaces::msg::PathUpdate>(
          servicePrefix + "/update_path", 1);
  clientCallbackGroup_ =
      callerNode.create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  startDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/start_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  closeDeviceClient_ = callerNode.create_client<std_srvs::srv::Trigger>(
      servicePrefix + "/close_device", rmw_qos_profile_services_default,
      clientCallbackGroup_);
  removePathClient_ =
      callerNode.create_client<laser_control_interfaces::srv::RemovePath>(
          servicePrefix + "/remove_path", rmw_qos_profile_services_default,
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

bool LaserControlClient::setPoint(uint32_t pathId, const LaserCoord& point) {
  if (point.x < 0.0 || point.x > 1.0 || point.y < 0.0 || point.y > 1.0) {
    return false;
  }

  auto msg{laser_control_interfaces::msg::PathUpdate()};
  msg.path_id = pathId;
  msg.destination.x = point.x;
  msg.destination.y = point.y;
  updatePathPublisher_->publish(std::move(msg));

  return true;
}

bool LaserControlClient::addWaypoint(uint32_t pathId,
                                     const LaserCoord& destination,
                                     double timestampSec) {
  if (destination.x < 0.0 || destination.x > 1.0 || destination.y < 0.0 ||
      destination.y > 1.0) {
    return false;
  }

  auto msg{laser_control_interfaces::msg::PathUpdate()};
  msg.path_id = pathId;
  msg.destination.x = destination.x;
  msg.destination.y = destination.y;
  msg.timestamp = common::toRosTime(timestampSec);
  updatePathPublisher_->publish(std::move(msg));

  return true;
}

bool LaserControlClient::removePath(uint32_t pathId) {
  auto request{
      std::make_shared<laser_control_interfaces::srv::RemovePath::Request>()};
  request->path_id = pathId;
  auto result{
      client_utils::callService<laser_control_interfaces::srv::RemovePath>(
          removePathClient_, request, timeoutSecs_, node_.get_logger())};
  return result && result->success;
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
