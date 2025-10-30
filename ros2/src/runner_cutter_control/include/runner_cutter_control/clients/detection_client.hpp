#pragma once

#include "detection_interfaces/msg/detection_result.hpp"
#include "detection_interfaces/msg/state.hpp"
#include "detection_interfaces/srv/get_detection_result.hpp"
#include "detection_interfaces/srv/get_positions.hpp"
#include "detection_interfaces/srv/get_state.hpp"
#include "detection_interfaces/srv/start_detection.hpp"
#include "detection_interfaces/srv/stop_detection.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "std_srvs/srv/trigger.hpp"

class DetectionClient {
 public:
  explicit DetectionClient(rclcpp::Node& callerNode,
                           const std::string& clientNodeName,
                           int timeoutSecs = 3);
  ~DetectionClient() = default;

  detection_interfaces::msg::DetectionResult::SharedPtr getDetection(
      uint8_t detectionType);
  bool startDetection(uint8_t detectionType,
                      const NormalizedPixelRect& normalizedBounds = {
                          0.0f, 0.0f, 1.0f, 1.0f});
  bool stopDetection(uint8_t detectionType);
  bool stopAllDetections();
  bool startRecordingVideo();
  bool stopRecordingVideo();
  bool setSaveDirectory(const std::string& saveDirectory);
  detection_interfaces::msg::State::SharedPtr getState();
  std::optional<std::vector<Position>> getPositions(
      const std::vector<NormalizedPixelCoord>& normalizedPixelCoords);

 private:
  rclcpp::Node& node_;
  int timeoutSecs_{0};

  std::shared_ptr<rclcpp::AsyncParametersClient> parametersClient_;
  rclcpp::CallbackGroup::SharedPtr clientCallbackGroup_;
  rclcpp::Client<detection_interfaces::srv::GetDetectionResult>::SharedPtr
      getDetectionClient_;
  rclcpp::Client<detection_interfaces::srv::StartDetection>::SharedPtr
      startDetectionClient_;
  rclcpp::Client<detection_interfaces::srv::StopDetection>::SharedPtr
      stopDetectionClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopAllDetectionsClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr startRecordingVideoClient_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stopRecordingVideoClient_;
  rclcpp::Client<detection_interfaces::srv::GetState>::SharedPtr
      getStateClient_;
  rclcpp::Client<detection_interfaces::srv::GetPositions>::SharedPtr
      getPositionsClient_;
};