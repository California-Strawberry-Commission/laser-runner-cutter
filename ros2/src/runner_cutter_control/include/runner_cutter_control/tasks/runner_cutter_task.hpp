#pragma once

#include <atomic>
#include <optional>
#include <string>

#include "common/event.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/camera_control_client.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/tasks/laser_targeting.hpp"
#include "runner_cutter_control/tracking/track.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"
#include "runner_cutter_control_interfaces/msg/tracks.hpp"

/**
 * The main runner-cutting automation loop. Arms the laser, starts detection,
 * repeatedly acquires the next pending/active track from the Tracker, aims the
 * laser at it, and burns it. Runs until stopped or auto-disarmed after a period
 * with no viable targets.
 */
class RunnerCutterTask {
 public:
  RunnerCutterTask(std::shared_ptr<CameraControlClient> camera,
                   std::shared_ptr<DetectionClient> detection,
                   std::shared_ptr<Calibration> calibration,
                   std::shared_ptr<Tracker> tracker,
                   std::shared_ptr<LaserTargeting> laserTargeting,
                   rclcpp::Logger logger,
                   rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
                       notificationsPublisher,
                   rclcpp::Publisher<
                       runner_cutter_control_interfaces::msg::Tracks>::SharedPtr
                       tracksPublisher);
  ~RunnerCutterTask() = default;

  void run(uint8_t detectionType, bool enableDetectionDuringBurn,
           bool enableAiming, float autoDisarmSecs, const std::string& saveDir,
           const LaserColor& trackingLaserColor,
           const LaserColor& burnLaserColor, float burnTimeSecs,
           std::atomic<bool>& stopSignal,
           common::Event& pendingTracksChangedEvent);

 private:
  /**
   * Get the next suitable target from the tracker.
   *
   * @return The target Track, if one is available.
   */
  std::optional<std::shared_ptr<const Track>> acquireNextTarget(
      std::atomic<bool>& stopSignal);

  runner_cutter_control_interfaces::msg::Tracks::UniquePtr getTracksMsg();
  void publishTracks();

  std::shared_ptr<CameraControlClient> camera_;
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  std::shared_ptr<Tracker> tracker_;
  std::shared_ptr<LaserTargeting> laserTargeting_;
  rclcpp::Logger logger_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
  rclcpp::Publisher<runner_cutter_control_interfaces::msg::Tracks>::SharedPtr
      tracksPublisher_;
};
