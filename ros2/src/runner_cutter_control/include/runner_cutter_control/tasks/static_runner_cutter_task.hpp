#pragma once

#include <atomic>
#include <optional>
#include <string>

#include "common/event.hpp"
#include "detection_interfaces/msg/detection_result.hpp"
#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/camera_control_client.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/tasks/callback_registry.hpp"
#include "runner_cutter_control/tasks/laser_targeting.hpp"
#include "runner_cutter_control/tracking/track.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"
#include "runner_cutter_control_interfaces/msg/tracks.hpp"

/**
 * A runner-cutting automation loop for static scene. Arms the laser, starts
 * detection, repeatedly acquires the next target from the Tracker, aims the
 * laser at it, and burns it. Runs until stopped or auto-disarmed after a period
 * with no viable targets.
 */
class StaticRunnerCutterTask {
 public:
  StaticRunnerCutterTask(
      std::shared_ptr<
          CallbackRegistry<detection_interfaces::msg::DetectionResult>>
          detectionCallbackRegistry,
      std::shared_ptr<CameraControlClient> camera,
      std::shared_ptr<DetectionClient> detection,
      std::shared_ptr<Calibration> calibration,
      std::shared_ptr<LaserControlClient> laser, rclcpp::Logger logger,
      rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
          notificationsPublisher,
      rclcpp::Publisher<runner_cutter_control_interfaces::msg::Tracks>::
          SharedPtr tracksPublisher);
  ~StaticRunnerCutterTask() = default;

  /**
   * Arm the laser and run the automation loop until stopped.
   *
   * @param trackMissTimeoutSecs Grace period, in seconds, for a PENDING or
   * ACTIVE track to go undetected before the Tracker marks it FAILED.
   * @param targetAttempts Max number of times a FAILED track may be requeued
   * as PENDING after being redetected. A negative value means no limit.
   * @param enableDetectionDuringBurn When false, RUNNER detection is stopped
   * for each target's aim/burn and restarted afterward. When true, it runs
   * continuously.
   * @param enableAiming When true, run a closed-loop laser aim-correction pass
   * before burning. When false, map the target's camera-space position straight
   * to a laser coordinate via calibration with no visual correction.
   * @param autoDisarmSecs If no new PENDING track appears within this many
   * seconds while there is no viable target, end the task (auto-disarm) with a
   * notification. A value <= 0 waits indefinitely.
   * @param saveDir Directory to save run data.
   * @param trackingLaserColor Laser color used during the aim pass (only when
   * enableAiming).
   * @param burnLaserColor Laser color used during the burn.
   * @param burnTimeSecs Burn duration, in seconds, per target.
   * @param stopSignal Set to true from another thread to end the task.
   */
  void run(float trackMissTimeoutSecs, int targetAttempts,
           bool enableDetectionDuringBurn, bool enableAiming,
           float autoDisarmSecs, const std::string& saveDir,
           const LaserColor& trackingLaserColor,
           const LaserColor& burnLaserColor, float burnTimeSecs,
           std::atomic<bool>& stopSignal);

 private:
  std::optional<std::shared_ptr<const Track>> acquireNextTarget();

  runner_cutter_control_interfaces::msg::Tracks::UniquePtr getTracksMsg();

  void publishTracks();

  bool waitForPendingTracks(float timeoutSecs, std::atomic<bool>& stopSignal);

  std::shared_ptr<CallbackRegistry<detection_interfaces::msg::DetectionResult>>
      detectionCallbackRegistry_;
  std::shared_ptr<CameraControlClient> camera_;
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  LaserTargeting laserTargeting_;
  rclcpp::Logger logger_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
  rclcpp::Publisher<runner_cutter_control_interfaces::msg::Tracks>::SharedPtr
      tracksPublisher_;
  std::shared_ptr<Tracker> tracker_;
  // Notifies when new pending tracks are detected
  common::Event pendingTracksChangedEvent_;
};
