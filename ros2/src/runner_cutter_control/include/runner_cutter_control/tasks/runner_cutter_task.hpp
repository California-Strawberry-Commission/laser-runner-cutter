#pragma once

#include <atomic>
#include <string>

#include "detection_interfaces/msg/detection_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/tasks/callback_registry.hpp"

/**
 * A runner-cutting automation loop for a moving scene. Arms the laser, starts
 * detection, repeatedly acquires the next target from the Tracker, and sends
 * lookahead waypoints to the laser.
 *
 * Unlike StaticRunnerCutterTask's static aim/burn, this predicts the target's
 * future position and sends it to the laser.
 */
class RunnerCutterTask {
 public:
  RunnerCutterTask(
      std::shared_ptr<
          CallbackRegistry<detection_interfaces::msg::DetectionResult>>
          detectionCallbackRegistry,
      std::shared_ptr<LaserControlClient> laser,
      std::shared_ptr<DetectionClient> detection,
      std::shared_ptr<Calibration> calibration, rclcpp::Logger logger);
  ~RunnerCutterTask() = default;

  /**
   * Arm the laser and run the automation loop until stopped.
   *
   * @param trackMissTimeoutSecs Grace period, in seconds, for a PENDING or
   * ACTIVE track to go undetected before the Tracker marks it FAILED.
   * @param targetAttempts Max number of times a FAILED track may be requeued
   * as PENDING after being redetected. A negative value means no limit.
   * @param lookaheadSecs How far ahead, in seconds, to predict the target's
   * position when placing laser waypoints.
   * @param burnLaserColor Laser color to emit while burning.
   * @param burnTimeSecs How long, in seconds, to burn each track before
   * marking it COMPLETED.
   * @param stopSignal Set to true from another thread to end the task.
   */
  void run(float trackMissTimeoutSecs, int targetAttempts, float lookaheadSecs,
           const LaserColor& burnLaserColor, float burnTimeSecs,
           std::atomic<bool>& stopSignal);

 private:
  std::shared_ptr<CallbackRegistry<detection_interfaces::msg::DetectionResult>>
      detectionCallbackRegistry_;
  std::shared_ptr<LaserControlClient> laser_;
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  rclcpp::Logger logger_;
};
