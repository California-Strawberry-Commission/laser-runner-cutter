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
 * Test/demo task that continuously follows a detected circle target: unlike
 * LaserTargeting's static aim/burn, this predicts the target's future
 * position and adds laser waypoints ahead of it, strobing the laser on
 * briefly at a fixed interval so the beam stays visible without a
 * continuous burn.
 */
class CircleFollowerTask {
 public:
  CircleFollowerTask(
      std::shared_ptr<
          CallbackRegistry<detection_interfaces::msg::DetectionResult>>
          detectionCallbackRegistry,
      std::shared_ptr<LaserControlClient> laser,
      std::shared_ptr<DetectionClient> detection,
      std::shared_ptr<Calibration> calibration, rclcpp::Logger logger);
  ~CircleFollowerTask() = default;

  void run(float trackMissTimeoutSecs, int targetAttempts, float lookaheadSecs,
           const LaserColor& laserColor, float laserIntervalSecs,
           std::atomic<bool>& stopSignal);

 private:
  std::shared_ptr<CallbackRegistry<detection_interfaces::msg::DetectionResult>>
      detectionCallbackRegistry_;
  std::shared_ptr<LaserControlClient> laser_;
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  rclcpp::Logger logger_;
};
