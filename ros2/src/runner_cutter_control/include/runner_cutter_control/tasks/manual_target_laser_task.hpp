#pragma once

#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/camera_control_client.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/tasks/laser_targeting.hpp"

/**
 * Targets a single, user-specified camera pixel coordinate on demand,
 * optionally aiming and/or burning it.
 */
class ManualTargetLaserTask {
 public:
  ManualTargetLaserTask(std::shared_ptr<LaserControlClient> laser,
                        std::shared_ptr<CameraControlClient> camera,
                        std::shared_ptr<DetectionClient> detection,
                        std::shared_ptr<Calibration> calibration,
                        rclcpp::Logger logger);
  ~ManualTargetLaserTask() = default;

  void run(const NormalizedPixelCoord& normalizedPixelCoord, bool shouldAim,
           bool shouldBurn, const LaserColor& trackingLaserColor,
           const LaserColor& burnLaserColor, float burnTimeSecs,
           std::atomic<bool>& stopSignal);

 private:
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  LaserTargeting laserTargeting_;
  rclcpp::Logger logger_;
};
