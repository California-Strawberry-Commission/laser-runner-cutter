#pragma once

#include <atomic>
#include <vector>

#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/common_types.hpp"

/**
 * Adds extra calibration point correspondences at a given set of camera
 * pixel coordinates, without redoing a full grid calibration. Used to
 * refine an existing calibration's accuracy in specific regions.
 */
class AddCalibrationPointsTask {
 public:
  AddCalibrationPointsTask(
      std::shared_ptr<DetectionClient> detection,
      std::shared_ptr<Calibration> calibration, rclcpp::Logger logger,
      rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
          notificationsPublisher);
  ~AddCalibrationPointsTask() = default;

  void run(const std::vector<NormalizedPixelCoord>& normalizedPixelCoords,
           bool saveImages, const LaserColor& trackingLaserColor,
           std::atomic<bool>& stopSignal);

 private:
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  rclcpp::Logger logger_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
};
