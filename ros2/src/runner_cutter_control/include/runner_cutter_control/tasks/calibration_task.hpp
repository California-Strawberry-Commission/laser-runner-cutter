#pragma once

#include <atomic>

#include "rcl_interfaces/msg/log.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/common_types.hpp"

/**
 * Runs the laser/camera calibration process.
 */
class CalibrationTask {
 public:
  CalibrationTask(std::shared_ptr<Calibration> calibration,
                  rclcpp::Logger logger,
                  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
                      notificationsPublisher);
  ~CalibrationTask() = default;

  void run(bool saveImages, const LaserColor& trackingLaserColor,
           std::pair<int, int> gridSize, std::pair<float, float> xBounds,
           std::pair<float, float> yBounds, std::atomic<bool>& stopSignal);

 private:
  std::shared_ptr<Calibration> calibration_;
  rclcpp::Logger logger_;
  rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
      notificationsPublisher_;
};
