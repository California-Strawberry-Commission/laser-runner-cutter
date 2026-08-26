#include "runner_cutter_control/tasks/calibration_task.hpp"

#include <fmt/core.h>

#include "common/ros_utils.hpp"

CalibrationTask::CalibrationTask(
    std::shared_ptr<Calibration> calibration, rclcpp::Logger logger,
    rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
        notificationsPublisher)
    : calibration_(std::move(calibration)),
      logger_(std::move(logger)),
      notificationsPublisher_(std::move(notificationsPublisher)) {}

void CalibrationTask::run(bool saveImages, const LaserColor& trackingLaserColor,
                          std::pair<int, int> gridSize,
                          std::pair<float, float> xBounds,
                          std::pair<float, float> yBounds,
                          std::atomic<bool>& stopSignal) {
  common::publishNotification(logger_, notificationsPublisher_,
                              "Calibration started");
  calibration_->calibrate(trackingLaserColor, gridSize, xBounds, yBounds,
                          saveImages, stopSignal);
  common::publishNotification(
      logger_, notificationsPublisher_,
      fmt::format("Calibration complete with {} point correspondences",
                  calibration_->getPointCorrespondencesCount()));
}
