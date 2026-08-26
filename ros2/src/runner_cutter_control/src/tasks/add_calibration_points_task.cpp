#include "runner_cutter_control/tasks/add_calibration_points_task.hpp"

#include <fmt/core.h>

#include <algorithm>

#include "common/ros_utils.hpp"

AddCalibrationPointsTask::AddCalibrationPointsTask(
    std::shared_ptr<DetectionClient> detection,
    std::shared_ptr<Calibration> calibration, rclcpp::Logger logger,
    rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
        notificationsPublisher)
    : detection_(std::move(detection)),
      calibration_(std::move(calibration)),
      logger_(std::move(logger)),
      notificationsPublisher_(std::move(notificationsPublisher)) {}

void AddCalibrationPointsTask::run(
    const std::vector<NormalizedPixelCoord>& normalizedPixelCoords,
    bool saveImages, const LaserColor& trackingLaserColor,
    std::atomic<bool>& stopSignal) {
  // For each camera pixel coord, find the 3D position wrt the camera
  auto positionsOpt{detection_->getPositions(normalizedPixelCoords)};
  if (!positionsOpt) {
    return;
  }
  auto positions{std::move(*positionsOpt)};

  // Filter out any invalid positions (x, y, and z are all negative)
  positions.erase(std::remove_if(positions.begin(), positions.end(),
                                 [](const Position& position) {
                                   return position.x < 0.0f &&
                                          position.y < 0.0f &&
                                          position.z < 0.0f;
                                 }),
                  positions.end());

  // Convert camera positions to laser pixels
  std::vector<LaserCoord> laserCoords;
  for (const auto& position : positions) {
    laserCoords.push_back(calibration_->cameraPositionToLaserCoord(position));
  }

  // Filter out laser coords that are out of bounds
  laserCoords.erase(
      std::remove_if(laserCoords.begin(), laserCoords.end(),
                     [](const LaserCoord& coord) {
                       return !(0.0f <= coord.x && coord.x <= 1.0f &&
                                0.0f <= coord.y && coord.y <= 1.0f);
                     }),
      laserCoords.end());

  std::size_t numPointsAdded{calibration_->addCalibrationPoints(
      laserCoords, trackingLaserColor, true, saveImages, stopSignal)};

  common::publishNotification(
      logger_, notificationsPublisher_,
      fmt::format("Added {} calibration point(s)", numPointsAdded));
}
