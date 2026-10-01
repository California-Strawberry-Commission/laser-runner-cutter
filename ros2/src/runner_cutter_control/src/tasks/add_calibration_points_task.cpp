#include "runner_cutter_control/tasks/add_calibration_points_task.hpp"

#include <fmt/core.h>

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
    const LaserColor& trackingLaserColor, bool saveImages,
    std::atomic<bool>& stopSignal) {
  // The existing calibration is needed to aim the laser at the requested
  // points
  if (!calibration_->isCalibrated()) {
    common::publishNotification(
        logger_, notificationsPublisher_,
        "Cannot add calibration point: no existing calibration.",
        rclcpp::Logger::Level::Warn);
    return;
  }

  // For each camera pixel coord, find the 3D position wrt the camera
  auto positionsOpt{detection_->getPositions(normalizedPixelCoords)};
  if (!positionsOpt) {
    common::publishNotification(
        logger_, notificationsPublisher_,
        "Cannot add calibration points: failed to get 3D positions.",
        rclcpp::Logger::Level::Warn);
    return;
  }
  const auto& positions{*positionsOpt};

  // Convert camera positions to laser coords
  std::vector<LaserCoord> laserCoords;
  for (std::size_t i = 0; i < normalizedPixelCoords.size(); ++i) {
    const auto& normalizedPixelCoord{normalizedPixelCoords[i]};
    // Invalid positions have x, y, and z all negative
    if (i >= positions.size() ||
        (positions[i].x < 0.0f && positions[i].y < 0.0f &&
         positions[i].z < 0.0f)) {
      common::publishNotification(
          logger_, notificationsPublisher_,
          fmt::format("Skipped ({:.3f}, {:.3f}): no valid depth at this pixel.",
                      normalizedPixelCoord.u, normalizedPixelCoord.v),
          rclcpp::Logger::Level::Warn);
      continue;
    }

    LaserCoord laserCoord{
        calibration_->cameraPositionToLaserCoord(positions[i])};
    if (!(0.0f <= laserCoord.x && laserCoord.x <= 1.0f &&
          0.0f <= laserCoord.y && laserCoord.y <= 1.0f)) {
      common::publishNotification(
          logger_, notificationsPublisher_,
          fmt::format("Skipped ({:.3f}, {:.3f}): outside the laser's "
                      "range ({:.3f}, {:.3f}).",
                      normalizedPixelCoord.u, normalizedPixelCoord.v,
                      laserCoord.x, laserCoord.y),
          rclcpp::Logger::Level::Warn);
      continue;
    }
    laserCoords.push_back(laserCoord);
  }

  std::size_t numPointsAdded{calibration_->collectCorrespondences(
      laserCoords, trackingLaserColor, saveImages, stopSignal)};
  if (numPointsAdded > 0) {
    calibration_->updateModel();
  }

  common::publishNotification(
      logger_, notificationsPublisher_,
      fmt::format("Added {} of {} calibration point(s). Total: {}",
                  numPointsAdded, normalizedPixelCoords.size(),
                  calibration_->getPointCorrespondencesCount()));
}
