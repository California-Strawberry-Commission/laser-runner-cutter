#include "runner_cutter_control/tasks/manual_target_laser_task.hpp"

#include <cmath>

ManualTargetLaserTask::ManualTargetLaserTask(
    std::shared_ptr<LaserControlClient> laser,
    std::shared_ptr<CameraControlClient> camera,
    std::shared_ptr<DetectionClient> detection,
    std::shared_ptr<Calibration> calibration, rclcpp::Logger logger)
    : detection_(std::move(detection)),
      calibration_(std::move(calibration)),
      laserTargeting_(std::move(laser), std::move(camera), detection_,
                      calibration_, logger),
      logger_(std::move(logger)) {}

void ManualTargetLaserTask::run(
    const NormalizedPixelCoord& normalizedPixelCoord, bool shouldAim,
    bool shouldBurn, const LaserColor& trackingLaserColor,
    const LaserColor& burnLaserColor, float burnTimeSecs,
    std::atomic<bool>& stopSignal) {
  // Find the 3D position wrt the camera
  std::vector<NormalizedPixelCoord> normalizedPixelCoords{normalizedPixelCoord};
  auto positionsOpt{detection_->getPositions(normalizedPixelCoords)};
  if (!positionsOpt) {
    return;
  }

  auto positions{std::move(*positionsOpt)};
  auto targetPosition{positions[0]};
  auto [frameWidth, frameHeight]{calibration_->getCameraFrameSize()};
  PixelCoord targetPixel{
      static_cast<int>(std::round(normalizedPixelCoord.u * frameWidth)),
      static_cast<int>(std::round(normalizedPixelCoord.v * frameHeight))};

  // Aim
  LaserCoord laserCoord;
  if (shouldAim) {
    auto laserCoordOpt{laserTargeting_.aim(0, targetPosition, targetPixel,
                                           trackingLaserColor, stopSignal)};
    if (!laserCoordOpt) {
      RCLCPP_INFO(logger_, "Failed to aim laser");
      return;
    }
    RCLCPP_INFO(logger_, "Aim laser successful");
    laserCoord = std::move(*laserCoordOpt);
  } else {
    laserCoord = calibration_->cameraPositionToLaserCoord(targetPosition);
  }

  // Burn
  if (shouldBurn) {
    laserTargeting_.burn(0, laserCoord, burnLaserColor, burnTimeSecs);
  }
}
