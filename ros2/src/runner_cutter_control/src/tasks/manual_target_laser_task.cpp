#include "runner_cutter_control/tasks/manual_target_laser_task.hpp"

#include <fmt/core.h>

#include <cmath>

#include "common/ros_utils.hpp"

ManualTargetLaserTask::ManualTargetLaserTask(
    std::shared_ptr<LaserControlClient> laser,
    std::shared_ptr<CameraControlClient> camera,
    std::shared_ptr<DetectionClient> detection,
    std::shared_ptr<Calibration> calibration, rclcpp::Logger logger,
    rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
        notificationsPublisher)
    : detection_(std::move(detection)),
      calibration_(std::move(calibration)),
      laserTargeting_(std::move(laser), std::move(camera), detection_,
                      calibration_, logger),
      logger_(std::move(logger)),
      notificationsPublisher_(std::move(notificationsPublisher)) {}

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
    auto aimResult{laserTargeting_.aim(0, targetPosition, targetPixel,
                                       trackingLaserColor, stopSignal)};
    if (aimResult.status != LaserTargeting::AimStatus::SUCCESS) {
      // Don't publish notif if aiming was interrupted by the stop signal
      if (aimResult.status != LaserTargeting::AimStatus::STOPPED) {
        common::publishNotification(
            logger_, notificationsPublisher_,
            fmt::format("Failed to aim laser: {}.",
                        LaserTargeting::describeAimStatus(aimResult.status)),
            rclcpp::Logger::Level::Warn);
      }
      return;
    }
    laserCoord = aimResult.laserCoord;
  } else {
    laserCoord = calibration_->cameraPositionToLaserCoord(targetPosition);
  }

  // Burn
  if (shouldBurn) {
    if (!laserTargeting_.burn(0, laserCoord, burnLaserColor, burnTimeSecs)) {
      common::publishNotification(
          logger_, notificationsPublisher_,
          "Failed to burn target: out of reach of laser.",
          rclcpp::Logger::Level::Warn);
    } else {
      common::publishNotification(logger_, notificationsPublisher_,
                                  "Successfully burned target.",
                                  rclcpp::Logger::Level::Info);
    }
  }
}
