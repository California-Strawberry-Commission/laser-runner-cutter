#include "runner_cutter_control/tasks/laser_targeting.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include "detection_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control/clients/laser_detection_context.hpp"

std::string LaserTargeting::describeAimStatus(AimStatus status) {
  switch (status) {
    case AimStatus::SUCCESS:
      return "success";
    case AimStatus::LASER_NOT_DETECTED:
      return "tracking laser was not detected";
    case AimStatus::LASER_COORD_OUT_OF_BOUNDS:
      return "target out of reach of laser";
    case AimStatus::MAX_ATTEMPTS_EXCEEDED:
      return "laser did not converge on the target";
    case AimStatus::STOPPED:
      return "aiming was stopped";
  }
  return "unknown";
}

LaserTargeting::LaserTargeting(std::shared_ptr<LaserControlClient> laser,
                               std::shared_ptr<CameraControlClient> camera,
                               std::shared_ptr<DetectionClient> detection,
                               std::shared_ptr<Calibration> calibration,
                               rclcpp::Logger logger)
    : laser_(std::move(laser)),
      camera_(std::move(camera)),
      detection_(std::move(detection)),
      calibration_(std::move(calibration)),
      logger_(std::move(logger)) {}

LaserTargeting::AimResult LaserTargeting::aim(
    uint32_t targetId, const Position& targetCameraPosition,
    const PixelCoord& targetCameraPixel, const LaserColor& trackingLaserColor,
    std::atomic<bool>& stopSignal) {
  LaserCoord initialLaserCoord{
      calibration_->cameraPositionToLaserCoord(targetCameraPosition)};
  if (initialLaserCoord.x < 0.0f || initialLaserCoord.x > 1.0f ||
      initialLaserCoord.y < 0.0f || initialLaserCoord.y > 1.0f) {
    RCLCPP_WARN(logger_,
                "[LaserTargeting][aim] Initial laser coord is outside of "
                "renderable area.");
    return {AimStatus::LASER_COORD_OUT_OF_BOUNDS, {}};
  }

  LaserDetectionContext context{laser_, camera_};
  laser_->setColor(trackingLaserColor);
  return correctLaser(targetId, initialLaserCoord, targetCameraPixel,
                      stopSignal);
}

bool LaserTargeting::burn(uint32_t targetTrackId, const LaserCoord& laserCoord,
                          const LaserColor& burnLaserColor,
                          float burnTimeSecs) {
  if (laserCoord.x < 0.0f || laserCoord.x > 1.0f || laserCoord.y < 0.0f ||
      laserCoord.y > 1.0f) {
    RCLCPP_WARN(
        logger_,
        "[LaserTargeting][burn] Laser coord is outside of renderable area.");
    return false;
  }

  LaserDetectionContext context{laser_, camera_};
  laser_->clearPaths();
  laser_->setColor(burnLaserColor);
  laser_->play();
  RCLCPP_INFO(logger_, "[LaserTargeting][burn] Burning track %u for %f secs",
              targetTrackId, burnTimeSecs);
  laser_->setPoint(targetTrackId, laserCoord);
  constexpr auto KEEPALIVE_PERIOD{std::chrono::milliseconds(100)};
  auto deadline{std::chrono::steady_clock::now() +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<float>(burnTimeSecs))};
  while (std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(KEEPALIVE_PERIOD);
    laser_->setPoint(targetTrackId, laserCoord);
  }
  laser_->clearPaths();
  laser_->stop();
  RCLCPP_INFO(logger_, "[LaserTargeting][burn] Burn complete on track %u",
              targetTrackId);
  return true;
}

LaserTargeting::AimResult LaserTargeting::correctLaser(
    uint32_t targetId, const LaserCoord& initialLaserCoord,
    const PixelCoord& targetCameraPixel, std::atomic<bool>& stopSignal,
    float pixelDistanceThreshold, int maxAttempts) {
  LaserCoord currentLaserCoord{initialLaserCoord};
  int attempt{0};
  while (attempt < maxAttempts && !stopSignal) {
    laser_->setPoint(targetId, currentLaserCoord);
    // Give sufficient time for the galvo to settle and for a new camera frame
    // to become available
    std::this_thread::sleep_for(std::chrono::duration<float>(0.15f));
    // Get detected camera pixel coord and camera-space position for laser
    auto detectResultOpt{detectLaser(stopSignal)};
    if (!detectResultOpt) {
      if (stopSignal) {
        return {AimStatus::STOPPED, {}};
      }
      RCLCPP_WARN(logger_,
                  "[LaserTargeting][correctLaser] Could not detect laser "
                  "during correction");
      return {AimStatus::LASER_NOT_DETECTED, {}};
    }

    // Calculate camera pixel distance
    auto [laserPixel, laserPosition]{std::move(*detectResultOpt)};
    PixelCoord cameraPixelDelta{targetCameraPixel.u - laserPixel.u,
                                targetCameraPixel.v - laserPixel.v};
    float dist{
        static_cast<float>(std::hypot(cameraPixelDelta.u, cameraPixelDelta.v))};
    RCLCPP_INFO(logger_,
                "[LaserTargeting][correctLaser] Aiming laser. Target camera "
                "pixel = (%d, %d), laser detected at = (%d, %d), dist = %f",
                targetCameraPixel.u, targetCameraPixel.v, laserPixel.u,
                laserPixel.v, dist);

    if (dist <= pixelDistanceThreshold) {
      RCLCPP_INFO(logger_,
                  "[LaserTargeting][correctLaser] Correction successful");
      return {AimStatus::SUCCESS, currentLaserCoord};
    }

    // Calculate new laser coord
    LaserCoord laserCoordCorrection{
        calibration_->cameraPixelDeltaToLaserCoordDelta(cameraPixelDelta)};
    LaserCoord newLaserCoord{currentLaserCoord.x + laserCoordCorrection.x,
                             currentLaserCoord.y + laserCoordCorrection.y};
    RCLCPP_INFO(logger_,
                "[LaserTargeting][correctLaser] Distance too large. Correcting "
                "laser. Camera pixel delta = (%d, %d), laser coord correction "
                "= (%f, %f). Current laser coord = (%f, %f), corrected laser "
                "coord = (%f, %f)",
                cameraPixelDelta.u, cameraPixelDelta.v, laserCoordCorrection.x,
                laserCoordCorrection.y, currentLaserCoord.x,
                currentLaserCoord.y, newLaserCoord.x, newLaserCoord.y);

    if (newLaserCoord.x < 0.0f || newLaserCoord.x > 1.0f ||
        newLaserCoord.y < 0.0f || newLaserCoord.y > 1.0f) {
      RCLCPP_WARN(logger_,
                  "[LaserTargeting][correctLaser] Corrected laser coord is "
                  "outside of renderable area.");
      return {AimStatus::LASER_COORD_OUT_OF_BOUNDS, {}};
    }

    currentLaserCoord = newLaserCoord;
    ++attempt;
  }

  return {stopSignal ? AimStatus::STOPPED : AimStatus::MAX_ATTEMPTS_EXCEEDED,
          {}};
}

std::optional<LaserTargeting::DetectLaserResult> LaserTargeting::detectLaser(
    std::atomic<bool>& stopSignal, int maxAttempts) {
  int attempt{0};
  while (attempt < maxAttempts && !stopSignal) {
    auto detectionResult{detection_->getDetection(
        detection_interfaces::msg::DetectionType::LASER)};
    auto instances{detectionResult->instances};
    if (instances.size() > 0) {
      // In case multiple lasers were detected, use the instance with the
      // highest confidence
      const auto& bestInstance = *std::max_element(
          instances.begin(), instances.end(), [](const auto& a, const auto& b) {
            return a.confidence < b.confidence;
          });
      return DetectLaserResult{
          {static_cast<int>(std::round(bestInstance.point.x)),
           static_cast<int>(std::round(bestInstance.point.y))},
          {static_cast<float>(bestInstance.position.x),
           static_cast<float>(bestInstance.position.y),
           static_cast<float>(bestInstance.position.z)}};
    }

    // No lasers detected. Try again.
    ++attempt;
  }

  return std::nullopt;
}
