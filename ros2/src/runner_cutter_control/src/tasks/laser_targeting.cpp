#include "runner_cutter_control/tasks/laser_targeting.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include "detection_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control/clients/laser_detection_context.hpp"

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

std::optional<LaserCoord> LaserTargeting::aim(
    uint32_t targetId, const Position& targetCameraPosition,
    const PixelCoord& targetCameraPixel, const LaserColor& trackingLaserColor,
    std::atomic<bool>& stopSignal) {
  LaserDetectionContext context{laser_, camera_};
  LaserCoord initialLaserCoord{
      calibration_->cameraPositionToLaserCoord(targetCameraPosition)};
  laser_->setColor(trackingLaserColor);
  return correctLaser(targetId, initialLaserCoord, targetCameraPixel,
                      stopSignal);
}

void LaserTargeting::burn(uint32_t targetTrackId, const LaserCoord& laserCoord,
                          const LaserColor& burnLaserColor,
                          float burnTimeSecs) {
  LaserDetectionContext context{laser_, camera_};
  laser_->clearPaths();
  laser_->setColor(burnLaserColor);
  laser_->play();
  RCLCPP_INFO(logger_, "Burning track %u for %f secs", targetTrackId,
              burnTimeSecs);
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
  RCLCPP_INFO(logger_, "Burn complete on track %u", targetTrackId);
}

std::optional<LaserCoord> LaserTargeting::correctLaser(
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
      RCLCPP_WARN(logger_, "Could not detect laser during correction");
      return std::nullopt;
    }

    // Calculate camera pixel distance
    auto [laserPixel, laserPosition]{std::move(*detectResultOpt)};
    PixelCoord cameraPixelDelta{targetCameraPixel.u - laserPixel.u,
                                targetCameraPixel.v - laserPixel.v};
    float dist{
        static_cast<float>(std::hypot(cameraPixelDelta.u, cameraPixelDelta.v))};
    RCLCPP_INFO(
        logger_,
        "Aiming laser. Target camera pixel = (%d, %d), laser detected at = "
        "(%d, %d), dist = %f",
        targetCameraPixel.u, targetCameraPixel.v, laserPixel.u, laserPixel.v,
        dist);

    if (dist <= pixelDistanceThreshold) {
      RCLCPP_INFO(logger_, "Correction successful");
      return currentLaserCoord;
    }

    // Calculate new laser coord
    LaserCoord laserCoordCorrection{
        calibration_->cameraPixelDeltaToLaserCoordDelta(cameraPixelDelta)};
    LaserCoord newLaserCoord{currentLaserCoord.x + laserCoordCorrection.x,
                             currentLaserCoord.y + laserCoordCorrection.y};
    RCLCPP_INFO(logger_,
                "Distance too large. Correcting laser. Camera pixel delta = "
                "(%d, %d), laser coord correction = (%f, %f). Current laser "
                "coord = (%f, %f), corrected laser coord = (%f, %f)",
                cameraPixelDelta.u, cameraPixelDelta.v, laserCoordCorrection.x,
                laserCoordCorrection.y, currentLaserCoord.x,
                currentLaserCoord.y, newLaserCoord.x, newLaserCoord.y);

    if (newLaserCoord.x > 1.0f || newLaserCoord.y > 1.0f ||
        newLaserCoord.x < 0.0f || newLaserCoord.y < 0.0f) {
      RCLCPP_INFO(logger_, "Laser coord is outside of renderable area.");
      return std::nullopt;
    }

    currentLaserCoord = newLaserCoord;
    ++attempt;
  }

  return std::nullopt;
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
