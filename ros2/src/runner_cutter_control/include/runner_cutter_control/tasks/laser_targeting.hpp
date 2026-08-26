#pragma once

#include <atomic>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/camera_control_client.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"
#include "runner_cutter_control/common_types.hpp"

/**
 * Encapsulates the laser aiming and burning logic shared between tasks.
 *
 * This targets a single static camera-space coordinate at a time. `aim`
 * iteratively corrects the laser onto a fixed target pixel, and `burn` dwells
 * on a fixed laser coordinate for a set duration.
 */
class LaserTargeting {
 public:
  LaserTargeting(std::shared_ptr<LaserControlClient> laser,
                 std::shared_ptr<CameraControlClient> camera,
                 std::shared_ptr<DetectionClient> detection,
                 std::shared_ptr<Calibration> calibration,
                 rclcpp::Logger logger);
  ~LaserTargeting() = default;

  /**
   * Attempt to incrementally guide the laser to a target camera pixel. The
   * target camera position is used to calculate the initial laser coordinate,
   * from which incremental corrections are applied until the laser reaches
   * the target camera pixel.
   *
   * @param targetId ID of the target.
   * @param targetCameraPosition Camera-space position of the target.
   * @param targetCameraPixel Camera pixel coordinate of the target.
   * @param trackingLaserColor Laser color to use while aiming.
   * @param stopSignal Flag to enable the aiming process to be prematurely
   * terminated when set to true.
   * @return The corrected laser coordinate that projects to the target camera
   * pixel.
   */
  std::optional<LaserCoord> aim(uint32_t targetId,
                                const Position& targetCameraPosition,
                                const PixelCoord& targetCameraPixel,
                                const LaserColor& trackingLaserColor,
                                std::atomic<bool>& stopSignal);

  /**
   * Burn a laser coordinate for a fixed duration.
   *
   * @param targetTrackId ID to associate with the laser path.
   * @param laserCoord Laser coordinate to burn.
   * @param burnLaserColor Laser color to use while burning.
   * @param burnTimeSecs Duration, in seconds, to burn for.
   */
  void burn(uint32_t targetTrackId, const LaserCoord& laserCoord,
            const LaserColor& burnLaserColor, float burnTimeSecs);

 private:
  struct DetectLaserResult {
    PixelCoord cameraPixel;
    Position cameraPosition;
  };

  /**
   * Find a laser coordinate that projects to the target camera pixel
   * coordinate to within a specified pixel distance. Starts from an initial
   * coordinate and incrementally calculates laser coordinates to attempt to
   * get within the threshold distance.
   *
   * @param targetId ID of the target.
   * @param initialLaserCoord Initial laser coordinate.
   * @param targetCameraPixel Target camera pixel coordinate.
   * @param stopSignal Flag to enable the correction process to be prematurely
   * terminated when set to true.
   * @param pixelDistanceThreshold Pixel distance threshold under which the
   * corrected laser coordinate is considered close enough to the target.
   * @param maxAttempts Maximum number of iterations.
   * @return The corrected laser coordinate that projects to the target camera
   * pixel.
   */
  std::optional<LaserCoord> correctLaser(uint32_t targetId,
                                         const LaserCoord& initialLaserCoord,
                                         const PixelCoord& targetCameraPixel,
                                         std::atomic<bool>& stopSignal,
                                         float pixelDistanceThreshold = 6.0f,
                                         int maxAttempts = 10);

  std::optional<DetectLaserResult> detectLaser(std::atomic<bool>& stopSignal,
                                               int maxAttempts = 3);

  std::shared_ptr<LaserControlClient> laser_;
  std::shared_ptr<CameraControlClient> camera_;
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  rclcpp::Logger logger_;
};
