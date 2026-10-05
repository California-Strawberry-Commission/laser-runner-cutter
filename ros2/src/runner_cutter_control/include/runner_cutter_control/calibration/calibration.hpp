#pragma once

#include <tuple>
#include <vector>

#include "runner_cutter_control/calibration/point_correspondences.hpp"
#include "runner_cutter_control/clients/camera_control_client.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"
#include "runner_cutter_control/common_types.hpp"

class Calibration {
 public:
  explicit Calibration(std::shared_ptr<LaserControlClient> laser,
                       std::shared_ptr<CameraControlClient> camera,
                       std::shared_ptr<DetectionClient> detection);
  ~Calibration() = default;

  FrameSize getCameraFrameSize() const { return cameraFrameSize_; }

  /**
   * @return The rect (min x, min y, width, height) representing the reach of
   * the laser, in terms of camera pixels, clipped to the camera frame.
   */
  PixelRect getLaserBounds() const;

  NormalizedPixelRect getNormalizedLaserBounds() const;

  bool isCalibrated() const { return pointCorrespondences_.hasModel(); }

  std::size_t getPointCorrespondencesCount() const {
    return pointCorrespondences_.size();
  }

  PointCorrespondences::FitStats getFitStats() const {
    return pointCorrespondences_.getFitStats();
  }

  /**
   * Clear all point correspondences and the fitted model.
   */
  void clear();

  /**
   * Find and add point correspondences for a grid of laser coords.
   *
   * 1. Shoot the laser at predetermined coords in a square grid pattern
   * 2. For each laser point, capture an image from the camera and identify the
   * corresponding pixel coord in the camera frame
   * 3. Add point correspondences to PointCorrespondences
   *
   * Note that this will append point correspondences to PointCorrespondences.
   * Call `clear()` before calling this to start fresh. This does not fit the
   * model; call `updateModel()` after all point correspondences have been
   * added.
   *
   * @param laserColor Laser color to shoot while calibrating.
   * @param gridSize Number of points in the x and y directions to use as
   * calibration points.
   * @param xBounds Min and max x for the calibration points
   * @param yBounds Min and max y for the calibration points
   * @param saveImages Whether to save an image at each calibration coordinate.
   * @param stopSignal Flag to enable the calibration process to be prematurely
   * terminated when set to true.
   * @return Number of point correspondences successfully added.
   */
  std::size_t collectGridCorrespondences(
      const LaserColor& laserColor, std::pair<int, int> gridSize = {7, 7},
      std::pair<float, float> xBounds = {0.0f, 1.0f},
      std::pair<float, float> yBounds = {0.0f, 1.0f}, bool saveImages = false,
      std::optional<std::reference_wrapper<std::atomic<bool>>> stopSignal =
          std::nullopt);

  /**
   * Find and add point correspondences by shooting the laser at each of
   * laserCoords. Like `collectGridCorrespondences()`, this appends to the
   * existing point correspondences and does not fit the model.
   *
   * @param laserCoords Laser coordinates to find point correspondences with.
   * @param laserColor Laser color to shoot while calibrating.
   * @param saveImages Whether to save an image at each laser coordinate.
   * @param stopSignal Flag to enable the calibration process to be prematurely
   * terminated when set to true.
   * @return Number of point correspondences successfully added.
   */
  std::size_t collectCorrespondences(
      const std::vector<LaserCoord>& laserCoords, const LaserColor& laserColor,
      bool saveImages = false,
      std::optional<std::reference_wrapper<std::atomic<bool>>> stopSignal =
          std::nullopt);

  /**
   * Fit the camera-space position to laser coord model to all point
   * correspondences, and update whether the calibration is usable. Fitting is
   * computationally expensive, so call this once after adding point
   * correspondences (e.g. after `collectGridCorrespondences()` at each depth),
   * rather than after each one.
   */
  void updateModel();

  /**
   * Transform a 3D position in camera-space to a laser coord.
   *
   * @param cameraPosition A 3D position (x, y, z) in camera-space.
   * @return (x, y) laser coordinates.
   */
  LaserCoord cameraPositionToLaserCoord(const Position& cameraPosition) const;

  /**
   * Transform a camera pixel coord delta to a laser coord delta.
   *
   * @param cameraPixelCoordDelta Camera pixel coordinate delta (dx, dy).
   * @return (dx, dy) laser coordinate delta.
   */
  LaserCoord cameraPixelDeltaToLaserCoordDelta(
      const PixelCoord& cameraPixelCoordDelta) const;

  /**
   * Save the current calibration data (specifically, point correspondences) to
   * a file.
   *
   * @param filePath Fully qualified file path where the calibration data will
   * be written to.
   * @return Whether the calibration data was saved successfully.
   */
  bool save(const std::string& filePath);

  /**
   * Load an existing calibration data file.
   *
   * @param filePath Fully qualified file path where the calibration data will
   * be loaded from.
   * @return Whether the calibration data was loaded successfully.
   */
  bool load(const std::string& filePath);

 private:
  struct FindPointCorrespondenceResult {
    PixelCoord cameraPixelCoord;
    Position cameraPosition;
  };

  /**
   * Detect the laser in numFrames distinct camera frames and average the
   * results.
   *
   * @param laserCoord Laser coord currently being shot (for logging).
   * @param numFrames Number of distinct frames to average across.
   * @param maxAttempts Max number of detection requests. Requests that return
   * no laser or an already-used frame count as attempts.
   * @param attemptIntervalSecs Time to wait after such a request.
   * @return Averaged point correspondence, or nullopt if the laser was not
   * detected in numFrames distinct frames within maxAttempts.
   */
  std::optional<FindPointCorrespondenceResult> findPointCorrespondence(
      const LaserCoord& laserCoord, int numFrames = 3, int maxAttempts = 10,
      float attemptIntervalSecs = 0.1f);

  void logFitStats() const;

  std::shared_ptr<LaserControlClient> laser_;
  std::shared_ptr<CameraControlClient> camera_;
  std::shared_ptr<DetectionClient> detection_;
  FrameSize cameraFrameSize_{0, 0};
  PointCorrespondences pointCorrespondences_{};
};
