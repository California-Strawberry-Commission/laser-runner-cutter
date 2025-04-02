#pragma once

#include <tuple>
#include <vector>

#include "runner_cutter_control_cpp/calibration/point_correspondences.hpp"
#include "runner_cutter_control_cpp/clients/camera_control_client.hpp"
#include "runner_cutter_control_cpp/clients/laser_control_client.hpp"

class Calibration {
 public:
  Calibration(std::shared_ptr<LaserControlClient> laser,
              std::shared_ptr<CameraControlClient> camera,
              std::tuple<float, float, float> laserColor);

  std::pair<int, int> getCameraFrameSize() const;
  std::tuple<int, int, int, int> getLaserBounds() const;
  std::tuple<float, float, float, float> getNormalizedLaserBounds() const;
  bool isCalibrated() const;
  void reset();

  /**
   * Use image correspondences to compute the transformation matrix from camera
   * to laser. Note that calling this resets the point correspondences.
   *
   * 1. Shoot the laser at predetermined points
   * 2. For each laser point, capture an image from the camera
   * 3. Identify the corresponding point in the camera frame
   * 4. Compute the transformation matrix from camera to laser
   *
   * @param gridSize Number of points in the x and y directions to use as
   * calibration points.
   * @return whether calibration was successful or not.
   */
  bool calibrate(std::pair<int, int> gridSize = {5, 5});

  /**
   * Transform a 3D position in camera-space to a laser coord.
   *
   * @param cameraPosition A 3D position (x, y, z) in camera-space.
   * @return (x, y) laser coordinates.
   */
  std::pair<float, float> cameraPositionToLaserCoord(
      std::tuple<float, float, float> cameraPosition);

  /**
   * Find and add additional point correspondences by shooting the laser at each
   * laserCoords and then optionally recalculate the transform.
   *
   * @param laserCoords Laser coordinates to find point correspondences with.
   * @param updateTransform Whether to recalculate the camera-space position to
   * laser coord transform.
   * @return Number of point correspondences successfully added.
   */
  std::size_t addCalibrationPoints(
      const std::vector<std::pair<float, float>>& laserCoords,
      bool updateTransform = false);

  /**
   * Add the point correspondence between laser coord and camera-space position
   * and optionally update the transform.
   *
   * @param laserCoord Laser coordinate (x, y) of the point correspondence.
   * @param cameraPixelCoord Camera pixel coordinate (x, y) of the point
   * correspondence.
   * @param cameraPosition Camera-space position (x, y, z) of the point
   * correspondence.
   * @param updateTransform Whether to update the transform matrix or not.
   */
  void addPointCorrespondence(std::pair<float, float> laserCoord,
                              std::pair<int, int> cameraPixelCoord,
                              std::tuple<float, float, float> cameraPosition,
                              bool updateTransform = false);

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
  static constexpr float EPSILON = 1e-6f;

  struct FindPointCorrespondenceResult {
    std::pair<int, int> cameraPixelCoord;
    std::tuple<float, float, float> cameraPosition;
  };
  std::optional<FindPointCorrespondenceResult> findPointCorrespondence(
      std::pair<float, float> laserCoord, int numAttempts = 3,
      float attemptIntervalSecs = 0.25f);

  std::shared_ptr<LaserControlClient> laser_;
  std::shared_ptr<CameraControlClient> camera_;
  std::tuple<float, float, float> laserColor_{0.0f, 0.0f, 0.0f};
  std::pair<int, int> cameraFrameSize_{0, 0};
  PointCorrespondences pointCorrespondences_;
  bool isCalibrated_{false};
};
