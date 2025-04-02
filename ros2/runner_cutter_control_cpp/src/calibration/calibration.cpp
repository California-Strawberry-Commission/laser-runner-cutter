#include "runner_cutter_control_cpp/calibration/calibration.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>

#include "camera_control_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control_cpp/clients/laser_detection_context.hpp"

Calibration::Calibration(std::shared_ptr<LaserControlClient> laser,
                         std::shared_ptr<CameraControlClient> camera,
                         std::tuple<float, float, float> laserColor)
    : laser_{laser},
      camera_{camera},
      laserColor_{laserColor},
      pointCorrespondences_{} {}

std::pair<int, int> Calibration::getCameraFrameSize() const {
  return cameraFrameSize_;
}

std::tuple<int, int, int, int> Calibration::getLaserBounds() const {
  return pointCorrespondences_.getLaserBounds();
}

std::tuple<float, float, float, float> Calibration::getNormalizedLaserBounds()
    const {
  auto [w, h]{getCameraFrameSize()};
  auto [boundsXMin, boundsYMin, boundsWidth, boundsHeight]{getLaserBounds()};
  return {(w > 0) ? boundsXMin / static_cast<float>(w) : 0.0f,
          (h > 0) ? boundsYMin / static_cast<float>(h) : 0.0f,
          (w > 0) ? boundsWidth / static_cast<float>(w) : 0.0f,
          (h > 0) ? boundsHeight / static_cast<float>(h) : 0.0f};
}

bool Calibration::isCalibrated() const { return isCalibrated_; }

void Calibration::reset() {
  pointCorrespondences_.clear();
  isCalibrated_ = false;
}

bool Calibration::calibrate(std::pair<int, int> gridSize) {
  reset();

  // Get color frame size
  auto frameOpt{camera_->getFrame()};
  if (!frameOpt) {
    return false;
  }
  auto colorFrame{frameOpt.value().colorFrame};
  cameraFrameSize_ = {colorFrame->width, colorFrame->height};

  // Get calibration points
  float xStep{1.0f / (gridSize.first - 1)};
  float yStep{1.0f / (gridSize.second - 1)};
  std::vector<std::pair<float, float>> pendingLaserCoords;
  for (int i = 0; i < gridSize.first; ++i) {
    for (int j = 0; j < gridSize.second; ++j) {
      pendingLaserCoords.emplace_back(i * xStep, j * yStep);
    }
  }

  // Get image correspondences
  spdlog::info("Getting image correspondences");
  addCalibrationPoints(pendingLaserCoords);
  spdlog::info("{} out of {} point correspondences found.",
               pointCorrespondences_.size(), pendingLaserCoords.size());
  if (pointCorrespondences_.size() < 3) {
    spdlog::warn(
        "Calibration failed: insufficient point correspondences found.");
    return false;
  }

  // Use linear least squares for an initial estimate, then refine using
  // nonlinear least squares
  pointCorrespondences_.updateTransformLinearLeastSquares();
  pointCorrespondences_.updateTransformNonlinearLeastSquares();

  isCalibrated_ = true;
  return true;
}

std::pair<float, float> Calibration::cameraPositionToLaserCoord(
    std::tuple<float, float, float> cameraPosition) {
  Eigen::Vector4f homogeneousCameraPosition{std::get<0>(cameraPosition),
                                            std::get<1>(cameraPosition),
                                            std::get<2>(cameraPosition), 1.0f};
  Eigen::Vector3f transformed{
      homogeneousCameraPosition.transpose() *
      pointCorrespondences_.getCameraToLaserTransform()};

  if (std::fabs(transformed[2]) < EPSILON) {
    return {-1.0f, -1.0f};
  }

  // Normalize by the third (homogeneous) coordinate to get (x, y)
  // coordinates
  Eigen::Vector3f homogeneousTransformed{transformed / transformed[2]};
  return {homogeneousTransformed[0], homogeneousTransformed[1]};
}

std::size_t Calibration::addCalibrationPoints(
    const std::vector<std::pair<float, float>>& laserCoords,
    bool updateTransform) {
  if (laserCoords.empty()) {
    return 0;
  }

  std::size_t numPointCorrespondencesAdded{0};
  laser_->clearPoints();
  laser_->setColor(0.0f, 0.0f, 0.0f, 0.0f);

  {
    // Prepare laser and camera for laser detection
    LaserDetectionContext context{laser_, camera_};
    laser_->play();
    for (const auto& laserCoord : laserCoords) {
      laser_->setPoints({laserCoord});
      auto [r, g, b]{laserColor_};
      laser_->setColor(r, g, b, 0.0f);

      auto resultOpt{findPointCorrespondence(laserCoord)};
      if (!resultOpt) {
        continue;
      }

      auto [cameraPixelCoord, cameraPosition]{resultOpt.value()};
      addPointCorrespondence(laserCoord, cameraPixelCoord, cameraPosition);
      ++numPointCorrespondencesAdded;

      // We use setColor() instead of stop() as it is faster to temporarily turn
      // off the laser
      laser_->setColor(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }

  // This is behind a flag as updating the transform is computationally
  // non-trivial
  if (updateTransform && numPointCorrespondencesAdded > 0) {
    pointCorrespondences_.updateTransformNonlinearLeastSquares();
  }

  return numPointCorrespondencesAdded;
}

void Calibration::addPointCorrespondence(
    std::pair<float, float> laserCoord, std::pair<int, int> cameraPixelCoord,
    std::tuple<float, float, float> cameraPosition, bool updateTransform) {
  pointCorrespondences_.add(laserCoord, cameraPixelCoord, cameraPosition);
  spdlog::info("Added point correspondence. {} total correspondences.",
               pointCorrespondences_.size());

  // This is behind a flag as updating the transform is computationally
  // non-trivial
  if (updateTransform) {
    pointCorrespondences_.updateTransformNonlinearLeastSquares();
  }
}

bool Calibration::save(const std::string& filePath) {
  if (!isCalibrated_) {
    return false;
  }

  std::filesystem::path fullPath{filePath};
  std::filesystem::path dirPath{fullPath.parent_path()};
  if (dirPath.empty()) {
    return false;
  }

  std::filesystem::create_directories(dirPath);
  std::ofstream ofs{filePath, std::ios::binary};
  if (!ofs) {
    spdlog::error("Failed to open file for saving calibration.");
    return false;
  }

  pointCorrespondences_.serialize(ofs);
  ofs.close();

  spdlog::info("Calibration saved to {}", filePath);
  return true;
}

bool Calibration::load(const std::string& filePath) {
  if (!std::filesystem::exists(filePath)) {
    spdlog::error("Could not find calibration file {}.", filePath);
    return false;
  }

  std::ifstream ifs{filePath, std::ios::binary};
  if (!ifs) {
    spdlog::error("Could not load calibration file {}.", filePath);
    return false;
  }

  pointCorrespondences_.deserialize(ifs);
  pointCorrespondences_.updateTransformLinearLeastSquares();
  pointCorrespondences_.updateTransformNonlinearLeastSquares();
  isCalibrated_ = true;
  spdlog::info("Successfully loaded calibration file {}", filePath);
  return true;
}

std::optional<Calibration::FindPointCorrespondenceResult>
Calibration::findPointCorrespondence(std::pair<float, float> laserCoord,
                                     int numAttempts,
                                     float attemptIntervalSecs) {
  for (int attempt = 0; attempt < numAttempts; ++attempt) {
    spdlog::info("Attempt {} to detect laser and find point correspondence.",
                 attempt);
    auto result{camera_->getDetection(
        camera_control_interfaces::msg::DetectionType::LASER, true)};
    if (result->instances.empty()) {
      std::this_thread::sleep_for(
          std::chrono::duration<float>(attemptIntervalSecs));
      continue;
    }

    // TODO: handle case where multiple lasers detected
    auto& instance{result->instances.front()};
    std::pair<int, int> cameraPixelCoord{std::round(instance.point.x),
                                         std::round(instance.point.y)};
    std::tuple<float, float, float> cameraPosition{
        instance.position.x, instance.position.y, instance.position.z};

    spdlog::info(
        "Found point correspondence: laser_coord = ({}, {}), pixel = ({}, "
        "{}), "
        "position = ({}, {}, {}).",
        laserCoord.first, laserCoord.second, cameraPixelCoord.first,
        cameraPixelCoord.second, instance.position.x, instance.position.y,
        instance.position.z);

    return FindPointCorrespondenceResult{cameraPixelCoord, cameraPosition};
  }

  return std::nullopt;
}
