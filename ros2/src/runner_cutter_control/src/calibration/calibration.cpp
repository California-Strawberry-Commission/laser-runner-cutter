#include "runner_cutter_control/calibration/calibration.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>

#include "camera_control_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control/clients/laser_detection_context.hpp"

Calibration::Calibration(std::shared_ptr<LaserControlClient> laser,
                         std::shared_ptr<CameraControlClient> camera)
    : laser_{std::move(laser)},
      camera_{std::move(camera)},
      pointCorrespondences_{} {}

FrameSize Calibration::getCameraFrameSize() const { return cameraFrameSize_; }

PixelRect Calibration::getLaserBounds() const {
  return pointCorrespondences_.getLaserBounds();
}

NormalizedPixelRect Calibration::getNormalizedLaserBounds() const {
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

bool Calibration::calibrate(
    const LaserColor& laserColor, std::pair<int, int> gridSize,
    std::pair<float, float> xBounds, std::pair<float, float> yBounds,
    bool saveImages,
    std::optional<std::reference_wrapper<std::atomic<bool>>> stopSignal) {
  reset();

  // Get color frame size
  auto frameOpt{camera_->getFrame()};
  if (!frameOpt) {
    return false;
  }
  auto frame{std::move(*frameOpt)};

  cameraFrameSize_ = {static_cast<int>(frame.colorFrame->width),
                      static_cast<int>(frame.colorFrame->height)};

  // Get calibration points
  float xMin{xBounds.first};
  float xMax{xBounds.second};
  float yMin{yBounds.first};
  float yMax{yBounds.second};
  float xStep{(xMax - xMin) / (gridSize.first - 1)};
  float yStep{(yMax - yMin) / (gridSize.second - 1)};
  std::vector<LaserCoord> pendingLaserCoords;
  for (int i = 0; i < gridSize.first; ++i) {
    for (int j = 0; j < gridSize.second; ++j) {
      float x{xMin + i * xStep};
      float y{yMin + j * yStep};
      pendingLaserCoords.emplace_back(LaserCoord{x, y});
    }
  }

  // Get image correspondences
  spdlog::info("Getting image correspondences");
  addCalibrationPoints(pendingLaserCoords, laserColor,
                       false,  // we will update transforms later
                       saveImages, stopSignal);
  spdlog::info("{} out of {} point correspondences found.",
               pointCorrespondences_.size(), pendingLaserCoords.size());
  if (pointCorrespondences_.size() < 3) {
    spdlog::warn(
        "Calibration failed: insufficient point correspondences found.");
    return false;
  }

  // Use linear least squares for an initial estimate...
  pointCorrespondences_.updateTransformLinearLeastSquares();
  // ...then refine using nonlinear least squares
  updateTransform();

  isCalibrated_ = true;
  return true;
}

LaserCoord Calibration::cameraPositionToLaserCoord(
    const Position& cameraPosition) const {
  Eigen::Vector4d homogeneousCameraPosition{
      static_cast<double>(cameraPosition.x),
      static_cast<double>(cameraPosition.y),
      static_cast<double>(cameraPosition.z), 1.0};
  Eigen::Vector3d transformed{
      homogeneousCameraPosition.transpose() *
      pointCorrespondences_.getCameraToLaserTransform()};

  if (std::fabs(transformed[2]) < EPSILON) {
    return {-1.0f, -1.0f};
  }

  // Normalize by the third (homogeneous) coordinate to get (x, y)
  // coordinates
  Eigen::Vector3d homogeneousTransformed{transformed / transformed[2]};
  return {static_cast<float>(homogeneousTransformed[0]),
          static_cast<float>(homogeneousTransformed[1])};
}

LaserCoord Calibration::cameraPixelDeltaToLaserCoordDelta(
    const PixelCoord& cameraPixelCoordDelta) const {
  Eigen::Vector2d cameraPixelDelta{
      static_cast<double>(cameraPixelCoordDelta.u),
      static_cast<double>(cameraPixelCoordDelta.v)};
  Eigen::Vector2d laserCoordDelta{
      pointCorrespondences_.getCameraPixelToLaserCoordJacobian() *
      cameraPixelDelta};
  return {static_cast<float>(laserCoordDelta[0]),
          static_cast<float>(laserCoordDelta[1])};
}

std::size_t Calibration::addCalibrationPoints(
    const std::vector<LaserCoord>& laserCoords, const LaserColor& laserColor,
    bool updateTransform, bool saveImages,
    std::optional<std::reference_wrapper<std::atomic<bool>>> stopSignal) {
  if (laserCoords.empty()) {
    return 0;
  }

  std::size_t numPointCorrespondencesAdded{0};
  laser_->setColor(laserColor);
  laser_->clearPoint();

  {
    // Prepare laser and camera for laser detection
    LaserDetectionContext context{laser_, camera_};
    for (const auto& laserCoord : laserCoords) {
      if (stopSignal && stopSignal->get()) {
        return 0;
      }

      laser_->setPoint(laserCoord);
      // Give sufficient time for galvo to settle.
      // TODO: This shouldn't be necessary in theory since getDetection waits
      // for several frames before running detection, so we'll need to figure
      // out why this helps.
      std::this_thread::sleep_for(std::chrono::duration<float>(0.1f));

      auto resultOpt{findPointCorrespondence(laserCoord)};

      if (saveImages) {
        camera_->saveImage();
      }

      if (!resultOpt) {
        continue;
      }

      auto [cameraPixelCoord, cameraPosition]{std::move(*resultOpt)};
      addPointCorrespondence(laserCoord, cameraPixelCoord, cameraPosition);
      ++numPointCorrespondencesAdded;

      laser_->clearPoint();
    }
  }

  // This is behind a flag as updating the transform is computationally
  // non-trivial
  if (updateTransform && numPointCorrespondencesAdded > 0) {
    this->updateTransform();
  }

  return numPointCorrespondencesAdded;
}

void Calibration::addPointCorrespondence(const LaserCoord& laserCoord,
                                         const PixelCoord& cameraPixelCoord,
                                         const Position& cameraPosition,
                                         bool updateTransform) {
  pointCorrespondences_.add(laserCoord, cameraPixelCoord, cameraPosition);
  spdlog::info("Added point correspondence. {} total correspondences.",
               pointCorrespondences_.size());

  // This is behind a flag as updating the transform is computationally
  // non-trivial
  if (updateTransform) {
    this->updateTransform();
  }
}

void Calibration::updateTransform() {
  pointCorrespondences_.updateTransformNonlinearLeastSquares();
  pointCorrespondences_.updateCameraPixelToLaserCoordJacobian();
  spdlog::info(
      "Updated transform. Reprojection error: {}, with {} total "
      "correspondences.",
      pointCorrespondences_.getReprojectionError(),
      pointCorrespondences_.size());
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

  try {
    std::ofstream ofs{filePath, std::ios::binary};
    if (!ofs) {
      spdlog::error("Failed to open file for saving calibration.");
      return false;
    }

    // Save the camera frame size
    ofs.write(reinterpret_cast<const char*>(&cameraFrameSize_.width),
              sizeof(cameraFrameSize_.width));
    ofs.write(reinterpret_cast<const char*>(&cameraFrameSize_.height),
              sizeof(cameraFrameSize_.height));

    // Save the point correspondences
    pointCorrespondences_.serialize(ofs);

    ofs.close();
  } catch (const std::exception& e) {
    spdlog::error("Failed to save calibration: {}", e.what());
    return false;
  }

  spdlog::info("Calibration saved to {}", filePath);
  return true;
}

bool Calibration::load(const std::string& filePath) {
  if (!std::filesystem::exists(filePath)) {
    spdlog::error("Could not find calibration file {}.", filePath);
    return false;
  }

  try {
    std::ifstream ifs{filePath, std::ios::binary};
    if (!ifs) {
      spdlog::error("Could not load calibration file {}.", filePath);
      return false;
    }

    // Load the camera frame size
    int cameraFrameWidth, cameraFrameHeight;
    ifs.read(reinterpret_cast<char*>(&cameraFrameWidth),
             sizeof(cameraFrameWidth));
    ifs.read(reinterpret_cast<char*>(&cameraFrameHeight),
             sizeof(cameraFrameHeight));
    cameraFrameSize_ = {cameraFrameWidth, cameraFrameHeight};

    // Load the point correspondences
    pointCorrespondences_.deserialize(ifs);

    ifs.close();
  } catch (const std::exception& e) {
    spdlog::error("Failed to load calibration file: {}", e.what());
    return false;
  }

  isCalibrated_ = true;
  spdlog::info(
      "Successfully loaded calibration file {}. Reprojection error: {}, with "
      "{} total "
      "correspondences.",
      filePath, pointCorrespondences_.getReprojectionError(),
      pointCorrespondences_.size());

  return true;
}

std::optional<Calibration::FindPointCorrespondenceResult>
Calibration::findPointCorrespondence(const LaserCoord& laserCoord,
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

    // In case multiple lasers were detected, use the instance with the highest
    // confidence
    const auto& bestInstance =
        *std::max_element(result->instances.begin(), result->instances.end(),
                          [](const auto& a, const auto& b) {
                            return a.confidence < b.confidence;
                          });
    PixelCoord cameraPixelCoord{
        static_cast<int>(std::round(bestInstance.point.x)),
        static_cast<int>(std::round(bestInstance.point.y))};
    Position cameraPosition{static_cast<float>(bestInstance.position.x),
                            static_cast<float>(bestInstance.position.y),
                            static_cast<float>(bestInstance.position.z)};

    spdlog::info(
        "Found point correspondence: laser_coord = ({}, {}), pixel = ({}, "
        "{}), position = ({}, {}, {}).",
        laserCoord.x, laserCoord.y, cameraPixelCoord.u, cameraPixelCoord.v,
        cameraPosition.x, cameraPosition.y, cameraPosition.z);

    return FindPointCorrespondenceResult{cameraPixelCoord, cameraPosition};
  }

  return std::nullopt;
}
