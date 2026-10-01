#include "runner_cutter_control/calibration/calibration.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>

#include "builtin_interfaces/msg/time.hpp"
#include "detection_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control/clients/laser_detection_context.hpp"

Calibration::Calibration(std::shared_ptr<LaserControlClient> laser,
                         std::shared_ptr<CameraControlClient> camera,
                         std::shared_ptr<DetectionClient> detection)
    : laser_{std::move(laser)},
      camera_{std::move(camera)},
      detection_{std::move(detection)},
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

std::size_t Calibration::collectGridCorrespondences(
    const LaserColor& laserColor, std::pair<int, int> gridSize,
    std::pair<float, float> xBounds, std::pair<float, float> yBounds,
    bool saveImages,
    std::optional<std::reference_wrapper<std::atomic<bool>>> stopSignal) {
  // Get color frame size
  auto state{camera_->getState()};
  cameraFrameSize_ = {static_cast<int>(state->color_width),
                      static_cast<int>(state->color_height)};
  if (cameraFrameSize_.width <= 0 || cameraFrameSize_.height <= 0) {
    spdlog::warn("Invalid camera frame size.");
    return 0;
  }

  if (gridSize.first < 2 || gridSize.second < 2) {
    spdlog::warn(
        "Invalid calibration grid size ({}, {}). Each dimension must be at "
        "least 2.",
        gridSize.first, gridSize.second);
    return 0;
  }

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
  std::size_t numAdded{collectCorrespondences(pendingLaserCoords, laserColor,
                                              saveImages, stopSignal)};
  spdlog::info(
      "{} out of {} point correspondences found. {} total correspondences.",
      numAdded, pendingLaserCoords.size(), pointCorrespondences_.size());
  return numAdded;
}

std::size_t Calibration::collectCorrespondences(
    const std::vector<LaserCoord>& laserCoords, const LaserColor& laserColor,
    bool saveImages,
    std::optional<std::reference_wrapper<std::atomic<bool>>> stopSignal) {
  if (laserCoords.empty()) {
    return 0;
  }

  std::size_t numPointCorrespondencesAdded{0};
  laser_->setColor(laserColor);
  laser_->clearPaths();

  {
    // Prepare laser and camera for laser detection
    LaserDetectionContext context{laser_, camera_};
    for (const auto& laserCoord : laserCoords) {
      if (stopSignal && stopSignal->get()) {
        return 0;
      }

      laser_->setPoint(0, laserCoord);
      // Give sufficient time for the galvo to settle and for a new camera frame
      // to become available
      std::this_thread::sleep_for(std::chrono::duration<float>(0.15f));

      auto resultOpt{findPointCorrespondence(laserCoord)};

      if (saveImages) {
        camera_->saveImage();
      }

      if (!resultOpt) {
        continue;
      }

      auto [cameraPixelCoord, cameraPosition]{std::move(*resultOpt)};
      pointCorrespondences_.add(laserCoord, cameraPixelCoord, cameraPosition);
      spdlog::info("Added point correspondence. {} total correspondences.",
                   pointCorrespondences_.size());

      ++numPointCorrespondencesAdded;

      laser_->clearPaths();
    }
  }

  return numPointCorrespondencesAdded;
}

void Calibration::updateModel() {
  pointCorrespondences_.updateModel();
  isCalibrated_ = pointCorrespondences_.hasModel();
  logFitStats();
}

LaserCoord Calibration::cameraPositionToLaserCoord(
    const Position& cameraPosition) const {
  auto laserCoordOpt{pointCorrespondences_.project(cameraPosition)};
  if (!laserCoordOpt) {
    return {-1.0f, -1.0f};
  }
  return *laserCoordOpt;
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

    // Load the point correspondences and fit the model
    pointCorrespondences_.deserialize(ifs);

    ifs.close();
  } catch (const std::exception& e) {
    spdlog::error("Failed to load calibration file: {}", e.what());
    return false;
  }

  spdlog::info("Loaded calibration file {} with {} correspondences.", filePath,
               pointCorrespondences_.size());
  isCalibrated_ = pointCorrespondences_.hasModel();
  logFitStats();
  return isCalibrated_;
}

std::optional<Calibration::FindPointCorrespondenceResult>
Calibration::findPointCorrespondence(const LaserCoord& laserCoord,
                                     int numFrames, int maxAttempts,
                                     float attemptIntervalSecs) {
  // Average the detected laser across multiple camera frames to reduce noise,
  // particularly in depth. Detection runs on the latest camera frame, so
  // consecutive requests may return the same frame. Use the frame timestamp to
  // only count each frame once.
  Eigen::Vector2d pixelSum{Eigen::Vector2d::Zero()};
  Eigen::Vector3d positionSum{Eigen::Vector3d::Zero()};
  int numFramesFound{0};
  std::optional<builtin_interfaces::msg::Time> lastTimestamp;
  for (int attempt = 0; attempt < maxAttempts && numFramesFound < numFrames;
       ++attempt) {
    auto result{detection_->getDetection(
        detection_interfaces::msg::DetectionType::LASER)};
    bool isNewFrame{!lastTimestamp || result->timestamp != *lastTimestamp};
    if (result->instances.empty() || !isNewFrame) {
      std::this_thread::sleep_for(
          std::chrono::duration<float>(attemptIntervalSecs));
      continue;
    }
    lastTimestamp = result->timestamp;

    // In case multiple lasers were detected, use the instance with the highest
    // confidence
    const auto& bestInstance{
        *std::max_element(result->instances.begin(), result->instances.end(),
                          [](const auto& a, const auto& b) {
                            return a.confidence < b.confidence;
                          })};
    pixelSum += Eigen::Vector2d{bestInstance.point.x, bestInstance.point.y};
    positionSum +=
        Eigen::Vector3d{bestInstance.position.x, bestInstance.position.y,
                        bestInstance.position.z};
    ++numFramesFound;
  }

  if (numFramesFound < numFrames) {
    spdlog::info(
        "Laser detected in only {} of {} required frames for laserCoord = "
        "({}, {}).",
        numFramesFound, numFrames, laserCoord.x, laserCoord.y);
    return std::nullopt;
  }

  Eigen::Vector2d pixel{pixelSum / numFramesFound};
  Eigen::Vector3d position{positionSum / numFramesFound};
  PixelCoord cameraPixelCoord{static_cast<int>(std::round(pixel.x())),
                              static_cast<int>(std::round(pixel.y()))};
  Position cameraPosition{static_cast<float>(position.x()),
                          static_cast<float>(position.y()),
                          static_cast<float>(position.z())};

  spdlog::info(
      "Found point correspondence averaged over {} frames: laserCoord = ({}, "
      "{}), cameraPixelCoord = ({}, {}), cameraPosition = ({}, {}, {}).",
      numFramesFound, laserCoord.x, laserCoord.y, cameraPixelCoord.u,
      cameraPixelCoord.v, cameraPosition.x, cameraPosition.y, cameraPosition.z);

  return FindPointCorrespondenceResult{cameraPixelCoord, cameraPosition};
}

void Calibration::logFitStats() const {
  if (!pointCorrespondences_.hasModel()) {
    spdlog::warn("Failed to fit calibration model with {} correspondences.",
                 pointCorrespondences_.size());
    return;
  }

  auto stats{pointCorrespondences_.getFitStats()};
  const auto& model{pointCorrespondences_.getModel()};
  spdlog::info(
      "Current model fit with {} of {} correspondences as inliers. \n"
      "\tLaser coord error: mean {:.5f}, max {:.5f}\n"
      "\t Position error: mean {:.2f}, max {:.2f}\n"
      "\t Depth range: [{:.0f}, {:.0f}]\n"
      "\t Mirror distance: {:.2f}\n"
      "\t {} mirror first",
      stats.numInliers, pointCorrespondences_.size(), stats.meanLaserError,
      stats.maxLaserError, stats.meanPositionError, stats.maxPositionError,
      stats.minDepth, stats.maxDepth, model.mirrorDistance,
      model.xMirrorFirst ? "x" : "y");
}
