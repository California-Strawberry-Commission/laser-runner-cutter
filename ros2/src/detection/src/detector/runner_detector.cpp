#include "detection/detector/runner_detector.hpp"

#include <spdlog/spdlog.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>

static inline float calculateIou(const cv::Rect2f& a, const cv::Rect2f& b) {
  float inter{(a & b).area()};
  float uni{a.area() + b.area() - inter};
  return (uni > 0.0f) ? inter / uni : 0.0f;
}

static inline cv::Rect2f toCvRect(const byte_track::Rect<float>& btRect) {
  return cv::Rect2f(btRect.x(), btRect.y(), btRect.width(), btRect.height());
}

/**
 * Map each Track ID -> index of matched detection (or -1 if none).
 * Greedy per-track, IoU threshold, tie-break by detection confidence.
 */
static std::unordered_map<int, int> matchTracksToDetections(
    const std::vector<byte_track::BYTETracker::STrackPtr>& tracks,
    const std::vector<Object>& detections, float iouThreshold = 0.8f) {
  std::unordered_map<int, int> trackToDetection;
  trackToDetection.reserve(tracks.size());

  if (tracks.empty()) {
    return trackToDetection;
  }

  std::vector<bool> detUsed(detections.size(), false);
  for (const auto& track : tracks) {
    int trackId{track->getTrackId()};
    auto trackRect{toCvRect(track->getRect())};

    int bestIdx{-1};
    float bestIou{-1.0f};
    float bestConf{-1.0f};
    for (int detectionIdx = 0;
         detectionIdx < static_cast<int>(detections.size()); ++detectionIdx) {
      // Prevent duplicate mappings
      if (detUsed[detectionIdx]) {
        continue;
      }

      // Check IoU
      float iou{calculateIou(trackRect, detections[detectionIdx].rect)};
      if (iou < iouThreshold) {
        continue;
      }

      if (iou > bestIou ||
          (iou == bestIou && detections[detectionIdx].conf > bestConf)) {
        bestIou = iou;
        bestConf = detections[detectionIdx].conf;
        bestIdx = detectionIdx;
      }
    }

    if (bestIdx >= 0) {
      trackToDetection[trackId] = bestIdx;
      detUsed[bestIdx] = true;
    } else {
      trackToDetection[trackId] = -1;
    }
  }

  return trackToDetection;
}

RunnerDetector::RunnerDetector() {
  // Locate the package share directory
  std::string packageShareDir{
      ament_index_cpp::get_package_share_directory("detection")};

  // Build path to models directory and engine file
  std::filesystem::path modelsDir{std::filesystem::path(packageShareDir) /
                                  "models"};
  std::filesystem::path enginePath{modelsDir / "RunnerSegYoloV8l.engine"};

  if (!std::filesystem::exists(enginePath)) {
    throw std::runtime_error("Model engine file not found at " +
                             enginePath.string());
  }

  // Create YoloV8 model
  model_ = std::make_unique<YoloV8>(enginePath.string());

  // Create ByteTrack
  tracker_ = std::make_unique<byte_track::BYTETracker>();
}

std::vector<Runner> RunnerDetector::track(const cv::Mat& imageRGB) {
  std::vector<Object> detections{model_->predict(imageRGB)};

  // Run through ByteTrack
  std::vector<byte_track::Object> btObjects;
  btObjects.reserve(detections.size());
  for (const auto& obj : detections) {
    const cv::Rect2f& rect{obj.rect};
    byte_track::Rect<float> btRect(rect.x, rect.y, rect.width, rect.height);
    btObjects.emplace_back(btRect, obj.label, obj.conf);
  }
  auto tracks{tracker_->update(btObjects)};

  // Associate ByteTrack tracks to detections
  auto tracksToDetections{matchTracksToDetections(tracks, detections)};

  // TODO: Determine associated point of mask using skeletonization

  std::vector<Runner> runners;
  return runners;
}