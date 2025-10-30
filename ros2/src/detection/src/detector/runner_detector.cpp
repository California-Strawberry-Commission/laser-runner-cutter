#include "detection/detector/runner_detector.hpp"

#include <spdlog/spdlog.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <opencv2/ximgproc.hpp>
#include <optional>

namespace {

std::optional<cv::Point> findRepresentativePoint(
    const cv::Rect& maskRect, const cv::Mat& mask,
    const std::optional<cv::Rect>& bounds = std::nullopt) {
  if (mask.empty()) {
    return std::nullopt;
  }

  cv::Mat skeleton{cv::Mat::zeros(mask.size(), CV_8U)};
  cv::ximgproc::thinning(mask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);

  // Find non-zero pixels (mask-local coordinates) in the skeleton
  std::vector<cv::Point> skeletonPoints;
  cv::findNonZero(skeleton, skeletonPoints);
  if (skeletonPoints.empty()) {
    return std::nullopt;
  }

  // Translate mask-local pixel coordinates into full-image coordinates
  for (auto& p : skeletonPoints) {
    p.x += maskRect.x;
    p.y += maskRect.y;
  }

  if (skeletonPoints.size() == 1) {
    return skeletonPoints.front();
  }

  // Filter skeleton points within ROI
  cv::Rect roi{maskRect};
  // If bounds are provided, intersect with the mask rect to get the ROI
  if (bounds) {
    roi = *bounds & maskRect;
    if (roi.empty()) {
      return std::nullopt;
    }
  }

  // Compute centroid of the mask
  cv::Moments m{cv::moments(mask, /*binaryImage=*/true)};
  if (m.m00 == 0.0) {
    // If there's no area, return first skeleton point
    return skeletonPoints.front();
  }
  cv::Point2d centroid{m.m10 / m.m00 + maskRect.x, m.m01 / m.m00 + maskRect.y};

  // Pick skeleton point in the ROI that is closest to the centroid of the mask
  double bestD2{std::numeric_limits<double>::infinity()};
  std::optional<cv::Point> best;
  for (const auto& p : skeletonPoints) {
    if (!roi.contains(p)) {
      continue;
    }

    double dx{p.x - centroid.x};
    double dy{p.y - centroid.y};
    double d2{dx * dx + dy * dy};
    if (d2 < bestD2) {
      bestD2 = d2;
      best = p;
    }
  }

  return best;
}

float calculateIou(const cv::Rect& a, const cv::Rect& b) {
  float intersectionArea{static_cast<float>((a & b).area())};
  float unionArea{a.area() + b.area() - intersectionArea};
  return (unionArea > 0.0f) ? intersectionArea / unionArea : 0.0f;
}

cv::Rect toCvRect(const byte_track::Rect<float>& btRect) {
  return cv::Rect{static_cast<int>(btRect.x()), static_cast<int>(btRect.y()),
                  static_cast<int>(btRect.width()),
                  static_cast<int>(btRect.height())};
}

/**
 * Map each Track ID -> index of matched detection (or -1 if none).
 * Greedy per-track, IoU threshold, tie-break by detection confidence.
 */
std::unordered_map<int, int> matchTracksToDetections(
    const std::vector<byte_track::BYTETracker::STrackPtr>& tracks,
    const std::vector<YoloV8::Object>& detections, float iouThreshold = 0.8f) {
  std::unordered_map<int, int> trackToDetection;
  trackToDetection.reserve(tracks.size());

  if (tracks.empty()) {
    return trackToDetection;
  }

  std::vector<bool> detUsed(detections.size(), false);
  for (const auto& track : tracks) {
    size_t trackId{track->getTrackId()};
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

}  // namespace

void RunnerDetector::drawDetections(
    cv::Mat& targetImage, const std::vector<RunnerDetector::Runner>& runners,
    const cv::Size& originalImageSize, cv::Scalar color, unsigned int scale) {
  if (targetImage.empty() || originalImageSize.width <= 0 ||
      originalImageSize.height <= 0) {
    return;
  }

  // The image we are drawing on may not be the size of the image that the
  // runner detection was run on. Thus, we'll need to scale the bbox, mask, and
  // point of each runner to the image we are drawing on.
  double xScale{static_cast<double>(targetImage.cols) /
                originalImageSize.width};
  double yScale{static_cast<double>(targetImage.rows) /
                originalImageSize.height};

  // Draw segmentation masks
  if (!runners.empty() && !runners[0].boxMask.empty()) {
    cv::Mat mask{targetImage.clone()};
    for (const auto& runner : runners) {
      // Scale rect to current image space
      cv::Rect rect{runner.rect};
      rect.x = rect.x * xScale;
      rect.y = rect.y * yScale;
      rect.width = rect.width * xScale;
      rect.height = rect.height * yScale;
      cv::Rect imgRect(0, 0, targetImage.cols, targetImage.rows);
      cv::Rect roi{rect & imgRect};  // clip to image
      if (roi.width <= 0 || roi.height <= 0) {
        continue;
      }

      // Scale boxMask to current image space
      cv::Mat boxMask{runner.boxMask};
      cv::Mat resizedBoxMask;
      cv::resize(boxMask, resizedBoxMask, cv::Size(roi.width, roi.height), 0, 0,
                 cv::INTER_NEAREST);

      mask(roi).setTo(color, resizedBoxMask);
    }
    // Add all the masks to our image
    cv::addWeighted(targetImage, 0.5, mask, 0.8, 1, targetImage);
  }

  // Bounding boxes and annotations
  double meanColor{cv::mean(color)[0]};
  cv::Scalar textColor{(meanColor > 128) ? cv::Scalar(0, 0, 0)
                                         : cv::Scalar(255, 255, 255)};
  cv::Scalar markerColor{255, 255, 255};
  for (const auto& runner : runners) {
    // Scale rect to current image space
    cv::Rect rect{runner.rect};
    rect.x = rect.x * xScale;
    rect.y = rect.y * yScale;
    rect.width = rect.width * xScale;
    rect.height = rect.height * yScale;
    cv::Rect imgRect(0, 0, targetImage.cols, targetImage.rows);
    cv::Rect roi{rect & imgRect};  // clip to image
    if (roi.width <= 0 || roi.height <= 0) {
      continue;
    }

    // Draw rectangles and text
    char text[256];
    sprintf(text, "%d: %.1f%%", runner.trackId, runner.conf * 100);
    int baseLine{0};
    cv::Size labelSize{cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5 * scale, scale, &baseLine)};
    cv::Scalar textBackgroundColor{color * 0.7};
    cv::rectangle(targetImage, roi, color, scale + 1);
    cv::rectangle(
        targetImage,
        cv::Rect(cv::Point2i(roi.x, roi.y),
                 cv::Size(labelSize.width, labelSize.height + baseLine)),
        textBackgroundColor, -1);
    cv::putText(targetImage, text, cv::Point2i(roi.x, roi.y + labelSize.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, textColor, scale);

    // Draw representative point
    if (runner.point.x >= 0 && runner.point.y >= 0) {
      int x{static_cast<int>(std::round(runner.point.x * xScale))};
      int y{static_cast<int>(std::round(runner.point.y * yScale))};
      cv::drawMarker(targetImage, cv::Point2i(x, y), markerColor,
                     cv::MARKER_TILTED_CROSS, 20, 2);
    }
  }
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

std::vector<RunnerDetector::Runner> RunnerDetector::track(
    const cv::Mat& imageRgb, const std::optional<cv::Rect>& bounds) {
  cv::cuda::GpuMat gpuImg;
  gpuImg.upload(imageRgb);
  return track(gpuImg, bounds);
}

std::vector<RunnerDetector::Runner> RunnerDetector::track(
    const cv::cuda::GpuMat& imageRgb, const std::optional<cv::Rect>& bounds) {
  std::vector<YoloV8::Object> predictionResult{model_->predict(imageRgb)};
  std::vector<Runner> runners;
  runners.reserve(predictionResult.size());

  // Determine representative point for each mask
  for (const auto& obj : predictionResult) {
    cv::Point point{-1, -1};
    // TODO: Improve performance of findRepresentativePoint
    auto repPointOpt{findRepresentativePoint(obj.rect, obj.boxMask, bounds)};
    if (repPointOpt) {
      point = std::move(*repPointOpt);
    }

    Runner runner;
    runner.conf = obj.conf;
    runner.rect = obj.rect;
    runner.boxMask = obj.boxMask;
    runner.point = point;
    runners.push_back(runner);
  }

  // Run through ByteTrack
  std::vector<byte_track::Object> btObjects;
  btObjects.reserve(predictionResult.size());
  for (const auto& obj : predictionResult) {
    const cv::Rect& rect{obj.rect};
    byte_track::Rect<float> btRect(rect.x, rect.y, rect.width, rect.height);
    btObjects.emplace_back(btRect, obj.label, obj.conf);
  }
  auto tracks{tracker_->update(btObjects)};

  // Associate ByteTrack tracks to detections
  auto tracksToDetections{matchTracksToDetections(tracks, predictionResult)};
  for (const auto& [trackId, objIdx] : tracksToDetections) {
    if (objIdx >= 0) {
      runners[objIdx].trackId = trackId;
    }
  }

  return runners;
}
