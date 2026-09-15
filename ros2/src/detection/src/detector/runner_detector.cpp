#include "detection/detector/runner_detector.hpp"

#include <spdlog/spdlog.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <optional>

namespace {

/**
 * Samples a float matrix at a sub-pixel position using bilinear interpolation,
 * clamping out-of-bounds coordinates to the edge.
 */
float bilinearSample(const cv::Mat& dist, float x, float y) {
  if (dist.rows == 0 || dist.cols == 0) {
    return 0.0f;
  }

  x = std::clamp(x, 0.0f, dist.cols - 1.0f);
  y = std::clamp(y, 0.0f, dist.rows - 1.0f);

  int x0{static_cast<int>(std::floor(x))};
  int y0{static_cast<int>(std::floor(y))};
  int x1{std::min(x0 + 1, dist.cols - 1)};
  int y1{std::min(y0 + 1, dist.rows - 1)};

  float ax{x - x0};
  float ay{y - y0};

  float v00{dist.at<float>(y0, x0)};
  float v01{dist.at<float>(y0, x1)};
  float v10{dist.at<float>(y1, x0)};
  float v11{dist.at<float>(y1, x1)};

  float v0{v00 + ax * (v01 - v00)};
  float v1{v10 + ax * (v11 - v10)};
  return v0 + ay * (v1 - v0);
}

/**
 * Extract the medial ridge by doing NMS along the gradient of the distance map.
 * Keep pixels that are local maxima along the gradient and above a small
 * distance threshold.
 */
void extractMedialRidge(const cv::Mat& dist, std::vector<cv::Point>& ridge,
                        float distThreshold = 0.5f) {
  // Gradients of distance transform
  cv::Mat gx, gy;
  cv::Sobel(dist, gx, CV_32F, 1, 0, 3);
  cv::Sobel(dist, gy, CV_32F, 0, 1, 3);

  ridge.clear();

  for (int y = 1; y < dist.rows - 1; ++y) {
    const float* drow{dist.ptr<float>(y)};
    const float* gxrow{gx.ptr<float>(y)};
    const float* gyrow{gy.ptr<float>(y)};
    for (int x = 1; x < dist.cols - 1; ++x) {
      float d{drow[x]};
      if (d < distThreshold) {
        continue;
      }

      float gxn{gxrow[x]};
      float gyn{gyrow[x]};
      float gnorm{std::sqrt(gxn * gxn + gyn * gyn)};
      if (gnorm < 1e-6f) {
        continue;
      }

      // Unit gradient direction
      gxn /= gnorm;
      gyn /= gnorm;

      // Sample distance slightly forward/backward along gradient line
      float dPlus{bilinearSample(dist, x + 0.5f * gxn, y + 0.5f * gyn)};
      float dMinus{bilinearSample(dist, x - 0.5f * gxn, y - 0.5f * gyn)};

      // Keep if it is a local maximum along gradient direction
      if (d >= dPlus && d >= dMinus) {
        ridge.emplace_back(x, y);
      }
    }
  }
}

/**
 * Calculates the medial-ridge point closest to a reference point, in image
 * coordinates. If `referencePoint` is given, the reference is `referencePoint`
 * blended with the mask centroid, weighted `centroidWeight` toward the
 * centroid. If no `referencePoint` is given, the reference is just the mask
 * centroid.
 */
std::optional<cv::Point> findRepresentativePoint(
    const cv::Rect& maskRect, const cv::Mat& mask,
    const std::optional<cv::Rect>& bounds = std::nullopt,
    const std::optional<cv::Point2f>& referencePoint = std::nullopt,
    float centroidWeight = 0.1f, float distThreshold = 0.5f) {
  if (mask.empty()) {
    return std::nullopt;
  }

  // If bounds are provided, intersect with the mask rect to get the ROI
  cv::Rect roi{bounds ? (*bounds & maskRect) : maskRect};
  // Convert ROI to mask-local coordinates
  cv::Rect localRoi{roi.x - maskRect.x, roi.y - maskRect.y, roi.width,
                    roi.height};
  if (localRoi.empty()) {
    return std::nullopt;
  }

  // Use the Euclidean distance transform to replace every foreground pixel's
  // value with its distance to the nearest background pixel
  cv::Mat dist;
  cv::distanceTransform(mask, dist, cv::DIST_L2, 3);

  // Extract medial ridge (in mask-local coordinates)
  std::vector<cv::Point> ridgePoints;
  extractMedialRidge(dist, ridgePoints, distThreshold);
  if (ridgePoints.empty()) {
    // If no medial ridge, fall back to distance transform's max point
    double maxVal{0.0};
    cv::Point maxLoc;
    cv::minMaxLoc(dist, nullptr, &maxVal, nullptr, &maxLoc);
    return cv::Point{maxLoc.x + maskRect.x, maxLoc.y + maskRect.y};
  }

  // Compute the centroid of the mask (in mask-local coordinates)
  cv::Moments m{cv::moments(mask, /*binaryImage=*/true)};
  if (m.m00 == 0.0) {
    // If there's no area, fall back to distance transform's max point
    double maxVal{0.0};
    cv::Point maxLoc;
    cv::minMaxLoc(dist, nullptr, &maxVal, nullptr, &maxLoc);
    return cv::Point{maxLoc.x + maskRect.x, maxLoc.y + maskRect.y};
  }
  cv::Point2d centroid{m.m10 / m.m00, m.m01 / m.m00};

  // Determine the reference point, in mask-local coordinates, that ridge
  // points are compared against
  cv::Point2d reference{centroid};
  if (referencePoint) {
    cv::Point2d localPrevious{referencePoint->x - maskRect.x,
                              referencePoint->y - maskRect.y};
    reference =
        (1.0 - centroidWeight) * localPrevious + centroidWeight * centroid;
  }

  // Pick the ridge point in the ROI that is closest to the reference point
  double bestD2{std::numeric_limits<double>::infinity()};
  std::optional<cv::Point> best;
  for (const auto& p : ridgePoints) {
    if (!localRoi.contains(p)) {
      continue;
    }

    double dx{p.x - reference.x};
    double dy{p.y - reference.y};
    double d2{dx * dx + dy * dy};
    if (d2 < bestD2) {
      bestD2 = d2;
      // Map mask-local coordinate back to image coordinate
      best = cv::Point(p.x + maskRect.x, p.y + maskRect.y);
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

RunnerDetector::RunnerDetector(const std::string& modelName) {
  // Locate the package share directory
  std::string packageShareDir{
      ament_index_cpp::get_package_share_directory("detection")};

  // Build path to models directory and engine file
  std::filesystem::path modelsDir{std::filesystem::path(packageShareDir) /
                                  "models"};
  std::filesystem::path enginePath{modelsDir / modelName};

  if (!std::filesystem::exists(enginePath)) {
    throw std::runtime_error("Model engine file not found at " +
                             enginePath.string());
  }

  spdlog::info("[RunnerDetector] Using model: {}", enginePath.string());

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

  // Run through ByteTrack
  std::vector<byte_track::Object> btObjects;
  btObjects.reserve(predictionResult.size());
  for (const auto& obj : predictionResult) {
    const cv::Rect& rect{obj.rect};
    byte_track::Rect<float> btRect(rect.x, rect.y, rect.width, rect.height);
    btObjects.emplace_back(btRect, obj.label, obj.conf);
  }
  auto tracks{tracker_->update(btObjects)};

  // Map each active track's ID to its current ByteTrack bounding box estimate.
  // We use this because it is Kalman filtered and provides a steady reference.
  std::unordered_map<int, cv::Rect> trackRects;
  trackRects.reserve(tracks.size());
  for (const auto& track : tracks) {
    trackRects[static_cast<int>(track->getTrackId())] =
        toCvRect(track->getRect());
  }

  // Associate ByteTrack tracks to detections, and invert into detection index
  // -> track ID
  auto trackToDetectionMap{matchTracksToDetections(tracks, predictionResult)};
  std::unordered_map<int, int> detectionToTrackMap;
  detectionToTrackMap.reserve(trackToDetectionMap.size());
  for (const auto& [trackId, objIdx] : trackToDetectionMap) {
    if (objIdx >= 0) {
      detectionToTrackMap[objIdx] = trackId;
    }
  }

  // Determine the representative point for each detected runner. For a tracked
  // instance, we reconstruct a reference point based on the normalized
  // coordinate within that track's bounding box from last frame, with the goal
  // to reduce jitter without drifting as the track moves.
  std::vector<Runner> runners;
  runners.reserve(predictionResult.size());
  for (int i = 0; i < static_cast<int>(predictionResult.size()); ++i) {
    const auto& obj{predictionResult[i]};

    // Find the current detection object's track ID, ByteTrack filtered bounding
    // box, and reference point.
    int trackId{-1};
    const cv::Rect* trackRect{nullptr};
    std::optional<cv::Point2f> referencePoint;
    if (auto it{detectionToTrackMap.find(i)}; it != detectionToTrackMap.end()) {
      trackId = it->second;
      if (auto rectIt{trackRects.find(trackId)}; rectIt != trackRects.end() &&
                                                 rectIt->second.width > 0 &&
                                                 rectIt->second.height > 0) {
        trackRect = &rectIt->second;
        if (auto fracIt{previousPointNormalizedInBbox_.find(trackId)};
            fracIt != previousPointNormalizedInBbox_.end()) {
          referencePoint =
              cv::Point2f{trackRect->x + fracIt->second.x * trackRect->width,
                          trackRect->y + fracIt->second.y * trackRect->height};
        }
      }
    }

    // Calculate the new representative point
    cv::Point point{-1, -1};
    auto repPointOpt{
        findRepresentativePoint(obj.rect, obj.boxMask, bounds, referencePoint)};
    if (repPointOpt) {
      point = *repPointOpt;
      if (trackId >= 0 && trackRect != nullptr) {
        previousPointNormalizedInBbox_[trackId] = cv::Point2f{
            (point.x - trackRect->x) / static_cast<float>(trackRect->width),
            (point.y - trackRect->y) / static_cast<float>(trackRect->height)};
      }
    }

    Runner runner;
    runner.conf = obj.conf;
    runner.rect = obj.rect;
    runner.boxMask = obj.boxMask;
    runner.point = point;
    runner.trackId = trackId;
    runners.push_back(runner);
  }

  // Prune point history for tracks that are no longer active
  for (auto it = previousPointNormalizedInBbox_.begin();
       it != previousPointNormalizedInBbox_.end();) {
    it = trackRects.count(it->first) ? std::next(it)
                                     : previousPointNormalizedInBbox_.erase(it);
  }

  return runners;
}

void RunnerDetector::reset() {
  tracker_ = std::make_unique<byte_track::BYTETracker>();
  previousPointNormalizedInBbox_.clear();
}
