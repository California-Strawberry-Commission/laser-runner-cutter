#pragma once

#include <ByteTrack/BYTETracker.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <optional>

#include "detection/detector/yolov8.hpp"

class RunnerDetector {
 public:
  struct Runner {
    // The detection's confidence probability
    float conf{0.0f};
    // The object bounding box rectangle (TLWH)
    cv::Rect rect;
    // Semantic segmentation mask, inside the bounding box
    cv::Mat boxMask;
    // The representative point of the detected object
    cv::Point point{-1, -1};
    // The detection's track ID
    int trackId{-1};
  };

  static void drawDetections(cv::Mat& targetImage,
                             const std::vector<Runner>& runners,
                             const cv::Size& originalImageSize,
                             cv::Scalar color = {255, 0, 0},
                             unsigned int scale = 1);

  explicit RunnerDetector();
  RunnerDetector(const RunnerDetector&) = delete;
  RunnerDetector& operator=(const RunnerDetector&) = delete;
  RunnerDetector(RunnerDetector&&) noexcept = default;
  RunnerDetector& operator=(RunnerDetector&&) noexcept = default;
  ~RunnerDetector() = default;

  std::vector<Runner> track(
      const cv::Mat& imageRgb,
      const std::optional<cv::Rect>& bounds = std::nullopt);
  std::vector<Runner> track(
      const cv::cuda::GpuMat& imageRgb,
      const std::optional<cv::Rect>& bounds = std::nullopt);

 private:
  std::unique_ptr<YoloV8> model_;
  std::unique_ptr<byte_track::BYTETracker> tracker_;
};