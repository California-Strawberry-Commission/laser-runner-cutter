#pragma once

#include <ByteTrack/BYTETracker.h>

#include <opencv2/opencv.hpp>

#include "detection/detector/yolov8.hpp"

struct Runner {
  // The detection's confidence probability
  float conf{};
  // The object bounding box rectangle (TLWH)
  cv::Rect2f rect;
  // Semantic segmentation mask, inside the bounding box
  cv::Mat boxMask;
  // The representative point of the detected object
  cv::Point point{-1, -1};
  // The detection's track ID
  int trackId{-1};
};

class RunnerDetector {
 public:
  explicit RunnerDetector();
  RunnerDetector(const RunnerDetector&) = delete;
  RunnerDetector& operator=(const RunnerDetector&) = delete;
  RunnerDetector(RunnerDetector&&) noexcept = default;
  RunnerDetector& operator=(RunnerDetector&&) noexcept = default;
  ~RunnerDetector() = default;

  std::vector<Runner> track(const cv::Mat& imageRGB);

 private:
  std::unique_ptr<YoloV8> model_;
  std::unique_ptr<byte_track::BYTETracker> tracker_;
};