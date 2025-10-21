#pragma once

#include <ByteTrack/BYTETracker.h>

#include <opencv2/opencv.hpp>

#include "detection/detector/yolov8.hpp"

struct Runner {
  // The detection's confidence probability
  float conf{};
  // The detection's track ID
  float trackId{-1};
  // The object bounding box rectangle (TLWH)
  cv::Rect2f rect;
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