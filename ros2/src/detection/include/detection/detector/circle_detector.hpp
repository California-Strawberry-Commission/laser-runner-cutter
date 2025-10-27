#pragma once

#include <opencv2/opencv.hpp>

struct Circle {
  // The detection's confidence probability
  float conf{0.0f};
  // The representative point of the detected object
  cv::Point point{-1, -1};
};

class CircleDetector {
 public:
  explicit CircleDetector();
  CircleDetector(const CircleDetector&) = delete;
  CircleDetector& operator=(const CircleDetector&) = delete;
  CircleDetector(CircleDetector&&) noexcept = default;
  CircleDetector& operator=(CircleDetector&&) noexcept = default;
  ~CircleDetector() = default;

  std::vector<Circle> detect(const cv::Mat& imageRgb);
};