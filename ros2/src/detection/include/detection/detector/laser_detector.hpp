#pragma once

#include <opencv2/opencv.hpp>

struct Laser {
  // The detection's confidence probability
  float conf{0.0f};
  // The representative point of the detected object
  cv::Point point{-1, -1};
};

class LaserDetector {
 public:
  explicit LaserDetector();
  LaserDetector(const LaserDetector&) = delete;
  LaserDetector& operator=(const LaserDetector&) = delete;
  LaserDetector(LaserDetector&&) noexcept = default;
  LaserDetector& operator=(LaserDetector&&) noexcept = default;
  ~LaserDetector() = default;

  std::vector<Laser> detect(const cv::Mat& imageRgb);
};