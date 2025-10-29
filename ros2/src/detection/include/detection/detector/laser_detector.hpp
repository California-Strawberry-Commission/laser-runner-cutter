#pragma once

#include <opencv2/opencv.hpp>

class LaserDetector {
 public:
  struct Laser {
    // The detection's confidence probability
    float conf{0.0f};
    // The representative point of the detected object
    cv::Point point{-1, -1};
  };

  static void drawDetections(cv::Mat& targetImage,
                             const std::vector<Laser>& lasers,
                             const cv::Size& originalImageSize,
                             cv::Scalar color = {255, 0, 255});

  explicit LaserDetector() = default;
  LaserDetector(const LaserDetector&) = delete;
  LaserDetector& operator=(const LaserDetector&) = delete;
  LaserDetector(LaserDetector&&) noexcept = default;
  LaserDetector& operator=(LaserDetector&&) noexcept = default;
  ~LaserDetector() = default;

  std::vector<Laser> detect(const cv::Mat& imageRgb);
};