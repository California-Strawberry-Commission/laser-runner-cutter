#pragma once

#include <opencv2/opencv.hpp>

class CircleDetector {
 public:
  struct Circle {
    // The detection's confidence probability
    float conf{0.0f};
    // The representative point of the detected object
    cv::Point point{-1, -1};
  };

  static void drawDetections(cv::Mat& targetImage,
                             const std::vector<Circle>& circles,
                             const cv::Size& originalImageSize,
                             cv::Scalar color = {255, 0, 255});

  explicit CircleDetector() = default;
  CircleDetector(const CircleDetector&) = delete;
  CircleDetector& operator=(const CircleDetector&) = delete;
  CircleDetector(CircleDetector&&) noexcept = default;
  CircleDetector& operator=(CircleDetector&&) noexcept = default;
  ~CircleDetector() = default;

  std::vector<Circle> detect(const cv::Mat& imageRgb);
};