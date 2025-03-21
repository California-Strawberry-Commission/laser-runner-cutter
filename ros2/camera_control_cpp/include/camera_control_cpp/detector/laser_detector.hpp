#pragma once

#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

class LaserDetector {
 public:
  LaserDetector();
  ~LaserDetector();
  std::vector<std::tuple<int, int>> detect(const cv::Mat& colorFrame,
                                           float confThreshold = 0.0f);
};