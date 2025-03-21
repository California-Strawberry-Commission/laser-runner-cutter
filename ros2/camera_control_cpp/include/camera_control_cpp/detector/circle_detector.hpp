#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

class CircleDetector {
 public:
  std::vector<std::tuple<int, int>> detect(const cv::Mat& colorFrame);
};