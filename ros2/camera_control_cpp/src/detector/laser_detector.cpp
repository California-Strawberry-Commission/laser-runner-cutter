#include "camera_control_cpp/detector/laser_detector.hpp"

LaserDetector::LaserDetector() {}
LaserDetector::~LaserDetector() {}

std::vector<std::tuple<int, int>> LaserDetector::detect(
    const cv::Mat& colorFrame, float confThreshold) {
  std::vector<std::tuple<int, int>> laserPoints;
  return laserPoints;
}
