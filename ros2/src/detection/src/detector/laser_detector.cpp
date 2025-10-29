#include "detection/detector/laser_detector.hpp"

void LaserDetector::drawDetections(
    cv::Mat& targetImage, const std::vector<LaserDetector::Laser>& lasers,
    const cv::Size& originalImageSize, cv::Scalar color) {
  // The image we are drawing on may not be the size of the image that the
  // runner detection was run on. Thus, we'll need to scale the bbox, mask, and
  // point of each runner to the image we are drawing on.
  double xScale{static_cast<double>(targetImage.cols) /
                originalImageSize.width};
  double yScale{static_cast<double>(targetImage.rows) /
                originalImageSize.height};

  for (const auto& laser : lasers) {
    if (laser.point.x >= 0 && laser.point.y >= 0) {
      int x{static_cast<int>(std::round(laser.point.x * xScale))};
      int y{static_cast<int>(std::round(laser.point.y * yScale))};
      cv::drawMarker(targetImage, cv::Point2i(x, y), color,
                     cv::MARKER_TILTED_CROSS, 20, 2);
    }
  }
}

std::vector<LaserDetector::Laser> LaserDetector::detect(
    const cv::Mat& imageRgb) {
  cv::Mat gray;
  cv::cvtColor(imageRgb, gray, cv::COLOR_RGB2GRAY);

  cv::Mat thresh;
  cv::threshold(gray, thresh, 191, 255, cv::THRESH_BINARY);

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Sort contours by area (descending)
  std::sort(
      contours.begin(), contours.end(),
      [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) > cv::contourArea(b);
      });

  std::vector<Laser> detections;

  // Assume only one detection at most for now
  if (!contours.empty()) {
    cv::Moments M{cv::moments(contours[0])};
    if (M.m00 != 0.0) {
      int cx{static_cast<int>(M.m10 / M.m00)};
      int cy{static_cast<int>(M.m01 / M.m00)};

      Laser laser;
      laser.point = cv::Point{cx, cy};
      laser.conf = 1.0f;
      detections.push_back(laser);
    }
  }

  return detections;
}