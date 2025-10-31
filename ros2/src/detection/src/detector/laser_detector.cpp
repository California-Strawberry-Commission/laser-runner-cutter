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
  // Note: this is an extremely simplistic approach and will only work well when
  // the camera exposure is as low as possible
  cv::Mat gray;
  cv::cvtColor(imageRgb, gray, cv::COLOR_RGB2GRAY);

  cv::Mat thresh;
  cv::threshold(gray, thresh, 191, 255, cv::THRESH_BINARY);

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  std::vector<Laser> detections;
  for (const auto& contour : contours) {
    cv::Moments m{cv::moments(contour)};
    if (m.m00 != 0.0) {
      int cx{static_cast<int>(m.m10 / m.m00)};
      int cy{static_cast<int>(m.m01 / m.m00)};

      Laser laser;
      laser.point = cv::Point{cx, cy};
      laser.conf = cv::contourArea(contour);
      detections.push_back(laser);
    }
  }

  return detections;
}