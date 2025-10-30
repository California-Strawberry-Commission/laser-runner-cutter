#include "detection/detector/circle_detector.hpp"

void CircleDetector::drawDetections(
    cv::Mat& targetImage, const std::vector<CircleDetector::Circle>& circles,
    const cv::Size& originalImageSize, cv::Scalar color) {
  // The image we are drawing on may not be the size of the image that the
  // runner detection was run on. Thus, we'll need to scale the bbox, mask, and
  // point of each runner to the image we are drawing on.
  double xScale{static_cast<double>(targetImage.cols) /
                originalImageSize.width};
  double yScale{static_cast<double>(targetImage.rows) /
                originalImageSize.height};

  for (const auto& circle : circles) {
    if (circle.point.x >= 0 && circle.point.y >= 0) {
      int x{static_cast<int>(std::round(circle.point.x * xScale))};
      int y{static_cast<int>(std::round(circle.point.y * yScale))};
      cv::drawMarker(targetImage, cv::Point2i(x, y), color,
                     cv::MARKER_TILTED_CROSS, 20, 2);
    }
  }
}

std::vector<CircleDetector::Circle> CircleDetector::detect(
    const cv::Mat& imageRgb) {
  cv::Mat gray;
  cv::cvtColor(imageRgb, gray, cv::COLOR_RGB2GRAY);

  // Apply Gaussian blur to smooth the image and reduce noise
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2);

  // Detect circles using HoughCircles
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT,
                   1.2,  // dp: inverse ratio of accumulator resolution
                   50,   // minDist: minimum distance between centers
                   100,  // param1: upper threshold for Canny edge detector
                   30,   // param2: accumulator threshold for circle detection
                   8,    // minRadius
                   50    // maxRadius
  );

  std::vector<Circle> detections;

  for (const auto& c : circles) {
    int x{cvRound(c[0])};
    int y{cvRound(c[1])};

    Circle circle;
    circle.point = cv::Point{x, y};
    circle.conf = 1.0f;
    detections.push_back(circle);
  }

  return detections;
}