#include "detection/detector/circle_detector.hpp"

std::vector<Circle> CircleDetector::detect(const cv::Mat& imageRgb) {
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