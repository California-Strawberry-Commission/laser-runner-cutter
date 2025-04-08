#include "camera_control_cpp/detector/circle_detector.hpp"

std::vector<std::tuple<int, int>> CircleDetector::detect(
    const cv::Mat& colorFrame) {
  // Convert to grayscale
  cv::Mat image;
  cv::cvtColor(colorFrame, image, cv::COLOR_BGR2GRAY);

  // Apply Gaussian blur to smooth the image and reduce noise
  cv::GaussianBlur(image, image, cv::Size(9, 9), 2);

  // Use HoughCircles to detect circles
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(
      image, circles, cv::HOUGH_GRADIENT,
      1.2,  // inverse ratio of resolution (1 means same resolution)
      50,   // minimum distance between detected centers
      100,  // higher threshold for Canny edge detector
      30,   // accumulator threshold for circle detection
      8,    // minimum circle radius
      50    // maximum circle radius
  );

  std::vector<std::tuple<int, int>> circleCenters;
  if (!circles.empty()) {
    for (const auto& circle : circles) {
      int x = cvRound(circle[0]);
      int y = cvRound(circle[1]);
      circleCenters.push_back(std::make_tuple(x, y));
    }
  }
  return circleCenters;
}
