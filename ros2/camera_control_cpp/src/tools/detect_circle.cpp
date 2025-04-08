#include <opencv2/opencv.hpp>

#include "camera_control_cpp/detector/circle_detector.hpp"

int main(int argc, char** argv) {
  // Ensure an image path is provided
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
    return -1;
  }

  std::string imagePath = argv[1];
  cv::Mat colorFrame = cv::imread(imagePath);
  if (colorFrame.empty()) {
    std::cerr << "Could not open or find the image: " << imagePath << std::endl;
    return -1;
  }

  CircleDetector detector;
  auto circleCenters = detector.detect(colorFrame);

  for (const auto& center : circleCenters) {
    int x, y;
    std::tie(x, y) = center;
    cv::circle(colorFrame, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
    std::cout << "Detected circle at (" << x << ", " << y << ")\n";
  }

  cv::imshow("Detected Circles", colorFrame);
  cv::waitKey(0);

  return 0;
}