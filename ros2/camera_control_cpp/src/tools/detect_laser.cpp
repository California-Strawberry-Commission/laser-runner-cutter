#include <opencv2/opencv.hpp>

#include "camera_control_cpp/detector/laser_detector.hpp"
#include "yolov8.h"

int main(int argc, char** argv) {
  // Ensure an image path is provided
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <path_to_trt_engine> <path_to_image>"
              << std::endl;
    return -1;
  }

  std::string onnxModelPath = argv[1];
  std::string trtEnginePath;
  std::string imagePath = argv[2];

  YoloV8Config config;
  config.nmsThreshold = 0.6;
  config.classNames = std::vector<std::string>{"runner"};
  // Config specific for yolov8l-seg model
  config.segChannels = 32;
  config.segH = 256;
  config.segW = 256;

  // Create the YoloV8 engine
  YoloV8 yoloV8(onnxModelPath, trtEnginePath, config);

  // Read the input image
  auto img = cv::imread(imagePath);
  if (img.empty()) {
    std::cout << "Error: Unable to read image at path '" << imagePath << "'"
              << std::endl;
    return -1;
  }

  // Run inference

  std::cout << "RUNNING INFERENCE" << std::endl;

  const auto objects = yoloV8.detectObjects(img);
  std::cout << "RAN INFERENCE" << std::endl;

  // Draw the bounding boxes on the image
  yoloV8.drawObjectLabels(img, objects);

  std::cout << "Detected " << objects.size() << " objects" << std::endl;

  // Save the image to disk
  const auto outputName =
      imagePath.substr(0, imagePath.find_last_of('.')) + "_annotated.jpg";
  cv::imwrite(outputName, img);
  std::cout << "Saved annotated image to: " << outputName << std::endl;

  return 0;
}