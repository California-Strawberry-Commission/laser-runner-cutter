#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "detection/detector/runner_detector.hpp"

namespace {

cv::Rect parseBounds(const std::string& input) {
  std::stringstream ss(input);
  std::string token;
  std::vector<int> vals;
  while (std::getline(ss, token, ',')) {
    vals.push_back(std::stoi(token));
  }
  if (vals.size() != 4) {
    throw CLI::ValidationError(
        "--bounds must have 4 comma-separated values: x,y,w,h");
  }
  return cv::Rect(vals[0], vals[1], vals[2], vals[3]);
}

}  // namespace

int main(int argc, char* argv[]) {
  CLI::App app{"Test runner detector"};

  std::string imageFile;
  std::string outputDir;
  std::string boundsStr;
  app.add_option("-i,--image_file", imageFile, "Input image file")->required();
  app.add_option("-o,--output_dir", outputDir,
                 "Path to the directory to write the annotated image to")
      ->required();
  app.add_option("-b,--bounds", boundsStr,
                 "Bounds as 'x,y,w,h' (optional; defaults to image rect)");

  CLI11_PARSE(app, argc, argv);

  // Read the input image
  auto img{cv::imread(imageFile)};
  if (img.empty()) {
    std::cout << "Error: Unable to read image at path '" << imageFile << "'"
              << std::endl;
    return -1;
  }
  cv::Mat rgbImage;
  cv::cvtColor(img, rgbImage, cv::COLOR_BGR2RGB);

  // Parse bounds
  cv::Rect bounds(0, 0, rgbImage.cols, rgbImage.rows);  // default to image rect
  if (!boundsStr.empty()) {
    try {
      bounds = parseBounds(boundsStr);
    } catch (const CLI::ValidationError& e) {
      std::cerr << e.what() << std::endl;
      return -1;
    }
  }

  RunnerDetector runnerDetector;

  int numIterations{20};
  for (int i = 0; i < numIterations; ++i) {
    runnerDetector.track(rgbImage, bounds);
  }

  auto runners{runnerDetector.track(rgbImage, bounds)};

  // Draw labels
  RunnerDetector::drawDetections(rgbImage, runners,
                                 cv::Size(img.cols, img.rows));
  // Draw bounds
  cv::rectangle(rgbImage, bounds, cv::Scalar(0, 0, 255), 2);

  // Save the image to disk
  std::filesystem::path imagePath{imageFile};
  std::filesystem::path filename{imagePath.stem()};
  std::filesystem::path outputFilename{filename.string() + "_annotated.jpg"};
  std::filesystem::path outputPath{std::filesystem::path(outputDir) /
                                   outputFilename};
  cv::Mat outImage;
  cv::cvtColor(rgbImage, outImage, cv::COLOR_RGB2BGR);
  cv::imwrite(outputPath.string(), outImage);
  std::cout << "Saved annotated image to: " << outputPath.string() << std::endl;

  return 0;
}