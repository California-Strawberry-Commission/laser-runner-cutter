#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "detection/detector/runner_detector.hpp"

int main(int argc, char* argv[]) {
  CLI::App app{"Test runner detector"};

  std::string imageFile;
  std::string outputDir;
  app.add_option("-i,--image_file", imageFile, "Input image file")->required();
  app.add_option("-o,--output_dir", outputDir,
                 "Path to the directory to write the annotated image to")
      ->required();

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

  RunnerDetector runnerDetector;

  int numIterations{20};
  for (int i = 0; i < numIterations; ++i) {
    runnerDetector.track(rgbImage);
  }

  auto runners{runnerDetector.track(rgbImage)};

  // Draw labels
  RunnerDetector::drawDetections(img, runners, cv::Size(img.cols, img.rows));

  // Save the image to disk
  std::filesystem::path imagePath{imageFile};
  std::filesystem::path filename{imagePath.stem()};
  std::filesystem::path outputFilename{filename.string() + "_annotated.jpg"};
  std::filesystem::path outputPath{std::filesystem::path(outputDir) /
                                   outputFilename};
  cv::imwrite(outputPath.string(), img);
  std::cout << "Saved annotated image to: " << outputPath.string() << std::endl;

  return 0;
}