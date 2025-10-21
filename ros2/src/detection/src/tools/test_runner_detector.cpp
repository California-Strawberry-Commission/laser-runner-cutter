#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "detection/detector/runner_detector.hpp"

void drawDetections(cv::Mat& image, const std::vector<Runner>& runners,
                    unsigned int scale = 1) {
  cv::Scalar color{0.0, 0.0, 1.0};

  // Draw segmentation masks
  if (!runners.empty() && !runners[0].boxMask.empty()) {
    cv::Mat mask{image.clone()};
    for (const auto& object : runners) {
      mask(object.rect).setTo(color * 255, object.boxMask);
    }
    // Add all the masks to our image
    cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
  }

  // Bounding boxes and annotations
  for (auto& runner : runners) {
    double meanColor{cv::mean(color)[0]};
    cv::Scalar textColor{(meanColor > 0.5) ? cv::Scalar(0, 0, 0)
                                           : cv::Scalar(255, 255, 255)};

    // Draw rectangles and text
    char text[256];
    sprintf(text, "%d: %.1f%%", runner.trackId, runner.conf * 100);
    int baseLine{0};
    cv::Size labelSize{cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5 * scale, scale, &baseLine)};
    cv::Scalar textBackgroundColor{color * 0.7 * 255};
    cv::rectangle(image, runner.rect, color * 255, scale + 1);
    int x{static_cast<int>(std::round(runner.rect.x))};
    int y{static_cast<int>(std::round(runner.rect.y)) + 1};
    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(labelSize.width, labelSize.height + baseLine)),
        textBackgroundColor, -1);
    cv::putText(image, text, cv::Point(x, y + labelSize.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, textColor, scale);

    // Draw representative point
    if (runner.point.x >= 0 && runner.point.y >= 0) {
      cv::Scalar markerColor{1.0, 1.0, 1.0};
      cv::drawMarker(image, runner.point, markerColor * 255,
                     cv::MARKER_TILTED_CROSS, 20, 2);
    }
  }
}

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
  drawDetections(img, runners);

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