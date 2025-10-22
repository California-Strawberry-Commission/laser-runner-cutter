#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "detection/detector/yolov8.hpp"

void drawObjectLabels(cv::Mat& image, const std::vector<Object>& objects,
                      unsigned int scale = 1) {
  cv::Scalar color{0.0, 0.0, 1.0};

  // Draw segmentation masks
  if (!objects.empty() && !objects[0].boxMask.empty()) {
    cv::Mat mask{image.clone()};
    for (const auto& object : objects) {
      mask(object.rect).setTo(color * 255, object.boxMask);
    }
    // Add all the masks to our image
    cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
  }

  // Bounding boxes and annotations
  for (auto& object : objects) {
    double meanColor{cv::mean(color)[0]};
    cv::Scalar textColor{(meanColor > 0.5) ? cv::Scalar(0, 0, 0)
                                           : cv::Scalar(255, 255, 255)};

    // Draw rectangles and text
    char text[256];
    sprintf(text, "%.1f%%", object.conf * 100);
    int baseLine{0};
    cv::Size labelSize{cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5 * scale, scale, &baseLine)};
    cv::Scalar textBackgroundColor{color * 0.7 * 255};
    cv::rectangle(image, object.rect, color * 255, scale + 1);
    int x{static_cast<int>(std::round(object.rect.x))};
    int y{static_cast<int>(std::round(object.rect.y)) + 1};
    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(labelSize.width, labelSize.height + baseLine)),
        textBackgroundColor, -1);
    cv::putText(image, text, cv::Point(x, y + labelSize.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, textColor, scale);
  }
}

int main(int argc, char* argv[]) {
  CLI::App app{"Test YoloV8"};

  std::string imageFile;
  std::string trtEngineFile;
  std::string outputDir;
  app.add_option("-i,--image_file", imageFile, "Input image file")->required();
  app.add_option("-t,--trt_engine", trtEngineFile, "TensorRT Engine file")
      ->required();
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

  YoloV8 yolo{trtEngineFile};

  // Warmup
  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < 10; ++i) {
    yolo.predict(rgbImage);
  }

  // Benchmarking
  std::cout << "Benchmarking..." << std::endl;
  double totalTimeMs{0.0};
  int numIterations{20};
  for (int i = 0; i < numIterations; ++i) {
    auto start{std::chrono::high_resolution_clock::now()};
    yolo.predict(rgbImage);
    auto end{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Prediction took " << duration.count() << " ms" << std::endl;
    totalTimeMs += duration.count();
  }
  std::cout << "Average prediction time: " << (totalTimeMs / numIterations)
            << " ms" << std::endl;

  // Draw labels
  auto objects{yolo.predict(rgbImage)};
  drawObjectLabels(img, objects);

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