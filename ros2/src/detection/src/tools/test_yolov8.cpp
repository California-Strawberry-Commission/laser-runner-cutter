#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "detection/detector/yolov8.hpp"

int main(int argc, char *argv[]) {
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
  yolo.drawObjectLabels(img, objects);

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