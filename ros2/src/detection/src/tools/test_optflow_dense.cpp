#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "detection/optflow/dense_optical_flow.hpp"

int main(int argc, char* argv[]) {
  CLI::App app{"Test DenseOpticalFlow"};

  std::string image1File;
  std::string image2File;
  app.add_option("-1,--image1", image1File, "First frame (prev)")->required();
  app.add_option("-2,--image2", image2File, "Second frame (curr)")->required();

  CLI11_PARSE(app, argc, argv);

  cv::Mat prevFrame{cv::imread(image1File)};
  if (prevFrame.empty()) {
    std::cout << "Error: Unable to read image at path '" << image1File << "'"
              << std::endl;
    return -1;
  }

  cv::Mat currFrame{cv::imread(image2File)};
  if (currFrame.empty()) {
    std::cout << "Error: Unable to read image at path '" << image2File << "'"
              << std::endl;
    return -1;
  }

  DenseOpticalFlow opticalFlow{4, VPI_OPTICAL_FLOW_QUALITY_MEDIUM};

  // Warmup
  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < 10; ++i) {
    opticalFlow.computeMeanFlow(prevFrame, currFrame);
  }

  // Benchmarking
  std::cout << "Benchmarking..." << std::endl;
  double totalTimeMs{0.0};
  int numIterations{60};
  cv::Point2f meanFlow;
  for (int i = 0; i < numIterations; ++i) {
    auto start{std::chrono::high_resolution_clock::now()};
    meanFlow = opticalFlow.computeMeanFlow(prevFrame, currFrame);
    auto end{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, std::milli> duration = end - start;
    totalTimeMs += duration.count();
  }
  std::cout << "Average computeMeanFlow time: " << (totalTimeMs / numIterations)
            << " ms" << std::endl;

  std::cout << "Mean displacement: dx=" << meanFlow.x
            << " px, dy=" << meanFlow.y << " px" << std::endl;

  return 0;
}
