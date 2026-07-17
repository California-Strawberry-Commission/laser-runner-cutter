#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "detection/optflow/sparse_optical_flow.hpp"

int main(int argc, char* argv[]) {
  CLI::App app{"Test SparseOpticalFlow"};

  std::string image1File;
  std::string image2File;
  int includeX{0}, includeY{0}, includeW{0}, includeH{0};
  app.add_option("-1,--image1", image1File, "First frame (prev)")->required();
  app.add_option("-2,--image2", image2File, "Second frame (curr)")->required();
  app.add_option("--include-x", includeX,
                 "X of region to restrict tracking to (default: none)");
  app.add_option("--include-y", includeY,
                 "Y of region to restrict tracking to (default: none)");
  app.add_option("--include-w", includeW,
                 "Width of region to restrict tracking to (default: none)");
  app.add_option("--include-h", includeH,
                 "Height of region to restrict tracking to (default: none)");

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

  SparseOpticalFlow opticalFlow{
      100, 3, cv::Rect{includeX, includeY, includeW, includeH}};

  // Warmup
  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < 10; ++i) {
    opticalFlow.computeFlow(prevFrame, currFrame);
  }

  // Benchmarking
  std::cout << "Benchmarking..." << std::endl;
  double totalTimeMs{0.0};
  int numIterations{20};
  cv::Point2f medianFlow;
  for (int i = 0; i < numIterations; ++i) {
    auto start{std::chrono::high_resolution_clock::now()};
    medianFlow = opticalFlow.computeFlow(prevFrame, currFrame);
    auto end{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, std::milli> duration = end - start;
    totalTimeMs += duration.count();
  }
  std::cout << "Average computeFlow time: " << (totalTimeMs / numIterations)
            << " ms" << std::endl;

  std::cout << "Median displacement: dx=" << medianFlow.x
            << " px, dy=" << medianFlow.y << " px" << std::endl;

  return 0;
}
