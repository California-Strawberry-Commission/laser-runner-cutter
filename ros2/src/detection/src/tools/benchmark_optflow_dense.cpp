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

  cv::cuda::GpuMat gpuPrevFrame;
  cv::cuda::GpuMat gpuCurrFrame;
  gpuPrevFrame.upload(prevFrame);
  gpuCurrFrame.upload(currFrame);

  DenseOpticalFlow opticalFlow{};

  // Warmup
  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < 10; ++i) {
    opticalFlow.computeFlow(gpuPrevFrame, gpuCurrFrame);
  }

  // Benchmarking
  std::cout << "Benchmarking..." << std::endl;
  double totalTimeMs{0.0};
  int numIterations{20};
  std::optional<cv::Point2f> medianFlow;
  for (int i = 0; i < numIterations; ++i) {
    auto start{std::chrono::high_resolution_clock::now()};
    medianFlow = opticalFlow.computeFlow(gpuPrevFrame, gpuCurrFrame);
    auto end{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, std::milli> duration = end - start;
    totalTimeMs += duration.count();
  }
  std::cout << "Average computeFlow time: " << (totalTimeMs / numIterations)
            << " ms" << std::endl;

  if (medianFlow) {
    std::cout << "Median displacement: dx=" << medianFlow->x
              << " px, dy=" << medianFlow->y << " px" << std::endl;
  } else {
    std::cout << "Median displacement: unavailable (empty motion vector grid)"
              << std::endl;
  }

  return 0;
}
