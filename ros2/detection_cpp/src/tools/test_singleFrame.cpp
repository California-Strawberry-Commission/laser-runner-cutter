#include "detection_cpp/tools/test_singleFrame.hpp"

class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    // suppress info-level messages
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} logger;

int main(int argc, char const *argv[]) {
  spdlog::info("Working");

  if (argc > 0) {
    std::stringstream ss;
    for (int i = 0; i < argc; ++i) {
      ss << argv[i];
      if (i < argc - 1) ss << " ";
    }
    spdlog::info("argv: {}", ss.str());
  }

  spdlog::info("TensorRT Version: {}.{}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);

  // Fix the missing slash in the path
  std::string inputDir = std::string(getenv("HOME")) + "/Documents/testing/Images/runnerExamples";
  std::string enginefile = std::string(getenv("HOME")) + "/Documents/laser-runner-cutter/ros2/camera_control/models/RunnerSegYoloV8l.engine.new.two";

  // Check if file exists first
  if (!std::filesystem::exists(enginefile)) {
    spdlog::error("Engine file does not exist: {}", enginefile);
    return -1;
  }

  std::ifstream file(enginefile, std::ios::binary);
  if (!file.is_open()) {
    spdlog::error("Failed to open engine file: {}", enginefile);
    return -1;
  }

  file.seekg(0, file.end);
  std::size_t fsize = file.tellg();

  spdlog::info("Engine file size: {} bytes", static_cast<size_t>(fsize));

  file.seekg(0, file.beg);
  std::vector<char> engineData(fsize);
  file.read(engineData.data(), fsize);
  file.close();

  nvinfer1::IRuntime *mRuntime{nvinfer1::createInferRuntime(logger)};
  if (!mRuntime) {
    spdlog::error("Failed to create TensorRT runtime.");
    return -1;
  } else {
    spdlog::info("TensorRT runtime created successfully.");
  }

  int numGPUs;
  cudaGetDeviceCount(&numGPUs);
  auto ret = cudaSetDevice(0);
  if (ret != 0) {
    spdlog::error("Unable to set GPU device index to: " + std::to_string(0) + ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).");
    return -1;
  } else {
    int device;
    cudaGetDevice(&device);
    spdlog::info("CUDA device set to: {}. Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).", device);
  }

  nvinfer1::ICudaEngine *mEngine = mRuntime->deserializeCudaEngine(engineData.data(), fsize);
  if (!mEngine) {
    spdlog::error("Failed to deserialize CUDA engine.");
    return -1;
  } else {
    spdlog::info("TensorRT engine created successfully.");
  }
  

  char const *input_name = "images";
  assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);

  // Get input tensor dimensions
  nvinfer1::Dims input_dims = mEngine->getTensorShape(input_name);
  std::stringstream dims_ss;
  dims_ss << "Input tensor dimensions: ";
  for (int i = 0; i < input_dims.nbDims; ++i) {
    dims_ss << input_dims.d[i] << " ";
  }
  spdlog::info(dims_ss.str());

  // Check if the input tensor has 3 channels (e.g., for RGB images)
  if (input_dims.nbDims >= 3) {
    int channels = input_dims.d[input_dims.nbDims - 3];
    int height = input_dims.d[input_dims.nbDims - 2];
    int width = input_dims.d[input_dims.nbDims - 1];
    spdlog::info("Channels: {}, Height: {}, Width: {}", channels, height,
                 width);
  }

  spdlog::info("Done.");
  return 0;
}
