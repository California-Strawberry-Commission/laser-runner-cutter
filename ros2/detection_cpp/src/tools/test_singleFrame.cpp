#include "detection_cpp/tools/test_singleFrame.hpp"

int main(int argc, char const *argv[]) {
  /*====================================*/
  #pragma region Setup
  /*====================================*/
  

  spdlog::info("Working");
  if (argc > 0) {
    std::stringstream ss;
    for (int i = 0; i < argc; ++i) {
      ss << argv[i];
      if (i < argc - 1) ss << " ";
    }
    spdlog::info("argv: {}", ss.str());
  }

  spdlog::info("OpenCV Version: {}.{}.{}", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
  spdlog::info("TensorRT Version: {}.{}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);

  std::string inputDir = std::string(getenv("HOME")) + "/Documents/testing/Images/runnerExamples";
  std::string enginefile = std::string(getenv("HOME")) + "/Documents/laser-runner-cutter/ros2/camera_control/models/RunnerSegYoloV8l.engine.pygen";

  int numGPUs;
  cudaGetDeviceCount(&numGPUs);
  int deviceNum = 0;
  auto ret = cudaSetDevice(deviceNum);
  if (ret != 0) {
    spdlog::error("Unable to set GPU device index to: {}. Note, your device has {} CUDA-capable GPU(s).", deviceNum, numGPUs);
    return -1;
  } else {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceNum);
    spdlog::info("CUDA device set to: {}. Note, your device has {} CUDA-capable GPU(s).", deviceProp.name, numGPUs);
  }


  /*====================================*/
  #pragma endregion Setup
  #pragma region File Loading
  /*====================================*/


  std::vector<cv::cuda::GpuMat> images;
  for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        cv::Mat img{cv::imread(entry.path().string())};
        if (!img.empty()) {
          cv::cuda::GpuMat gpuImg;
          gpuImg.upload(img);
          images.push_back(gpuImg);
          spdlog::info("Loaded RGB image: {} || size = {} x {}, channels = {}", entry.path().filename().string(), img.cols, img.rows, img.channels());
        }
      }
    }
  }

  if (images.size() < 1) {
    spdlog::error("Not enough images loaded from: {}", inputDir);
    return -1;
  } else {
    spdlog::info("Total images loaded: {}", images.size());
  }

    // Check if engine file exists
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

  /*====================================*/
  #pragma endregion File Loading
  #pragma region Image Preprocess
  /*====================================*/


  // OpenCV loads images as HWC (height, width, channels)
  spdlog::info("Original image shape (HWC): ({}, {}, {})", images[0].rows, images[0].cols, images[0].channels());

  // Resize images for smaller tensor
  cv::Size resultSize(1024, 768);
  std::vector<cv::cuda::GpuMat> resizedImages(images.size());
  for (size_t i = 0; i < images.size(); i++) {
    cv::cuda::resize(images[i], resizedImages[i], resultSize, 0.0, 0.0, cv::INTER_LINEAR);
  }
  spdlog::info("Resized image shape (HWC): ({}, {}, {})", resizedImages[0].rows, resizedImages[0].cols, resizedImages[0].channels());

  //Convert HWC to NCHW
  std::vector<cv::cuda::GpuMat> nchwImages;
  for (const cv::cuda::GpuMat& img : resizedImages) {
    // Split HWC to 3 single-channel HW on GPU
    std::vector<cv::cuda::GpuMat> channels(3);
    cv::cuda::split(img, channels);
    // Concatenate channels along rows (simulate vconcat on GPU)
    cv::cuda::GpuMat chw(channels[0].rows * 3, channels[0].cols, channels[0].type());
    for (int c = 0; c < 3; c++) {
      cv::cuda::GpuMat roi(chw, cv::Rect(0, c * channels[0].rows, channels[0].cols, channels[0].rows));
      channels[c].copyTo(roi);
    }
    // Create a GpuMat with NCHW shape (1, 3, 768, 1024)
    // Note: OpenCV GpuMat does not support 4D shapes directly, so we keep as (3*768, 1024)
    nchwImages.push_back(chw);
  }
  spdlog::info("New NCHW shape: ({}, {}, {}, {})", 1, 3, nchwImages[0].rows / 3, nchwImages[0].cols);

  spdlog::info("Image matrix type: {}", cv::typeToString(nchwImages[0].type()));
  for (cv::cuda::GpuMat& img : nchwImages) {
    img.convertTo(img, CV_32FC1, 1.0 / 255.0);
  }
  spdlog::info("Matrix types converted to : {}", cv::typeToString(nchwImages[0].type()));


  /*====================================*/
  #pragma endregion Image Preprocess
  #pragma region Engine Creation
  /*====================================*/


  nvinfer1::IRuntime *mRuntime{nvinfer1::createInferRuntime(logger)};
  if (!mRuntime) {
    spdlog::error("Failed to create TensorRT runtime.");
    return -1;
  } else {
    spdlog::info("TensorRT runtime created successfully.");
  }

  nvinfer1::ICudaEngine *mEngine = mRuntime->deserializeCudaEngine(engineData.data(), fsize);
  if (!mEngine) {
    spdlog::error("Failed to deserialize CUDA engine.");
    return -1;
  } else {
    spdlog::info("TensorRT engine created successfully.");
  }

  nvinfer1::IExecutionContext *mContext = mEngine->createExecutionContext();
  if (!mContext) {
    spdlog::error("Failed to create inference context");
    return -1;
  } else {
    spdlog::info("TensorRT context created successfully.");
  }


  /*====================================*/
  #pragma endregion Engine Creation
  #pragma region Inference
  /*====================================*/


  char const *input_name = "images";

  // Get input tensor dimensions
  nvinfer1::Dims input_dims = mEngine->getTensorShape(input_name);
  std::stringstream dims_ss;
  dims_ss << "Input tensor dimensions: ";
  for (int i = 0; i < input_dims.nbDims; ++i) {
    dims_ss << input_dims.d[i] << " ";
  }
  spdlog::info(dims_ss.str());

  if (nchwImages[0].channels() != input_dims.d[0] || 
      3 != input_dims.d[1] || 
      nchwImages[0].rows / 3 != input_dims.d[2] || 
      nchwImages[0].cols != input_dims.d[3]) {
    spdlog::error("Image shape does NOT match model input shape. You may need to preprocess/reshape.");
    return -1;
  } else {
    spdlog::info("Image and model input shapes match.");
  }

  assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
  if (nchwImages[0].type() != CV_32FC1) {
    spdlog::error("NCHW image type is not CV_32FC1 (float). Actual type: {}", cv::typeToString(nchwImages[0].type()));
    return -1;
  } else {
    spdlog::info("NCHW image type valid: {}", cv::typeToString(nchwImages[0].type()));
  }




  /*====================================*/
  #pragma endregion Inference
  /*====================================*/


  spdlog::info("Done.");
  return 0;
}
