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

  // Assume output0 tensor shape: [1, 37, 16128] (float32)
  // For each image, allocate a single Mat with 37 channels, each of length 16128
  std::vector<cv::Mat> output0Tensors;
  for (size_t i = 0; i < nchwImages.size(); i++) {
    // Create a Mat of size 16128 x 37, type CV_32FC1
    output0Tensors.emplace_back(37, 16128, CV_32FC1);
  }
  spdlog::info("Output0 tensors allocated: {} matrices, each size = {} x {}, channels = {}", output0Tensors.size(), output0Tensors[0].rows, output0Tensors[0].cols, output0Tensors[0].channels());

  // Assume output tensor shape: [1, 32, 192, 256] (float32)
  // For each image, allocate a single Mat with 32 channels
  std::vector<cv::Mat> outputMasks;
  for (size_t i = 0; i < nchwImages.size(); i++) {
    // Create a Mat of size 192x256 with 32 channels, type CV_32FC1
    outputMasks.emplace_back(32 * 192, 256, CV_32FC1);
  }
  spdlog::info("Output masks allocated: {} matrices, each size = {} x {}, channels = {}", outputMasks.size(), outputMasks[0].rows / 32, outputMasks[0].cols, 32);


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
  #pragma region Inference Setup
  /*====================================*/


  char const *input_name = "images";
  char const *output0_name = "output0";  // TODO: CLARIFY WHAT THIS IS
  char const *output1_name = "output1";  // Looking for mask output

  // Get input tensor dimensions
  nvinfer1::Dims input_dims = mEngine->getTensorShape(input_name);
  std::stringstream in_dims_ss;
  in_dims_ss << "Input tensor dimensions: ";
  for (int i = 0; i < input_dims.nbDims; ++i) {
    in_dims_ss << input_dims.d[i] << " ";
  }
  spdlog::info(in_dims_ss.str());

  if (nchwImages[0].channels() != input_dims.d[0] || 
      3 != input_dims.d[1] || 
      nchwImages[0].rows / 3 != input_dims.d[2] || 
      nchwImages[0].cols != input_dims.d[3]) {
    spdlog::error("Image shape does NOT match model input shape: ({}, {}, {}, {})", nchwImages[0].channels(), 3, nchwImages[0].rows / 3, nchwImages[0].cols);
    return -1;
  } else {
    spdlog::info("Image and model input shapes match.");
  }

  // Get output0 tensor dimensions
  nvinfer1::Dims output0_dims = mEngine->getTensorShape(output0_name);
  std::stringstream out0_dims_ss;
  out0_dims_ss << "Output0 tensor dimensions: ";
  for (int i = 0; i < output0_dims.nbDims; ++i) {
    out0_dims_ss << output0_dims.d[i] << " ";
  }
  spdlog::info(out0_dims_ss.str());

  if (output0Tensors[0].channels() != output0_dims.d[0] || 
      output0Tensors[1].rows != output0_dims.d[1] || 
      output0Tensors[2].cols != output0_dims.d[2]) {
    spdlog::error("output0Tensors shape does NOT match model input shape: ({}, {}, {})", output0Tensors[0].channels(), output0Tensors[0].rows, output0Tensors[0].cols);
    return -1;
  } else {
    spdlog::info("Allocated output0Tensors and model output0 shapes match.");
  }


  // Get output1 tensor dimensions
  nvinfer1::Dims output1_dims = mEngine->getTensorShape(output1_name);
  std::stringstream out1_dims_ss;
  out1_dims_ss << "Output1 tensor dimensions: ";
  for (int i = 0; i < output1_dims.nbDims; ++i) {
    out1_dims_ss << output1_dims.d[i] << " ";
  }
  spdlog::info(out1_dims_ss.str());

  if (outputMasks[0].channels() != output1_dims.d[0] || 
      32 != output1_dims.d[1] || 
      outputMasks[2].rows / 32 != output1_dims.d[2] || 
      outputMasks[3].cols != output1_dims.d[3]) {
    spdlog::error("outMasks shape does NOT match model input shape: ({}, {}, {}, {})", outputMasks[0].channels(), 32, outputMasks[0].rows / 32, outputMasks[0].cols);
    return -1;
  } else {
    spdlog::info("Allocated outMasks and model output1 shapes match.");
  }

  assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
  if (nchwImages[0].type() != CV_32FC1) {
    spdlog::error("NCHW image type is not CV_32FC1 (float). Actual type: {}", cv::typeToString(nchwImages[0].type()));
    return -1;
  } else {
    spdlog::info("NCHW image type valid: {}", cv::typeToString(nchwImages[0].type()));
  }

  assert(mEngine->getTensorDataType(output0_name) == nvinfer1::DataType::kFLOAT);
  if (output0Tensors[0].type() != CV_32FC1) {
    spdlog::error("output0Tensors type is not CV_32FC1 (float). Actual type: {}", cv::typeToString(output0Tensors[0].type()));
    return -1;
  } else {
    spdlog::info("Allocated output0Tensors type valid: {}", cv::typeToString(output0Tensors[0].type()));
  }

  assert(mEngine->getTensorDataType(output1_name) == nvinfer1::DataType::kFLOAT);
  if (outputMasks[0].type() != CV_32FC1) {
    spdlog::error("outMasks type is not CV_32FC1 (float). Actual type: {}", cv::typeToString(outputMasks[0].type()));
    return -1;
  } else {
    spdlog::info("Allocated output masks type valid: {}", cv::typeToString(outputMasks[0].type()));
  }


  /*====================================*/
  #pragma endregion Inference Setup
  #pragma region Inferencing
  /*====================================*/


  // Allocate output tensor sizes
  size_t output0_size = output0_dims.d[0] * output0_dims.d[1] * output0_dims.d[2] * sizeof(float);
  size_t output1_size = output1_dims.d[0] * output1_dims.d[1] * output1_dims.d[2] * output1_dims.d[3] * sizeof(float);
  void *outputMem0, *outputMem1;

  int successCount = 0;
  for (size_t img_idx = 0; img_idx < nchwImages.size(); img_idx++) {
    spdlog::info("Processing image {} of {}", img_idx + 1, nchwImages.size());
    void* inputMem = nchwImages[img_idx].ptr<void>();
    cudaMalloc(&outputMem0, output0_size);
    cudaMemset(outputMem0, 0, output0_size);
    cudaMalloc(&outputMem1, output1_size);
    cudaMemset(outputMem1, 0, output1_size);

    // Set tensor addresses
    mContext->setTensorAddress(input_name, inputMem);
    mContext->setTensorAddress(output0_name, outputMem0);
    mContext->setTensorAddress(output1_name, outputMem1);

    // Create bindings
    void* bindings[2];
    bindings[0] = inputMem;   // Input binding
    bindings[1] = outputMem0;  // Output0 binding
    bindings[2] = outputMem1;  // Output1 binding

    // Run inference
    auto start = std::chrono::high_resolution_clock::now();
    bool success = mContext->executeV2(bindings);
    if (!success) {
        spdlog::error("TensorRT inference failed for image {}", img_idx + 1);
        cudaFree(outputMem0);
        cudaFree(outputMem1);
        continue;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info("Inference completed successfully for image {} in {} ms", img_idx + 1, duration);
    successCount++;

    // Copy results back to host
    cudaMemcpy(output0Tensors[img_idx].data, outputMem0, output0_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputMasks[img_idx].data, outputMem1, output1_size, cudaMemcpyDeviceToHost);

    // Cleanup output memory for this image
    cudaFree(outputMem0);
    cudaFree(outputMem1);
  }

  spdlog::info("Total successful inferences: {} out of {}", successCount, nchwImages.size());


  /*====================================*/
  #pragma endregion Inferencing
  #pragma region Display Results
  /*====================================*/

  std::string first_image_path, mask_path;

  bool testDisplay = false;
  if (testDisplay) {
    system("figlet IMAGE");
    // Display the first example image using chafa
    // Save the first image in the array to a temporary PNG file and display it with chafa
    first_image_path = "/tmp/first_image.png";
    cv::Mat first_img_host;
    images[0].download(first_img_host); // Download from GPU to host
    cv::imwrite(first_image_path, first_img_host);
    system(("chafa " + first_image_path).c_str());

    mask_path = "/tmp/output_mask.png";
    for (int chnl=0; chnl<32; chnl++) {
      try {
        system(("figlet MASK" + std::to_string(chnl)).c_str());
        // Save the first output mask to a temporary PNG file and display it with chafa
        cv::Mat mask;
        // Reshape outputMasks[0] to (32, 192, 256) and select the first channel (mask)
        mask = outputMasks[0].rowRange(chnl, 192).clone(); // Take first channel (assuming row-major)
        mask = mask.reshape(1, 192); // Reshape to 192x256
        cv::normalize(mask, mask, 0, 255, cv::NORM_MINMAX);
        mask.convertTo(mask, CV_8UC1);
        cv::imwrite(mask_path, mask);
        system(("chafa " + mask_path).c_str());
      } catch (const cv::Exception& e) {
        spdlog::error("OpenCV error displaying mask channel {}:\n{}", chnl, e.what());
      }
    }
  }




  /*====================================*/
  #pragma endregion Display Results
  #pragma region Post-Process
  /*====================================*/

  // output0: [1, 37, 16128] - detections
  // output1: [1, 32, 192, 256] - mask prototypes

  // output0 shape: [batch=1, features=37, anchors=16128]
  // Each of the 16128 anchor points has 37 values:

  // For each anchor point (16128 total):
  // [0-3]:   Bounding box (x_center, y_center, width, height)
  // [4]:     Object confidence score
  // [5-36]:  32 mask coefficients (used to combine the 32 prototype masks)

  
  float confidence_threshold = 0.1f;
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<std::vector<float>> mask_coefficients;
  
  for (const cv::Mat& output0 : output0Tensors) {
    boxes.clear();
    confidences.clear();
    mask_coefficients.clear();
    // Process each anchor point
    for (int i = 0; i < 16128; ++i) {
        // Extract confidence score (index 4)
        float confidence = output0.at<float>(4, i);
        
        if (confidence > confidence_threshold) {
            // Extract bounding box (indices 0-3)
            float x = output0.at<float>(0, i);
            float y = output0.at<float>(1, i);
            float w = output0.at<float>(2, i);
            float h = output0.at<float>(3, i);
            
            boxes.push_back(cv::Rect(x, y, w, h));
            confidences.push_back(confidence);
            
            // Extract mask coefficients (indices 5-36)
            std::vector<float> coeffs;
            for (int j = 5; j < 37; ++j) {
                coeffs.push_back(output0.at<float>(j, i));
            }
            mask_coefficients.push_back(coeffs);
        }
    }
    
    spdlog::info("Found {} runners", boxes.size());
    for (size_t i=0; i<boxes.size(); i++) {
        spdlog::info("Runner {}: Box=({},{},{},{}), Confidence={:.3f}", 
                      i, boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height, confidences[i]);
    }
  }

  /*====================================*/
  #pragma endregion Post-Process
  #pragma region Cleanup
  /*====================================*/


  // Cleanup tmp files
  if (testDisplay) {
    std::remove(first_image_path.c_str());
    std::remove(mask_path.c_str());
  }

  // Cleanup TensorRT objects
  delete mContext;
  delete mEngine;
  delete mRuntime;


  /*====================================*/
  #pragma endregion Cleanup
  /*====================================*/


  spdlog::info("Done.");
  return 0;
}
