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
  
  std::string inputDir = std::string(getenv("HOME")) + "/Documents/testing/Images/runnerExamples";
  std::string enginefile = std::string(getenv("HOME")) + "/Documents/laser-runner-cutter/ros2/camera_control/models/RunnerSegYoloV8l.trtexec.nonhalf.engine";
  
  spdlog::info("OpenCV Version: {}.{}.{}", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
  spdlog::info("TensorRT Version: {}.{}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);

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
          cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);
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
  char const *output0_name = "output0";  // Bounding Boxes and Confidences
  char const *output1_name = "output1";  // Masks

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
      output0Tensors[0].rows != output0_dims.d[1] || 
      output0Tensors[0].cols != output0_dims.d[2]) {
    spdlog::error("output0Tensors shape does NOT match model output0 shape: ({}, {}, {})", output0Tensors[0].channels(), output0Tensors[0].rows, output0Tensors[0].cols);
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
      outputMasks[0].rows / 32 != output1_dims.d[2] || 
      outputMasks[0].cols != output1_dims.d[3]) {
    spdlog::error("outMasks shape does NOT match model output1 shape: ({}, {}, {}, {})", outputMasks[0].channels(), 32, outputMasks[0].rows / 32, outputMasks[0].cols);
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


  // output0: [1, 37, 16128] - detections
  // output1: [1, 32, 192, 256] - mask prototypes
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

    // Create bindings
    void* bindings[3];
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
  #pragma region Post-Process
  /*====================================*/


  // output0 shape: [batch=1, features=37, anchors=16128]
  // Each of the 16128 anchor points has 37 values:
  // For each anchor point (16128 total):
  // [0-3]:   Bounding box (x_center, y_center, width, height)
  // [4]:     Object confidence score
  // [5-36]:  32 mask coefficients (used to combine the 32 prototype masks)
  
  float confidence_threshold = 0.5;
  float nms_threshold = 0.45f;

  // Vectors to store all results across all images
  std::vector<std::vector<cv::Rect>> all_bounding_boxes;      // [image_idx][detection_idx]
  std::vector<std::vector<float>> all_confidences;           // [image_idx][detection_idx]
  std::vector<std::vector<cv::Mat>> all_masks;               // [image_idx][detection_idx]
  std::vector<int> detection_counts;                         // Number of detections per image


  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<std::vector<float>> mask_coefficients;

  std::vector<cv::Rect> final_boxes;
  std::vector<float> final_confidences;
  std::vector<cv::Mat> final_masks;
  
  for (size_t img_idx = 0; img_idx < output0Tensors.size(); img_idx++) {
    spdlog::info("Processing image {} results...", img_idx);
    const cv::Mat& output0 = output0Tensors[img_idx];
    const cv::Mat& prototypes = outputMasks[img_idx];

    boxes.clear();
    confidences.clear();
    mask_coefficients.clear();
    // Process each anchor point
    for (int i = 0; i < 16128; ++i) {
        // Extract confidence score (index 4)
        float confidence = output0.at<float>(4, i);
        
        if (confidence > confidence_threshold) {
            // Extract bounding box (indices 0-3)
            float x_center = output0.at<float>(0, i);
            float y_center = output0.at<float>(1, i);
            float width = output0.at<float>(2, i);
            float height = output0.at<float>(3, i);

            // Convert center format to corner format
            int x = static_cast<int>(x_center - width/2) * 2;   // Top-left x
            int y = static_cast<int>(y_center - height/2) * 2;  // Top-left y
            int w = static_cast<int>(width) * 2;
            int h = static_cast<int>(height) * 2;
            
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
    spdlog::info("Found {} potential detections before NMS", boxes.size());

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, nms_indices);
    spdlog::info("--Found {} runners after NMS", nms_indices.size());

    final_boxes.clear();
    final_confidences.clear();
    final_masks.clear();

    for (int idx : nms_indices) {
        final_boxes.push_back(boxes[idx]);
        final_confidences.push_back(confidences[idx]);
        
        // Generate mask for this detection
        const std::vector<float>& coeffs = mask_coefficients[idx];
        
        // Combine prototype masks using coefficients
        cv::Mat final_mask = cv::Mat::zeros(192, 256, CV_32F);
        for (int c = 0; c < 32; ++c) {
            cv::Mat prototype = prototypes.rowRange(c * 192, (c + 1) * 192).clone();
            prototype = prototype.reshape(1, 192);
            final_mask += coeffs[c] * prototype;
        }
        
        // Apply sigmoid and threshold
        cv::Mat sigmoid_mask;
        cv::exp(-final_mask, sigmoid_mask);
        sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
        
        // Resize to input image size and binarize
        cv::Mat resized_mask, binary_mask;
        cv::resize(sigmoid_mask, resized_mask, cv::Size(1024, 768));
        cv::threshold(resized_mask, binary_mask, 0.5, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8UC1);
        
        final_masks.push_back(binary_mask);
    }

    // Store results for this image
    all_bounding_boxes.push_back(final_boxes);
    all_confidences.push_back(final_confidences);
    all_masks.push_back(final_masks);
    detection_counts.push_back(final_boxes.size());
    
    // spdlog::info("Image {}: Found {} runners", img_idx, final_boxes.size());
  }

  // Summary results
  spdlog::info("=== DETECTION SUMMARY ===");
  for (size_t img_idx = 0; img_idx < detection_counts.size(); ++img_idx) {
      spdlog::info("Image {}: {} detections", img_idx, detection_counts[img_idx]);
      
      // Optionally log individual detection details
      for (size_t det_idx = 0; det_idx < all_bounding_boxes[img_idx].size(); ++det_idx) {
          const cv::Rect& box = all_bounding_boxes[img_idx][det_idx];
          float conf = all_confidences[img_idx][det_idx];
          spdlog::info("  Runner {}: Box=({},{},{},{}), Confidence={:.3f}", 
                      det_idx, box.x, box.y, box.width, box.height, conf);
      }
  }


  /*====================================*/
  #pragma endregion Post-Process
  #pragma region Observe Results
  /*====================================*/
  
  
  bool sendResults = true;

  std::string overlays_dir = "/tmp/overlays/";
  std::string laptop_address = "paul@10.42.0.1:/home/paul/Desktop/testing/";
  
  std::vector<cv::Mat> images_with_boxes;
  for (size_t img_idx = 0; img_idx < all_bounding_boxes.size(); ++img_idx) {
    // Download original image from GPU to CPU
    cv::Mat img_cpu;
    images[img_idx].download(img_cpu);
    cv::cvtColor(img_cpu, img_cpu, cv::COLOR_RGB2BGR);

    // Draw boxes
    for (const auto& box : all_bounding_boxes[img_idx]) {
      cv::rectangle(img_cpu, box, cv::Scalar(0, 255, 0), 2);
      // Draw confidence label
      std::string label = cv::format("%.2f", all_confidences[img_idx][&box - &all_bounding_boxes[img_idx][0]]);
      int baseLine = 0;
      cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      int top = std::max(box.y, labelSize.height);
      cv::rectangle(img_cpu, cv::Point(box.x, top - labelSize.height),
            cv::Point(box.x + labelSize.width, top + baseLine),
            cv::Scalar(0, 255, 0), cv::FILLED);
      cv::putText(img_cpu, label, cv::Point(box.x, top),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    images_with_boxes.push_back(img_cpu);
  }

  if (sendResults) {
    std::filesystem::create_directories(overlays_dir);
    for (size_t img_idx = 0; img_idx < images_with_boxes.size(); ++img_idx) {
      std::string img_filename = overlays_dir + "img_with_boxes_" + std::to_string(img_idx) + ".png";
      cv::imwrite(img_filename, images_with_boxes[img_idx]);
      spdlog::info("Wrote image with boxes: {}", img_filename);
    }
    system(("rsync -avz --progress " + overlays_dir + " " + laptop_address).c_str());
  }


  /*====================================*/
  #pragma endregion Observe Results
  #pragma region Cleanup
  /*====================================*/


  // Cleanup tmp files
  if (sendResults) {
    std::filesystem::remove_all(overlays_dir);
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
