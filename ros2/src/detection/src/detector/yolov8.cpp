#include "detection/detector/yolov8.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "detection/detector/stopwatch.hpp"

void Logger::log(Severity severity, const char* msg) noexcept {
  switch (severity) {
    case Severity::kVERBOSE:
      spdlog::debug(msg);
      break;
    case Severity::kINFO:
      spdlog::info(msg);
      break;
    case Severity::kWARNING:
      spdlog::warn(msg);
      break;
    case Severity::kERROR:
      spdlog::error(msg);
      break;
    case Severity::kINTERNAL_ERROR:
      spdlog::critical(msg);
      break;
    default:
      spdlog::info("Unexpected severity level");
  }
}

inline bool fileExists(const std::string& filepath) {
  std::ifstream f(filepath.c_str());
  return f.good();
}

inline void checkCudaErrorCode(cudaError_t code) {
  if (code != cudaSuccess) {
    throw std::runtime_error(
        "CUDA operation failed with code: " + std::to_string(code) + " (" +
        cudaGetErrorName(code) +
        "), with message: " + cudaGetErrorString(code));
  }
}

size_t getDataTypeSize(nvinfer1::DataType dataType) {
  switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kBOOL:
      return 1;
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT64:
      return 8;
    default:
      throw std::runtime_error("Unhandled TensorRT DataType enum value: " +
                               std::to_string(int(dataType)));
  }
}

YoloV8::YoloV8(const std::string& trtEngineFile) {
  // TODO: Currently only handles segmentation models

  // Read the serialized model from disk
  if (!fileExists(trtEngineFile)) {
    throw std::runtime_error("Unable to read TensorRT engine file: " +
                             trtEngineFile);
  }

  spdlog::info("Loading TensorRT engine file at path: {}", trtEngineFile);

  std::ifstream file(trtEngineFile, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engineData(size);
  if (!file.read(engineData.data(), size)) {
    throw std::runtime_error("Unable to read engine file");
  }
  file.close();

  // Create a runtime to deserialize the engine file
  nvRuntime_ = std::unique_ptr<nvinfer1::IRuntime>{
      nvinfer1::createInferRuntime(logger_)};
  // Create an engine, a representation of the optimized model
  nvEngine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
      nvRuntime_->deserializeCudaEngine(engineData.data(), engineData.size()));
  // The execution context contains all of the state associated with a
  // particular invocation
  nvContext_ = std::unique_ptr<nvinfer1::IExecutionContext>(
      nvEngine_->createExecutionContext());

  inputType_ = nvEngine_->getTensorDataType(INPUT_TENSOR_NAME);
  // Model input tensor (1, 3, input image height, input image width)
  inputDims_ = nvEngine_->getTensorShape(INPUT_TENSOR_NAME);
  // Detection tensor (YOLOv8 head)
  // For YoloV8, it is (num batches, [x, y, w, h, conf, 32 mask coeffs], num
  // anchors), which is (1, 37, 16128)
  output0Dims_ = nvEngine_->getTensorShape(OUTPUT0_TENSOR_NAME);
  // Mask prototype (segmentation features)
  // For YoloV8, it is (num batches, num mask prototype channels, mask feature
  // spatial size), which is (1, 32, 192, 256)
  output1Dims_ = nvEngine_->getTensorShape(OUTPUT1_TENSOR_NAME);

  // Allocate memory on device
  checkCudaErrorCode(cudaMalloc(&devOutput0Ptr_, getOutput0Size()));
  checkCudaErrorCode(cudaMalloc(&devOutput1Ptr_, getOutput1Size()));

  // Allocate memory on host
  checkCudaErrorCode(cudaMallocHost(&hostOutput0Ptr_, getOutput0Size()));
  checkCudaErrorCode(cudaMallocHost(&hostOutput1Ptr_, getOutput1Size()));

  nvContext_->setTensorAddress(OUTPUT0_TENSOR_NAME, devOutput0Ptr_);
  nvContext_->setTensorAddress(OUTPUT1_TENSOR_NAME, devOutput1Ptr_);
}

YoloV8::~YoloV8() {
  // Make sure no GPU work is in flight
  if (cvStream_) {
    auto cudaStream{cv::cuda::StreamAccessor::getStream(cvStream_)};
    cudaStreamSynchronize(cudaStream);
  }

  // Tear down TensorRT objects
  nvContext_.reset();
  nvEngine_.reset();
  nvRuntime_.reset();

  // Free device and host-pinned memory
  if (devOutput0Ptr_) {
    cudaFree(devOutput0Ptr_);
    devOutput0Ptr_ = nullptr;
  }
  if (devOutput1Ptr_) {
    cudaFree(devOutput1Ptr_);
    devOutput1Ptr_ = nullptr;
  }
  if (hostOutput0Ptr_) {
    cudaFreeHost(hostOutput0Ptr_);
    hostOutput0Ptr_ = nullptr;
  }
  if (hostOutput1Ptr_) {
    cudaFreeHost(hostOutput1Ptr_);
    hostOutput1Ptr_ = nullptr;
  }
}

std::vector<Object> YoloV8::predict(const cv::Mat& imageRGB,
                                    float confThreshold, float nmsThreshold,
                                    float segmentationThreshold,
                                    int maxDetections) {
  // Validate
  if (imageRGB.empty() ||
      (imageRGB.type() != CV_8UC3 && imageRGB.type() != CV_32FC3)) {
    throw std::invalid_argument(
        "Expected non-empty CV_8UC3 or CV_32FC3 RGB image");
  }

  int imageWidth{imageRGB.cols};
  int imageHeight{imageRGB.rows};

#ifdef ENABLE_BENCHMARKS
  static int numIterations{1};
  preciseStopwatch s1;
#endif

  // Note: be sure to run all GPU operations on the same CUDA stream for best
  // performance
  auto cudaStream{cv::cuda::StreamAccessor::getStream(cvStream_)};

  // Upload the image to GPU memory
  cv::cuda::GpuMat gpuImg;
  gpuImg.upload(imageRGB, cvStream_);

  ////////////////
  // PREPROCESSING
  ////////////////

  // Resize image to the model input size
  int inputWidth{static_cast<int>(inputDims_.d[3])};
  int inputHeight{static_cast<int>(inputDims_.d[2])};
  if (gpuImg.rows != inputHeight || gpuImg.cols != inputWidth) {
    cv::cuda::resize(gpuImg, gpuImg, cv::Size(inputWidth, inputHeight), 0.0,
                     0.0, cv::INTER_LINEAR, cvStream_);
  }

  // Convert input to float and normalize to [0, 1]
  if (gpuImg.type() == CV_8UC3) {
    gpuImg.convertTo(gpuImg, CV_32FC3, 1.0 / 255.0, 0.0, cvStream_);
  }

  // Convert the image from NHWC (cv image) to NCHW (what TensorRT expects) by
  // channel-splitting and stacking (zero extra host copies)
  // TODO: Can we keep NHWC but add shuffle in engine?
  std::vector<cv::cuda::GpuMat> channels(3);
  cv::cuda::split(gpuImg, channels, cvStream_);
  cv::cuda::GpuMat nchw(channels[0].rows * 3, channels[0].cols,
                        channels[0].type());
  for (int channelIdx = 0; channelIdx < 3; ++channelIdx) {
    cv::cuda::GpuMat roi(nchw, cv::Rect(0, channelIdx * channels[0].rows,
                                        channels[0].cols, channels[0].rows));
    channels[channelIdx].copyTo(roi, cvStream_);
  }
  nvContext_->setTensorAddress(INPUT_TENSOR_NAME, nchw.ptr<void>());

  ////////////
  // INFERENCE
  ////////////

  if (!nvContext_->enqueueV3(cudaStream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  // Copy outputs back to host and sync CUDA stream
  cudaMemcpyAsync(hostOutput0Ptr_, devOutput0Ptr_, getOutput0Size(),
                  cudaMemcpyDeviceToHost, cudaStream);
  cudaMemcpyAsync(hostOutput1Ptr_, devOutput1Ptr_, getOutput1Size(),
                  cudaMemcpyDeviceToHost, cudaStream);
  cudaStreamSynchronize(cudaStream);

  // Wrap host buffers as cv::Mat (no copy)
  // output0 (detections): [num attrs x num anchors]
  int numAttrs{static_cast<int>(output0Dims_.d[1])};
  int numAnchors{static_cast<int>(output0Dims_.d[2])};
  cv::Mat detectionsMat(numAttrs, numAnchors, CV_32F, hostOutput0Ptr_);
  // output1 (seg masks): [segChannels x (segHeight * segWidth)]
  int segChannels{static_cast<int>(output1Dims_.d[1])};
  int segHeight{static_cast<int>(output1Dims_.d[2])};
  int segWidth{static_cast<int>(output1Dims_.d[3])};
  cv::Mat maskProtosMat(segChannels, segHeight * segWidth, CV_32F,
                        hostOutput1Ptr_);

#ifdef ENABLE_BENCHMARKS
  static long long inferenceTimeUs{0};
  inferenceTimeUs += s1.elapsedTime<long long, std::chrono::microseconds>();
  spdlog::info("Avg preprocessing + inference time: {} ms",
               (inferenceTimeUs / numIterations) / 1000.f);
#endif

  /////////////////
  // POSTPROCESSING
  /////////////////

#ifdef ENABLE_BENCHMARKS
  preciseStopwatch s2;
#endif

  int numClasses{numAttrs - segChannels - 4};
  std::vector<int> labels;
  std::vector<float> confs;
  std::vector<cv::Rect> bboxes;
  std::vector<std::vector<float>> maskCoeffs;

  // Go through all anchor points in the output tensor looking for
  // detections of confidence greater than our designated threshold. Where
  // we find detections, the box details are extrapolated and scaled to
  // full-size.
  for (int anchorIdx = 0; anchorIdx < numAnchors; ++anchorIdx) {
    // Use pointer arithmetic here since cv::Mat::at has overhead
    // The detections output is numAttrs x numAnchors (attribute-major)
    // Attrs: [x, y, w, h, class_0, class_1, ..., mask_0, mask_1, ...]
    const float* colPtr{detectionsMat.ptr<float>(0) + anchorIdx};
    // Attribute stride = numAnchors (since cols are contiguous across anchors)
    const int stride{numAnchors};

    // Get max confidence across classes
    float maxConf{-std::numeric_limits<float>::infinity()};
    int maxClass{-1};
    for (int classIdx = 0; classIdx < numClasses; ++classIdx) {
      float conf{colPtr[(4 + classIdx) * stride]};
      if (conf > maxConf) {
        maxConf = conf;
        maxClass = classIdx;
      }
    }

    if (maxConf > confThreshold) {
      // bbox
      float x{colPtr[0 * stride]};
      float y{colPtr[1 * stride]};
      float w{colPtr[2 * stride]};
      float h{colPtr[3 * stride]};

      // Scale bbox to original image size
      float xRatio{static_cast<float>(imageWidth) / inputWidth};
      float yRatio{static_cast<float>(imageHeight) / inputHeight};
      float x0{std::clamp((x - 0.5f * w) * xRatio, 0.f,
                          static_cast<float>(imageWidth))};
      float y0{std::clamp((y - 0.5f * h) * yRatio, 0.f,
                          static_cast<float>(imageHeight))};
      float x1{std::clamp((x + 0.5f * w) * xRatio, 0.f,
                          static_cast<float>(imageWidth))};
      float y1{std::clamp((y + 0.5f * h) * yRatio, 0.f,
                          static_cast<float>(imageHeight))};

      cv::Rect2f bbox;
      bbox.x = x0;
      bbox.y = y0;
      bbox.width = x1 - x0;
      bbox.height = y1 - y0;

      // Mask coeffs
      std::vector<float> anchorMaskCoeffs(segChannels);
      for (int maskCoeffIdx = 0; maskCoeffIdx < segChannels; ++maskCoeffIdx) {
        float maskCoeff{colPtr[(4 + numClasses + maskCoeffIdx) * stride]};
        anchorMaskCoeffs[maskCoeffIdx] = maskCoeff;
      }

      labels.push_back(maxClass);
      bboxes.push_back(bbox);
      confs.push_back(maxConf);
      maskCoeffs.push_back(anchorMaskCoeffs);
    }
  }

  // Run NMS to eliminate duplicate detections
  std::vector<int> nmsIndices;
  cv::dnn::NMSBoxesBatched(bboxes, confs, labels, confThreshold, nmsThreshold,
                           nmsIndices);

  // Prepare output for just the NMS detections
  cv::Mat detectedObjectsMaskCoeffs;
  std::vector<Object> detectedObjects;
  int detectionsCount{0};
  for (int nmsIdx : nmsIndices) {
    if (detectionsCount >= maxDetections) {
      break;
    }
    Object obj;
    obj.label = labels[nmsIdx];
    obj.rect = bboxes[nmsIdx];
    obj.conf = confs[nmsIdx];
    detectedObjects.push_back(obj);
    cv::Mat maskCoeffsRow(1, static_cast<int>(maskCoeffs[nmsIdx].size()),
                          CV_32F,
                          const_cast<float*>(maskCoeffs[nmsIdx].data()));
    detectedObjectsMaskCoeffs.push_back(maskCoeffsRow);
    ++detectionsCount;
  }

  // For every NMS detection, combine the prototype masks using the mask coeffs
  // to get the final mask for that detection. YoloV8-seg predicts shared mask
  // prototypes (32 channels x (segHeight * segWidth)); each detected object
  // then only needs 32 coefficients to generate its mask using those shared
  // prototypes.
  if (!detectedObjectsMaskCoeffs.empty()) {
    // detectedObjectsMaskCoeffs is (N x 32) where N is the number of NMS
    // detections maskProtosMat is (32 x (segHeight * segWidth))
    cv::Mat maskLogits{detectedObjectsMaskCoeffs *
                       maskProtosMat};  // (N x (segHeight * segWidth))

    // Apply sigmoid to convert logits to probabilities
    cv::Mat neg, expNeg, denom, maskProbs;
    cv::multiply(maskLogits, -1, neg);
    cv::exp(neg, expNeg);
    cv::add(expNeg, 1.0, denom);
    cv::divide(1.0, denom, maskProbs);  // values are now in [0, 1]

    // Reshape each row (detection mask) to image dims
    for (int nmsIdx = 0; nmsIdx < maskProbs.rows; ++nmsIdx) {
      cv::Mat mask{maskProbs.row(nmsIdx).reshape(
          1, segHeight)};  // (segHeight x segWidth)

      // Upscale to original image size
      cv::resize(mask, mask, cv::Size(imageWidth, imageHeight),
                 cv::INTER_LINEAR);
      // Threshold to CV_8U bitmask, and limit to only inside the bbox
      detectedObjects[nmsIdx].boxMask =
          mask(detectedObjects[nmsIdx].rect) > segmentationThreshold;
    }
  }

#ifdef ENABLE_BENCHMARKS
  static long long postprocessingTimeUs{0};
  postprocessingTimeUs +=
      s2.elapsedTime<long long, std::chrono::microseconds>();
  spdlog::info("Avg postprocessing time: {} ms",
               (postprocessingTimeUs / numIterations) / 1000.f);
  ++numIterations;
#endif

  spdlog::info("Detected {} objects", detectedObjects.size());

  return detectedObjects;
}

void YoloV8::drawObjectLabels(cv::Mat& image,
                              const std::vector<Object>& objects,
                              unsigned int scale) {
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

size_t YoloV8::getOutput0Size() const {
  auto output0Type{nvEngine_->getTensorDataType(OUTPUT0_TENSOR_NAME)};
  return output0Dims_.d[0] * output0Dims_.d[1] * output0Dims_.d[2] *
         getDataTypeSize(output0Type);
}

size_t YoloV8::getOutput1Size() const {
  auto output1Type{nvEngine_->getTensorDataType(OUTPUT1_TENSOR_NAME)};
  return output1Dims_.d[0] * output1Dims_.d[1] * output1Dims_.d[2] *
         output1Dims_.d[3] * getDataTypeSize(output1Type);
}