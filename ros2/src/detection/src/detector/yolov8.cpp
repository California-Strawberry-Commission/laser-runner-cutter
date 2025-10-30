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

namespace {

bool fileExists(const std::string& filepath) {
  std::ifstream f(filepath.c_str());
  return f.good();
}

void checkCudaErrorCode(cudaError_t code) {
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

}  // namespace

void YoloV8::Logger::log(Severity severity, const char* msg) noexcept {
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

std::vector<YoloV8::Object> YoloV8::predict(const cv::Mat& imageRgb,
                                            float confThreshold,
                                            float nmsThreshold,
                                            float segmentationThreshold,
                                            int maxDetections) {
  cv::cuda::GpuMat gpuImg;
  gpuImg.upload(imageRgb, cvStream_);
  return predict(gpuImg, confThreshold, nmsThreshold, segmentationThreshold,
                 maxDetections);
}

std::vector<YoloV8::Object> YoloV8::predict(const cv::cuda::GpuMat& imageRgb,
                                            float confThreshold,
                                            float nmsThreshold,
                                            float segmentationThreshold,
                                            int maxDetections) {
  // Validate image
  if (imageRgb.empty() ||
      (imageRgb.type() != CV_8UC3 && imageRgb.type() != CV_32FC3)) {
    throw std::invalid_argument(
        "Expected non-empty CV_8UC3 or CV_32FC3 RGB image");
  }

  int imageWidth{imageRgb.cols};
  int imageHeight{imageRgb.rows};

#ifdef ENABLE_BENCHMARKS
  static int numIterations{1};
  preciseStopwatch s1;
#endif

  // Use a single CUDA stream for prediction to enable concurrency for CUDA
  // operations and host <-> device copies
  auto cudaStream{cv::cuda::StreamAccessor::getStream(cvStream_)};

  ////////////////
  // PREPROCESSING
  ////////////////

  // We reuse temp buffers across predictions in order to minimize allocation
  // overhead and memory fragmentation

  // Resize image to the model input size
  int inputWidth{static_cast<int>(inputDims_.d[3])};
  int inputHeight{static_cast<int>(inputDims_.d[2])};
  cv::Size inputSize(inputWidth, inputHeight);
  const cv::cuda::GpuMat* rgbResized{&imageRgb};
  if (imageRgb.size() != inputSize) {
    // Note: GpuMat::create is a no-op if size/type already matches
    resizeBuf_.create(inputSize, imageRgb.type());
    cv::cuda::resize(imageRgb, resizeBuf_, cv::Size(inputWidth, inputHeight),
                     0.0, 0.0, cv::INTER_LINEAR, cvStream_);
    rgbResized = &resizeBuf_;
  }

  // Convert input to float and normalize to [0, 1]
  const cv::cuda::GpuMat* rgbFloat{rgbResized};
  if (rgbResized->type() == CV_8UC3) {
    floatBuf_.create(inputSize, CV_32FC3);
    rgbResized->convertTo(floatBuf_, CV_32FC3, 1.0 / 255.0, 0.0, cvStream_);
    rgbFloat = &floatBuf_;
  }

  // Convert from NHWC to NCHW (what TensorRT expects) by channel-splitting and
  // stacking
  // TODO: Can we keep NHWC but add shuffle in engine?

  // Ensure temp buffer (a single contiguous buffer with 3 planes stacked
  // vertically) size/type is what we want
  nchwBuf_.create(3 * inputHeight, inputWidth, CV_32FC1);

  // Update plane headers to point into nchwBuf_ (no copy)
  for (int cIdx = 0; cIdx < 3; ++cIdx) {
    float* basePtr{
        reinterpret_cast<float*>(nchwBuf_.ptr<float>(cIdx * inputHeight))};
    nchwPlanes_[cIdx] = cv::cuda::GpuMat(inputHeight, inputWidth, CV_32FC1,
                                         basePtr, nchwBuf_.step);
  }

  // Split directly into NCHW planes
  cv::cuda::split(*rgbFloat, nchwPlanes_.data(), cvStream_);

  nvContext_->setTensorAddress(INPUT_TENSOR_NAME, nchwBuf_.ptr<void>());

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
    // Use pointer arithmetic here since cv::Mat::at has overhead. Note
    // that a full transpose of detectionsMat is expensive, so we stick with
    // attribute-major (numAttrs x numAnchors). So, each column of detectionsMat
    // is [x, y, w, h, class_0, class_1, ..., mask_0, mask_1, ...]
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
      float x{colPtr[0 * stride]};  // x center of bbox
      float y{colPtr[1 * stride]};  // y center of bbox
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

      cv::Rect bbox;
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
    for (int objIdx = 0; objIdx < detectedObjectsMaskCoeffs.rows; ++objIdx) {
      // Note: we only calculate logits and create the mask for the area inside
      // the bbox for performance reasons

      // Scale down bbox (which is at original image scale) to mask prototype
      // scale (segWidth, segHeight)
      const cv::Rect& bbox{detectedObjects[objIdx].rect};
      int px0{std::max(0, int(std::floor(bbox.x * static_cast<float>(segWidth) /
                                         imageWidth)))};
      int py0{
          std::max(0, int(std::floor(bbox.y * static_cast<float>(segHeight) /
                                     imageHeight)))};
      int px1{std::min(
          segWidth, int(std::ceil((bbox.x + bbox.width) *
                                  static_cast<float>(segWidth) / imageWidth)))};
      int py1{std::min(segHeight, int(std::ceil((bbox.y + bbox.height) *
                                                static_cast<float>(segHeight) /
                                                imageHeight)))};
      int pw{std::max(1, px1 - px0)};
      int ph{std::max(1, py1 - py0)};

      // Accumulator for logits inside the ROI (ph x pw)
      cv::Mat acc(ph, pw, CV_32F, cv::Scalar(0));
      float* accBasePtr{acc.ptr<float>(0)};

      // For each mask coeff, multiply it to the corresponding mask proto, and
      // accumulate into acc
      const float* maskCoeffsPtr{detectedObjectsMaskCoeffs.ptr<float>(objIdx)};
      for (int protoIdx = 0; protoIdx < segChannels; ++protoIdx) {
        const float* protoPtr{maskProtosMat.ptr<float>(protoIdx)};
        const float maskCoeff{maskCoeffsPtr[protoIdx]};

        // Add maskCoeff * proto[ROI] into acc
        for (int ry = 0; ry < ph; ++ry) {
          const int srcOffset{(py0 + ry) * segWidth + px0};
          const float* src{protoPtr + srcOffset};
          float* dst{accBasePtr + ry * pw};

          for (int rx = 0; rx < pw; ++rx) {
            dst[rx] += maskCoeff * src[rx];
          }
        }
      }

      // Sigmoid on ROI to convert logits into probabilities
      for (int ry = 0; ry < ph; ++ry) {
        float* accRow{acc.ptr<float>(ry)};
        for (int rx = 0; rx < pw; ++rx) {
          // Sigmoid converts logits into probabilities
          const float s{1.0f / (1.0f + std::exp(-accRow[rx]))};
          accRow[rx] = s;
        }
      }

      // Upscale to bbox, apply thresholding, and convert to U8
      // Note that we intentionally upscale first before thresholding to smooth
      // the mask contour
      cv::Mat boxMask;
      cv::resize(acc, boxMask, cv::Size(bbox.width, bbox.height), 0, 0,
                 cv::INTER_LINEAR);
      cv::threshold(boxMask, boxMask, segmentationThreshold, 1.0,
                    cv::THRESH_BINARY);
      boxMask.convertTo(boxMask, CV_8U, 255.0);
      detectedObjects[objIdx].boxMask = boxMask;
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