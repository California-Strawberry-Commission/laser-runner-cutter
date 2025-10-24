#pragma once

#include <NvInfer.h>

#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <string>

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override;
};

struct Object {
  // The object class
  int label{};
  // The detection's confidence probability
  float conf{};
  // The object bounding box rectangle (TLWH)
  cv::Rect2f rect;
  // Semantic segmentation mask, inside the bounding box
  cv::Mat boxMask;
};

class YoloV8 {
 public:
  explicit YoloV8(const std::string& trtEngineFile);
  ~YoloV8();
  YoloV8(const YoloV8&) = delete;
  YoloV8& operator=(const YoloV8&) = delete;

  std::vector<Object> predict(const cv::Mat& imageRgb,
                              float confThreshold = 0.25f,
                              float nmsThreshold = 0.6f,
                              float segmentationThreshold = 0.5f,
                              int maxDetections = 300);
  std::vector<Object> predict(const cv::cuda::GpuMat& imageRgb,
                              float confThreshold = 0.25f,
                              float nmsThreshold = 0.6f,
                              float segmentationThreshold = 0.5f,
                              int maxDetections = 300);

 private:
  static inline constexpr const char* INPUT_TENSOR_NAME{"images"};
  static inline constexpr const char* OUTPUT0_TENSOR_NAME{"output0"};
  static inline constexpr const char* OUTPUT1_TENSOR_NAME{"output1"};

  size_t getOutput0Size() const;
  size_t getOutput1Size() const;

  std::unique_ptr<nvinfer1::IRuntime> nvRuntime_;
  std::unique_ptr<nvinfer1::ICudaEngine> nvEngine_;
  std::unique_ptr<nvinfer1::IExecutionContext> nvContext_;
  Logger logger_;
  nvinfer1::DataType inputType_;
  nvinfer1::Dims inputDims_;
  nvinfer1::Dims output0Dims_;
  nvinfer1::Dims output1Dims_;
  cv::cuda::Stream cvStream_;
  // Device memory pointers
  void* devOutput0Ptr_{nullptr};
  void* devOutput1Ptr_{nullptr};
  // Host pinned memory pointers
  void* hostOutput0Ptr_{nullptr};
  void* hostOutput1Ptr_{nullptr};
  // Reused temp buffers for input preprocessing
  cv::cuda::GpuMat resizeBuf_;  // for resizing to model input size
  cv::cuda::GpuMat floatBuf_;   // for conversion to float
  cv::cuda::GpuMat nchwBuf_;    // for NHWC -> NCHW conversion
  std::array<cv::cuda::GpuMat, 3> nchwPlanes_;  // views into nchwBuf_
};