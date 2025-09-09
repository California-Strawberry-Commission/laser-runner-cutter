#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include "spdlog/spdlog.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "camera_control_interfaces/msg/lucid_frame_images.hpp"

#include "BYTETracker.h"
#include "NvInfer.h"
// Building fp16 mode with trtexec somehow enables cross-compatibility with input
// #include "detection_cpp/tools/fp16_utils.hpp"

#include <filesystem>
#include <fstream>


class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    // suppress info-level messages
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} logger;

class DetectionNode : public rclcpp::Node {
  public:
    DetectionNode();
    void frame_callback(const camera_control_interfaces::msg::LucidFrameImages::SharedPtr msg);

  private:
    float confidence_threshold, nms_threshold, tracker_fps, tracker_buffer_size;

    rclcpp::Subscription<camera_control_interfaces::msg::LucidFrameImages>::SharedPtr frameSubscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlayFramePublisher_;

    int deviceNum = 0;
    char const *input_name = "images";
    char const *output0_name = "output0";
    char const *output1_name = "output1";

    BYTETracker tracker;
    cudaDeviceProp deviceProp;
    cudaStream_t stream;
    nvinfer1::IRuntime *mRuntime;
    nvinfer1::ICudaEngine *mEngine;
    nvinfer1::IExecutionContext *mContext;
    nvinfer1::Dims input_dims;
    nvinfer1::Dims output0_dims;
    nvinfer1::Dims output1_dims;
    nvinfer1::DataType inputType;
    bool isHalfPrecision;
    size_t dtype_size;
    size_t output0_size;
    size_t output1_size;
    void *inputMem, *outputMem0, *outputMem1;
    cv::Mat output0, output1;
};