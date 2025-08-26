#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include "spdlog/spdlog.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

#include "BYTETracker.h"
#include "NvInfer.h"
#include "detection_cpp/tools/fp16_utils.hpp"

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
    DetectionNode() : Node("detection_node") {
      // Constructor implementation (if needed)
      declare_parameter<float>("confidence_threshold", 0.5f);
      declare_parameter<float>("nms_threshold", 0.45f);
      declare_parameter<float>("tracker_fps", 30.0f);
      declare_parameter<float>("tracker_buffer_size", 30.0f);

      confidence_threshold = this->get_parameter("confidence_threshold").as_double();
      nms_threshold = this->get_parameter("nms_threshold").as_double();
      tracker_fps = this->get_parameter("tracker_fps").as_double();
      tracker_buffer_size = this->get_parameter("tracker_buffer_size").as_double();

      frameSubscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera0/frame",
        rclcpp::SensorDataQoS(),
        std::bind(&DetectionNode::frame_callback, this, std::placeholders::_1)
      );

      overlayFramePublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/overlay_frame", rclcpp::SensorDataQoS());

      std::string engineFilePath{"../detection_cpp/models/RunnerSegYoloV8l.engine"};

      std::ifstream file(engineFilePath, std::ios::binary);
      file.seekg(0, file.end);
      std::size_t fsize = file.tellg();
      file.seekg(0, file.beg);
      std::vector<char> engineData(fsize);
      file.read(engineData.data(), fsize);
      file.close();

      tracker = BYTETracker(tracker_fps, tracker_buffer_size);

      cudaSetDevice(deviceNum);
      cudaGetDeviceProperties(&deviceProp, deviceNum);
      spdlog::info("CUDA device set to: {}.", deviceProp.name);

      cudaStreamCreate(&stream);

      mRuntime = nvinfer1::createInferRuntime(logger);
      mEngine = mRuntime->deserializeCudaEngine(engineData.data(), fsize);
      mContext = mEngine->createExecutionContext();

      input_dims = mEngine->getTensorShape(input_name);
      output0_dims = mEngine->getTensorShape(output0_name);
      output1_dims = mEngine->getTensorShape(output1_name);

      inputType = mEngine->getTensorDataType(input_name);
      isHalfPrecision = (inputType == nvinfer1::DataType::kHALF);
      spdlog::info("Model uses {} precision", isHalfPrecision ? "FP16" : "FP32");

      dtype_size = isHalfPrecision ? sizeof(uint16_t) : sizeof(float);
      output0_size = output0_dims.d[0] * output0_dims.d[1] * output0_dims.d[2] * dtype_size;
      output1_size = output1_dims.d[0] * output1_dims.d[1] * output1_dims.d[2] * output1_dims.d[3] * dtype_size;

      cudaMalloc(&outputMem0, output0_size);
      cudaMalloc(&outputMem1, output1_size);

      output0 = cv::Mat(37, 16128, CV_32FC1);
    }


    void frame_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
      // spdlog::info("===============================================");
      // spdlog::info("Frame timestamp: sec={}, nanosec={}", msg->header.stamp.sec, msg->header.stamp.nanosec);
      // spdlog::info("Image size: {}x{}, step={}", msg->width, msg->height, msg->step);

      cv::Mat image(msg->height, msg->width, CV_8UC3, const_cast<uint8_t*>(msg->data.data()), msg->step);
      // spdlog::info("cv::Mat image dimensions: rows={}, cols={}, channels={}, type={}", 
      //      image.rows, image.cols, image.channels(), image.type());

      boxes.clear();
      confidences.clear();
      objects.clear();
      
      cv::cuda::GpuMat gpuImg;
      gpuImg.upload(image);

      cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);
      cv::cuda::resize(gpuImg, gpuImg, cv::Size(1024, 768), 0.0, 0.0, cv::INTER_LINEAR);
      gpuImg.convertTo(gpuImg, CV_32FC1, 1.0 / 255.0);

      std::vector<cv::cuda::GpuMat> channels(3);
      cv::cuda::split(gpuImg, channels);
      cv::cuda::GpuMat nchw(channels[0].rows * 3, channels[0].cols, channels[0].type());
      for (int c = 0; c < 3; c++) {
        cv::cuda::GpuMat roi(nchw, cv::Rect(0, c * channels[0].rows, channels[0].cols, channels[0].rows));
        channels[c].copyTo(roi);
      }

      if (isHalfPrecision) {
        int numElems = nchw.rows * nchw.cols;
        __half* inputFp16;
        cudaMalloc((void **)&inputFp16, numElems * sizeof(__half));
        convertFp32ToFp16(nchw, inputFp16, numElems);
        this->inputMem = inputFp16;
      } else {
        this->inputMem = nchw.ptr<void>();
      }

      cudaMemset(outputMem0, 0, output0_size);
      cudaMemset(outputMem1, 0, output1_size);

      mContext->setTensorAddress(input_name, inputMem);
      mContext->setTensorAddress(output0_name, outputMem0);
      mContext->setTensorAddress(output1_name, outputMem1);

      bool success = mContext->enqueueV3(stream);
      if (!success) {
        spdlog::error("TensorRT inference failed");
      }
      cudaStreamSynchronize(stream);
      
      if (isHalfPrecision) {
        cv::Mat half_output0(37, 16128, CV_16FC1);
        cudaMemcpy(half_output0.data, outputMem0, output0_size, cudaMemcpyDeviceToHost);
        half_output0.convertTo(output0, CV_32FC1);
      } else {
        cudaMemcpy(output0.data, outputMem0, output0_size, cudaMemcpyDeviceToHost);
      }

      for (int i = 0; i < 16128; ++i) {
        float confidence = output0.at<float>(4, i);
        if (confidence > confidence_threshold) {
          spdlog::info("Detection found: confidence = {:.2f}, index = {}", confidence, i);
          float x_center = output0.at<float>(0, i);
          float y_center = output0.at<float>(1, i);
          float width = output0.at<float>(2, i);
          float height = output0.at<float>(3, i);

          int x = static_cast<int>(x_center - width / 2) * 2;
          int y = static_cast<int>(y_center - height / 2) * 2;
          int w = static_cast<int>(width) * 2;
          int h = static_cast<int>(height) * 2;

          Object obj;
          obj.rect = cv::Rect(x, y, w, h);        // x, y, w, h
          obj.prob = confidence;                  // confidence
          obj.label = 0;                          // optional (class index)

          boxes.push_back(obj.rect);
          confidences.push_back(obj.prob);
          objects.push_back(obj);
        }
      }

      std::vector<int> nms_indices;
      cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, nms_indices);

      std::vector<Object> nms_objects;
      for (int idx : nms_indices) {
        nms_objects.push_back(objects[idx]);
      }

      // std::vector<STrack> tracked = tracker.update(nms_objects);

      // for (const auto& track : tracked) {
      //   if (!track.is_activated) continue;
      //   int baseLine = 0;
      //   std::string label = cv::format("ID: %d | Conf: %.2f", track.track_id, track.score);
      //   cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      //   cv::Rect box(track.tlwh[0], track.tlwh[1], track.tlwh[2], track.tlwh[3]);
      //   int top = std::max(box.y, labelSize.height);

      //   cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
      //   cv::rectangle(image, cv::Point(box.x, top - labelSize.height), cv::Point(box.x + labelSize.width, top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
      //   cv::putText(image, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 2);
      // }

      for (const Object& obj : nms_objects) {
        int baseLine = 0;
        std::string label = cv::format("ID: %d | Conf: %.2f", 0, obj.prob);
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        // cv::Rect box(track.tlwh[0], track.tlwh[1], track.tlwh[2], track.tlwh[3]);
        int top = std::max((int)(obj.rect.y), labelSize.height);

        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(image, cv::Point(obj.rect.x, top - labelSize.height), cv::Point(obj.rect.x + labelSize.width, top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, obj.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 2);
      }

      cv::resize(image, image, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);

      auto header{std_msgs::msg::Header()};
      header.stamp.sec = msg->header.stamp.sec;
      header.stamp.nanosec = msg->header.stamp.nanosec;
      sensor_msgs::msg::Image::SharedPtr overlayFrameMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
      overlayFramePublisher_->publish(*overlayFrameMsg);
      
    }
  
  private:
    float confidence_threshold, nms_threshold, tracker_fps, tracker_buffer_size;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr frameSubscription_;
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
    cv::Mat output0;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<Object> objects;
};


int main(int argc, char const *argv[])
{
  rclcpp::init(argc, argv);

  try {
    // MultiThreadedExecutor allows callbacks to run in parallel
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node{std::make_shared<DetectionNode>()};
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& e) {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
