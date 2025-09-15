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

#include <rclcpp_components/register_node_macro.hpp>

/*
Creating a runtime inference model with nvinfer means passing it a logger to use:
this is just a simple model of doing so that throws everything except "INFO" status logs.
*/
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
    /*
    Adding NodeOptions enables toggling specific settings around how the node is configured on provisioning,
    in this case, we specify in the runner-cutter-control package's launch.py that we want to toggle intra-process communications for this node.
    Intra-Process comms are pretty straight forward, and at a base level it just means that ros2 knows the memory is accessible by both publisher and subscriber "processes"
    so it just passes ack and forth memory pointers to the message data instead of copying and serializing/deserializing the data every time.
    */
    DetectionNode(const rclcpp::NodeOptions & options) : Node("detection_node", options) {
      /*
      I more or less replicated the template for using parameters.yaml I saw in your code, and it seems to work fine in containerized/composable nodes.
      */
      declare_parameter<float>("confidence_threshold", 0.5f);
      declare_parameter<float>("nms_threshold", 0.45f);
      declare_parameter<float>("tracker_fps", 15.0f);
      declare_parameter<float>("tracker_buffer_size", 15.0f);

      confidence_threshold = this->get_parameter("confidence_threshold").as_double();
      nms_threshold = this->get_parameter("nms_threshold").as_double();
      tracker_fps = this->get_parameter("tracker_fps").as_double();
      tracker_buffer_size = this->get_parameter("tracker_buffer_size").as_double();


      /*
      While this was a debugging message, it's useful to keep in as some messaging protocols (especially revolving message loaning) 
      are only enabled on specific middlewares. If we ever were to switch in the future or if this were to get launched on an improperly 
      setup system, showing the RMW eliminates a big chunk of potential debugging as to what may have gone wrong.
      */
      spdlog::info("Using RMW: {}", rmw_get_implementation_identifier());

      /*
      Because the actual state of data in memory being pointed to and passed around in intra-process comms can't be guaranteed 
      without the middleware copy, topic QOS's need to be designated as "durability volatile" or the process simply won't launch. 
      This is just a simple qos that tries to keep stability while also designating volatility.
      */
      rclcpp::QoS qos(rclcpp::KeepLast(10));
      qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      frameSubscription_ = this->create_subscription<camera_control_interfaces::msg::LucidFrameImages>(
        "/camera0/frame",
        qos,
        std::bind(&DetectionNode::frame_callback, this, std::placeholders::_1)
      );

      overlayFramePublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "~/overlay_frame", rclcpp::SensorDataQoS());

      /*
      Assuming either my script is assimilated into setup or activated explicitly, either way it should know to put the engine file
      into the below designated folder with that specific name. I've never run into trouble of calling ./run_ros.sh from random dirs 
      and the active dir being messed up as it should always be in the same dir by the time this script is run.
      */
      std::string engineFilePath{"../detection_cpp/models/RunnerSegYoloV8l.engine"};

      std::ifstream file(engineFilePath, std::ios::binary);
      file.seekg(0, file.end);
      std::size_t fsize = file.tellg();
      file.seekg(0, file.beg);
      std::vector<char> engineData(fsize);
      file.read(engineData.data(), fsize);
      file.close();

      /*
      This works the way you think it does lol.
      */
      tracker = BYTETracker(tracker_fps, tracker_buffer_size);

      /* 
      I currently have deviceNum hardcoded to 0, designating the Orin gpu on the box. This should be fine for the foreseeable future
      UNLESS we decide to switch platforms, NVIDIA adds another gpu, or NVIDIA pulls a funny prank and changes the numbering scheme (ie making cpu 0).
      */
      cudaSetDevice(deviceNum);
      cudaGetDeviceProperties(&deviceProp, deviceNum);
      spdlog::info("CUDA device set to: {}.", deviceProp.name);

      /*
      This creates the asynchronous gpu stream to which we enque tasks like the gpuimg copy and the inference command.
      */
      cudaStreamCreate(&stream);

      /*
      This just sets up the engine into an in-memory execution context which we enque inference commands to.
      */
      mRuntime = nvinfer1::createInferRuntime(logger);
      mEngine = mRuntime->deserializeCudaEngine(engineData.data(), fsize);
      mContext = mEngine->createExecutionContext();

      /*
      This also does what you think it does. 
      NOTE: If we were to switch resolutions and wanted to resize based on input tensor rather than hardcoded values,
      this is where we could find the input res to adjust to from. To save future grief input tensors are typically in 
      nchw format which is PROBABLY NOT what you can expect your image/cv::Mat format to be in.
      */
      input_dims = mEngine->getTensorShape(input_name);
      output0_dims = mEngine->getTensorShape(output0_name);
      output1_dims = mEngine->getTensorShape(output1_name);

      inputType = mEngine->getTensorDataType(input_name);
      isHalfPrecision = (inputType == nvinfer1::DataType::kHALF);

      dtype_size = isHalfPrecision ? sizeof(uint16_t) : sizeof(float);
      output0_size = output0_dims.d[0] * output0_dims.d[1] * output0_dims.d[2] * dtype_size;
      output1_size = output1_dims.d[0] * output1_dims.d[1] * output1_dims.d[2] * output1_dims.d[3] * dtype_size;

      /*
      Here we allocate space for the output responses of the asynchronous inference calls and also pre-generate the cpu-memory-side
      objects that these outputs will be memcopied into.
      */
      cudaMalloc(&outputMem0, output0_size);
      cudaMalloc(&outputMem1, output1_size);

      output0 = cv::Mat(37, 16128, CV_32FC1);
      output1 = cv::Mat(32 * 192, 256, CV_32FC1);
    }


    void frame_callback(const camera_control_interfaces::msg::LucidFrameImages::SharedPtr msg) {
      /*
      With the latest version of the LucidFrameImages message type, frames are uploaded to gpu memory by CameraControlNode
      and then just the pointer to the the object is sent via intro-process comms. The memory is then asynchronously downloaded to a 
      pinned memory buffer so that the image can be used to generate the final overlayFrame to be published.
      */
      cv::cuda::GpuMat* gpuImgPtr = reinterpret_cast<cv::cuda::GpuMat*>(msg->color_gpu_data_ptr);
      cv::cuda::GpuMat& gpuImg = *gpuImgPtr;

      // Allocate pinned host memory for async copy
      cv::cuda::HostMem pinned_buf(
        cv::Size(msg->color_width, msg->color_height),
        CV_8UC3,
        cv::cuda::HostMem::PAGE_LOCKED
      );

      // Create a Mat header pointing to pinned memory
      cv::Mat image = pinned_buf.createMatHeader();

      // Start async download from GPU → pinned host buffer
      cudaMemcpyAsync(image.data, gpuImg.ptr(), gpuImg.rows * gpuImg.step, cudaMemcpyDeviceToHost, stream);


      /*
      Create vectors for detection data. Yes I could have made maskCoefs a vector of arrays, but it didn't work
      with my first guess of std::vector<float [32]> and I figured saving 10 whole bytes by not using a vector
      wasn't a huge deal.
      */
      std::vector<cv::Rect> boxes;
      std::vector<float> confidences;
      std::vector<Object> objects;
      std::vector<std::vector<float>> maskCoeffs;
      
      /*
      Conver image from BGR input to RGB since the model is trained on and requires RGB inputs for inference.
      The image alse needs to be resized to 1024 x 768 and converted to float. All of which is done gpu-side via cv::cuda.
      */
      cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);
      cv::cuda::resize(gpuImg, gpuImg, cv::Size(1024, 768), 0.0, 0.0, cv::INTER_LINEAR);
      gpuImg.convertTo(gpuImg, CV_32FC1, 1.0 / 255.0);

      /*
      The following is some code I put together off of examples I found online for reformatting an image from nhwc to nchw.
      Unfortunately, there is an unavoidable "copy" here as the memory can't be moved around without destroying internal data otherwise.
      */
      std::vector<cv::cuda::GpuMat> channels(3);
      cv::cuda::split(gpuImg, channels);
      cv::cuda::GpuMat nchw(channels[0].rows * 3, channels[0].cols, channels[0].type());
      for (int c = 0; c < 3; c++) {
        cv::cuda::GpuMat roi(nchw, cv::Rect(0, c * channels[0].rows, channels[0].cols, channels[0].rows));
        channels[c].copyTo(roi);
      }

      /*
      We do economize a copy here however by just pointing the input buffer to the current image adress since its already on the gpu.
      The output buffers however do need to be cleared, but not reset since they get allocated on initialization.
      */
      this->inputMem = nchw.ptr<void>();
      
      cudaMemset(outputMem0, 0, output0_size);
      cudaMemset(outputMem1, 0, output1_size);

      /*
      This does what you think it does.
      */
      mContext->setTensorAddress(input_name, inputMem);
      mContext->setTensorAddress(output0_name, outputMem0);
      mContext->setTensorAddress(output1_name, outputMem1);

      bool success = mContext->enqueueV3(stream);
      if (!success) {
        spdlog::error("TensorRT inference failed");
      }
      cudaStreamSynchronize(stream);

      cudaMemcpy(output0.data, outputMem0, output0_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(output1.data, outputMem1, output1_size, cudaMemcpyDeviceToHost);

      /*
      The following code goes through all of the 16128 anchor points in the output tensor looking for detections of confidence
      greater than our designated threshold. Where we find detections, the box details are extrapolated and scaled to full-size 
      and then thrown into "object" objects (the actual class is called Object lol) since that's what ByteTrack requires.
      NOTE: We don't technically need to filter by confidence here since we do nms anyway, but it theoretically saves us some 
      efficiency by not wasting time pulling, computing, and creating objects for detections we won't care about anyway.
      */
      for (int i = 0; i < 16128; ++i) {
        float confidence = output0.at<float>(4, i);
        if (confidence > confidence_threshold) {
          // spdlog::info("Detection found: confidence = {:.2f}, index = {}", confidence, i);
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

          std::vector<float> coeffs;
          for (int j = 5; j < 37; j++) {
              coeffs.push_back(output0.at<float>(j, i));
          }

          boxes.push_back(obj.rect);
          confidences.push_back(obj.prob);
          objects.push_back(obj);
          maskCoeffs.push_back(coeffs);
        }
      }

      /*
      This next bit of code is a bit of a mess, but in theory it does three simplified tasks.
       -1- Before entering the loop, NMS is run on the accumulated references and the first line in the loop
           is exclusively just creating a new vector of objects that passed nms.
       -2- For every nms detection, all of that detection's output prototype masks are combined via the stored coeffs
           into a dedicated "detection mask" which is resized and inevitably added to a "combined mask" showing segmentation
           masks on all detections in the frame.
       -3- Before the mask is lost to the full frame combination though, a loop checks the middle slice of the bounding box
           for detection mask flags in order to identify potential cut points. These potential cut points are added to an array 
           that can be further published but are not used for the time being.
      NOTE: Task #3 has only been tested/used on example cases where no other mask/white will show in the middle slice beyond 
      the original detection. If another white were to be present, it would likely set the point in between the first and last white pixels as cutX.
      */
      std::vector<int> nms_indices;
      cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, nms_indices);
      std::vector<Point2i> cutPoints;
      std::vector<Object> nms_objects;
      cv::Mat combinedMask = cv::Mat::zeros(cv::Size(2048, 1536), CV_8UC1);
      for (int idx : nms_indices) {
        nms_objects.push_back(objects[idx]);

        const std::vector<float>& coeffs = maskCoeffs[idx];
        cv::Mat mask = cv::Mat::zeros(192, 256, CV_32F);
        for (int c = 0; c < 32; ++c) {
            cv::Mat prototype = output1.rowRange(c * 192, (c + 1) * 192).clone();
            prototype = prototype.reshape(1, 192);
            mask += coeffs[c] * prototype;
        }
        cv::resize(mask, mask, cv::Size(2048, 1536));
        cv::threshold(mask, mask, 0.5, 255, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8UC1);

        // Search the horizontal slice in the middle of the bounding box for the first white pixel in the mask
        cv::Rect rect = nms_objects.back().rect;
        int midY = rect.y + rect.height / 2;
        int cutX = -1;
        if (midY >= 0 && midY < mask.rows) {
          int whiteStart = -1, whiteEnd = -1;
          for (int x = rect.x; x < rect.x + rect.width && x < mask.cols; x++) {
            if (x >= 0 && mask.at<uchar>(midY, x) == 255) {
              if (whiteStart == -1) whiteStart = x;
              whiteEnd = x;
            }
          }
          if (whiteStart != -1 && whiteEnd != -1) {
            cutX = (whiteStart + whiteEnd) / 2;
          }
        }
        if (cutX != -1) {
          cutPoints.push_back(cv::Point(cutX, midY));
          cv::circle(image, cv::Point(cutX, midY), 5, cv::Scalar(0, 0, 255), -1);
        }

        cv::add(combinedMask, mask, combinedMask);
      }

      /*
      This just converts the fully combined mask into BGR format so it can be overlayed with the output frame via addWeighted.
      */
      cv::Mat combinedMaskBGR;
      cv::cvtColor(combinedMask, combinedMaskBGR, cv::COLOR_GRAY2BGR);
      double alpha = 0.4; // transparency factor
      cv::addWeighted(combinedMaskBGR, alpha, image, 1.0 - alpha, 0, image);

      /*
      This code has two parts.
       -1- ByteTracker is updated via the vector of objects that passed nms and the newly tracked and IDed objects are iterated on.
       -2- For every tracked detection, a bounding box and label is placed on the overlayFrame.
      NOTE: The combinedMask and cutPoints will show on the overlayFrame whenever there is a detection, even if that detection doesn't
      get through byteTracker for some reason. Given more time, the ideal next would've been consolidating adding any overlays to this loop.
      */
      std::vector<STrack> tracked = tracker.update(nms_objects);
      for (const auto& track : tracked) {
        // if (!track.is_activated) continue;
        int baseLine = 0;
        std::string label = cv::format("ID: %d | Conf: %.2f", track.track_id, track.score);
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::Rect box(track.tlwh[0], track.tlwh[1], track.tlwh[2], track.tlwh[3]);
        int top = std::max(box.y, labelSize.height);

        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(image, cv::Point(box.x, top - labelSize.height), cv::Point(box.x + labelSize.width, top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 2);
      }

      /*
      The output overlayFrame image is consolidated into a message and published without any form of zero-copy efficienies.
      */
      auto header{std_msgs::msg::Header()};
      header.stamp.sec = msg->stamp.sec;
      header.stamp.nanosec = msg->stamp.nanosec;
      sensor_msgs::msg::Image::SharedPtr overlayFrameMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
      overlayFramePublisher_->publish(*overlayFrameMsg);

      /*
      Because of how the gpuImg pointer is passed from CameraControlNode to detectionNode, it is entirely on 
      this program to make sure the memory is freed.
      */
      delete gpuImgPtr;
    }
  
  
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

// int main(int argc, char const *argv[])
// {
//   rclcpp::init(argc, argv);
//   try {
//     // MultiThreadedExecutor allows callbacks to run in parallel
//     rclcpp::executors::MultiThreadedExecutor executor;
//     auto node{std::make_shared<DetectionNode>(rclcpp::NodeOptions())};
//     executor.add_node(node);
//     executor.spin();
//   } catch (const std::exception& e) {
//     rclcpp::shutdown();
//     return 1;
//   }

//   rclcpp::shutdown();
//   return 0;
// }

/*
This macro line is needed to compile the node as a composableNode that can be run in the launch.py container.
*/
RCLCPP_COMPONENTS_REGISTER_NODE(DetectionNode)

