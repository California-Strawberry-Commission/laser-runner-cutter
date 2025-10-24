#include <cv_bridge/cv_bridge.h>

#include <atomic>
#include <condition_variable>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "detection/detector/runner_detector.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

void drawRunners(const cv::Mat& image, const std::vector<Runner>& runners,
                 const cv::Size& runnerImageSize,
                 cv::Scalar color = {255, 0, 0}, unsigned int scale = 1) {
  if (image.empty() || runnerImageSize.width <= 0 ||
      runnerImageSize.height <= 0) {
    return;
  }

  // The image we are drawing on may not be the size of the image that the
  // runner detection was run on. Thus, we'll need to scale the bbox, mask, and
  // point of each runner to the image we are drawing on.
  double xScale{static_cast<double>(image.cols) / runnerImageSize.width};
  double yScale{static_cast<double>(image.rows) / runnerImageSize.height};

  // Draw segmentation masks
  if (!runners.empty() && !runners[0].boxMask.empty()) {
    cv::Mat mask{image.clone()};
    for (const auto& runner : runners) {
      // Scale rect to current image space
      cv::Rect2f rect{runner.rect};
      rect.x = static_cast<float>(rect.x * xScale);
      rect.y = static_cast<float>(rect.y * yScale);
      rect.width = static_cast<float>(rect.width * xScale);
      rect.height = static_cast<float>(rect.height * yScale);
      cv::Rect rectInt = rect;  // implicit float -> int conversion
      cv::Rect imgRect(0, 0, image.cols, image.rows);
      cv::Rect roi{rectInt & imgRect};  // clip to image
      if (roi.width <= 0 || roi.height <= 0) {
        continue;
      }

      // Scale boxMask to current image space
      cv::Mat boxMask{runner.boxMask};
      cv::Mat resizedBoxMask;
      cv::resize(boxMask, resizedBoxMask, cv::Size(roi.width, roi.height), 0, 0,
                 cv::INTER_NEAREST);

      mask(roi).setTo(color, resizedBoxMask);
    }
    // Add all the masks to our image
    cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
  }

  // Bounding boxes and annotations
  double meanColor{cv::mean(color)[0]};
  cv::Scalar textColor{(meanColor > 128) ? cv::Scalar(0, 0, 0)
                                         : cv::Scalar(255, 255, 255)};
  cv::Scalar markerColor{255, 255, 255};
  for (const auto& runner : runners) {
    // Scale rect to current image space
    cv::Rect2f rect{runner.rect};
    rect.x = static_cast<float>(rect.x * xScale);
    rect.y = static_cast<float>(rect.y * yScale);
    rect.width = static_cast<float>(rect.width * xScale);
    rect.height = static_cast<float>(rect.height * yScale);
    cv::Rect rectInt = rect;  // implicit float -> int conversion
    cv::Rect imgRect(0, 0, image.cols, image.rows);
    cv::Rect roi{rectInt & imgRect};  // clip to image
    if (roi.width <= 0 || roi.height <= 0) {
      continue;
    }

    // Draw rectangles and text
    char text[256];
    sprintf(text, "%d: %.1f%%", runner.trackId, runner.conf * 100);
    int baseLine{0};
    cv::Size labelSize{cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5 * scale, scale, &baseLine)};
    cv::Scalar textBackgroundColor{color * 0.7};
    cv::rectangle(image, roi, color, scale + 1);
    cv::rectangle(
        image,
        cv::Rect(cv::Point(roi.x, roi.y),
                 cv::Size(labelSize.width, labelSize.height + baseLine)),
        textBackgroundColor, -1);
    cv::putText(image, text, cv::Point(roi.x, roi.y + labelSize.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, textColor, scale);

    // Draw representative point
    if (runner.point.x >= 0 && runner.point.y >= 0) {
      int x{static_cast<int>(std::round(runner.point.x * xScale))};
      int y{static_cast<int>(std::round(runner.point.y * yScale))};
      cv::drawMarker(image, cv::Point2i(x, y), markerColor,
                     cv::MARKER_TILTED_CROSS, 20, 2);
    }
  }
}

/**
 * Concurrency primitive that provides a shared flag that can be set and waited
 * on.
 */
class Event {
 public:
  Event() : flag_(false) {}

  void set() {
    std::lock_guard<std::mutex> lock(mtx_);
    flag_ = true;
    cv_.notify_all();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    flag_ = false;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return flag_; });
  }

  bool wait_for(float timeoutSecs) {
    std::unique_lock<std::mutex> lock(mtx_);
    return cv_.wait_for(lock, std::chrono::duration<float>(timeoutSecs),
                        [this] { return flag_; });
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  bool flag_;
};

class DetectionNode : public rclcpp::Node {
 public:
  explicit DetectionNode(const rclcpp::NodeOptions& options)
      : Node("detection_node", options) {
    /////////////
    // Parameters
    /////////////
    declare_parameter<int>("debug_image_width", 640);

    /////////////
    // Publishers
    /////////////
    // TODO: State
    debugImagePublisher_ = create_publisher<sensor_msgs::msg::Image>(
        "debug_image", rclcpp::SensorDataQoS());

    //////////////
    // Subscribers
    //////////////
    rclcpp::SubscriptionOptions subOptions;
    subscriberCallbackGroup_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    subOptions.callback_group = subscriberCallbackGroup_;
    colorImageSubscriber_ = create_subscription<sensor_msgs::msg::Image>(
        "color_image", rclcpp::SensorDataQoS(),
        std::bind(&DetectionNode::onColorImage, this, std::placeholders::_1),
        subOptions);

    ///////////
    // Services
    ///////////
    // TODO

    runnerDetector_ = std::make_unique<RunnerDetector>();

    detectionThread_ = std::thread(&DetectionNode::detectionThreadFn, this);
  }

  ~DetectionNode() {
    // Signal stop and join detection thread
    detectionStopSignal_ = true;
    colorImageEvent_.set();
    if (detectionThread_.joinable()) {
      detectionThread_.join();
    }
  }

 private:
#pragma region Param Helpers

  int getParamDebugImageWidth() {
    return static_cast<int>(get_parameter("debug_image_width").as_int());
  }

#pragma endregion

#pragma region Callbacks

  void onColorImage(const sensor_msgs::msg::Image::ConstSharedPtr imgMsg) {
    std::lock_guard<std::mutex> lock(lastColorImageMutex_);
    lastColorImage_ = imgMsg;
    colorImageEvent_.set();
  }

  void detectionThreadFn() {
    // Note: we use separate CUDA streams due to branching
    // GPU upload -> demosaic ---> runner detection
    //        (stream A)       |     (stream B)
    //                         └-> resize for debug frame
    //                               (stream C)
    cv::cuda::Stream cvStream0_;
    cv::cuda::Stream cvStream1_;

    // Allocate once up front to avoid overhead and fragmentation
    cv::cuda::GpuMat gpuRaw;
    cv::cuda::GpuMat gpuRgb;
    cv::cuda::GpuMat gpuDebugImage;
    cv::Mat debugImage;

    while (!detectionStopSignal_) {
      colorImageEvent_.wait();
      sensor_msgs::msg::Image::ConstSharedPtr imgMsg{nullptr};
      {
        std::lock_guard<std::mutex> lock(lastColorImageMutex_);
        imgMsg = lastColorImage_;
      }
      colorImageEvent_.clear();

      if (!imgMsg) {
        continue;
      }

      // Wrap raw Bayer bytes with stride (no copy)
      cv::Mat raw(imgMsg->height, imgMsg->width, CV_8UC1,
                  const_cast<uint8_t*>(imgMsg->data.data()),
                  static_cast<size_t>(imgMsg->step));

      // Upload image (BayerRG8) to GPU
      // Note: GpuMat::create is a no-op if size/type matches
      gpuRaw.create(imgMsg->height, imgMsg->width, CV_8UC1);
      gpuRaw.upload(raw, cvStream0_);

      // Demosaic on GPU
      gpuRgb.create(imgMsg->height, imgMsg->width, CV_8UC3);
      cv::cuda::demosaicing(gpuRaw, gpuRgb, cv::COLOR_BayerRG2RGB, -1,
                            cvStream0_);

      cvStream0_.waitForCompletion();

      ////////////
      // Detection
      ////////////
      // TODO: Control this via service calls
      auto runners{runnerDetector_->track(gpuRgb)};

      /////////////////////
      // Create debug image
      /////////////////////
      // Downscale using INTER_NEAREST for best perf
      double aspectRatio{static_cast<double>(imgMsg->height) / imgMsg->width};
      int debugImageWidth{getParamDebugImageWidth()};
      int debugImageHeight{
          static_cast<int>(std::round(debugImageWidth * aspectRatio))};
      gpuDebugImage.create(debugImageHeight, debugImageWidth, CV_8UC3);
      cv::cuda::resize(gpuRgb, gpuDebugImage,
                       cv::Size(debugImageWidth, debugImageHeight), 0.0, 0.0,
                       cv::INTER_NEAREST, cvStream1_);

      // Copy back to host
      debugImage.create(debugImageHeight, debugImageWidth, CV_8UC3);
      gpuDebugImage.download(debugImage, cvStream1_);

      // Synchronize stream before using debugImage
      cvStream1_.waitForCompletion();

      drawRunners(debugImage, runners, cv::Size(imgMsg->width, imgMsg->height));

      // Publish debug image
      auto debugImageMsg{
          cv_bridge::CvImage(imgMsg->header, "rgb8", debugImage).toImageMsg()};
      debugImagePublisher_->publish(*debugImageMsg);
    }
  }

#pragma endregion

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugImagePublisher_;
  rclcpp::CallbackGroup::SharedPtr subscriberCallbackGroup_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      colorImageSubscriber_;

  std::unique_ptr<RunnerDetector> runnerDetector_;
  std::thread detectionThread_;
  std::atomic<bool> detectionStopSignal_{false};
  sensor_msgs::msg::Image::ConstSharedPtr lastColorImage_{nullptr};
  std::mutex lastColorImageMutex_;
  // Notifies the detection thread that a new image is available
  Event colorImageEvent_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(DetectionNode)