#include "camera_control_cpp/camera/lucid_camera.hpp"

#include <chrono>

#include "BS_thread_pool.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "spdlog/spdlog.h"

namespace {

std::optional<Arena::DeviceInfo> findFirstDeviceWithModelPrefix(
    std::vector<Arena::DeviceInfo>& deviceInfos,
    const std::vector<std::string>& modelPrefixes) {
  auto it{std::find_if(deviceInfos.begin(), deviceInfos.end(),
                       [&modelPrefixes](Arena::DeviceInfo& deviceInfo) {
                         return std::any_of(
                             modelPrefixes.begin(), modelPrefixes.end(),
                             [&deviceInfo](const std::string& prefix) {
                               return std::strncmp(
                                          deviceInfo.ModelName().c_str(),
                                          prefix.c_str(), prefix.length()) == 0;
                             });
                       })};
  if (it != deviceInfos.end()) {
    return *it;
  }
  return std::nullopt;
}

std::optional<Arena::DeviceInfo> findDeviceWithSerial(
    std::vector<Arena::DeviceInfo>& deviceInfos,
    const std::string& serialNumber) {
  auto it{std::find_if(
      deviceInfos.begin(), deviceInfos.end(),
      [&serialNumber](Arena::DeviceInfo& deviceInfo) {
        return deviceInfo.SerialNumber().length() == serialNumber.length() &&
               std::strncmp(deviceInfo.SerialNumber().c_str(),
                            serialNumber.c_str(), serialNumber.length()) == 0;
      })};

  if (it != deviceInfos.end()) {
    return *it;
  }
  return std::nullopt;
}

}  // namespace

LucidCamera::LucidCamera(std::optional<std::string> colorCameraSerialNumber,
                         std::optional<std::string> depthCameraSerialNumber,
                         std::pair<int, int> colorFrameSize,
                         StateChangeCallback stateChangeCallback)
    : arena_(Arena::OpenSystem()),
      colorCameraSerialNumber_(std::move(colorCameraSerialNumber)),
      depthCameraSerialNumber_(std::move(depthCameraSerialNumber)),
      colorFrameSize_(colorFrameSize),
      stateChangeCallback_(std::move(stateChangeCallback)) {}

LucidCamera::~LucidCamera() {
  stop();
  Arena::CloseSystem(arena_);
}

LucidCamera::State LucidCamera::getState() const {
  std::lock_guard<std::mutex> lock(deviceMutex_);
  if (colorDevice_ && depthDevice_) {
    return LucidCamera::State::STREAMING;
  } else if (isRunning_) {
    return LucidCamera::State::CONNECTING;
  } else {
    return LucidCamera::State::DISCONNECTED;
  }
}

void LucidCamera::start(CaptureMode captureMode, double exposureUs,
                        double gainDb, ColorCallback colorCallback,
                        DepthCallback depthCallback) {
  if (isRunning_) {
    return;
  }

  isRunning_ = true;
  callStateChangeCallback();

  // Start connection thread and acquisition thread
  connectionThread_ = std::thread(&LucidCamera::connectionThreadFn, this,
                                  captureMode, exposureUs, gainDb);
  // We don't use an acquisition loop when doing single frame captures
  if (captureMode != CaptureMode::SINGLE_FRAME) {
    acquisitionThread_ = std::thread(&LucidCamera::acquisitionThreadFn, this,
                                     colorCallback, depthCallback);
  }
}

void LucidCamera::stop() {
  if (!isRunning_) {
    return;
  }

  isRunning_ = false;
  callStateChangeCallback();

  {
    // Notify all waiting threads to wake up
    std::lock_guard<std::mutex> lock(cvMutex_);
    cv_.notify_all();
  }

  // Join the threads to ensure they have finished
  if (connectionThread_.joinable()) {
    connectionThread_.join();
  }
  if (acquisitionThread_.joinable()) {
    acquisitionThread_.join();
  }
}

void LucidCamera::connectionThreadFn(CaptureMode captureMode, double exposureUs,
                                     double gainDb) {
  bool deviceConnected{false};
  bool deviceWasEverConnected{false};

  std::unique_lock<std::mutex> lock(cvMutex_);
  while (isRunning_) {
    if (deviceConnected) {
      spdlog::info("Devices found. Signaling acquisition thread");
      cv_.notify_one();
      cv_.wait(lock);
    }

    // Clean up existing connection if needed
    stopStream();

    // Create new connection
    if (isRunning_) {
      deviceConnected = false;

      // Get device infos
      arena_->UpdateDevices(1000);
      std::vector<Arena::DeviceInfo> deviceInfos{arena_->GetDevices()};

      std::optional<Arena::DeviceInfo> colorDeviceInfo{std::nullopt};
      std::optional<Arena::DeviceInfo> depthDeviceInfo{std::nullopt};

      // If we don't have a serial number of a device, attempt to find one among
      // connected devices. Otherwise, find the DeviceInfo with the desired
      // serial number.
      if (!colorCameraSerialNumber_) {
        colorDeviceInfo = findFirstDeviceWithModelPrefix(
            deviceInfos, LucidCamera::COLOR_CAMERA_MODEL_PREFIXES);
        if (colorDeviceInfo) {
          colorCameraSerialNumber_ = colorDeviceInfo.value().SerialNumber();
        }
      } else {
        colorDeviceInfo =
            findDeviceWithSerial(deviceInfos, colorCameraSerialNumber_.value());
      }
      if (!depthCameraSerialNumber_) {
        depthDeviceInfo = findFirstDeviceWithModelPrefix(
            deviceInfos, LucidCamera::DEPTH_CAMERA_MODEL_PREFIXES);
        if (depthDeviceInfo) {
          depthCameraSerialNumber_ = depthDeviceInfo.value().SerialNumber();
        }
      } else {
        depthDeviceInfo =
            findDeviceWithSerial(deviceInfos, depthCameraSerialNumber_.value());
      }

      // If the devices are connected, set up and start streaming
      if (colorDeviceInfo && depthDeviceInfo) {
        spdlog::info(
            "Device (color, model={}, serial={}, firmware={}) and device "
            "(depth, model={}, serial={}, firmware={}) found",
            colorDeviceInfo.value().ModelName(),
            colorDeviceInfo.value().SerialNumber(),
            colorDeviceInfo.value().DeviceVersion(),
            depthDeviceInfo.value().ModelName(),
            depthDeviceInfo.value().SerialNumber(),
            depthDeviceInfo.value().DeviceVersion());
        // Start the stream. Only set exposure/gain if this is the first time
        // the device is connected
        if (deviceWasEverConnected) {
          startStream(colorDeviceInfo.value(), depthDeviceInfo.value(),
                      captureMode);
        } else {
          startStream(colorDeviceInfo.value(), depthDeviceInfo.value(),
                      captureMode, exposureUs, gainDb);
        }
        deviceWasEverConnected = true;
        deviceConnected = true;
        {
          std::lock_guard<std::mutex> lock(streamingStateCvMutex_);
          streamingStateCv_.notify_all();
        }
      } else {
        spdlog::warn("Either color device or depth device was not found");
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
    }
  }

  // Clean up existing connection
  stopStream();
  spdlog::info("Terminating connection thread");
}

void LucidCamera::startStream(const Arena::DeviceInfo& colorDeviceInfo,
                              const Arena::DeviceInfo& depthDeviceInfo,
                              CaptureMode captureMode,
                              std::optional<double> exposureUs,
                              std::optional<double> gainDb) {
  {
    std::lock_guard<std::mutex> lock(deviceMutex_);
    colorDevice_ = arena_->CreateDevice(colorDeviceInfo);
    depthDevice_ = arena_->CreateDevice(depthDeviceInfo);
  }

  // Note: A camera's node list can be found in the camera's Technical Reference
  // Manual such as https://support.thinklucid.com/triton-tri032s/.
  // To get more in depth information about a node (such as type and accepted
  // values), you can use the Arena SDK's precompiled example
  // `C_Explore_NodeTypes`:
  // `/opt/ArenaSDK/ArenaSDK_Linux_x64/precompiledExamples/C_Explore_NodeTypes`

  /////////////////////////////////
  // Configure color device nodemap
  /////////////////////////////////
  setNetworkSettings(colorDevice_);
  // Set frame size and pixel format
  // Use BayerRG (RGGB pattern) to achieve streaming at 30 FPS at max
  // resolution. We will demosaic to RGB on the host device
  Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                         "PixelFormat", "BayerRG8");
  // Reset ROI (Region of Interest) offset, as it persists on the device
  Arena::SetNodeValue<int64_t>(colorDevice_->GetNodeMap(), "OffsetX", 0);
  Arena::SetNodeValue<int64_t>(colorDevice_->GetNodeMap(), "OffsetY", 0);
  auto maxWidth{
      Arena::GetNodeMax<int64_t>(colorDevice_->GetNodeMap(), "Width")};
  auto maxHeight{
      Arena::GetNodeMax<int64_t>(colorDevice_->GetNodeMap(), "Height")};
  // Check that the desired color frame size is valid before attempting to set
  if (colorFrameSize_.first <= 0 || maxWidth < colorFrameSize_.first ||
      colorFrameSize_.second <= 0 || maxHeight < colorFrameSize_.second) {
    std::ostringstream errorMsg;
    errorMsg << "Invalid color frame size specified: (" << colorFrameSize_.first
             << ", " << colorFrameSize_.second << "). Max size is (" << maxWidth
             << ", " << maxHeight << ")";
    throw std::invalid_argument(errorMsg.str());
  }
  Arena::SetNodeValue<int64_t>(colorDevice_->GetNodeMap(), "Width",
                               colorFrameSize_.first);
  Arena::SetNodeValue<int64_t>(colorDevice_->GetNodeMap(), "Height",
                               colorFrameSize_.second);
  // Set the ROI offset to be the center of the full frame
  colorFrameOffset_ = {(maxWidth - colorFrameSize_.first) / 2,
                       (maxHeight - colorFrameSize_.second) / 2};
  Arena::SetNodeValue<int64_t>(colorDevice_->GetNodeMap(), "OffsetX",
                               colorFrameOffset_.first);
  Arena::SetNodeValue<int64_t>(colorDevice_->GetNodeMap(), "OffsetY",
                               colorFrameOffset_.second);

  /////////////////////////////////
  // Configure depth device nodemap
  /////////////////////////////////
  setNetworkSettings(depthDevice_);
  // Set pixel format
  depthFrameSize_ = {
      Arena::GetNodeValue<int64_t>(depthDevice_->GetNodeMap(), "Width"),
      Arena::GetNodeValue<int64_t>(depthDevice_->GetNodeMap(), "Height")};
  Arena::SetNodeValue<GenICam::gcstring>(depthDevice_->GetNodeMap(),
                                         "PixelFormat", "Coord3D_ABCY16");
  // Set Scan 3D node values
  Arena::SetNodeValue<GenICam::gcstring>(depthDevice_->GetNodeMap(),
                                         "Scan3dOperatingMode",
                                         "Distance3000mmSingleFreq");
  Arena::SetNodeValue<GenICam::gcstring>(depthDevice_->GetNodeMap(),
                                         "ExposureTimeSelector", "Exp88Us");
  xyzScale_ = Arena::GetNodeValue<double>(depthDevice_->GetNodeMap(),
                                          "Scan3dCoordinateScale");
  Arena::SetNodeValue<GenICam::gcstring>(
      depthDevice_->GetNodeMap(), "Scan3dCoordinateSelector", "CoordinateA");
  double xOffset{Arena::GetNodeValue<double>(depthDevice_->GetNodeMap(),
                                             "Scan3dCoordinateOffset")};
  Arena::SetNodeValue<GenICam::gcstring>(
      depthDevice_->GetNodeMap(), "Scan3dCoordinateSelector", "CoordinateB");
  double yOffset{Arena::GetNodeValue<double>(depthDevice_->GetNodeMap(),
                                             "Scan3dCoordinateOffset")};
  Arena::SetNodeValue<GenICam::gcstring>(
      depthDevice_->GetNodeMap(), "Scan3dCoordinateSelector", "CoordinateC");
  double zOffset{Arena::GetNodeValue<double>(depthDevice_->GetNodeMap(),
                                             "Scan3dCoordinateOffset")};
  xyzOffset_ = {xOffset, yOffset, zOffset};
  // Set confidence threshold
  Arena::SetNodeValue<bool>(depthDevice_->GetNodeMap(),
                            "Scan3dConfidenceThresholdEnable", true);
  Arena::SetNodeValue<int64_t>(depthDevice_->GetNodeMap(),
                               "Scan3dConfidenceThresholdMin", 500);

  /////////////////
  // Configure sync
  /////////////////
  Arena::SetNodeValue<bool>(colorDevice_->GetNodeMap(), "PtpEnable", false);
  Arena::SetNodeValue<bool>(depthDevice_->GetNodeMap(), "PtpEnable", false);
  Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                         "AcquisitionStartMode", "Normal");
  Arena::SetNodeValue<GenICam::gcstring>(depthDevice_->GetNodeMap(),
                                         "AcquisitionStartMode", "Normal");
  // TODO: Once cameras are hardware synced via trigger signals, set appropriate
  // packet delay and frame transmission delay. See
  // https://support.thinklucid.com/app-note-bandwidth-sharing-in-multi-camera-systems/
  // With packet size of 9000 bytes on a 1 Gbps link, the packet delay is:
  // (9000 bytes * 8 ns/byte) * 1.1 buffer = 79200 ns
  Arena::SetNodeValue<int64_t>(colorDevice_->GetNodeMap(), "GevSCFTD", 0);
  Arena::SetNodeValue<int64_t>(depthDevice_->GetNodeMap(), "GevSCFTD", 0);
  Arena::SetNodeValue<int64_t>(colorDevice_->GetNodeMap(), "GevSCPD", 80);
  Arena::SetNodeValue<int64_t>(depthDevice_->GetNodeMap(), "GevSCPD", 80);
  // Select GPIO line to output strobe signal on color camera
  // See https://support.thinklucid.com/app-note-using-gpio-on-lucid-cameras/
  Arena::SetNodeValue<GenICam::gcstring>(
      colorDevice_->GetNodeMap(), "LineSelector",
      "Line3");  // non-isolated bi-directional GPIO channel
  Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(), "LineMode",
                                         "Output");
  Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                         "LineSource", "ExposureActive");
  Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                         "LineSelector",
                                         "Line1");  // opto-isolated output
  Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(), "LineMode",
                                         "Output");
  Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                         "LineSource", "ExposureActive");
  // TODO: Enable trigger mode on depth camera. See
  // https://support.thinklucid.com/app-note-using-gpio-on-lucid-cameras/#config

  captureMode_ = captureMode;
  Arena::SetNodeValue<GenICam::gcstring>(
      colorDevice_->GetNodeMap(), "AcquisitionMode",
      captureMode == CaptureMode::SINGLE_FRAME ? "SingleFrame" : "Continuous");
  Arena::SetNodeValue<GenICam::gcstring>(
      depthDevice_->GetNodeMap(), "AcquisitionMode",
      captureMode == CaptureMode::SINGLE_FRAME ? "SingleFrame" : "Continuous");

  ////////////////////////
  // Set exposure and gain
  ////////////////////////
  if (exposureUs) {
    setExposureUs(exposureUs.value());
  }
  if (gainDb) {
    setGainDb(gainDb.value());
  }

  ////////////////
  // Start streams
  ////////////////
  size_t numBuffers{
      static_cast<size_t>(captureMode == CaptureMode::SINGLE_FRAME ? 1 : 10)};
  colorDevice_->StartStream(numBuffers);
  spdlog::info(
      "Device (color, serial={}) is now streaming with resolution ({}, {})",
      colorCameraSerialNumber_.value(), colorFrameSize_.first,
      colorFrameSize_.second);
  depthDevice_->StartStream(numBuffers);
  spdlog::info(
      "Device (depth, serial={}) is now streaming with resolution ({}, {})",
      depthCameraSerialNumber_.value(), depthFrameSize_.first,
      depthFrameSize_.second);
  callStateChangeCallback();
}

void LucidCamera::setNetworkSettings(Arena::IDevice* device) {
  // Setting the buffer handling mode to "NewestOnly" ensures the most recent
  // image is delivered, even if it means skipping frames
  Arena::SetNodeValue<GenICam::gcstring>(
      device->GetTLStreamNodeMap(), "StreamBufferHandlingMode", "NewestOnly");
  // Enable stream auto negotiate packet size, which instructs the camera to
  // receive the largest packet size that the system will allow
  Arena::SetNodeValue<bool>(device->GetTLStreamNodeMap(),
                            "StreamAutoNegotiatePacketSize", true);
  // Enable stream packet resend. If a packet is missed while receiving an
  // image, a packet resend is requested which retrieves and redelivers the
  // missing packet in the correct order.
  Arena::SetNodeValue<bool>(device->GetTLStreamNodeMap(),
                            "StreamPacketResendEnable", true);
  // Set the following when Persistent IP is set on the camera
  Arena::SetNodeValue<bool>(device->GetNodeMap(),
                            "GevPersistentARPConflictDetectionEnable", false);
}

LucidCamera::CaptureMode LucidCamera::getCaptureMode() const {
  return captureMode_;
}

std::pair<int, int> LucidCamera::getColorFrameSize() const {
  return colorFrameSize_;
}

std::pair<int, int> LucidCamera::getDepthFrameSize() const {
  return depthFrameSize_;
}

double LucidCamera::getExposureUs() const {
  if (getState() != LucidCamera::State::STREAMING) {
    return 0.0;
  }

  return exposureUs_;
}

void LucidCamera::setExposureUs(double exposureUs) {
  if (getState() != LucidCamera::State::STREAMING) {
    return;
  }

  if (exposureUs < 0.0) {
    // Set auto exposure
    exposureUs_ = -1.0;
    Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                           "ExposureAuto", "Continuous");
    spdlog::info("Auto exposure set");
  } else if (colorDevice_->GetNodeMap()->GetNode("ExposureTime")) {
    Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                           "ExposureAuto", "Off");
    exposureUs_ = std::clamp(
        exposureUs,
        Arena::GetNodeMin<double>(colorDevice_->GetNodeMap(), "ExposureTime"),
        Arena::GetNodeMax<double>(colorDevice_->GetNodeMap(), "ExposureTime"));
    Arena::SetNodeValue<double>(colorDevice_->GetNodeMap(), "ExposureTime",
                                exposureUs_);
    spdlog::info("Exposure set to {} us", exposureUs_);
  }
}

std::pair<double, double> LucidCamera::getExposureUsRange() const {
  if (getState() != LucidCamera::State::STREAMING ||
      !colorDevice_->GetNodeMap()->GetNode("ExposureTime")) {
    return {0.0, 0.0};
  }

  return {
      Arena::GetNodeMin<double>(colorDevice_->GetNodeMap(), "ExposureTime"),
      Arena::GetNodeMax<double>(colorDevice_->GetNodeMap(), "ExposureTime")};
}

double LucidCamera::getGainDb() const {
  if (getState() != LucidCamera::State::STREAMING) {
    return 0.0;
  }

  return gainDb_;
}

void LucidCamera::setGainDb(double gainDb) {
  if (getState() != LucidCamera::State::STREAMING) {
    return;
  }

  if (gainDb < 0.0) {
    // Set auto gain
    gainDb_ = -1.0;
    Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                           "GainAuto", "Continuous");
    spdlog::info("Auto gain set");
  } else if (colorDevice_->GetNodeMap()->GetNode("Gain")) {
    Arena::SetNodeValue<GenICam::gcstring>(colorDevice_->GetNodeMap(),
                                           "GainAuto", "Off");
    gainDb_ = std::clamp(
        gainDb, Arena::GetNodeMin<double>(colorDevice_->GetNodeMap(), "Gain"),
        Arena::GetNodeMax<double>(colorDevice_->GetNodeMap(), "Gain"));
    Arena::SetNodeValue<double>(colorDevice_->GetNodeMap(), "Gain", gainDb_);
    spdlog::info("Gain set to {} dB", gainDb_);
  }
}

std::pair<double, double> LucidCamera::getGainDbRange() const {
  if (getState() != LucidCamera::State::STREAMING ||
      !colorDevice_->GetNodeMap()->GetNode("Gain")) {
    return {0.0, 0.0};
  }

  return {Arena::GetNodeMin<double>(colorDevice_->GetNodeMap(), "Gain"),
          Arena::GetNodeMax<double>(colorDevice_->GetNodeMap(), "Gain")};
}

double LucidCamera::getColorDeviceTemperature() const {
  if (getState() != LucidCamera::State::STREAMING ||
      !colorDevice_->GetNodeMap()->GetNode("DeviceTemperature")) {
    return 0.0;
  }

  return Arena::GetNodeValue<double>(colorDevice_->GetNodeMap(),
                                     "DeviceTemperature");
}

double LucidCamera::getDepthDeviceTemperature() const {
  if (getState() != LucidCamera::State::STREAMING ||
      !depthDevice_->GetNodeMap()->GetNode("DeviceTemperature")) {
    return 0.0;
  }

  return Arena::GetNodeValue<double>(depthDevice_->GetNodeMap(),
                                     "DeviceTemperature");
}

void LucidCamera::waitForStreaming() {
  std::unique_lock<std::mutex> lock(streamingStateCvMutex_);
  if (getState() != State::STREAMING) {
    // Wait up to 10 seconds
    streamingStateCv_.wait_for(lock, std::chrono::seconds(10), [this]() {
      return getState() == State::STREAMING;
    });
  }

  if (getState() != State::STREAMING) {
    throw std::runtime_error("Camera failed to start streaming");
  }
}

std::optional<LucidCamera::Frame> LucidCamera::getNextFrame() {
  if (getState() != LucidCamera::State::STREAMING) {
    return std::nullopt;
  }

  // When in SingleFrame mode, manually fire AcquisitionStart and
  // AcquisitionStop signals
  if (captureMode_ == CaptureMode::SINGLE_FRAME) {
    Arena::ExecuteNode(colorDevice_->GetNodeMap(), "AcquisitionStart");
    Arena::ExecuteNode(depthDevice_->GetNodeMap(), "AcquisitionStart");
    // Sleep a bit to let the acquisition signal propagate
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
  }

  std::optional<Frame> frameOpt;
  try {
    auto colorFrame{getColorFrame()};
    auto depthFrameOpt{getDepthFrame()};
    if (colorFrame && depthFrameOpt) {
      frameOpt =
          Frame{std::move(colorFrame), std::move(depthFrameOpt.value().xyz),
                std::move(depthFrameOpt.value().intensity)};
    }
  } catch (const std::exception& e) {
    spdlog::error(
        "There was an issue with the camera: {}. Signaling connection thread",
        e.what());
    cv_.notify_one();
    return std::nullopt;
  }

  if (!frameOpt.has_value()) {
    spdlog::error("No frame available. Signaling connection thread");
    cv_.notify_one();
    return std::nullopt;
  }

  if (captureMode_ == CaptureMode::SINGLE_FRAME) {
    Arena::ExecuteNode(colorDevice_->GetNodeMap(), "AcquisitionStop");
    Arena::ExecuteNode(depthDevice_->GetNodeMap(), "AcquisitionStop");
  }

  return frameOpt;
}

void LucidCamera::stopStream() {
  {
    std::lock_guard<std::mutex> lock(deviceMutex_);
    if (colorDevice_) {
      arena_->DestroyDevice(colorDevice_);
      colorDevice_ = nullptr;
    }
    if (depthDevice_) {
      arena_->DestroyDevice(depthDevice_);
      depthDevice_ = nullptr;
    }
  }
  callStateChangeCallback();
}

void LucidCamera::callStateChangeCallback() {
  if (stateChangeCallback_) {
    stateChangeCallback_(getState());
  }
}

void LucidCamera::acquisitionThreadFn(ColorCallback colorCallback,
                                      DepthCallback depthCallback) {
  BS::thread_pool threadPool;

  std::unique_lock<std::mutex> lock(cvMutex_);
  while (isRunning_) {
    try {
      auto colorFrameFuture{threadPool.submit_task([this, colorCallback] {
        auto colorFrame{getColorFrame()};
        if (!colorFrame) {
          return false;
        }

        if (colorCallback) {
          colorCallback(std::move(colorFrame));
        }
        return true;
      })};

      auto depthFrameFuture{threadPool.submit_task([this, depthCallback] {
        auto depthFrameOpt{getDepthFrame()};
        if (!depthFrameOpt) {
          return false;
        }

        if (depthCallback) {
          depthCallback(std::move(depthFrameOpt.value().xyz),
                        std::move(depthFrameOpt.value().intensity));
        }
        return true;
      })};

      bool colorFrameSuccess{colorFrameFuture.get()};
      bool depthFrameSuccess{depthFrameFuture.get()};

      if (!colorFrameSuccess || !depthFrameSuccess) {
        spdlog::error("No frame available. Signaling connection thread");
        cv_.notify_one();
        cv_.wait(lock);
      }
    } catch (...) {
      spdlog::error(
          "There was an issue with the camera. Signaling connection thread");
      cv_.notify_one();
      cv_.wait(lock);
      continue;
    }
  }

  spdlog::info("Terminating acquisition thread");
}

sensor_msgs::msg::Image::UniquePtr LucidCamera::getColorFrame() {
  if (!colorDevice_) {
    return nullptr;
  }

  // Note: `Arena::IDevice::GetImage()` must be called after
  // `Arena::IDevice::StartStream()` and before `Arena::IDevice::StopStream()`
  // (or `Arena::ISystem::DestroyDevice()`), and buffers must be requeued
  Arena::IImage* image{colorDevice_->GetImage(10000)};
  const size_t height{image->GetHeight()};
  const size_t width{image->GetWidth()};
  const uint8_t* imageData{static_cast<const uint8_t*>(image->GetData())};

  // Note: we only do a single copy from the Arena buffer into an Image message,
  // and we leverage zero-copy intra-process comms to publish the image data to
  // other nodes
  auto imageMsg{std::make_unique<sensor_msgs::msg::Image>()};
  imageMsg->header.stamp = rclcpp::Clock().now();
  imageMsg->header.frame_id = "triton_color_camera";
  imageMsg->height = height;
  imageMsg->width = width;
  imageMsg->encoding = sensor_msgs::image_encodings::BAYER_RGGB8;
  imageMsg->is_bigendian = false;
  imageMsg->step = width;  // 1 byte per pixel for BayerRG8

  // Copy image data from Arena buffer to imageMsg
  const size_t totalBytes{imageMsg->step * height};
  imageMsg->data.resize(totalBytes);
  std::memcpy(imageMsg->data.data(), imageData, totalBytes);

  // Return buffer to Arena
  colorDevice_->RequeueBuffer(image);

  return imageMsg;
}

std::optional<LucidCamera::GetDepthFrameResult> LucidCamera::getDepthFrame() {
  if (!depthDevice_) {
    return std::nullopt;
  }

  // Note: `Arena::IDevice::GetImage()` must be called after
  // `Arena::IDevice::StartStream()` and before `Arena::IDevice::StopStream()`
  // (or `Arena::ISystem::DestroyDevice()`), and buffers must be requeued
  Arena::IImage* image{depthDevice_->GetImage(10000)};
  const size_t height{image->GetHeight()};
  const size_t width{image->GetWidth()};
  // Note: For Coord3D_ABCY16, each pixel is 4 U16 values
  const uint16_t* imageData{
      reinterpret_cast<const uint16_t*>(image->GetData())};

  // Note: we only do a single copy from the Arena buffer into Image messages,
  // and we leverage zero-copy intra-process comms to publish the data to
  // other nodes

  // Prepare Image messages
  auto xyzMsg{std::make_unique<sensor_msgs::msg::Image>()};
  xyzMsg->header.stamp = rclcpp::Clock().now();
  xyzMsg->header.frame_id = "helios_depth_camera";
  xyzMsg->height = height;
  xyzMsg->width = width;
  xyzMsg->encoding = sensor_msgs::image_encodings::TYPE_32FC3;
  xyzMsg->is_bigendian = false;
  xyzMsg->step = static_cast<uint32_t>(width * 3 * sizeof(float));

  auto intensityMsg{std::make_unique<sensor_msgs::msg::Image>()};
  intensityMsg->header = xyzMsg->header;
  intensityMsg->height = height;
  intensityMsg->width = width;
  intensityMsg->encoding = sensor_msgs::image_encodings::MONO16;
  intensityMsg->is_bigendian = false;
  intensityMsg->step = static_cast<uint32_t>(width * sizeof(uint16_t));

  // Prepare destination data buffers
  xyzMsg->data.resize(xyzMsg->step * height);
  intensityMsg->data.resize(intensityMsg->step * height);
  float* xyzPtr{reinterpret_cast<float*>(xyzMsg->data.data())};
  uint16_t* intensityPtr{
      reinterpret_cast<uint16_t*>(intensityMsg->data.data())};

  float scale{static_cast<float>(xyzScale_)};
  auto [xOffset, yOffset, zOffset]{xyzOffset_};
  size_t numPixels{static_cast<size_t>(width) * height};
  for (size_t pxIdx = 0; pxIdx < numPixels; ++pxIdx) {
    const uint16_t rawX{imageData[pxIdx * 4 + 0]};
    const uint16_t rawY{imageData[pxIdx * 4 + 1]};
    const uint16_t rawZ{imageData[pxIdx * 4 + 2]};
    const uint16_t rawI{imageData[pxIdx * 4 + 3]};

    intensityPtr[pxIdx] = rawI;

    // In unsigned pixel formats (such as ABCY16), values below the confidence
    // threshold will have their x, y, z, and intensity values set to 0xFFFF
    // (denoting invalid). For these invalid pixels, set (x, y, z) to (-1, -1,
    // -1).
    if (rawI == 0xFFFFu) {
      // Invalid
      xyzPtr[pxIdx * 3 + 0] = -1.0f;
      xyzPtr[pxIdx * 3 + 1] = -1.0f;
      xyzPtr[pxIdx * 3 + 2] = -1.0f;
    } else {
      // Apply scale, then offset
      xyzPtr[pxIdx * 3 + 0] = rawX * scale + static_cast<float>(xOffset);
      xyzPtr[pxIdx * 3 + 1] = rawY * scale + static_cast<float>(yOffset);
      xyzPtr[pxIdx * 3 + 2] = rawZ * scale + static_cast<float>(zOffset);
    }
  }

  // Return buffer to Arena
  depthDevice_->RequeueBuffer(image);

  return GetDepthFrameResult{std::move(xyzMsg), std::move(intensityMsg)};
}
