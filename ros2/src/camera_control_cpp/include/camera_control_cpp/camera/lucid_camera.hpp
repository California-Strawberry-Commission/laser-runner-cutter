#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

#include "ArenaApi.h"
#include "BS_thread_pool.hpp"
#include "sensor_msgs/msg/image.hpp"

struct Frame {
  sensor_msgs::msg::Image::UniquePtr colorImage;
  sensor_msgs::msg::Image::UniquePtr depthXyz;
  sensor_msgs::msg::Image::UniquePtr depthIntensity;
};

class LucidCamera {
 public:
  const std::vector<std::string> COLOR_CAMERA_MODEL_PREFIXES{
      "ATL", "ATX", "PHX", "TRI", "TRT"};
  const std::vector<std::string> DEPTH_CAMERA_MODEL_PREFIXES{"HTP", "HLT",
                                                             "HTR", "HTW"};

  enum class State { STREAMING, CONNECTING, DISCONNECTED };
  enum class CaptureMode { CONTINUOUS, SINGLE_FRAME };

  using StateChangeCallback = std::function<void(State)>;
  LucidCamera(std::optional<std::string> colorCameraSerialNumber = std::nullopt,
              std::optional<std::string> depthCameraSerialNumber = std::nullopt,
              std::pair<int, int> colorFrameSize = {2048, 1536},
              StateChangeCallback stateChangeCallback = nullptr);
  ~LucidCamera();

  State getState() const;

  /**
   * Connects device and starts streaming.
   *
   * @param exposureUs Exposure time in microseconds. A negative value sets auto
   * exposure.
   * @param gainDb Gain level in dB. A negative value sets auto gain.
   * @param frameCallback Callback that gets called when a new frame is
   * available.
   */
  using FrameCallback = std::function<void(Frame)>;
  void start(CaptureMode captureMode = CaptureMode::CONTINUOUS,
             double exposureUs = -1.0, double gainDb = -1.0,
             FrameCallback frameCallback = nullptr);

  /**
   * Stops streaming and disconnects device.
   */
  void stop();

  /**
   * @return Capture mode of the cameras.
   */
  CaptureMode getCaptureMode() const;

  /**
   * @return Exposure time of the color camera in microseconds.
   */
  double getExposureUs() const;

  /**
   * Set the exposure time of the color camera.
   *
   * @param exposureUs Exposure time in microseconds. A negative value sets auto
   * exposure.
   */
  void setExposureUs(double exposureUs);

  /**
   * @return (min, max) exposure times of the color camera in microseconds.
   */
  std::pair<double, double> getExposureUsRange() const;

  /**
   * @return Gain level of the color camera in dB.
   */
  double getGainDb() const;

  /**
   * Set the gain level of the color camera.
   *
   * @param gainDb Gain level in dB.
   */
  void setGainDb(double gainDb);

  /**
   * @return (min, max) gain levels of the color camera in dB.
   */
  std::pair<double, double> getGainDbRange() const;

  /**
   * @return Color device temperature, in Celsius.
   */
  double getColorDeviceTemperature() const;

  /**
   * @return Depth device temperature, in Celsius.
   */
  double getDepthDeviceTemperature() const;

  /**
   * Wait until the cameras are streaming.
   */
  void waitForStreaming();

  /**
   * Get the next available frame. Call this method to trigger frame acquisition
   * when in SingleFrame mode.
   *
   * @return The next available frame.
   */
  std::optional<Frame> getNextFrame();

 private:
  std::condition_variable
      cv_;  // Used for syncing acquisition and connection threads
  std::mutex cvMutex_;
  std::thread connectionThread_;
  std::thread acquisitionThread_;
  std::condition_variable
      streamingStateCv_;  // Used for notifying that cameras are streaming
  std::mutex streamingStateCvMutex_;
  Arena::ISystem* arena_{nullptr};
  std::atomic<bool> isRunning_{false};
  std::optional<std::string> colorCameraSerialNumber_{std::nullopt};
  std::optional<std::string> depthCameraSerialNumber_{std::nullopt};
  std::pair<int, int> colorFrameSize_{0, 0};
  StateChangeCallback stateChangeCallback_ = nullptr;
  mutable std::mutex deviceMutex_;
  Arena::IDevice* colorDevice_ = nullptr;
  Arena::IDevice* depthDevice_ = nullptr;
  std::pair<int, int> colorFrameOffset_{0, 0};
  std::pair<int, int> depthFrameSize_{0, 0};
  double xyzScale_{0.0};
  std::tuple<double, double, double> xyzOffset_{0.0, 0.0, 0.0};
  CaptureMode captureMode_{CaptureMode::CONTINUOUS};
  std::atomic<double> exposureUs_{0.0};
  std::atomic<double> gainDb_{0.0};

  void connectionThreadFn(CaptureMode captureMode, double exposureUs,
                          double gainDb);
  std::optional<Arena::DeviceInfo> findFirstDeviceWithModelPrefix(
      std::vector<Arena::DeviceInfo>& deviceInfos,
      const std::vector<std::string>& modelPrefixes);
  std::optional<Arena::DeviceInfo> findDeviceWithSerial(
      std::vector<Arena::DeviceInfo>& deviceInfos,
      const std::string& serialNumber);
  void startStream(const Arena::DeviceInfo& colorDeviceInfo,
                   const Arena::DeviceInfo& depthDeviceInfo,
                   CaptureMode captureMode = CaptureMode::CONTINUOUS,
                   std::optional<double> exposureUs = std::nullopt,
                   std::optional<double> gainDb = std::nullopt);
  void setNetworkSettings(Arena::IDevice* device);
  void stopStream();
  void callStateChangeCallback();
  void acquisitionThreadFn(FrameCallback frameCallback = nullptr);
  sensor_msgs::msg::Image::UniquePtr getColorFrame();
  struct GetDepthFrameResult {
    sensor_msgs::msg::Image::UniquePtr xyz;
    sensor_msgs::msg::Image::UniquePtr intensity;
  };
  std::optional<GetDepthFrameResult> getDepthFrame();
};