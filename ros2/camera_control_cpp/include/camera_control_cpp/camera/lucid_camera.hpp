#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "ArenaApi.h"
#include "BS_thread_pool.hpp"
#include "camera_control_cpp/camera/lucid_frame.hpp"

class LucidCamera {
 public:
  const std::vector<std::string> COLOR_CAMERA_MODEL_PREFIXES{
      "ATL", "ATX", "PHX", "TRI", "TRT"};
  const std::vector<std::string> DEPTH_CAMERA_MODEL_PREFIXES{"HTP", "HLT",
                                                             "HTR", "HTW"};

  enum class State { STREAMING, CONNECTING, DISCONNECTED };

  using StateChangeCallback = std::function<void(State)>;
  LucidCamera(const cv::Mat& colorCameraIntrinsicMatrix,
              const cv::Mat& colorCameraDistortionCoeffs,
              const cv::Mat& depthCameraIntrinsicMatrix,
              const cv::Mat& depthCameraDistortionCoeffs,
              const cv::Mat& xyzToColorCameraExtrinsicMatrix,
              const cv::Mat& xyzToDepthCameraExtrinsicMatrix,
              std::optional<std::string> colorCameraSerialNumber = std::nullopt,
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
  using FrameCallback = std::function<void(std::shared_ptr<LucidFrame>)>;
  void start(double exposureUs = -1.0, double gainDb = -1.0,
             FrameCallback frameCallback = nullptr);

  /**
   * Stops streaming and disconnects device.
   */
  void stop();

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

 private:
  std::condition_variable cv_;
  std::mutex cvMutex_;
  std::thread connectionThread_;
  std::thread acquisitionThread_;
  Arena::ISystem* arena_{nullptr};
  std::atomic<bool> isRunning_{false};
  cv::Mat colorCameraIntrinsicMatrix_;
  cv::Mat colorCameraDistortionCoeffs_;
  cv::Mat depthCameraIntrinsicMatrix_;
  cv::Mat depthCameraDistortionCoeffs_;
  cv::Mat xyzToColorCameraExtrinsicMatrix_;
  cv::Mat xyzToDepthCameraExtrinsicMatrix_;
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
  std::atomic<double> exposureUs_{0.0};
  std::atomic<double> gainDb_{0.0};

  void connectionThreadFn(double exposureUs, double gainDb);
  std::optional<Arena::DeviceInfo> findFirstDeviceWithModelPrefix(
      std::vector<Arena::DeviceInfo>& deviceInfos,
      const std::vector<std::string>& modelPrefixes);
  std::optional<Arena::DeviceInfo> findDeviceWithSerial(
      std::vector<Arena::DeviceInfo>& deviceInfos,
      const std::string& serialNumber);
  void startStream(const Arena::DeviceInfo& colorDeviceInfo,
                   const Arena::DeviceInfo& depthDeviceInfo,
                   std::optional<double> exposureUs = std::nullopt,
                   std::optional<double> gainDb = std::nullopt);
  void setNetworkSettings(Arena::IDevice* device);
  void stopStream();
  void callStateChangeCallback();
  void acquisitionThreadFn(FrameCallback frameCallback = nullptr);
  std::optional<cv::Mat> getColorFrame();
  std::optional<std::pair<cv::Mat, cv::Mat>> getDepthFrame();
  LucidFrame getRgbdFrame(const cv::Mat& colorFrame,
                          const cv::Mat& depthFrameXyz);
};