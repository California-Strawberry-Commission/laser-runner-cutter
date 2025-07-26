#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include "spdlog/spdlog.h"

#include "camera_control_cpp/camera/calibrate.hpp"

// General min and max possible depths
#define DEPTH_MIN_MM 500
#define DEPTH_MAX_MM 10000

class LucidFrame {
 public:
  LucidFrame(const cv::Mat& colorFrame, const cv::Mat& depthFrameXyz,
             double timestampMillis, const cv::Mat& colorCameraIntrinsicMatrix,
             const cv::Mat& colorCameraDistortionCoeffs,
             const cv::Mat& depthCameraIntrinsicMatrix,
             const cv::Mat& depthCameraDistortionCoeffs,
             const cv::Mat& xyzToColorCameraExtrinsicMatrix,
             const cv::Mat& xyzToDepthCameraExtrinsicMatrix,
             std::pair<int, int> colorFrameOffset = {0, 0});
  ~LucidFrame();

  const cv::Mat& getColorFrame() const { return colorFrame_; }
  cv::Mat getDepthFrame() const;
  const cv::Mat& getDepthFrameXyz() const { return depthFrameXyz_; }
  double getTimestampMillis() const { return timestampMillis_; }


  cv::Point2i getCorrespondingDepthPixel(const cv::Point2i& colorPixel) const;

  /**
   * Given an (x, y) coordinate in the color frame, return the (x, y, z)
   * position with respect to the camera.
   *
   * @param colorPixel (x, y) coordinate in the color frame.
   * @return (x, y, z) position with respect to the camera, or nullopt if
   * the position could not be determined.
   */
  std::optional<std::tuple<double, double, double>> getPosition(
      std::pair<int, int> colorPixel) const;

 private:
  cv::Mat colorFrame_;
  cv::Mat depthFrameXyz_;
  double timestampMillis_{0.0};
  cv::Mat colorCameraIntrinsicMatrix_;
  cv::Mat colorCameraDistortionCoeffs_;
  cv::Mat depthCameraIntrinsicMatrix_;
  cv::Mat depthCameraDistortionCoeffs_;
  cv::Mat xyzToColorCameraExtrinsicMatrix_;
  cv::Mat xyzToDepthCameraExtrinsicMatrix_;
  cv::Mat colorToDepthExtrinsicMatrix_;
  std::pair<int, int> colorFrameOffset_;

  cv::Vec3f deprojectPixel(const cv::Point2i& pixel, float depth,
                           const cv::Mat& cameraMatrix,
                           const cv::Mat& distCoeffs) const;
  cv::Vec3f transformPosition(const cv::Vec3f& position,
                              const cv::Mat& extrinsic) const;
  cv::Point2i projectPosition(const cv::Vec3f& position,
                              const cv::Mat& cameraMatrix,
                              const cv::Mat& distCoeffs,
                              const cv::Mat& extrinsicMatrix = cv::Mat()) const;
  cv::Point2i adjustPixelToBounds(const cv::Point2i& pixel, int width,
                                  int height) const;
  cv::Point2i nextPixelInLine(const cv::Point2i& curr, const cv::Point2i& start,
                              const cv::Point2i& end) const;
  bool isPixelInLine(const cv::Point2i& curr, const cv::Point2i& start,
                     const cv::Point2i& end) const;
};