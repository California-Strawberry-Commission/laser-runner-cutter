#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

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

  /**
   * Given an (x, y) coordinate in the color frame, return the (x, y, z)
   * position with respect to the camera.
   *
   * @param colorPixel (x, y) coordinate in the color frame.
   * @return (x, y, z) position with respect to the camera, or nullopt if
   * the position could not be determined.
   */
  std::optional<cv::Vec3f> getPosition(const cv::Point2i& colorPixel) const;

 private:
  static constexpr float DEPTH_MIN_MM{500};
  static constexpr float DEPTH_MAX_MM{10000};

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

  std::optional<cv::Point2i> getCorrespondingDepthPixel(
      const cv::Point2i& colorPixel) const;
};