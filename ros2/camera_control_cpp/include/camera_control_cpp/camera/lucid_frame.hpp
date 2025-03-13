#pragma once

#include <opencv2/opencv.hpp>

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
  std::pair<int, int> colorFrameOffset_;
};