#include "camera_control_cpp/camera/lucid_frame.hpp"

LucidFrame::LucidFrame(const cv::Mat& colorFrame, const cv::Mat& depthFrameXyz,
                       double timestampMillis,
                       const cv::Mat& colorCameraIntrinsicMatrix,
                       const cv::Mat& colorCameraDistortionCoeffs,
                       const cv::Mat& depthCameraIntrinsicMatrix,
                       const cv::Mat& depthCameraDistortionCoeffs,
                       const cv::Mat& xyzToColorCameraExtrinsicMatrix,
                       const cv::Mat& xyzToDepthCameraExtrinsicMatrix,
                       std::pair<int, int> colorFrameOffset)
    : colorFrame_(colorFrame),
      depthFrameXyz_(depthFrameXyz),
      timestampMillis_(timestampMillis),
      colorCameraIntrinsicMatrix_(colorCameraIntrinsicMatrix),
      colorCameraDistortionCoeffs_(colorCameraDistortionCoeffs),
      depthCameraIntrinsicMatrix_(depthCameraIntrinsicMatrix),
      depthCameraDistortionCoeffs_(depthCameraDistortionCoeffs),
      xyzToColorCameraExtrinsicMatrix_(xyzToColorCameraExtrinsicMatrix),
      xyzToDepthCameraExtrinsicMatrix_(xyzToDepthCameraExtrinsicMatrix),
      colorFrameOffset_(colorFrameOffset) {}

LucidFrame::~LucidFrame() {}

cv::Mat LucidFrame::getDepthFrame() const {
  // Split into separate X, Y, Z components
  std::vector<cv::Mat> channels(3);
  cv::split(depthFrameXyz_, channels);

  // Compute the L2 norm
  cv::Mat depthFrame;
  depthFrame = channels[0].mul(channels[0]) + channels[1].mul(channels[1]) +
               channels[2].mul(channels[2]);
  cv::sqrt(depthFrame, depthFrame);

  // Convert to 16-bit unsigned integer (mono16 format)
  cv::Mat depthFrameMono16;
  depthFrame.convertTo(depthFrameMono16, CV_16U);

  return depthFrameMono16;
}

std::optional<std::tuple<double, double, double>> LucidFrame::getPosition(
    std::pair<int, int> colorPixel) const {
  // TODO: implement this
  return std::nullopt;
}