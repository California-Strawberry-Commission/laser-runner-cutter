#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

class RgbdAlignment {
 public:
  static constexpr float DEPTH_MIN_MM{400};
  static constexpr float DEPTH_MAX_MM{2000};

  RgbdAlignment(const cv::Mat& colorCameraIntrinsicMatrix,
                const cv::Mat& colorCameraDistortionCoeffs,
                const cv::Mat& depthCameraIntrinsicMatrix,
                const cv::Mat& depthCameraDistortionCoeffs,
                const cv::Mat& xyzToColorCameraExtrinsicMatrix,
                const cv::Mat& xyzToDepthCameraExtrinsicMatrix,
                std::pair<int, int> colorFrameOffset = {0, 0});
  ~RgbdAlignment() = default;

  /**
   * Given an (x, y) coordinate in the color frame, return the corresponding
   * (x, y) coordinate in the depth frame.
   *
   * @param colorPixel (x, y) coordinate in the color frame.
   * @param depthXyz Depth camera xyz data, with shape (h x w x 3)
   * @return Corresponding (x, y) coordinate in the depth frame, or nullopt
   * if it could not be determined.
   */
  std::optional<cv::Point2i> getCorrespondingDepthPixel(
      const cv::Point2i& colorPixel, const cv::Mat& depthXyz) const;

  /**
   * Given an (x, y) coordinate in the color frame, return the (x, y, z)
   * position with respect to the camera.
   *
   * @param colorPixel (x, y) coordinate in the color frame.
   * @param depthXyz Depth camera xyz data, with shape (h x w x 3)
   * @return (x, y, z) position with respect to the camera, or nullopt if
   * it could not be determined.
   */
  std::optional<cv::Vec3f> getPosition(const cv::Point2i& colorPixel,
                                       const cv::Mat& depthXyz) const;

  /**
   * Project a 3D position onto a camera's image plane.
   *
   * @param position (x, y, z) position to project.
   * @param cameraMatrix Camera intrinsic matrix.
   * @param distCoeffs Camera distortion coefficients.
   * @param extrinsicMatrix 4x4 extrinsic matrix transforming `position`'s
   * coordinate space into the camera's coordinate space. If empty, `position`
   * is assumed to already be in the camera's coordinate space.
   * @return (x, y) pixel coordinate, or nullopt if it could not be
   * determined.
   */
  static std::optional<cv::Point2i> projectPosition(
      const cv::Vec3f& position, const cv::Mat& cameraMatrix,
      const cv::Mat& distCoeffs, const cv::Mat& extrinsicMatrix = cv::Mat());

 private:
  cv::Mat colorCameraIntrinsicMatrix_;
  cv::Mat colorCameraDistortionCoeffs_;
  cv::Mat depthCameraIntrinsicMatrix_;
  cv::Mat depthCameraDistortionCoeffs_;
  cv::Mat xyzToColorCameraExtrinsicMatrix_;
  cv::Mat xyzToDepthCameraExtrinsicMatrix_;
  cv::Mat colorToDepthExtrinsicMatrix_;
  std::pair<int, int> colorFrameOffset_;
};
