#include "camera_control_cpp/camera/lucid_frame.hpp"

#include "camera_control_cpp/camera/calibration.hpp"
#include "spdlog/spdlog.h"

namespace {

std::optional<cv::Vec3f> deprojectPixel(const cv::Point2i& pixel, float depth,
                                        const cv::Mat& cameraMatrix,
                                        const cv::Mat& distCoeffs) {
  std::vector<cv::Point2f> src{cv::Point2f(pixel)};
  std::vector<cv::Point2f> undistorted;
  cv::undistortPoints(src, undistorted, cameraMatrix, distCoeffs);

  // cv::undistortPoints can return nan or inf values
  if (undistorted.empty() || std::isnan(undistorted[0].x) ||
      std::isinf(undistorted[0].x) || std::isnan(undistorted[0].y) ||
      std::isinf(undistorted[0].y)) {
    return std::nullopt;
  }

  return cv::Vec3f{undistorted[0].x * depth, undistorted[0].y * depth, depth};
}

cv::Vec3f transformPosition(const cv::Vec3f& position,
                            const cv::Mat& extrinsic) {
  cv::Vec4d homogeneousPosition{position[0], position[1], position[2], 1.0};
  cv::Vec4d transformed;
  for (int i = 0; i < 4; ++i) {
    transformed[i] = 0;
    for (int j = 0; j < 4; ++j) {
      transformed[i] += extrinsic.at<double>(i, j) * homogeneousPosition[j];
    }
  }
  return cv::Vec3f{static_cast<float>(transformed[0]),
                   static_cast<float>(transformed[1]),
                   static_cast<float>(transformed[2])};
}

std::optional<cv::Point2i> projectPosition(
    const cv::Vec3f& position, const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs, const cv::Mat& extrinsicMatrix = cv::Mat()) {
  cv::Mat rvec, tvec;
  if (!extrinsicMatrix.empty()) {
    // Extract rotation (3x3) and translation (3x1) from extrinsic
    cv::Mat R = extrinsicMatrix(cv::Range(0, 3), cv::Range(0, 3));
    cv::Mat t = extrinsicMatrix(cv::Range(0, 3), cv::Range(3, 4));

    // Convert rotation matrix to rotation vector
    cv::Rodrigues(R, rvec);
    tvec = t.clone();
  } else {
    rvec = cv::Mat::zeros(3, 1, CV_64F);
    tvec = cv::Mat::zeros(3, 1, CV_64F);
  }

  std::vector<cv::Point3f> objectPoints{
      {cv::Point3f{position[0], position[1], position[2]}}};
  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs,
                    imagePoints);

  // cv::projectPoints can return nan or inf values
  if (imagePoints.empty() || std::isnan(imagePoints[0].x) ||
      std::isinf(imagePoints[0].x) || std::isnan(imagePoints[0].y) ||
      std::isinf(imagePoints[0].y)) {
    return std::nullopt;
  }

  return cv::Point2i{static_cast<int>(std::round(imagePoints[0].x)),
                     static_cast<int>(std::round(imagePoints[0].y))};
}

cv::Point2i adjustPixelToBounds(const cv::Point2i& pixel, int width,
                                int height) {
  int x{
      std::max(0, std::min(static_cast<int>(std::round(pixel.x)), width - 1))};
  int y{
      std::max(0, std::min(static_cast<int>(std::round(pixel.y)), height - 1))};
  return cv::Point2i{x, y};
}

cv::Point2i nextPixelInLine(const cv::Point2i& curr, const cv::Point2i& end) {
  cv::Point2f direction{cv::Point2f(end - curr)};
  double norm{cv::norm(direction)};
  if (norm == 0.0) {
    // Already at the end
    return cv::Point2i(static_cast<int>(std::round(curr.x)),
                       static_cast<int>(std::round(curr.y)));
  }
  direction /= norm;
  cv::Point2f next{cv::Point2f(curr) + direction};
  return cv::Point2i(static_cast<int>(std::round(next.x)),
                     static_cast<int>(std::round(next.y)));
}

bool isPixelInLine(const cv::Point2i& curr, const cv::Point2i& start,
                   const cv::Point2i& end) {
  int minX{std::min(start.x, end.x)};
  int maxX{std::max(start.x, end.x)};
  int minY{std::min(start.y, end.y)};
  int maxY{std::max(start.y, end.y)};
  return (minX <= curr.x && curr.x <= maxX) &&
         (minY <= curr.y && curr.y <= maxY);
}

}  // namespace

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
      colorFrameOffset_(colorFrameOffset) {
  colorToDepthExtrinsicMatrix_ =
      xyzToDepthCameraExtrinsicMatrix_ *
      calibration::invertExtrinsicMatrix(xyzToColorCameraExtrinsicMatrix_);
}

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

std::optional<cv::Point2i> LucidFrame::getCorrespondingDepthPixel(
    const cv::Point2i& colorPixel) const {
  /*
  Given an (x, y) coordinate in the color frame, return the corresponding (x, y)
  coordinate in the depth frame.

  The general approach is as follows:
      1. Deproject the color image pixel coordinate to two positions in the
  color camera-space: one that corresponds to the position at the minimum depth,
  and one at the maximum depth.
      2. Transform the two positions from color camera-space to depth
  camera-space.
      3. Project the two positions to their respective depth image pixel
  coordinates.
      4. The target lies somewhere along the line formed by the two pixel
  coordinates found in the previous step. We iteratively move pixel by pixel
  along this line. For each depth image pixel, we grab the xyz data at the
  pixel, project it onto the color image plane, and see how far it is from the
  original color pixel coordinate. We find and return the closest match.
  Note that in order to achieve the above, we require two extrinsic matrices -
  one for projecting the xyz positions to the color camera image plane, and one
  for projecting the xyz positions to the depth camera image plane.
  */

  // Apply frame ROI offset. The calibration matrices and coefficients were
  // calculated based on the max camera frame size, and if there was a reduced
  // ROI set when capturing the frame, the pixel coordinate for the frame must
  // be converted to that of the full size frame.
  cv::Point2i fullFrameColorPixel{
      colorPixel +
      cv::Point2i(colorFrameOffset_.first, colorFrameOffset_.second)};

  // Min-depth and max-depth positions in color camera-space
  auto minDepthPositionColorSpaceOpt{deprojectPixel(
      fullFrameColorPixel, DEPTH_MIN_MM, colorCameraIntrinsicMatrix_,
      colorCameraDistortionCoeffs_)};
  if (!minDepthPositionColorSpaceOpt) {
    return std::nullopt;
  }
  cv::Vec3f minDepthPositionColorSpace{minDepthPositionColorSpaceOpt.value()};
  auto maxDepthPositionColorSpaceOpt{deprojectPixel(
      fullFrameColorPixel, DEPTH_MAX_MM, colorCameraIntrinsicMatrix_,
      colorCameraDistortionCoeffs_)};
  if (!maxDepthPositionColorSpaceOpt) {
    return std::nullopt;
  }
  cv::Vec3f maxDepthPositionColorSpace{maxDepthPositionColorSpaceOpt.value()};

  // Min-depth and max-depth positions in depth camera-space
  cv::Vec3f minDepthPositionDepthSpace{transformPosition(
      minDepthPositionColorSpace, colorToDepthExtrinsicMatrix_)};
  cv::Vec3f maxDepthPositionDepthSpace{transformPosition(
      maxDepthPositionColorSpace, colorToDepthExtrinsicMatrix_)};

  // Project depth camera-space positions to depth pixels
  auto minDepthPixelOpt{projectPosition(minDepthPositionDepthSpace,
                                        depthCameraIntrinsicMatrix_,
                                        depthCameraDistortionCoeffs_)};
  if (!minDepthPixelOpt) {
    return std::nullopt;
  }
  cv::Point2i minDepthPixel{minDepthPixelOpt.value()};
  auto maxDepthPixelOpt{projectPosition(maxDepthPositionDepthSpace,
                                        depthCameraIntrinsicMatrix_,
                                        depthCameraDistortionCoeffs_)};
  if (!maxDepthPixelOpt) {
    return std::nullopt;
  }
  cv::Point2i maxDepthPixel{maxDepthPixelOpt.value()};

  // Make sure pixel coords are in boundary
  cv::Point2i minDepthPixelAdjusted{adjustPixelToBounds(
      minDepthPixel, depthFrameXyz_.cols, depthFrameXyz_.rows)};
  cv::Point2i maxDepthPixelAdjusted{adjustPixelToBounds(
      maxDepthPixel, depthFrameXyz_.cols, depthFrameXyz_.rows)};

  // Search along the line for the depth pixel for which its corresponding
  // projected color pixel is the closest to the target color pixel
  int minDist{-1};
  cv::Point2i closestDepthPixel{minDepthPixelAdjusted};
  cv::Point2i currDepthPixel{minDepthPixelAdjusted};
  while (true) {
    cv::Vec3f xyzmm{
        depthFrameXyz_.at<cv::Vec3f>(currDepthPixel.y, currDepthPixel.x)};
    auto currColorPixelOpt{projectPosition(xyzmm, colorCameraIntrinsicMatrix_,
                                           colorCameraDistortionCoeffs_,
                                           xyzToColorCameraExtrinsicMatrix_)};
    if (!currColorPixelOpt) {
      return std::nullopt;
    }
    cv::Point2i currColorPixel{currColorPixelOpt.value()};
    double distance{cv::norm(cv::Point2f(currColorPixel) -
                             cv::Point2f(fullFrameColorPixel))};
    if (distance < minDist || minDist < 0) {
      minDist = distance;
      closestDepthPixel = currDepthPixel;
    }
    // Stop if we've processed the maxDepthPixel
    if (currDepthPixel.x == maxDepthPixel.x &&
        currDepthPixel.y == maxDepthPixel.y) {
      break;
    }
    // Otherwise, find the next pixel along the line that we shold try
    currDepthPixel = nextPixelInLine(currDepthPixel, maxDepthPixel);
    if (!isPixelInLine(currDepthPixel, minDepthPixel, maxDepthPixel)) {
      break;
    }
  }

  return closestDepthPixel;
}

std::optional<cv::Vec3f> LucidFrame::getPosition(
    const cv::Point2i& colorPixel) const {
  auto depthPixelOpt{getCorrespondingDepthPixel(colorPixel)};
  if (!depthPixelOpt) {
    return std::nullopt;
  }
  cv::Point2i depthPixel{depthPixelOpt.value()};

  cv::Vec3f position{depthFrameXyz_.at<cv::Vec3f>(depthPixel.y, depthPixel.x)};
  // Negative depth indicates an invalid position. Depth greater than 2^14 - 1
  // also indicates invalid position.
  if (position[2] < 0.0f || position[2] > ((1 << 14) - 1)) {
    spdlog::warn("Invalid depth value at pixel ({}, {}): {}", depthPixel.x,
                 depthPixel.y, position[2]);
    return std::nullopt;
  }

  return position;
}
