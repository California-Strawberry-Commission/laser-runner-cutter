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

cv::Vec3f LucidFrame::deprojectPixel(const cv::Point2i& pixel, float depth,
                                     const cv::Mat& cameraMatrix,
                                     const cv::Mat& distCoeffs) const {
  std::vector<cv::Point2f> src{{cv::Point2f(pixel)}};
  std::vector<cv::Point2f> undistorted;
  cv::undistortPoints(src, undistorted, cameraMatrix, distCoeffs);
  return cv::Vec3f(undistorted[0].x * depth, undistorted[0].y * depth, depth);
}

cv::Vec3f LucidFrame::transformPosition(const cv::Vec3f& position,
                                        const cv::Mat extrinsic) const {
  cv::Mat homogeneousPosition{
      (cv::Mat_<double>(4, 1) << position[0], position[1], position[2], 1.0)};
  cv::Mat transformed{homogeneousPosition * extrinsic};
  return cv::Vec3f(transformed.at<double>(0), transformed.at<double>(1),
                   transformed.at<double>(2));
}

cv::Point2i LucidFrame::projectPosition(
    const cv::Vec3f& position, const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    const cv::Mat& extrinsicMatrix = cv::Mat()) const {
  cv::Mat rvec, tvec;
  if (extrinsicMatrix.empty()) {
    rvec = cv::Mat::eye(3, 1, CV_64F);
    tvec = cv::Mat::zeros(3, 1, CV_64F);
  } else {
    Rodrigues(extrinsicMatrix(cv::Rect(0, 0, 3, 3)), rvec);
    tvec = {extrinsicMatrix(cv::Rect(3, 0, 1, 3)).clone()};
  }

  std::vector<cv::Point3f> objectPoints{
      {cv::Point3f(position[0], position[1], position[2])}};
  std::vector<cv::Point3f> imagePoints;

  cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs,
                    imagePoints);

  return cv::Point2i(cvRound(imagePoints[0].x), cvRound(imagePoints[0].x));
}

cv::Point2i LucidFrame::adjustPixelToBounds(const cv::Point2i& pixel, int width,
                                            int height) const {
  int x{std::max(0, std::min(cvRound(pixel.x), width - 1))};
  int y{std::max(0, std::min(cvRound(pixel.y), height - 1))};
  return cv::Point2i(x, y);
}

cv::Point2i LucidFrame::nextPixelInLine(const cv::Point2i& curr,
                                        const cv::Point2i& start,
                                        const cv::Point2i& end) const {
  cv::Point2f direction{cv::Point2f(end - curr)};
  float length(cv::norm(direction));
  direction /= length;
  cv::Point2f next{cv::Point2f(curr) + direction};
  return cv::Point2i(cvRound(next.x), cvRound(next.y));
}

bool LucidFrame::isPixelInLine(const cv::Point2i& curr,
                               const cv::Point2i& start,
                               const cv::Point2i& end) const {
  int min_x{std::min(start.x, end.x)};
  int max_x{std::max(start.x, end.x)};
  int min_y{std::min(start.y, end.y)};
  int max_y{std::max(start.y, end.y)};
  return (min_x <= curr.x && curr.x <= max_x) &&
         (min_y <= curr.y && curr.y <= max_y);
}

cv::Point2i LucidFrame::getCorrespondingDepthPixel(
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

  Args:
      color_pixel (Sequence[int]): (x, y) coordinate in the color frame.

  Returns:
      Tuple[int, int]: (x, y) coordinate in the depth frame.
  */

  // Apply frame ROI offset. The calibration matrices and coefficients were
  // calculated based on the max camera frame size, and if there was a reduced
  // ROI set when capturing the frame, the pixel coordinate for the frame must
  // be converted to that of the full size frame.

  cv::Point2i fullFrameColorPixel{
      colorPixel +
      cv::Point2i(colorFrameOffset_.first, colorFrameOffset_.second)};

  // Min-depth and max-depth positions in color camera-space
  cv::Vec3f minDepthPositionColorSpace{deprojectPixel(
      fullFrameColorPixel, DEPTH_MIN_MM, colorCameraIntrinsicMatrix_,
      colorCameraDistortionCoeffs_)};
  cv::Vec3f maxDepthPositionColorSpace{deprojectPixel(
      fullFrameColorPixel, DEPTH_MAX_MM, colorCameraIntrinsicMatrix_,
      colorCameraDistortionCoeffs_)};

  // Min-depth and max-depth positions in depth camera-space
  cv::Vec3f minDepthPositionDepthSpace{transformPosition(
      minDepthPositionColorSpace, xyzToDepthCameraExtrinsicMatrix_)};
  cv::Vec3f maxDepthPositionDepthSpace{transformPosition(
      maxDepthPositionColorSpace, xyzToDepthCameraExtrinsicMatrix_)};

  // Project depth camera-space positions to depth pixels
  cv::Point2i minDepthPixel{projectPosition(minDepthPositionDepthSpace,
                                            depthCameraIntrinsicMatrix_,
                                            depthCameraDistortionCoeffs_)};
  cv::Point2i maxDepthPixel{projectPosition(maxDepthPositionDepthSpace,
                                            depthCameraIntrinsicMatrix_,
                                            depthCameraDistortionCoeffs_)};

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
    cv::Point2i currColorPixel = projectPosition(
        xyzmm, colorCameraIntrinsicMatrix_, colorCameraDistortionCoeffs_,
        xyzToColorCameraExtrinsicMatrix_);
    double distance = cv::norm(cv::Point2f(currColorPixel) -
                               cv::Point2f(fullFrameColorPixel));
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
    currDepthPixel =
        nextPixelInLine(currDepthPixel, minDepthPixel, maxDepthPixel);
    if (!isPixelInLine(currDepthPixel, minDepthPixel, maxDepthPixel)) {
      break;
    }
  }
  return closestDepthPixel;
}

std::optional<std::tuple<double, double, double>> LucidFrame::getPosition(
    std::pair<int, int> colorPixel) const {
  cv::Point2i colorPixelPoint{cv::Point2i(colorPixel.first, colorPixel.second)};
  cv::Point2i depthPixel = getCorrespondingDepthPixel(colorPixelPoint);
  cv::Vec3f position{depthFrameXyz_.at<cv::Vec3f>(depthPixel.y, depthPixel.x)};
  // Negative depth indicates an invalid position. Depth greater than 2^14 - 1
  // also indicates invalid position.
  if (position[2] < 0.0f || position[2] > ((1 << 14) - 1)) {
    return std::nullopt;
  }
  return std::make_tuple(static_cast<double>(position[0]),
                         static_cast<double>(position[1]),
                         static_cast<double>(position[2]));
}