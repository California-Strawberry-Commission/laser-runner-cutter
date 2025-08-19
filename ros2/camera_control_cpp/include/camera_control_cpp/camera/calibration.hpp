#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

namespace calibration {

struct ReprojectErrors {
  float meanError;
  std::vector<float> perImageErrors;
};

struct CalculateIntrinsicsResult {
  cv::Mat intrinsicMatrix;
  cv::Mat distCoeffs;
};

/**
 * Finds the camera intrinsic parameters and distortion coefficients from
 * several views of a calibration pattern.
 *
 * @param monoImages Grayscale images, each containing the calibration pattern.
 * @param gridSize Size of the calibration pattern (columns, rows).
 * @param gridType One of:
 *        cv::CALIB_CB_SYMMETRIC_GRID - symmetric pattern of circles,
 *        cv::CALIB_CB_ASYMMETRIC_GRID - asymmetric pattern of circles,
 *        cv::CALIB_CB_CLUSTERING - robust to perspective distortions but
 * sensitive to clutter.
 * @param blobDetector Feature detector for blobs (e.g., dark circles on light
 * background). If nullptr, a default implementation is used.
 * @return std::optional<calibration::CalibrationMetrics> Struct containing
 * (camera intrinsic matrix, distortion coefficients), or std::nullopt if
 * calibration was unsuccessful.
 */
std::optional<calibration::CalculateIntrinsicsResult> calculateIntrinsics(
    const std::vector<cv::Mat>& monoImages, const cv::Size& gridSize,
    const int gridType = cv::CALIB_CB_SYMMETRIC_GRID,
    const cv::Ptr<cv::FeatureDetector> blobDetector = NULL);

/**
 * Constructs a 4x4 extrinsic transformation matrix from rotation
 * and translation vectors.
 *
 * @param rvec Rotation vector (Rodrigues representation), as a cv::Mat.
 * @param tvec Translation vector, as a cv::Mat.
 * @return cv::Mat The resulting 4x4 extrinsic transformation matrix.
 */
cv::Mat constructExtrinsicMatrix(const cv::Mat& rvec, const cv::Mat& tvec);

/**
 * Extract rotation and translation vectors from a 4x4 extrinsic matrix.
 *
 * @param extrinsicMatrix 4x4 extrinsic matrix.
 * @return {rvec, tvec}
 */
std::pair<cv::Mat, cv::Mat> extractPoseFromExtrinsic(
    const cv::Mat& extrinsicMatrix);

/**
 * Inverts a 4x4 extrinsic transformation matrix.
 *
 * @param extrinsic The 4x4 extrinsic transformation matrix (CV_64F).
 * @return The inverted 4x4 extrinsic transformation matrix.
 */
cv::Mat invertExtrinsicMatrix(const cv::Mat& extrinsic);

/**
 * Applies lens distortion to undistorted pixel coordinates using camera
 * intrinsics and distortion coefficients.
 *
 * @param undistortedPixelCoords The undistorted pixel coordinates (input).
 * @param intrinsicMatrix The camera intrinsic matrix (3x3, CV_64F).
 * @param distCoeffs The distortion coefficients (1xN, where N >= 4, CV_64F).
 * @return std::optional<cv::Point2f> The distorted pixel coordinates, or
 * std::nullopt on failure.
 */
std::optional<cv::Point2f> distortPixelCoords(
    const cv::Point2f& undistortedPixelCoords, const cv::Mat& intrinsicMatrix,
    const cv::Mat& distCoeffs);

/**
 * Scale a grayscale image so that it uses the full 8-bit range.
 *
 * @param monoImage The grayscale image.
 * @return Scaled uint8 grayscale image.
 */
cv::Mat scaleGrayscaleImage(const cv::Mat& monoImage);

/**
 * Creates and returns a pointer to a configured SimpleBlobDetector.
 *
 * @return cv::Ptr<cv::SimpleBlobDetector> A smart pointer to the created
 * SimpleBlobDetector instance.
 */
cv::Ptr<cv::Feature2D> createBlobDetector();

/**
 * Compute the reprojection error.
 *
 * @param objectPoints List of object points in real-world space.
 * @param imagePoints List of corresponding image points detected in images.
 * @param rvecs List of rotation vectors returned by cv::calibrateCamera.
 * @param tvecs List of translation vectors returned by cv::calibrateCamera.
 * @param cameraMatrix Camera matrix.
 * @param distCoeffs Distortion coefficients.
 * @return ReprojectErrors struct containing mean reprojection error and
 * per-image errors.
 */
ReprojectErrors _calcReprojectionError(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);

}  // namespace calibration