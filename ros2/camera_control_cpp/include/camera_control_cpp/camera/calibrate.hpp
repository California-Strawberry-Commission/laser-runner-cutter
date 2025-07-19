#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

#include "ArenaApi.h"
#include "GenApi/GenApi.h"

namespace calibration {

std::string calibrationParamsDir =
    std::string(std::getenv("PWD")) + "/camera_control_cpp/calibration_params/";

struct ReprojectErrors {
  float meanError;
  std::vector<float> perImageErrors;
};

struct CalibrationMetrics {
  cv::Mat intrinsicMatrix;
  cv::Mat distCoeffs;
};

struct ExtrinsicMetrics {
  cv::Mat rvec;
  cv::Mat tvec;
};

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
std::optional<calibration::CalibrationMetrics> calibrateCamera(
    const std::vector<cv::Mat>& monoImages, const cv::Size& gridSize,
    const int gridType = cv::CALIB_CB_SYMMETRIC_GRID,
    const cv::Ptr<cv::FeatureDetector> blobDetector = NULL);

/**
 * Retrieves the calibration metrics for the depth camera.
 *
 * @return std::optional<calibration::CalibrationMetrics>
 *         The calibration metrics if available, otherwise std::nullopt.
 */
std::optional<calibration::CalibrationMetrics> getDepthCameraCalibration();

/**
 * Scales a grayscale image to enhance its intensity range.
 *
 * @param monoImage The input grayscale image (CV_8UC1 or CV_16UC1).
 * @return cv::Mat The scaled grayscale image with adjusted intensity values.
 */
cv::Mat scaleGrayscaleImage(const cv::Mat& monoImage);

/**
 * Finds the rotation and translation vectors that describe the conversion from
 Helios 3D coordinates to the Triton camera's coordinate system.
 *
 * @param tritonMonoImage Grayscale image from the Triton camera containing
 calibration pattern.
 * @param heliosIntensityImage Intensity image from the Helios camera containing
 calibration pattern.
 * @param heliosXyzImage 3D point cloud image from the Helios camera (XYZ
 coordinates).
 * @param tritonIntrinsicMatrix Intrinsic matrix of the Triton camera.
 * @param tritonDistortionCoeffs Distortion coefficients of the Triton camera.
 * @param gridSize Size of the calibration grid (number of inner corners per
 chessboard row and column).
 * @param gridType Type of calibration grid (default:
 cv::CALIB_CB_SYMMETRIC_GRID).
 * @param blobDetector Optional custom blob detector for grid detection.
 * @return std::optional<calibration::ExtrinsicMetrics> Estimated extrinsic
 metrics if calibration is successful, std::nullopt otherwise.
 */
std::optional<calibration::ExtrinsicMetrics> getExtrinsics(
    const cv::Mat& tritonMonoImage, const cv::Mat& heliosIntensityImage,
    const cv::Mat& heliosXyzImage, const cv::Mat& tritonIntrinsicMatrix,
    const cv::Mat& tritonDistortionCoeffs, const cv::Size& gridSize,
    const int& gridType = cv::CALIB_CB_SYMMETRIC_GRID,
    const cv::Ptr<cv::FeatureDetector>& blobDetector = nullptr);



/**
 * Saves the extrinsic calibration matrices for both cameras.
 *
 * @param colorMetrics The extrinsic metrics containing calibration data for the color camera.
 * @return int Returns 0 on success, or a negative error code on failure.
 */
int calibration::saveExtrinsics(calibration::ExtrinsicMetrics colorMetrics);

/**
 * Saves the calibration metrics for color and depth cameras.
 *
 * @param colorMetrics Calibration metrics for the color camera.
 * @param depthMetrics Calibration metrics for the depth camera.
 * @return int Returns 0 on success, or a non-zero error code on failure.
 */
int saveMetrics(const calibration::CalibrationMetrics& colorMetrics,
                const calibration::CalibrationMetrics& depthMetrics);

/**
 * Tests the undistortion of an image using the provided camera matrix and
 * distortion coefficients.
 *
 * @param cameraMatrix The camera intrinsic matrix (3x3) used for undistortion.
 * @param distCoeffs The distortion coefficients vector (typically 1x5 or 1x8)
 * for the camera.
 * @param img The input distorted image to be undistorted.
 */
void testUndistortion(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                      const cv::Mat& img);

}  // namespace calibration