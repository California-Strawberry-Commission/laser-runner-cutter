#include "camera_control_cpp/camera/calibration.hpp"

#include <filesystem>

#include "spdlog/spdlog.h"

std::optional<calibration::IntrinsicsResult> calibration::calculateIntrinsics(
    const std::vector<cv::Mat>& monoImages, const cv::Size& gridSize,
    const int gridType, const cv::Ptr<cv::FeatureDetector> blobDetector) {
  // Prepare calibration pattern points,
  // These points are in the calibration pattern coordinate space. Since the
  // calibration grid is on a flat plane, we can set the Z coordinates as 0.

  std::vector<cv::Point3f> calibrationPoints;
  if (gridType == cv::CALIB_CB_SYMMETRIC_GRID) {
    for (int i = 0; i < gridSize.height; i++) {
      for (int j = 0; j < gridSize.width; j++) {
        calibrationPoints.emplace_back(j, i, 0);
      }
    }
  } else if (gridType == cv::CALIB_CB_ASYMMETRIC_GRID) {
    for (int i = 0; i < gridSize.height; i++) {
      for (int j = 0; j < gridSize.width; j++) {
        calibrationPoints.emplace_back((2 * j + i % 2), i, 0);
      }
    }
  } else {
    spdlog::error("Unsupported grid type.");
    return std::nullopt;
  }

  if (monoImages.size() < 3) {
    spdlog::error(
        "At least 3 images are required to calculate the camera intrinsics.");
    return std::nullopt;
  }

  std::vector<std::vector<cv::Point3f>> objPoints;
  std::vector<std::vector<cv::Point2f>> imgPoints;
  bool found;
  for (const cv::Mat& image : monoImages) {
    std::vector<cv::Point2f> centers;
    found = findCirclesGrid(image, gridSize, centers, gridType, blobDetector);
    if (found) {
      objPoints.push_back(calibrationPoints);
      imgPoints.push_back(centers);
    } else {
      spdlog::warn("Could not get circle centers. Ignoring Image.");
    }
  }

  try {
    std::vector<cv::Mat> rvecs, tvecs;
    calibration::IntrinsicsResult result;
    result.intrinsicMatrix = cv::Mat::eye(3, 3, CV_64F);
    result.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    float retval = cv::calibrateCamera(
        objPoints, imgPoints, monoImages[0].size(), result.intrinsicMatrix,
        result.distCoeffs, rvecs, tvecs);
    if (retval) {
      calibration::ReprojectErrors projErrors{
          calibration::_calcReprojectionError(objPoints, imgPoints, rvecs,
                                              tvecs, result.intrinsicMatrix,
                                              result.distCoeffs)};
      spdlog::info(
          "Calibration successful. Used {} images. Mean reprojection error: {}",
          objPoints.size(), projErrors.meanError);
      return result;
    }
  } catch (const std::exception& e) {
    spdlog::warn("Exception during calibration: \n{}", e.what());
  }

  spdlog::warn("Calibration unsuccessful");
  return std::nullopt;
}

cv::Mat calibration::constructExtrinsicMatrix(const cv::Mat& rvec,
                                              const cv::Mat& tvec) {
  cv::Mat R;

  // Convert rotation vector to rotation matric using Rodrigues' formula
  cv::Rodrigues(rvec, R);

  // Create extrinsic matrix
  cv::Mat extrinsic{cv::Mat::eye(4, 4, R.type())};
  R.copyTo(extrinsic(cv::Range(0, 3), cv::Range(0, 3)));
  tvec.copyTo(extrinsic(cv::Range(0, 3), cv::Range(3, 4)));

  return extrinsic;
}

std::pair<cv::Mat, cv::Mat> calibration::extractPoseFromExtrinsic(
    const cv::Mat& extrinsicMatrix) {
  // Extract rotation (3x3) and translation (3x1) from the 4x4 extrinsic matrix
  cv::Mat R{extrinsicMatrix(cv::Rect(0, 0, 3, 3)).clone()};     // top-left 3x3
  cv::Mat tvec{extrinsicMatrix(cv::Rect(3, 0, 1, 3)).clone()};  // 3x1 column

  // Convert rotation matrix to rotation vector (Rodrigues)
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);

  return {rvec, tvec};
}

cv::Mat calibration::invertExtrinsicMatrix(const cv::Mat& extrinsic) {
  // Extract the rotation matrix and the translation vetor
  cv::Mat R{extrinsic(cv::Range(0, 3), cv::Range(0, 3))};
  cv::Mat t{extrinsic(cv::Range(0, 3), cv::Range(3, 4))};

  // Compute the inverse rotation matrix
  cv::Mat R_inv{R.t()};

  // Compute the new translation vector
  cv::Mat t_inv{-R_inv * t};

  // Construct the new extrinsic matrix
  cv::Mat extrinsic_inv{cv::Mat::eye(4, 4, R.type())};
  R_inv.copyTo(extrinsic_inv(cv::Range(0, 3), cv::Range(0, 3)));
  t_inv.copyTo(extrinsic_inv(cv::Range(0, 3), cv::Range(3, 4)));

  return extrinsic_inv;
}

std::optional<cv::Point2f> calibration::distortPixelCoords(
    const cv::Point2f& undistortedPixelCoords, const cv::Mat& intrinsicMatrix,
    const cv::Mat& distCoeffs) {
  if (intrinsicMatrix.type() != CV_64F) {
    spdlog::warn("Invalid type for Intrinsic Camera Matrix: {}",
                 intrinsicMatrix.type());
    return std::nullopt;
  }
  if (distCoeffs.type() != CV_64F) {
    spdlog::warn("Invalid type for Distortion Coefficients Matrix: {}",
                 distCoeffs.type());
    return std::nullopt;
  }

  // Extract focal length, principal point, etc.
  double fx{intrinsicMatrix.at<double>(0, 0)};
  double fy{intrinsicMatrix.at<double>(1, 1)};
  double cx{intrinsicMatrix.at<double>(0, 2)};
  double cy{intrinsicMatrix.at<double>(1, 2)};

  // Normalize Points & create 3d
  std::vector<cv::Point3f> normalizedPoints{
      {cv::Point3f((undistortedPixelCoords.x - cx) / fx,
                   (undistortedPixelCoords.y - cy) / fy, 1.0f)}};

  // Designate no rotation or translation
  cv::Mat rvec{cv::Mat::zeros(3, 1, CV_64F)};
  cv::Mat tvec{cv::Mat::zeros(3, 1, CV_64F)};

  // Project using distortion
  std::vector<cv::Point2f> distortedPoints;
  cv::projectPoints(normalizedPoints, rvec, tvec, intrinsicMatrix, distCoeffs,
                    distortedPoints);

  return distortedPoints[0];
}

cv::Mat calibration::scaleGrayscaleImage(const cv::Mat& monoImage) {
  // Ensure grayscale
  CV_Assert(monoImage.channels() == 1);

  // Convert to float to avoid overflow or underflow issues
  cv::Mat floatImg;
  monoImage.convertTo(floatImg, CV_32F);

  // Find the minimum and maximum values in the image
  double minVal, maxVal;
  cv::minMaxLoc(floatImg, &minVal, &maxVal);

  // Normalize the image to the range 0 to 1
  cv::Mat normalized;
  if (maxVal > minVal) {
    // Scale to [0,1]
    normalized = (floatImg - minVal) / (maxVal - minVal);
  } else {
    // Avoid division by zero, just shift
    normalized = floatImg - minVal;
  }

  // Scale to [0,255] as uint8
  cv::Mat scaled;
  normalized.convertTo(scaled, CV_8U, 255.0);

  return scaled;
}

cv::Ptr<cv::Feature2D> calibration::createBlobDetector() {
  cv::SimpleBlobDetector::Params params;

  // Filter By Color
  params.filterByColor = true;
  params.blobColor = 255;

  // Filter By Area
  params.filterByArea = true;
  params.minArea = 10.0;
  params.maxArea = 10000.0;

  return cv::SimpleBlobDetector::create(params);
}

calibration::ReprojectErrors calibration::_calcReprojectionError(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
  calibration::ReprojectErrors retVals;
  float totalError{0};
  float err;
  for (size_t i = 0; i < objectPoints.size(); i++) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix,
                      distCoeffs, projected);
    err = norm(imagePoints[i], projected, cv::NORM_L2) / projected.size();
    retVals.perImageErrors.push_back(err);
    totalError += err;
  }
  retVals.meanError = totalError / objectPoints.size();
  return retVals;
}

std::string calibration::expandUser(const std::string& path) {
  if (path.empty() || path[0] != '~') {
    return path;
  }

  const char* home{std::getenv("HOME")};
  if (!home) {
    throw std::runtime_error("HOME environment variable not set");
  }

  return std::string(home) + path.substr(1);
}

std::optional<cv::Mat> calibration::readXyzFile(const std::string& xyzFile) {
  std::filesystem::path filePath{calibration::expandUser(xyzFile)};
  if (!std::filesystem::exists(filePath)) {
    spdlog::error("File does not exist: {}", filePath.string());
    return std::nullopt;
  }
  cv::FileStorage fs{filePath, cv::FileStorage::READ};
  if (!fs.isOpened() || fs["xyz"].isNone()) {
    spdlog::error("Could not read XYZ data file: {}", filePath.string());
    return std::nullopt;
  }

  cv::Mat xyz;
  fs["xyz"] >> xyz;
  fs.release();

  return xyz;
}

std::optional<calibration::IntrinsicsResult> calibration::readIntrinsicsFile(
    const std::string& intrinsicsFile) {
  std::filesystem::path filePath{calibration::expandUser(intrinsicsFile)};
  if (!std::filesystem::exists(filePath)) {
    spdlog::error("Intrinsics file does not exist: {}", filePath.string());
    return std::nullopt;
  }
  cv::FileStorage fs{filePath, cv::FileStorage::READ};
  if (!fs.isOpened() || fs["intrinsicMatrix"].isNone() ||
      fs["distCoeffs"].isNone()) {
    spdlog::error("Invalid intrinsics file: {}", filePath.string());
    return std::nullopt;
  }

  cv::Mat intrinsicMatrix, distCoeffs;
  fs["intrinsicMatrix"] >> intrinsicMatrix;
  fs["distCoeffs"] >> distCoeffs;
  fs.release();

  calibration::IntrinsicsResult result;
  result.intrinsicMatrix = intrinsicMatrix;
  result.distCoeffs = distCoeffs;

  return result;
}

std::optional<cv::Mat> calibration::readExtrinsicsFile(
    const std::string& extrinsicsFile) {
  std::filesystem::path filePath{calibration::expandUser(extrinsicsFile)};
  if (!std::filesystem::exists(filePath)) {
    spdlog::error("Extrinsics file does not exist: {}", filePath.string());
    return std::nullopt;
  }
  cv::FileStorage fs{filePath, cv::FileStorage::READ};
  if (!fs.isOpened() || fs["extrinsicMatrix"].isNone()) {
    spdlog::error("Invalid extrinsics file: {}", filePath.string());
    return std::nullopt;
  }

  cv::Mat extrinsicMatrix;
  fs["extrinsicMatrix"] >> extrinsicMatrix;
  fs.release();

  return extrinsicMatrix;
}