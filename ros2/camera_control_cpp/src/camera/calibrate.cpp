#include "camera_control_cpp/camera/calibrate.hpp"

#include <filesystem>

#include "spdlog/spdlog.h"

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
    spdlog::warn("Invalid type for Intrinsic Camera Matrix: {}", intrinsicMatrix.type());
    return std::nullopt;
  }
  if (distCoeffs.type() != CV_64F) {
    spdlog::warn("Invalid type for Distortion Coefficients Matrix: {}", distCoeffs.type());
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
    cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs,
                  projected);
    err = norm(imagePoints[i], projected, cv::NORM_L2) / projected.size();
    retVals.perImageErrors.push_back(err);
    totalError += err;
  }
  retVals.meanError = totalError / objectPoints.size();
  return retVals;
}

std::optional<calibration::CalibrationMetrics> calibration::calibrateCamera(
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
    spdlog::warn("Unsupported grid type.");
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
    calibration::CalibrationMetrics metrics;
    metrics.intrinsicMatrix = cv::Mat::eye(3, 3, CV_64F);
    metrics.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    float retval = cv::calibrateCamera(
        objPoints, imgPoints, monoImages[0].size(), metrics.intrinsicMatrix,
        metrics.distCoeffs, rvecs, tvecs);
    if (retval) {
      calibration::ReprojectErrors projErrors{
          calibration::_calcReprojectionError(objPoints, imgPoints, rvecs,
                                              tvecs, metrics.intrinsicMatrix,
                                              metrics.distCoeffs)};
      spdlog::info(
          "Calibration successful. Used {} images. Mean reprojection error: {}",
          objPoints.size(), projErrors.meanError);
      return metrics;
    }
  } catch (const std::exception& e) {
    spdlog::warn("Exception during calibration: \n{}", e.what());
  }

  spdlog::warn("Calibration unsuccessful");
  return std::nullopt;
}

void calibration::testUndistortion(const cv::Mat& cameraMatrix,
                                   const cv::Mat& distCoeffs,
                                   const cv::Mat& img) {
  cv::Mat newCameraMatrix, undistorted;
  cv::Rect roi;
  newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs,
                                              img.size(), 1, img.size(), &roi);
  cv::undistort(img, undistorted, cameraMatrix, distCoeffs, newCameraMatrix);
  undistorted = undistorted(roi);
  std::string filename{"undistortion_test.png"};
  bool success{cv::imwrite(filename, undistorted)};
  if (success) {
    std::cout << "Image saved successfully as " << filename << std::endl;
  } else {
    std::cerr << "Error saving image to " << filename << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    spdlog::error(
        "Invalid call: Arguments must (exclusively) include calibration image "
        "dir");
    return -1;
  }

  std::filesystem::path dirPath(argv[1]);
  if (!exists(dirPath) || !is_directory(dirPath)) {
    spdlog::error("Error: Provided path is not a valid directory.");
    return -1;
  }

  std::vector<cv::Mat> images;
  size_t imageCount{0};
  for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        cv::Mat img{cv::imread(entry.path().string())};
        if (!img.empty()) {
          cvtColor(img, img, cv::COLOR_BGR2GRAY);
          images.push_back(img);
          ++imageCount;
        }
      }
    }
  }
  if (imageCount < 9) {
    spdlog::error(
        "Error: Directory must contain at least 9 image (.png, .jpg, .jpeg) "
        "files.");
    return -1;
  } else {
    spdlog::info("Directory opened successfully: \n{}", dirPath.string());
  }

  cv::Size gSize{cv::Size(5, 4)};
  std::optional<calibration::CalibrationMetrics> calibVals{
      calibration::calibrateCamera(images, gSize, cv::CALIB_CB_SYMMETRIC_GRID,
                                   calibration::createBlobDetector())};
  if (calibVals == std::nullopt) {
    spdlog::error("Calibration failed. Exiting.");
    return -1;
  } else {
    std::ostringstream oss1, oss2;
    oss1 << (*calibVals).intrinsicMatrix;
    oss2 << (*calibVals).distCoeffs;
    spdlog::info("Calibrated intrins: \n{}", oss1.str());
    spdlog::info("Distortion coeffs: \n{}", oss2.str());
  }

  return 0;
}