#include "camera_control_cpp/camera/calibrate.hpp"

cv::Mat constructExtrinsicMatrix(const cv::Mat& rvec, const cv::Mat& tvec) {
  /*
      Assemble 4x4 extrinsic matrix
  */

  cv::Mat R;

  // Convert rotation vector to rotation matric using Rodrigues' formula
  Rodrigues(rvec, R);

  // Create extrinsic matrix
  cv::Mat extrinsic{cv::Mat::eye(4, 4, R.type())};
  R.copyTo(extrinsic(cv::Range(0, 3), cv::Range(0, 3)));
  tvec.copyTo(extrinsic(cv::Range(0, 3), cv::Range(3, 4)));

  return extrinsic;
}

cv::Mat invertExtrinsicMatrix(const cv::Mat& extrinsic) {
  /*
      Invert a 4x4 extrinsic matrix (rotation + translation)
  */

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

cv::Point2f distortPixelCoords(const cv::Point2f& undistortedPixelCoords,
                               const cv::Mat& intrinsicMatrix,
                               const cv::Mat& distCoeffs) {
  // Extract focal length, principal point, etc.
  float fx{intrinsicMatrix.at<float>(0, 0)};
  float fy{intrinsicMatrix.at<float>(1, 1)};
  float cx{intrinsicMatrix.at<float>(0, 2)};
  float cy{intrinsicMatrix.at<float>(1, 2)};

  // Normalize Points & create 3d
  std::vector<cv::Point3f> normalizedPoints{
      {cv::Point3f((undistortedPixelCoords.x - cx) / fx,
                   (undistortedPixelCoords.y - cy) / fy, 1.0f)}};

  // Designate no rotation or translation
  cv::Mat rvec{cv::Mat::zeros(3, 1, CV_64F)};
  cv::Mat tvec{cv::Mat::zeros(3, 1, CV_64F)};

  // Project using distortion
  std::vector<cv::Point2f> distortedPoints;
  projectPoints(normalizedPoints, rvec, tvec, intrinsicMatrix, distCoeffs,
                distortedPoints);

  return distortedPoints[0];
}

cv::Ptr<cv::SimpleBlobDetector> createBlobDetector() {
  /*
      Blob detector for white circles on black background
  */

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

std::tuple<float, std::vector<float>> _calcReprojectionError(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
  /*
  Compute the reprojection error.

  Args:
      object_points (List[np.ndarray]): List of object points in real-world
  space image_points (List[np.ndarray]): List of corresponding image points
  detected in images rvecs (np.ndarray): List of rotation vectors returned by
  cv2.calibrateCamera tvecs (np.ndarray): List of translation vectors returned
  by cv2.calibrateCamera camera_matrix (np.ndarray): Camera matrix dist_coeffs
  (np.ndarray): Distortion coefficients

  Returns:
      float: Tuple of (mean reprojection error, list of per-image errors)
  */
  std::vector<float> errors;
  float totalError{0};
  float err;
  for (size_t i = 0; i < objectPoints.size(); i++) {
    std::vector<cv::Point2f> projected;
    projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs,
                  projected);
    err = norm(imagePoints[i], projected, cv::NORM_L2) / projected.size();
    errors.push_back(err);
    totalError += err;
  }
  float meanErr{totalError / objectPoints.size()};
  return make_tuple(meanErr, errors);
}

std::optional<std::tuple<cv::Mat, cv::Mat>> calibrateCamera(
    const std::vector<cv::Mat> monoImages, const cv::Size& gridSize,
    int gridType, const cv::Ptr<cv::FeatureDetector> blobDetector) {
  /*
  Finds the camera intrinsic parameters and distortion coefficients from several
  views of a calibration pattern.

  Args:
      mono_images (List[np.ndarray]): Grayscale images each containing the
  calibration pattern. grid_size (Tuple[int, int]): (# cols, # rows) of the
  calibration pattern. grid_type (int): One of the following:
          cv2.CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
          cv2.CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
          cv2.CALIB_CB_CLUSTERING uses a special algorithm for grid detection.
  It is more robust to perspective distortions but much more sensitive to
  background clutter. blobDetector: Feature detector that finds blobs, like dark
  circalibrationPointscles on light background. If None then a default
  implementation is used.

  Returns:
      Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple of (camera
  intrisic matrix, distortion coefficients), or (None, None) if calibration was
  unsuccessful.
  */

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
  for (cv::Mat image : monoImages) {
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
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    float retval = calibrateCamera(objPoints, imgPoints, monoImages[0].size(),
                                   cameraMatrix, distCoeffs, rvecs, tvecs);
    if (retval) {
      std::tuple<float, std::vector<float>> projErrors = _calcReprojectionError(
          objPoints, imgPoints, rvecs, tvecs, cameraMatrix, distCoeffs);
      spdlog::info(
          "Calibration successful. Used {} images. Mean reprojection error: {}",
          objPoints.size(), std::get<0>(projErrors));
      return std::make_tuple(cameraMatrix, distCoeffs);
    }
  } catch (const std::exception& e) {
    spdlog::warn("Exception during calibration: \n{}", e.what());
  }

  spdlog::warn("Calibration unsuccessful");
  return std::nullopt;
}

void testUndistortion(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                      const cv::Mat& img) {
  cv::Mat newCameraMatrix, undistorted;
  cv::Rect roi;
  newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs,
                                              img.size(), 1, img.size(), &roi);
  undistort(img, undistorted, cameraMatrix, distCoeffs, newCameraMatrix);
  undistorted = undistorted(roi);
  std::string filename{"undistortion_test.png"};
  bool success{imwrite(filename, img)};
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

  std::filesystem::path dir_path(argv[1]);
  if (!exists(dir_path) || !is_directory(dir_path)) {
    spdlog::error("Error: Provided path is not a valid directory.");
    return -1;
  }

  std::vector<cv::Mat> images;
  size_t image_count{0};
  for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        cv::Mat img{cv::imread(entry.path().string())};
        if (!img.empty()) {
          cvtColor(img, img, cv::COLOR_BGR2GRAY);
          images.push_back(img);
          ++image_count;
        }
      }
    }
  }
  if (image_count < 9) {
    spdlog::error(
        "Error: Directory must contain at least 9 image (.png, .jpg, .jpeg) "
        "files.");
    return -1;
  } else {
    spdlog::info("Directory opened successfully: \n{}", dir_path.string());
  }

  cv::Size gSize{cv::Size(5, 4)};
  std::optional<std::tuple<cv::Mat, cv::Mat>> calibVals{calibrateCamera(
      images, gSize, cv::CALIB_CB_SYMMETRIC_GRID, createBlobDetector())};
  if (calibVals == std::nullopt) {
    spdlog::error("Calibration failed. Exiting.");
    return -1;
  } else {
    cv::Mat cameraMatrix{std::get<0>(*calibVals)};
    cv::Mat distCoeffs{std::get<1>(*calibVals)};
    std::ostringstream oss1, oss2;
    oss1 << cameraMatrix;
    oss2 << distCoeffs;
    spdlog::info("Calibrated intrins: \n{}", oss1.str());
    spdlog::info("Distortion coeffs: \n{}", oss2.str());
  }

  return 0;
}