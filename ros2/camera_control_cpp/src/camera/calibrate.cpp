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
    metrics.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
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

std::optional<calibration::CalibrationMetrics>
calibration::getDepthCameraCalibration() {  // DEPRECATED (Used internal factory
                                            // calibration)
  try {
    Arena::ISystem* pSystem = Arena::OpenSystem();
    std::vector<Arena::DeviceInfo> deviceInfos = pSystem->GetDevices();
    // Wait for at least one camera to be detected, with a timeout
    constexpr int maxRetries = 20;
    constexpr int retryDelayMs = 1000;
    int retries = 0;
    while (deviceInfos.empty() && retries < maxRetries) {
      spdlog::info(
          "No devices found. Waiting for camera to be detected... (attempt "
          "{}/{})",
          retries + 1, maxRetries);
      pSystem->UpdateDevices(retryDelayMs);
      deviceInfos = pSystem->GetDevices();
      ++retries;
    }
    int depthDeviceNum = -1;
    if (deviceInfos.empty()) {
      spdlog::warn("No devices found after waiting.");
      Arena::CloseSystem(pSystem);
      return std::nullopt;
    } else {
      for (size_t i = 0; i < deviceInfos.size(); ++i) {
        spdlog::info(
            "Found Camera [{}] | Model: {} | Serial: {} | Vendor: {} | Device "
            "Version: {}",
            i, deviceInfos[i].ModelName(), deviceInfos[i].SerialNumber(),
            deviceInfos[i].VendorName(), deviceInfos[i].DeviceVersion());
        if (deviceInfos[i].ModelName().substr(0, 3) == "HTR") {
          depthDeviceNum = i;
        }
      }
    }

    if (depthDeviceNum != -1) {
      spdlog::info("Found depth camera! [Camera #{}]", depthDeviceNum);
    } else {
      spdlog::warn("Didn't find any depth camera attached.");
      return std::nullopt;
    }

    Arena::IDevice* pDevice = pSystem->CreateDevice(deviceInfos[1]);
    GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();

    calibration::CalibrationMetrics metrics;
    double fx = Arena::GetNodeValue<double>(pNodeMap, "CalibFocalLengthX");
    double fy = Arena::GetNodeValue<double>(pNodeMap, "CalibFocalLengthY");
    double cx = Arena::GetNodeValue<double>(pNodeMap, "CalibOpticalCenterX");
    double cy = Arena::GetNodeValue<double>(pNodeMap, "CalibOpticalCenterY");
    metrics.intrinsicMatrix =
        (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Getting distortion coeffs by modifying selector value
    std::vector<double> coeffs;
    GenApi::CEnumerationPtr selector =
        pNodeMap->GetNode("CalibLensDistortionValueSelector");
    GenApi::CFloatPtr value = pNodeMap->GetNode("CalibLensDistortionValue");
    if (!GenApi::IsAvailable(selector) || !GenApi::IsReadable(selector) ||
        !GenApi::IsAvailable(value) || !GenApi::IsReadable(value)) {
      spdlog::warn("Distortion selector or value not available.");
      return std::nullopt;
    }
    GenApi::NodeList_t entries;
    selector->GetEntries(entries);
    for (size_t i = 0; i < entries.size(); ++i) {
      GenApi::CEnumEntryPtr entry = entries[i];
      if (GenApi::IsAvailable(entry) && GenApi::IsReadable(entry)) {
        selector->SetIntValue(entry->GetValue());
        coeffs.push_back(value->GetValue());
      }
    }
    metrics.distCoeffs = cv::Mat(coeffs);

    Arena::CloseSystem(pSystem);
    return metrics;

  } catch (GenICam::GenericException& e) {
    spdlog::warn("GenICam exception: {}", e.GetDescription());
    return std::nullopt;
  } catch (std::exception& e) {
    spdlog::warn("Standard exception: {}", e.what());
    return std::nullopt;
  }
  return std::nullopt;
}

cv::Mat calibration::scaleGrayscaleImage(const cv::Mat& monoImage) {
  cv::Mat floatImage;
  monoImage.convertTo(floatImage, CV_32F);
  double minVal, maxVal;
  cv::minMaxLoc(floatImage, &minVal, &maxVal);
  if (maxVal > minVal) {
    floatImage = (floatImage - minVal) / (maxVal - minVal);
  } else {
    floatImage = floatImage - minVal;
  }
  floatImage *= 255.0;
  cv::Mat scaledImage;
  floatImage.convertTo(scaledImage, CV_8U);
  return scaledImage;
}

std::optional<calibration::ExtrinsicMetrics> calibration::getExtrinsics(
    const cv::Mat& fromImage, const cv::Mat& toImage, const cv::Mat& xyzImage,
    const cv::Mat& fromIntrinsicMatrix, const cv::Mat& fromDistortionCoeffs,
    const cv::Size& gridSize, const int& gridType,
    const cv::Ptr<cv::FeatureDetector>& blobDetector) {
  if (fromImage.empty()) {
    spdlog::warn("Invalid fromImage matrix.");
    return std::nullopt;
  }
  if (toImage.empty()) {
    spdlog::warn("Invalid toImage matrix.");
    return std::nullopt;
  }
  if (xyzImage.empty() || xyzImage.type() != CV_32FC3) {
    spdlog::warn("Invalid xyzImage matrix.");
    return std::nullopt;
  }
  if (fromIntrinsicMatrix.empty()) {
    spdlog::warn("\"From\" intrinsic matrix is invalid.");
    return std::nullopt;
  }
  if (fromDistortionCoeffs.empty()) {
    spdlog::warn("\"From\" distortion coefficients are invalid.");
    return std::nullopt;
  }

  cv::Mat scaledFromImage = calibration::scaleGrayscaleImage(fromImage);
  cv::Mat scaledToImage = calibration::scaleGrayscaleImage(toImage);

  std::vector<cv::Point2f> fromCircleCenters;
  bool fromRetval{cv::findCirclesGrid(
      scaledFromImage, gridSize, fromCircleCenters, gridType, blobDetector)};
  if (fromRetval == false) {
    spdlog::warn("Could not get circle centers from Triton mono image.");
    return std::nullopt;
  }

  std::vector<cv::Point2f> toCircleCenters;
  bool toRetval{cv::findCirclesGrid(scaledToImage, gridSize, toCircleCenters,
                                    gridType, blobDetector)};
  if (toRetval == false) {
    spdlog::warn("Could not get circle centers from Helios intensity image.");
    return std::nullopt;
  }

  std::vector<cv::Point> toPixelCoords;
  for (const cv::Point2f& pt : toCircleCenters) {
    toPixelCoords.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
  }

  std::vector<cv::Point3f> toCirclePositions;
  for (const cv::Point& pt : toPixelCoords) {
    const cv::Vec3f& vec{xyzImage.at<cv::Vec3f>(pt.x, pt.y)};
    toCirclePositions.emplace_back(cv::Point3f(vec));
  }

  calibration::ExtrinsicMetrics metrics;
  /*  "FROM" universe --> "TO" universe */
  bool pnpRetval{cv::solvePnP(toCirclePositions, fromCircleCenters,
                              fromIntrinsicMatrix, fromDistortionCoeffs,
                              metrics.rvec, metrics.tvec)};
  if (pnpRetval) {
    return metrics;
  } else {
    spdlog::warn("Error from solvePnP.");
    return std::nullopt;
  }
}

int calibration::saveMetrics(
    const calibration::CalibrationMetrics& rgbMetrics,
    const calibration::CalibrationMetrics& depthMetrics,
    const cv::Mat& tritonToHeliosExtrinsicMatrix,
    const cv::Mat& heliosToTritonExtrinsicMatrix) {
  if (!std::filesystem::exists(calibration::calibrationParamsDir)) {
    spdlog::warn("Output dir for saving metrics DNE: {}",
                 calibration::calibrationParamsDir);
    return -1;
  } else {
    spdlog::info("Saving calibration metrics to: {}",
                 calibration::calibrationParamsDir);
  }

  try {
    cv::FileStorage tritonIntrinsicFile(
        calibration::calibrationParamsDir + "triton_intrinsic_matrix.yml",
        cv::FileStorage::WRITE);
    tritonIntrinsicFile.write("Intrinsic Matrix", rgbMetrics.intrinsicMatrix);
    tritonIntrinsicFile.release();
  } catch (const cv::Exception& e) {
    spdlog::warn("OpenCV error saving triton intrinsic matrix: {}", e.what());
    return -1;
  }
  try {
    cv::FileStorage tritonDistortionFile(
        calibration::calibrationParamsDir + "triton_distortion_coeffs.yml",
        cv::FileStorage::WRITE);
    tritonDistortionFile.write("Distortion Coefficients",
                               rgbMetrics.distCoeffs);
    tritonDistortionFile.release();
  } catch (const cv::Exception& e) {
    spdlog::error("OpenCV error saving triton distortion coefficients: {}",
                  e.what());
    return -1;
  }

  try {
    cv::FileStorage heliosIntrinsicFile(
        calibration::calibrationParamsDir + "helios_intrinsic_matrix.yml",
        cv::FileStorage::WRITE);
    heliosIntrinsicFile.write("Intrinsic Matrix", depthMetrics.intrinsicMatrix);
    heliosIntrinsicFile.release();
  } catch (const cv::Exception& e) {
    spdlog::warn("OpenCV error saving helios intrinsic matrix: {}", e.what());
    return -1;
  }
  try {
    cv::FileStorage heliosDistortionFile(
        calibration::calibrationParamsDir + "helios_distortion_coeffs.yml",
        cv::FileStorage::WRITE);
    heliosDistortionFile.write("Distortion Coefficients",
                               depthMetrics.distCoeffs);
    heliosDistortionFile.release();
  } catch (const cv::Exception& e) {
    spdlog::error("OpenCV error saving helios distortion coefficients: {}",
                  e.what());
    return -1;
  }

  try {
    cv::FileStorage tritonToHeliosExtrinsicMatrixFile(
        calibration::calibrationParamsDir +
            "triton_to_helios_extrinsic_matrix.yml",
        cv::FileStorage::WRITE);
    tritonToHeliosExtrinsicMatrixFile.write("Extrinsic Matrix",
                                            tritonToHeliosExtrinsicMatrix);
    tritonToHeliosExtrinsicMatrixFile.release();
  } catch (const cv::Exception& e) {
    spdlog::error("OpenCV error saving triton to helios extrinsic matrix: {}",
                  e.what());
    return -1;
  }

  try {
    cv::FileStorage heliosToTritonExtrinsicMatrixFile(
        calibration::calibrationParamsDir +
            "helios_to_triton_extrinsic_matrix.yml",
        cv::FileStorage::WRITE);
    heliosToTritonExtrinsicMatrixFile.write("Extrinsic Matrix",
                                            heliosToTritonExtrinsicMatrix);
    heliosToTritonExtrinsicMatrixFile.release();
  } catch (const cv::Exception& e) {
    spdlog::error("OpenCV error saving helios to triton extrinsic matrix: {}",
                  e.what());
    return -1;
  }

  return 0;
}

std::optional<cv::Mat> calibration::boostIntensity(
    const cv::Mat& intensityImage) {
  if (intensityImage.empty() || intensityImage.channels() != 1) {
    spdlog::warn("Input intensity image is empty or not single-channel.");
    return std::nullopt;
  }
  cv::Mat boosted;
  double minVal, maxVal;
  cv::minMaxLoc(intensityImage, &minVal, &maxVal);
  if (maxVal > minVal) {
    // Linear stretch
    double alpha = 255.0 / (maxVal - minVal);
    double beta = -minVal * alpha;
    intensityImage.convertTo(boosted, -1, alpha, beta);
    spdlog::info("Boosting intensity image with alpha: {}, beta: {}", alpha,
                 beta);
    // Aggressive gamma correction
    cv::Mat lookupTable(1, 256, CV_8U);
    uchar* p = lookupTable.ptr();
    double gamma = 0.4;  // Very aggressive
    for (int i = 0; i < 256; ++i) {
      p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::LUT(boosted, lookupTable, boosted);
    spdlog::info("Applied gamma correction with gamma: {}", gamma);
  } else {
    intensityImage.convertTo(boosted, -1);
    spdlog::info(
        "Max and min intensity values are equal. Intensity not boosted.");
  }
  return boosted;
}

cv::Mat calibration::testUndistortion(const cv::Mat& cameraMatrix,
                                      const cv::Mat& distCoeffs,
                                      const cv::Mat& img) {
  cv::Mat newCameraMatrix, undistorted;
  cv::Rect roi;
  newCameraMatrix = cv::getOptimalNewCameraMatrix(
      cameraMatrix, distCoeffs, img.size(), 1, img.size(), &roi);
  cv::undistort(img, undistorted, cameraMatrix, distCoeffs, newCameraMatrix);
  undistorted = undistorted(roi);
  return undistorted;
}

int calibration::showWithChafa(const cv::Mat& img, const std::string& label) {
  bool hasChafa = (std::system("which chafa > /dev/null 2>&1") == 0);
  bool hasFiglet = (std::system("which figlet > /dev/null 2>&1") == 0);
  if (!hasChafa) {
    spdlog::warn("chafa not found in PATH. Skipping image display for '{}'.",
                 label);
    return -1;
  }
  if (img.empty()) {
    spdlog::warn("Image for {} is empty, skipping chafa display.", label);
    return -1;
  }
  if (!hasFiglet) {
    spdlog::warn(
        "figlet not found in PATH. Label will not be stylized for '{}'.",
        label);
    std::string tmpfile = "/tmp/" + label + ".png";
    cv::imwrite(tmpfile, img);
    std::string cmd = "chafa " + tmpfile;
    std::cout << "=== " << label << " ===" << std::endl;
    std::system(cmd.c_str());
    std::remove(tmpfile.c_str());
  } else {
    std::string tmpfile = "/tmp/" + label + ".png";
    cv::imwrite(tmpfile, img);
    std::string figletCmd = "figlet \"" + label + "\"";
    std::string chafaCmd = "chafa " + tmpfile;
    std::cout << std::endl;
    std::system(figletCmd.c_str());
    std::system(chafaCmd.c_str());
    std::remove(tmpfile.c_str());
  }
  return 0;
};

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

  std::vector<cv::Mat> rgbImages, intensityImages, xyzImages;
  std::vector<std::filesystem::directory_entry> entries;
  for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
    entries.push_back(entry);
  }
  std::sort(entries.begin(), entries.end(),
            [](const std::filesystem::directory_entry& a,
               const std::filesystem::directory_entry& b) {
              return a.path().filename().string() <
                     b.path().filename().string();
            });
  for (const auto& entry : entries) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      std::string filename = entry.path().filename().string();
      std::transform(filename.begin(), filename.end(), filename.begin(),
                     ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        cv::Mat img{cv::imread(entry.path().string())};
        if (!img.empty()) {
          if (filename.find("rgb") != std::string::npos) {
            cvtColor(img, img, cv::COLOR_BGR2GRAY);
            rgbImages.push_back(img);
            spdlog::info("Loaded RGB image: {}",
                         entry.path().filename().string());
          } else if (filename.find("intensity") != std::string::npos) {
            cvtColor(img, img, cv::COLOR_BGR2GRAY);
            intensityImages.push_back(img);
            spdlog::info("Loaded intensity image: {}",
                         entry.path().filename().string());
          }
        }
      } else if (ext == ".yml") {
        if (filename.find("xyz") != std::string::npos) {
          cv::FileStorage fs(entry.path().string(), cv::FileStorage::READ);
          if (fs.isOpened()) {
            cv::Mat xyz;
            fs["xyz"] >> xyz;
            if (!xyz.empty()) {
              xyzImages.push_back(xyz);
              spdlog::info("Loaded XYZ image: {}",
                           entry.path().filename().string());
            } else {
              spdlog::warn("Failed to load xyz data from {}",
                           entry.path().string());
            }
            fs.release();
          } else {
            spdlog::warn("Could not open xyz file: {}", entry.path().string());
          }
        }
      }
    }
  }
  if (rgbImages.size() < 9 || intensityImages.size() < 9 ||
      xyzImages.size() < 9) {
    spdlog::error("Error: Directory must contain at least 9 image files.");
    return -1;
  } else {
    spdlog::info("Directory opened successfully: \n{}", dirPath.string());
  }

  spdlog::info("Loaded {} RGB images, {} Intensity images, {} XYZ images",
               rgbImages.size(), intensityImages.size(), xyzImages.size());
  if (!rgbImages.empty()) {
    spdlog::info("First RGB image channels: {}", rgbImages[0].channels());
  }
  if (!intensityImages.empty()) {
    spdlog::info("First Intensity image channels: {}",
                 intensityImages[0].channels());
  }
  if (!xyzImages.empty()) {
    spdlog::info("First XYZ image channels: {}, type: {}, size: {}x{}",
                 xyzImages[0].channels(), xyzImages[0].type(),
                 xyzImages[0].cols, xyzImages[0].rows);
  }

  cv::Size gSize{cv::Size(5, 4)};
  std::optional<calibration::CalibrationMetrics> rgbMetrics{
      calibration::calibrateCamera(rgbImages, gSize,
                                   cv::CALIB_CB_SYMMETRIC_GRID,
                                   calibration::createBlobDetector())};

  if (rgbMetrics == std::nullopt) {
    spdlog::error("RGB Calibration failed.");
  } else {
    std::ostringstream ossi1, ossd1;
    ossi1 << (*rgbMetrics).intrinsicMatrix;
    ossd1 << (*rgbMetrics).distCoeffs;
    spdlog::info("RGB Intrinsic Matrix: \n{}", ossi1.str());
    spdlog::info("RGB Distortion Coeffs: \n{}", ossd1.str());
  }

  std::vector<cv::Mat> boostedIntensityImages;
  for (cv::Mat img : intensityImages) {
    std::optional<cv::Mat> boosted = calibration::boostIntensity(img);
    if (boosted != std::nullopt) {
      boostedIntensityImages.push_back(*boosted);
    } else {
      spdlog::warn("Failed to boost an intensity image.");
    }
  }

  std::optional<calibration::CalibrationMetrics> depthMetrics{
      calibration::calibrateCamera(boostedIntensityImages, gSize,
                                   cv::CALIB_CB_SYMMETRIC_GRID,
                                   calibration::createBlobDetector())};

  if (depthMetrics == std::nullopt) {
    spdlog::error("Could not get depth camera calibration metrics.");
  } else {
    std::ostringstream ossi2, ossd2;
    ossi2 << (*depthMetrics).intrinsicMatrix;
    ossd2 << (*depthMetrics).distCoeffs;
    spdlog::info("Depth Intrinsic Matrix: \n{}", ossi2.str());
    spdlog::info("Depth Distortion Coeffs: \n{}", ossd2.str());
  }

  cv::Mat distorted{rgbImages[5]};
  cv::Mat undistorted{calibration::testUndistortion(
      (*rgbMetrics).intrinsicMatrix, (*rgbMetrics).distCoeffs, distorted)};

  calibration::showWithChafa(distorted, "Distorted");
  calibration::showWithChafa(undistorted, "Undistorted");

  int imageIndex{0};
  cv::Mat tritonMonoImage{rgbImages[imageIndex]};
  cv::Mat heliosIntensityImage{boostedIntensityImages[imageIndex]};
  cv::Mat heliosXyzImage{xyzImages[imageIndex]};

  calibration::showWithChafa(tritonMonoImage, "triton_mono");
  calibration::showWithChafa(heliosIntensityImage, "helios_intensity");

  // For xyz image, visualize the Z channel as grayscale
  if (!heliosXyzImage.empty() && heliosXyzImage.type() == CV_32FC3) {
    std::vector<cv::Mat> xyz_channels;
    cv::split(heliosXyzImage, xyz_channels);
    cv::Mat z_vis;
    double minz, maxz;
    cv::minMaxLoc(xyz_channels[2], &minz, &maxz);
    if (maxz > minz) {
      xyz_channels[2].convertTo(z_vis, CV_8U, 255.0 / (maxz - minz),
                                -minz * 255.0 / (maxz - minz));
      calibration::showWithChafa(z_vis, "helios_xyz_flat");
    }
  }

  std::optional<cv::Mat> tritonToHeliosExtrinsicMatrix{std::nullopt},
      heliosToTritonExtrinsicMatrix{std::nullopt};

  std::optional<calibration::ExtrinsicMetrics> tritonToHeliosExtrinsicMetrics{
      calibration::getExtrinsics(
          tritonMonoImage, heliosIntensityImage, heliosXyzImage,
          (*rgbMetrics).intrinsicMatrix, (*rgbMetrics).distCoeffs, gSize,
          cv::CALIB_CB_SYMMETRIC_GRID, calibration::createBlobDetector())};

  if (tritonToHeliosExtrinsicMetrics == std::nullopt) {
    spdlog::error("Extrinsic calibration (triton to helios) failed.");
    return -1;
  } else {
    spdlog::info("Extrinsic calibration (triton to helios) successful.");
    tritonToHeliosExtrinsicMatrix = calibration::constructExtrinsicMatrix(
        tritonToHeliosExtrinsicMetrics->rvec,
        tritonToHeliosExtrinsicMetrics->tvec);
    std::ostringstream osse1;
    osse1 << *tritonToHeliosExtrinsicMatrix;
    spdlog::info("Constructed Extrinsic Matrix (triton to helios):\n{}",
                 osse1.str());
  }

  std::optional<calibration::ExtrinsicMetrics> heliosToTritonExtrinsicMetrics{
      calibration::getExtrinsics(
          heliosIntensityImage, tritonMonoImage, heliosXyzImage,
          (*depthMetrics).intrinsicMatrix, (*depthMetrics).distCoeffs, gSize,
          cv::CALIB_CB_SYMMETRIC_GRID, calibration::createBlobDetector())};

  if (heliosToTritonExtrinsicMetrics == std::nullopt) {
    spdlog::error("Extrinsic calibration (helios to triton) failed.");
  } else {
    spdlog::info("Extrinsic calibration (helios to triton) successful.");
    heliosToTritonExtrinsicMatrix = calibration::constructExtrinsicMatrix(
        heliosToTritonExtrinsicMetrics->rvec,
        heliosToTritonExtrinsicMetrics->tvec);
    std::ostringstream osse2;
    osse2 << *heliosToTritonExtrinsicMatrix;
    spdlog::info("Constructed Extrinsic Matrix (helios to triton):\n{}",
                 osse2.str());
  }

  if (rgbMetrics != std::nullopt && depthMetrics != std::nullopt &&
      tritonToHeliosExtrinsicMetrics != std::nullopt &&
      heliosToTritonExtrinsicMetrics != std::nullopt) {
    if (calibration::saveMetrics(*rgbMetrics, *depthMetrics,
                                 *tritonToHeliosExtrinsicMatrix,
                                 *heliosToTritonExtrinsicMatrix) < 0) {
      spdlog::error("Failed to save Calibration Metrics");
    } else {
      spdlog::info("Successfully saved Calibration Metrics");
    }
  }

  return 0;
}