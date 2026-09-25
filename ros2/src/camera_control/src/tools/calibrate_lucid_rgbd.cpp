#include <CLI/CLI.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "camera_control/camera/calibration.hpp"
#include "camera_control/camera/lucid_camera.hpp"
#include "common/utils.hpp"
#include "spdlog/spdlog.h"

// Reads an image in its native format and converts it to a scaled grayscale
// image based on its actual channel count. We don't force IMREAD_GRAYSCALE here
// as it would silently truncate 16-bit mono images (e.g. Helios intensity
// images) down to 8-bit. Returns an empty cv::Mat if the image could not be
// read or has an unsupported number of channels.
cv::Mat readGrayscaleImage(const std::string& imagePath,
                           double claheClipLimit = 5.0,
                           cv::Size claheTileGridSize = cv::Size(7, 7)) {
  cv::Mat raw{cv::imread(imagePath, cv::IMREAD_UNCHANGED)};
  if (raw.empty()) {
    return cv::Mat();
  }

  cv::Mat gray;
  if (raw.channels() == 1) {
    gray = raw;
  } else if (raw.channels() == 3) {
    cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);
  } else if (raw.channels() == 4) {
    cv::cvtColor(raw, gray, cv::COLOR_BGRA2GRAY);
  } else {
    spdlog::error("Unsupported number of channels ({}): {}", raw.channels(),
                  imagePath);
    return cv::Mat();
  }

  cv::Mat scaled{calibration::scaleGrayscaleImage(gray)};

  // Correct for uneven illumination (e.g. Helios intensity images get
  // noticeably dimmer away from the image center) using CLAHE. CLAHE equalizes
  // contrast within local tiles instead of globally, so dim corners get boosted
  // independently of the bright center.
  cv::Ptr<cv::CLAHE> clahe{cv::createCLAHE(claheClipLimit, claheTileGridSize)};
  cv::Mat corrected;
  clahe->apply(scaled, corrected);

  return corrected;
}

void captureFrame(double exposureUs, double gainDb,
                  const std::string& outputDir) {
  LucidCamera camera;
  camera.start(LucidCamera::CaptureMode::SINGLE_FRAME, exposureUs, gainDb);
  camera.waitForStreaming();

  auto frameOpt{camera.getNextFrame()};

  if (!frameOpt) {
    spdlog::error("Could not capture frame");
    return;
  }

  std::filesystem::path outputDirExpandedPath{common::expandUser(outputDir)};
  std::filesystem::create_directories(outputDirExpandedPath);

  LucidCamera::Frame frame{std::move(*frameOpt)};

  // Demosaic color image (which is BayerRG8) and write to file
  cv::Mat raw(frame.colorImage->height, frame.colorImage->width, CV_8UC1,
              const_cast<uint8_t*>(frame.colorImage->data.data()),
              frame.colorImage->step);
  cv::Mat bgr;
  cv::cvtColor(raw, bgr, cv::COLOR_BayerRGGB2BGR);
  std::filesystem::path colorImagePath{
      std::filesystem::path(outputDirExpandedPath) / "triton.png"};
  cv::imwrite(colorImagePath, bgr);
  spdlog::info("Saved color camera image to: {}", colorImagePath.string());

  // Depth intensity is MONO16
  // Wrap image buffer as cv::Mat
  cv::Mat intens(frame.depthIntensity->height, frame.depthIntensity->width,
                 CV_16UC1,
                 const_cast<uint8_t*>(frame.depthIntensity->data.data()),
                 frame.depthIntensity->step);
  std::filesystem::path depthIntensityImagePath{
      std::filesystem::path(outputDirExpandedPath) / "helios_intensity.png"};
  cv::imwrite(depthIntensityImagePath, intens);
  spdlog::info("Saved depth camera intensity image to: {}",
               depthIntensityImagePath.string());

  // Wrap image buffer as cv::Mat
  cv::Mat xyzMat(frame.depthXyz->height, frame.depthXyz->width, CV_32FC3,
                 const_cast<uint8_t*>(frame.depthXyz->data.data()),
                 frame.depthXyz->step);
  std::filesystem::path xyzPath{std::filesystem::path(outputDirExpandedPath) /
                                "helios_xyz.yml"};
  cv::FileStorage fs{xyzPath, cv::FileStorage::WRITE};
  fs << "xyz" << xyzMat;
  fs.release();
  spdlog::info("Saved xyz data to: {}", xyzPath.string());
}

// Evaluates a set of intrinsics against images of the calibration pattern and
// writes evaluation images to outputDir.
void validateIntrinsics(const std::vector<std::string>& imagePaths,
                        const std::vector<cv::Mat>& images,
                        const cv::Size& gridSize,
                        const cv::Mat& intrinsicMatrix,
                        const cv::Mat& distCoeffs,
                        const std::filesystem::path& outputDir) {
  spdlog::info("Evaluating intrinsics...");

  std::filesystem::create_directories(outputDir);

  // Object points use unit grid spacing. Reprojection error is independent of
  // the grid's physical scale.
  std::vector<cv::Point3f> objectPoints;
  for (int i = 0; i < gridSize.height; ++i) {
    for (int j = 0; j < gridSize.width; ++j) {
      objectPoints.emplace_back(j, i, 0);
    }
  }

  auto blobDetector{calibration::createBlobDetector()};
  cv::Size imageSize{images.front().size()};
  int markerSize{std::max(1, imageSize.width / 400)};
  double fontScale{0.4 * markerSize};
  int fontThickness{std::max(1, markerSize / 2)};
  cv::Point textOrigin{10, 15 * markerSize};
  cv::Mat coverageImg{cv::Mat::zeros(imageSize, CV_8UC3)};
  double residualsArrowScale{20.};

  for (size_t i = 0; i < images.size(); ++i) {
    std::string name{std::filesystem::path(imagePaths[i]).stem().string()};
    auto centersOpt{calibration::findCircleGridCenters(
        images[i], gridSize, cv::CALIB_CB_SYMMETRIC_GRID, blobDetector)};
    if (!centersOpt) {
      spdlog::warn("[validateIntrinsics] Could not get circle centers from {}",
                   imagePaths[i]);
      continue;
    }
    const std::vector<cv::Point2f>& detected{*centersOpt};

    cv::Mat rvec, tvec;
    if (!cv::solvePnP(objectPoints, detected, intrinsicMatrix, distCoeffs, rvec,
                      tvec)) {
      spdlog::warn("[validateIntrinsics] Could not solve pose for {}",
                   imagePaths[i]);
      continue;
    }
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(objectPoints, rvec, tvec, intrinsicMatrix, distCoeffs,
                      reprojected);

    cv::Mat residualsImg;
    cv::cvtColor(images[i], residualsImg, cv::COLOR_GRAY2BGR);
    double sqError{0.0};
    double maxError{0.0};
    for (size_t p = 0; p < detected.size(); ++p) {
      double err{cv::norm(reprojected[p] - detected[p])};
      sqError += err * err;
      maxError = std::max(maxError, err);

      cv::circle(residualsImg, detected[p], 4 * markerSize,
                 cv::Scalar(0, 255, 0), markerSize, cv::LINE_AA);
      cv::drawMarker(residualsImg, reprojected[p], cv::Scalar(0, 0, 255),
                     cv::MARKER_CROSS, 6 * markerSize, markerSize, cv::LINE_AA);
      cv::arrowedLine(
          residualsImg, detected[p],
          detected[p] + (reprojected[p] - detected[p]) * residualsArrowScale,
          cv::Scalar(255, 0, 255), markerSize, cv::LINE_AA, 0, 0.2);

      // Draw a circle on the coverage image where green is error == 0 and red
      // is error >= 1
      double t{std::clamp(err, 0.0, 1.0)};
      cv::circle(coverageImg, detected[p], 3 * markerSize,
                 cv::Scalar(0, 255 * (1.0 - t), 255 * t), cv::FILLED,
                 cv::LINE_AA);
    }
    double rmsError{std::sqrt(sqError / detected.size())};
    spdlog::info(
        "[validateIntrinsics] {}: RMS error {:.4f}px, max error {:.4f}px", name,
        rmsError, maxError);
    cv::putText(residualsImg,
                fmt::format("RMS {:.3f}px (arrows scaled x{})", rmsError,
                            residualsArrowScale),
                textOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(255, 0, 255), fontThickness, cv::LINE_AA);
    cv::imwrite(outputDir / (name + "_residuals.png"), residualsImg);
  }

  cv::putText(coverageImg, "Reprojection error: green=0px, red>=1px",
              textOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale,
              cv::Scalar(255, 255, 255), fontThickness, cv::LINE_AA);
  cv::imwrite(outputDir / "coverage.png", coverageImg);
  spdlog::info("Saved intrinsics validation images to: {}", outputDir.string());
}

void calculateIntrinsics(const std::string& imagesDir,
                         const std::string& outputDir) {
  std::filesystem::path imagesDirExpandedPath{common::expandUser(imagesDir)};
  if (!std::filesystem::exists(imagesDirExpandedPath) ||
      !std::filesystem::is_directory(imagesDirExpandedPath)) {
    spdlog::error("Provided path is not a valid directory: {}",
                  imagesDirExpandedPath.string());
    return;
  }

  std::vector<std::string> candidatePaths;
  for (const auto& entry :
       std::filesystem::directory_iterator(imagesDirExpandedPath)) {
    if (entry.is_regular_file()) {
      auto ext{entry.path().extension().string()};
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        candidatePaths.push_back(entry.path().string());
      }
    }
  }
  std::sort(candidatePaths.begin(), candidatePaths.end());

  std::vector<std::string> imagePaths;
  std::vector<cv::Mat> images;
  for (const auto& path : candidatePaths) {
    cv::Mat img{readGrayscaleImage(path)};
    if (!img.empty()) {
      imagePaths.push_back(path);
      images.push_back(img);
    }
  }

  if (images.size() < 9) {
    spdlog::error(
        "Directory must contain at least 9 image (.png, .jpg, .jpeg) files.");
    return;
  }

  spdlog::info("Found {} images in {}", images.size(),
               imagesDirExpandedPath.string());
  auto calibrateResultsOpt{calibration::calculateIntrinsics(
      images, cv::Size(5, 4), cv::CALIB_CB_SYMMETRIC_GRID,
      calibration::createBlobDetector())};
  if (!calibrateResultsOpt) {
    spdlog::error("Calibration failed");
    return;
  }

  auto calibrateResults{std::move(*calibrateResultsOpt)};
  std::ostringstream oss1, oss2;
  oss1 << calibrateResults.intrinsicMatrix;
  oss2 << calibrateResults.distCoeffs;
  spdlog::info("Calibrated intrins: \n{}", oss1.str());
  spdlog::info("Distortion coeffs: \n{}", oss2.str());

  std::filesystem::path outputDirExpandedPath{common::expandUser(outputDir)};
  std::filesystem::create_directories(outputDirExpandedPath);

  std::filesystem::path intrinsicsPath{
      std::filesystem::path(outputDirExpandedPath) / "intrinsics.yml"};
  cv::FileStorage fs{intrinsicsPath, cv::FileStorage::WRITE};
  fs << "intrinsicMatrix" << calibrateResults.intrinsicMatrix;
  fs << "distCoeffs" << calibrateResults.distCoeffs;
  fs.release();
  spdlog::info("Saved intrinsics data to: {}", intrinsicsPath.string());

  validateIntrinsics(imagePaths, images, cv::Size(5, 4),
                     calibrateResults.intrinsicMatrix,
                     calibrateResults.distCoeffs,
                     outputDirExpandedPath / "intrinsics_validation");
}

std::optional<Arena::DeviceInfo> findFirstDeviceWithModelPrefix(
    std::vector<Arena::DeviceInfo>& deviceInfos,
    const std::vector<std::string>& modelPrefixes) {
  auto it{std::find_if(deviceInfos.begin(), deviceInfos.end(),
                       [&modelPrefixes](Arena::DeviceInfo& deviceInfo) {
                         return std::any_of(
                             modelPrefixes.begin(), modelPrefixes.end(),
                             [&deviceInfo](const std::string& prefix) {
                               return std::strncmp(
                                          deviceInfo.ModelName().c_str(),
                                          prefix.c_str(), prefix.length()) == 0;
                             });
                       })};
  if (it != deviceInfos.end()) {
    return *it;
  }
  return std::nullopt;
}

std::optional<Arena::DeviceInfo> findDeviceWithSerial(
    std::vector<Arena::DeviceInfo>& deviceInfos,
    const std::string& serialNumber) {
  auto it{std::find_if(
      deviceInfos.begin(), deviceInfos.end(),
      [&serialNumber](Arena::DeviceInfo& deviceInfo) {
        return deviceInfo.SerialNumber().length() == serialNumber.length() &&
               std::strncmp(deviceInfo.SerialNumber().c_str(),
                            serialNumber.c_str(), serialNumber.length()) == 0;
      })};
  if (it != deviceInfos.end()) {
    return *it;
  }
  return std::nullopt;
}

// Reads the a camera's built-in factory intrinsic calibration from its
// GenICam device nodes. See:
// https://support.thinklucid.com/knowledgebase/projecting-3d-image-to-and-from-helios-to-2d-image/
calibration::IntrinsicsResult readIntrinsicsFromDevice(
    Arena::IDevice* device, int numDistortionCoeffs = 5) {
  GenApi::INodeMap* nodeMap{device->GetNodeMap()};

  cv::Mat intrinsicMatrix{cv::Mat::eye(3, 3, CV_64F)};
  intrinsicMatrix.at<double>(0, 0) =
      Arena::GetNodeValue<double>(nodeMap, "CalibFocalLengthX");
  intrinsicMatrix.at<double>(1, 1) =
      Arena::GetNodeValue<double>(nodeMap, "CalibFocalLengthY");
  intrinsicMatrix.at<double>(0, 2) =
      Arena::GetNodeValue<double>(nodeMap, "CalibOpticalCenterX");
  intrinsicMatrix.at<double>(1, 2) =
      Arena::GetNodeValue<double>(nodeMap, "CalibOpticalCenterY");

  cv::Mat distCoeffs{cv::Mat::zeros(numDistortionCoeffs, 1, CV_64F)};
  for (int i = 0; i < numDistortionCoeffs; ++i) {
    Arena::SetNodeValue<GenICam::gcstring>(
        nodeMap, "CalibLensDistortionValueSelector",
        GenICam::gcstring(("Value" + std::to_string(i)).c_str()));
    distCoeffs.at<double>(i, 0) =
        Arena::GetNodeValue<double>(nodeMap, "CalibLensDistortionValue");
  }

  return calibration::IntrinsicsResult{intrinsicMatrix, distCoeffs};
}

// Pulls the Helios camera's factory intrinsic matrix and distortion
// coefficients directly from its GenICam device nodes.
void getHeliosDeviceIntrinsics(const std::optional<std::string>& serialNumber,
                               const std::string& outputDir) {
  Arena::ISystem* arena{Arena::OpenSystem()};

  try {
    arena->UpdateDevices(1000);
    std::vector<Arena::DeviceInfo> deviceInfos{arena->GetDevices()};

    std::optional<Arena::DeviceInfo> depthDeviceInfo{
        serialNumber
            ? findDeviceWithSerial(deviceInfos, *serialNumber)
            : findFirstDeviceWithModelPrefix(
                  deviceInfos, LucidCamera::DEPTH_CAMERA_MODEL_PREFIXES)};
    if (!depthDeviceInfo) {
      spdlog::error(
          "Could not find a Helios (depth) camera device ({} device(s) "
          "enumerated)",
          deviceInfos.size());
    } else {
      spdlog::info("Connecting to Helios device (model={}, serial={})",
                   depthDeviceInfo->ModelName(),
                   depthDeviceInfo->SerialNumber());
      Arena::IDevice* depthDevice{arena->CreateDevice(*depthDeviceInfo)};

      calibration::IntrinsicsResult result{
          readIntrinsicsFromDevice(depthDevice)};

      arena->DestroyDevice(depthDevice);

      std::ostringstream oss1, oss2;
      oss1 << result.intrinsicMatrix;
      oss2 << result.distCoeffs;
      spdlog::info("Device intrinsic matrix: \n{}", oss1.str());
      spdlog::info("Device distortion coeffs: \n{}", oss2.str());

      std::filesystem::path outputDirExpandedPath{
          common::expandUser(outputDir)};
      std::filesystem::create_directories(outputDirExpandedPath);

      std::filesystem::path intrinsicsPath{
          std::filesystem::path(outputDirExpandedPath) / "intrinsics.yml"};
      cv::FileStorage fs{intrinsicsPath, cv::FileStorage::WRITE};
      fs << "intrinsicMatrix" << result.intrinsicMatrix;
      fs << "distCoeffs" << result.distCoeffs;
      fs.release();
      spdlog::info("Saved intrinsics data to: {}", intrinsicsPath.string());
    }
  } catch (const GenICam::GenericException& e) {
    spdlog::error("GenICam exception: {}", e.what());
  } catch (const std::exception& e) {
    spdlog::error("Exception: {}", e.what());
  }

  Arena::CloseSystem(arena);
}

void undistortImage(const std::string& intrinsicsFile,
                    const std::string& imageFile,
                    const std::string& outputFile) {
  // Parse intrinsics file
  auto intrinsicsOpt{calibration::readIntrinsicsFile(intrinsicsFile)};
  if (!intrinsicsOpt) {
    return;
  }
  auto [intrinsicMatrix, distCoeffs]{std::move(*intrinsicsOpt)};

  // Read image file
  std::filesystem::path imageFileExpandedPath{common::expandUser(imageFile)};
  cv::Mat img{cv::imread(imageFileExpandedPath)};

  cv::Rect roi;
  cv::Mat newCameraMatrix{cv::getOptimalNewCameraMatrix(
      intrinsicMatrix, distCoeffs, img.size(), 1, img.size(), &roi)};
  cv::Mat undistorted;
  cv::undistort(img, undistorted, intrinsicMatrix, distCoeffs, newCameraMatrix);
  undistorted = undistorted(roi);

  std::filesystem::path outputFileExpandedPath{common::expandUser(outputFile)};
  cv::imwrite(outputFileExpandedPath, undistorted);
}

struct ViewCorrespondences {
  std::string name;
  std::string cameraImagePath;
  std::vector<cv::Point2f> cameraPts;
  std::vector<cv::Point3f> xyzPts;
};

// For each camera image, finds the Helios intensity image and XYZ data with the
// same file stem, detects the circle grid in both images, and pairs each camera
// circle center with the XYZ position at the corresponding Helios circle
// center. Circles whose XYZ position is invalid (non-finite or z <= 0) are
// dropped. Returns std::nullopt if any of the directories are invalid or an
// XYZ file could not be read.
std::optional<std::vector<ViewCorrespondences>>
collectXyzToCameraCorrespondences(const std::string& cameraImagesDir,
                                  const std::string& heliosImagesDir,
                                  const std::string& heliosXyzDir) {
  // Find camera image paths
  std::filesystem::path cameraImagesExpandedPath{
      common::expandUser(cameraImagesDir)};
  if (!std::filesystem::exists(cameraImagesExpandedPath) ||
      !std::filesystem::is_directory(cameraImagesExpandedPath)) {
    spdlog::error("Provided path is not a valid directory: {}",
                  cameraImagesExpandedPath.string());
    return std::nullopt;
  }
  std::vector<std::string> cameraImagePaths;
  for (auto& entry :
       std::filesystem::directory_iterator(cameraImagesExpandedPath)) {
    if (entry.is_regular_file()) {
      auto ext{entry.path().extension().string()};
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        cameraImagePaths.push_back(entry.path().string());
      }
    }
  }
  std::sort(cameraImagePaths.begin(), cameraImagePaths.end());

  // Ensure Helios intensity image and XYZ data directories are valid
  std::filesystem::path heliosImagesExpandedPath{
      common::expandUser(heliosImagesDir)};
  if (!std::filesystem::exists(heliosImagesExpandedPath) ||
      !std::filesystem::is_directory(heliosImagesExpandedPath)) {
    spdlog::error("Provided path is not a valid directory: {}",
                  heliosImagesExpandedPath.string());
    return std::nullopt;
  }
  std::filesystem::path heliosXyzExpandedPath{common::expandUser(heliosXyzDir)};
  if (!std::filesystem::exists(heliosXyzExpandedPath) ||
      !std::filesystem::is_directory(heliosXyzExpandedPath)) {
    spdlog::error("Provided path is not a valid directory: {}",
                  heliosXyzExpandedPath.string());
    return std::nullopt;
  }

  std::vector<ViewCorrespondences> views;
  auto blobDetector{calibration::createBlobDetector()};

  // For each camera image, find the corresponding Helios intensity image and
  // XYZ data
  for (const auto& cameraImagePath : cameraImagePaths) {
    std::string baseName{
        std::filesystem::path(cameraImagePath).stem().string()};

    // Find corresponding Helios image
    std::filesystem::path heliosImagePath;
    for (auto ext : {".jpg", ".jpeg", ".png"}) {
      std::filesystem::path candidate{heliosImagesExpandedPath /
                                      (baseName + ext)};
      if (std::filesystem::exists(candidate)) {
        heliosImagePath = candidate;
        break;
      }
    }
    if (heliosImagePath.empty()) {
      spdlog::warn(
          "Could not find corresponding Helios intensity image for {}. "
          "Skipping image.",
          cameraImagePath);
      continue;
    }

    // Find corresponding Helios XYZ data
    std::filesystem::path heliosXyzFilePath;
    for (auto ext : {".yml", ".yaml"}) {
      std::filesystem::path candidate{heliosXyzExpandedPath / (baseName + ext)};
      if (std::filesystem::exists(candidate)) {
        heliosXyzFilePath = candidate;
        break;
      }
    }
    if (heliosXyzFilePath.empty()) {
      spdlog::warn(
          "Could not find corresponding Helios XYZ data for {}. Skipping "
          "image.",
          cameraImagePath);
      continue;
    }

    spdlog::info("Processing {}", baseName);
    spdlog::info("  Camera image file: {}", cameraImagePath);
    spdlog::info("  Helios image file: {}", heliosImagePath.string());
    spdlog::info("  Helios XYZ file: {}", heliosXyzFilePath.string());

    // Get circle centers in camera image
    cv::Mat cameraImg{readGrayscaleImage(cameraImagePath)};
    auto circleCoordsOpt{calibration::findCircleGridCenters(
        cameraImg, cv::Size(5, 4), cv::CALIB_CB_SYMMETRIC_GRID, blobDetector)};
    if (!circleCoordsOpt) {
      spdlog::warn("Could not get circle centers from {}", cameraImagePath);
      continue;
    }
    std::vector<cv::Point2f> circleCoords{std::move(*circleCoordsOpt)};

    // Get circle centers in Helios image
    cv::Mat heliosImg{readGrayscaleImage(heliosImagePath.string())};
    auto heliosCircleCoordsOpt{calibration::findCircleGridCenters(
        heliosImg, cv::Size(5, 4), cv::CALIB_CB_SYMMETRIC_GRID, blobDetector)};
    if (!heliosCircleCoordsOpt) {
      spdlog::warn("Could not get circle centers from {}",
                   heliosImagePath.string());
      continue;
    }
    std::vector<cv::Point2f> heliosCircleCoords{
        std::move(*heliosCircleCoordsOpt)};

    // Parse XYZ data file
    cv::FileStorage xyzFileFs{heliosXyzFilePath, cv::FileStorage::READ};
    if (!xyzFileFs.isOpened() || xyzFileFs["xyz"].isNone()) {
      spdlog::error("Could not read XYZ file: {}", heliosXyzFilePath.string());
      return std::nullopt;
    }
    cv::Mat heliosXyz;
    xyzFileFs["xyz"] >> heliosXyz;
    xyzFileFs.release();

    // Get corresponding XYZ value from the XYZ data, dropping circles without
    // a valid depth measurement
    ViewCorrespondences view{baseName, cameraImagePath, {}, {}};
    for (size_t i = 0; i < heliosCircleCoords.size(); ++i) {
      const cv::Point2f& pt{heliosCircleCoords[i]};
      // Access XYZ at [y, x]
      cv::Vec3f xyz{heliosXyz.at<cv::Vec3f>(cvRound(pt.y), cvRound(pt.x))};
      if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) ||
          !std::isfinite(xyz[2]) || xyz[2] <= 0.0f) {
        continue;
      }
      view.cameraPts.push_back(circleCoords[i]);
      view.xyzPts.emplace_back(xyz[0], xyz[1], xyz[2]);
    }
    size_t numDropped{heliosCircleCoords.size() - view.cameraPts.size()};
    if (numDropped > 0) {
      spdlog::warn("  Dropped {} circle(s) with invalid XYZ data", numDropped);
    }
    if (view.cameraPts.empty()) {
      continue;
    }

    views.push_back(std::move(view));
  }

  return views;
}

void calculateExtrinsicsXyzToCamera(const std::string& cameraIntrinsicsFile,
                                    const std::string& cameraImagesDir,
                                    const std::string& heliosImagesDir,
                                    const std::string& heliosXyzDir,
                                    const std::string& outputDir) {
  // Parse intrinsics file
  auto intrinsicsOpt{calibration::readIntrinsicsFile(cameraIntrinsicsFile)};
  if (!intrinsicsOpt) {
    return;
  }
  auto [intrinsicMatrix, distCoeffs]{std::move(*intrinsicsOpt)};

  auto viewsOpt{collectXyzToCameraCorrespondences(
      cameraImagesDir, heliosImagesDir, heliosXyzDir)};
  if (!viewsOpt) {
    return;
  }
  std::vector<ViewCorrespondences> views{std::move(*viewsOpt)};

  std::filesystem::path outputDirExpandedPath{common::expandUser(outputDir)};
  std::filesystem::create_directories(outputDirExpandedPath);

  std::vector<cv::Point2f> allCircleCoords;
  std::vector<cv::Point3f> allCircleXyzPositions;
  for (const auto& view : views) {
    allCircleCoords.insert(allCircleCoords.end(), view.cameraPts.begin(),
                           view.cameraPts.end());
    allCircleXyzPositions.insert(allCircleXyzPositions.end(),
                                 view.xyzPts.begin(), view.xyzPts.end());
  }

  if (allCircleCoords.empty() || allCircleXyzPositions.empty()) {
    spdlog::error("No suitable correspondences found.");
    return;
  }

  spdlog::info("Total number of point correspondences found: {}",
               allCircleCoords.size());

  cv::Mat rvec, tvec;
  bool ok{cv::solvePnP(allCircleXyzPositions, allCircleCoords, intrinsicMatrix,
                       distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE)};
  // Note: solvePnPRansac may perform better when we have extreme outliers
  if (!ok) {
    spdlog::error("Could not calculate extrinsic matrix.");
    return;
  }

  // Construct extrinsic matrix and write to file
  cv::Mat extrinsicMatrix{calibration::constructExtrinsicMatrix(rvec, tvec)};

  std::filesystem::path extrinsicsPath{
      std::filesystem::path(outputDirExpandedPath) / "extrinsics.yml"};
  cv::FileStorage fs{extrinsicsPath, cv::FileStorage::WRITE};
  fs << "extrinsicMatrix" << extrinsicMatrix;
  fs.release();
  spdlog::info("Saved extrinsics data to: {}", extrinsicsPath.string());

  // Calculate reprojection error, overall and per view, and write a residuals
  // image for each view
  spdlog::info("Evaluating extrinsics...");
  std::filesystem::path validationDir{outputDirExpandedPath /
                                      "extrinsics_validation"};
  std::filesystem::create_directories(validationDir);
  double residualsArrowScale{20.};
  double totalError{0.0};
  std::vector<double> viewMeanErrors;
  std::vector<double> viewMaxErrors;
  for (const auto& view : views) {
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(view.xyzPts, rvec, tvec, intrinsicMatrix, distCoeffs,
                      reprojected);

    cv::Mat cameraImg{readGrayscaleImage(view.cameraImagePath)};
    cv::Mat residualsImg;
    cv::cvtColor(cameraImg, residualsImg, cv::COLOR_GRAY2BGR);
    int markerSize{std::max(1, residualsImg.cols / 400)};
    double fontScale{0.4 * markerSize};
    int fontThickness{std::max(1, markerSize / 2)};
    cv::Point textOrigin{10, 15 * markerSize};

    double viewTotalError{0.0};
    double viewMaxError{0.0};
    for (size_t i = 0; i < view.cameraPts.size(); ++i) {
      double err{cv::norm(view.cameraPts[i] - reprojected[i])};
      viewTotalError += err;
      viewMaxError = std::max(viewMaxError, err);

      cv::circle(residualsImg, view.cameraPts[i], 4 * markerSize,
                 cv::Scalar(0, 255, 0), markerSize, cv::LINE_AA);
      cv::drawMarker(residualsImg, reprojected[i], cv::Scalar(0, 0, 255),
                     cv::MARKER_CROSS, 6 * markerSize, markerSize, cv::LINE_AA);
      cv::arrowedLine(residualsImg, view.cameraPts[i],
                      view.cameraPts[i] + (reprojected[i] - view.cameraPts[i]) *
                                              residualsArrowScale,
                      cv::Scalar(255, 0, 255), markerSize, cv::LINE_AA, 0, 0.2);
    }
    double viewMeanError{viewTotalError / view.cameraPts.size()};
    totalError += viewTotalError;
    viewMeanErrors.push_back(viewMeanError);
    viewMaxErrors.push_back(viewMaxError);

    cv::putText(residualsImg,
                fmt::format("Mean {:.3f}px, max {:.3f}px (arrows scaled x{})",
                            viewMeanError, viewMaxError, residualsArrowScale),
                textOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(255, 0, 255), fontThickness, cv::LINE_AA);
    cv::imwrite(validationDir / (view.name + "_residuals.png"), residualsImg);
  }
  spdlog::info("Saved extrinsics validation images to: {}",
               validationDir.string());
  double meanError{totalError / allCircleCoords.size()};

  spdlog::info("Per-view reprojection error:");
  spdlog::info("  {:<16} {:>8} {:>10} {:>10}", "view", "points", "mean (px)",
               "max (px)");
  for (size_t v = 0; v < views.size(); ++v) {
    spdlog::info("  {:<16} {:>8} {:>10.4f} {:>10.4f}", views[v].name,
                 views[v].cameraPts.size(), viewMeanErrors[v],
                 viewMaxErrors[v]);
  }
  spdlog::info("Reprojection error (mean): {}", meanError);
  spdlog::info("Translation magnitude: {:.2f} mm", cv::norm(tvec));
  spdlog::info("Rotation angle: {:.3f} deg", cv::norm(rvec) * 180.0 / CV_PI);
}

void visualizeExtrinsics(const std::string& cameraImageFile,
                         const std::string& heliosIntensityImageFile,
                         const std::string& heliosXyzFile,
                         const std::string& cameraIntrinsicsFile,
                         const std::string& xyzToCameraExtrinsicsFile,
                         const std::string& outputFile) {
  // Parse intrinsics file
  auto intrinsicsOpt{calibration::readIntrinsicsFile(cameraIntrinsicsFile)};
  if (!intrinsicsOpt) {
    return;
  }
  auto [intrinsicMatrix, distCoeffs]{std::move(*intrinsicsOpt)};

  // Parse extrinsics file
  auto extrinsicMatrixOpt{
      calibration::readExtrinsicsFile(xyzToCameraExtrinsicsFile)};
  if (!extrinsicMatrixOpt) {
    return;
  }
  cv::Mat extrinsicMatrix{std::move(*extrinsicMatrixOpt)};
  auto [rvec, tvec]{calibration::extractPoseFromExtrinsic(extrinsicMatrix)};

  // Read image files
  std::filesystem::path cameraImageFileExpandedPath{
      common::expandUser(cameraImageFile)};
  cv::Mat cameraImg{readGrayscaleImage(cameraImageFileExpandedPath.string())};
  std::filesystem::path heliosIntensityimageFileExpandedPath{
      common::expandUser(heliosIntensityImageFile)};
  cv::Mat heliosIntensityImg{
      readGrayscaleImage(heliosIntensityimageFileExpandedPath.string())};

  // Read XYZ file
  auto heliosXyzOpt{calibration::readXyzFile(heliosXyzFile)};
  if (!heliosXyzOpt) {
    return;
  }
  auto heliosXyz{std::move(*heliosXyzOpt)};

  // Prepare object points
  int h{heliosXyz.rows};
  int w{heliosXyz.cols};
  int c{heliosXyz.channels()};
  heliosXyz = heliosXyz.reshape(c, h * w);

  // Project points
  cv::Mat projectedPoints;
  cv::projectPoints(heliosXyz, rvec, tvec, intrinsicMatrix, distCoeffs,
                    projectedPoints);
  projectedPoints = projectedPoints.reshape(2, h * w);

  // Generate image
  int cameraH{cameraImg.rows};
  int cameraW{cameraImg.cols};
  cv::Mat projectionImg{cv::Mat::zeros(cameraH, cameraW, CV_8UC3)};
  // Render camera frame as red
  for (int r = 0; r < cameraH; ++r) {
    for (int c = 0; c < cameraW; ++c) {
      projectionImg.at<cv::Vec3b>(r, c)[2] = 255 - cameraImg.at<uint8_t>(r, c);
    }
  }
  // Render projected XYZ points as green
  heliosIntensityImg = heliosIntensityImg.reshape(1, h * w);  // flatten
  for (int i = 0; i < h * w; ++i) {
    cv::Point2f pt{projectedPoints.at<cv::Point2f>(i)};
    int col{cvRound(pt.x)};
    int row{cvRound(pt.y)};
    if (0 <= col && col < cameraW && 0 <= row && row < cameraH) {
      uint8_t intensity{heliosIntensityImg.at<uint8_t>(i)};
      projectionImg.at<cv::Vec3b>(row, col)[1] = intensity;
    }
  }

  std::filesystem::path outputFileExpandedPath{common::expandUser(outputFile)};
  cv::imwrite(outputFileExpandedPath, projectionImg);
}

int main(int argc, char* argv[]) {
  CLI::App app{"Helper program for calibrating LUCID cameras"};

  auto captureFrameCommand{
      app.add_subcommand("capture_frame", "Capture a frame")};
  double exposureUs{-1.0};
  double gainDb{-1.0};
  std::string outputDir;
  captureFrameCommand
      ->add_option("--exposure_us", exposureUs, "Exposure time in microseconds")
      ->default_val("-1.0");
  captureFrameCommand->add_option("--gain_db", gainDb, "Gain (dB)")
      ->default_val("-1.0");
  captureFrameCommand
      ->add_option("-o,--output_dir", outputDir, "Directory to save images")
      ->required();

  auto calculateIntrinsicsCommand{
      app.add_subcommand("calculate_intrinsics",
                         "Calculate camera intrinsics and distortion "
                         "coefficients from images of a calibration pattern")};
  std::string imagesDir;
  calculateIntrinsicsCommand
      ->add_option("-i,--images_dir", imagesDir,
                   "Path to the directory containing grayscale images of a "
                   "calibration pattern")
      ->required();
  calculateIntrinsicsCommand
      ->add_option("-o,--output_dir", outputDir,
                   "Path to the directory to write intrinsic parameters to")
      ->required();

  auto getHeliosDeviceIntrinsicsCommand{app.add_subcommand(
      "get_helios_device_intrinsics",
      "Pull the Helios camera's factory intrinsic matrix and distortion "
      "coefficients from the device")};
  std::string serialNumber;
  getHeliosDeviceIntrinsicsCommand
      ->add_option("--serial_number", serialNumber,
                   "Serial number of the Helios device to query. If "
                   "omitted, the first connected Helios-model device is "
                   "used.")
      ->default_val("");
  getHeliosDeviceIntrinsicsCommand
      ->add_option("-o,--output_dir", outputDir,
                   "Path to the directory to write intrinsic parameters to")
      ->required();

  auto undistortImageCommand{app.add_subcommand(
      "undistort_image", "Undistort an image using the camera intrinsics")};
  std::string intrinsicsFile;
  std::string imageFile;
  std::string outputFile;
  undistortImageCommand
      ->add_option("--intrinsics_file", intrinsicsFile,
                   "yml file containing camera intrinsics")
      ->required();
  undistortImageCommand
      ->add_option("-i,--image_file", imageFile, "Image to undistort")
      ->required();
  undistortImageCommand
      ->add_option("-o,--output_file", outputFile,
                   "Path to write undistorted image")
      ->required();

  auto calculateExtrinsicsXyzToTritonCommand{app.add_subcommand(
      "calculate_extrinsics_xyz_to_triton",
      "Calculate extrinsics that describe the orientation of Triton relative "
      "to Helios XYZ from images of a calibration pattern")};
  std::string tritonImagesDir;
  std::string heliosImagesDir;
  std::string heliosXyzDir;
  calculateExtrinsicsXyzToTritonCommand
      ->add_option("--triton_intrinsics_file", intrinsicsFile,
                   "yml file containing Triton camera intrinsics")
      ->required();
  calculateExtrinsicsXyzToTritonCommand
      ->add_option("--triton_images_dir", tritonImagesDir,
                   "Path to directory containing Triton images")
      ->required();
  calculateExtrinsicsXyzToTritonCommand
      ->add_option("--helios_images_dir", heliosImagesDir,
                   "Path to directory containing Helios intensity images")
      ->required();
  calculateExtrinsicsXyzToTritonCommand
      ->add_option("--helios_xyz_dir", heliosXyzDir,
                   "Path to directory containing Helios XYZ data")
      ->required();
  calculateExtrinsicsXyzToTritonCommand
      ->add_option("-o,--output_dir", outputDir,
                   "Path to the directory to write extrinsic parameters to")
      ->required();

  auto visualizeExtrinsicsCommand{
      app.add_subcommand("visualize_extrinsics",
                         "Verify extrinsics between a camera and Helios XYZ by "
                         "projecting XYZ onto the camera image")};
  std::string cameraImageFile;
  std::string heliosIntensityImageFile;
  std::string heliosXyzFile;
  std::string extrinsicsFile;
  visualizeExtrinsicsCommand->add_option("--camera_image_file", cameraImageFile)
      ->required();
  visualizeExtrinsicsCommand
      ->add_option("--helios_intensity_image_file", heliosIntensityImageFile)
      ->required();
  visualizeExtrinsicsCommand->add_option("--helios_xyz_file", heliosXyzFile)
      ->required();
  visualizeExtrinsicsCommand->add_option("--intrinsics_file", intrinsicsFile)
      ->required();
  visualizeExtrinsicsCommand->add_option("--extrinsics_file", extrinsicsFile)
      ->required();
  visualizeExtrinsicsCommand
      ->add_option("-o,--output_file", outputFile,
                   "Path to write visualization image")
      ->required();

  CLI11_PARSE(app, argc, argv);

  if (*captureFrameCommand) {
    captureFrame(exposureUs, gainDb, outputDir);
  } else if (*calculateIntrinsicsCommand) {
    calculateIntrinsics(imagesDir, outputDir);
  } else if (*getHeliosDeviceIntrinsicsCommand) {
    getHeliosDeviceIntrinsics(
        serialNumber.empty() ? std::nullopt : std::make_optional(serialNumber),
        outputDir);
  } else if (*undistortImageCommand) {
    undistortImage(intrinsicsFile, imageFile, outputFile);
  } else if (*calculateExtrinsicsXyzToTritonCommand) {
    calculateExtrinsicsXyzToCamera(intrinsicsFile, tritonImagesDir,
                                   heliosImagesDir, heliosXyzDir, outputDir);
  } else if (*visualizeExtrinsicsCommand) {
    visualizeExtrinsics(cameraImageFile, heliosIntensityImageFile,
                        heliosXyzFile, intrinsicsFile, extrinsicsFile,
                        outputFile);
  }

  return 0;
}