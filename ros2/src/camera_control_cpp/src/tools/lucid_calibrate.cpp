#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "camera_control_cpp/camera/calibration.hpp"
#include "camera_control_cpp/camera/lucid_camera.hpp"
#include "spdlog/spdlog.h"

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

  std::filesystem::path outputDirExpandedPath{
      calibration::expandUser(outputDir)};
  std::filesystem::create_directories(outputDirExpandedPath);

  LucidCamera::Frame frame{std::move(*frameOpt)};

  // Demosaic color image (which is BayerRG8) and write to file
  cv::Mat raw(frame.colorImage->height, frame.colorImage->width, CV_8UC1,
              const_cast<uint8_t*>(frame.colorImage->data.data()),
              frame.colorImage->step);
  cv::Mat rgb;
  cv::cvtColor(raw, rgb, cv::COLOR_BayerRG2RGB);
  std::filesystem::path colorImagePath{
      std::filesystem::path(outputDirExpandedPath) / "triton.png"};
  cv::imwrite(colorImagePath, rgb);
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

void calculateIntrinsics(const std::string& imagesDir,
                         const std::string& outputDir) {
  std::filesystem::path imagesDirExpandedPath{
      calibration::expandUser(imagesDir)};
  if (!std::filesystem::exists(imagesDirExpandedPath) ||
      !std::filesystem::is_directory(imagesDirExpandedPath)) {
    spdlog::error("Provided path is not a valid directory: {}",
                  imagesDirExpandedPath.string());
    return;
  }

  std::vector<cv::Mat> images;
  for (const auto& entry :
       std::filesystem::directory_iterator(imagesDirExpandedPath)) {
    if (entry.is_regular_file()) {
      auto ext{entry.path().extension().string()};
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        cv::Mat img{calibration::scaleGrayscaleImage(
            cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE))};
        if (!img.empty()) {
          images.push_back(img);
        }
      }
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

  std::filesystem::path outputDirExpandedPath{
      calibration::expandUser(outputDir)};
  std::filesystem::create_directories(outputDirExpandedPath);

  std::filesystem::path intrinsicsPath{
      std::filesystem::path(outputDirExpandedPath) / "intrinsics.yml"};
  cv::FileStorage fs{intrinsicsPath, cv::FileStorage::WRITE};
  fs << "intrinsicMatrix" << calibrateResults.intrinsicMatrix;
  fs << "distCoeffs" << calibrateResults.distCoeffs;
  fs.release();
  spdlog::info("Saved intrinsics data to: {}", intrinsicsPath.string());
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
  std::filesystem::path imageFileExpandedPath{
      calibration::expandUser(imageFile)};
  cv::Mat img{cv::imread(imageFileExpandedPath)};

  cv::Rect roi;
  cv::Mat newCameraMatrix{cv::getOptimalNewCameraMatrix(
      intrinsicMatrix, distCoeffs, img.size(), 1, img.size(), &roi)};
  cv::Mat undistorted;
  cv::undistort(img, undistorted, intrinsicMatrix, distCoeffs, newCameraMatrix);
  undistorted = undistorted(roi);

  std::filesystem::path outputFileExpandedPath{
      calibration::expandUser(outputFile)};
  cv::imwrite(outputFileExpandedPath, undistorted);
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

  // Find camera image paths
  std::filesystem::path cameraImagesExpandedPath{
      calibration::expandUser(cameraImagesDir)};
  if (!std::filesystem::exists(cameraImagesExpandedPath) ||
      !std::filesystem::is_directory(cameraImagesExpandedPath)) {
    spdlog::error("Provided path is not a valid directory: {}",
                  cameraImagesExpandedPath.string());
    return;
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
      calibration::expandUser(heliosImagesDir)};
  if (!std::filesystem::exists(heliosImagesExpandedPath) ||
      !std::filesystem::is_directory(heliosImagesExpandedPath)) {
    spdlog::error("Provided path is not a valid directory: {}",
                  heliosImagesExpandedPath.string());
    return;
  }
  std::filesystem::path heliosXyzExpandedPath{
      calibration::expandUser(heliosXyzDir)};
  if (!std::filesystem::exists(heliosXyzExpandedPath) ||
      !std::filesystem::is_directory(heliosXyzExpandedPath)) {
    spdlog::error("Provided path is not a valid directory: {}",
                  heliosXyzExpandedPath.string());
    return;
  }

  std::vector<cv::Point2f> allCircleCoords;
  std::vector<cv::Point3f> allCircleXyzPositions;
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
      spdlog::warn("Could not find corresponding Helios intensity image for {}",
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
      spdlog::warn("Could not find corresponding Helios XYZ data for {}",
                   cameraImagePath);
      continue;
    }

    spdlog::info("Processing {}", baseName);
    spdlog::info("  Camera image file: {}", cameraImagePath);
    spdlog::info("  Helios image file: {}", heliosImagePath.string());
    spdlog::info("  Helios XYZ file: {}", heliosXyzFilePath.string());

    // Get circle centers in camera image
    cv::Mat cameraImg{calibration::scaleGrayscaleImage(
        cv::imread(cameraImagePath, cv::IMREAD_GRAYSCALE))};
    std::vector<cv::Point2f> circleCoords;
    bool found{cv::findCirclesGrid(cameraImg, cv::Size(5, 4), circleCoords,
                                   cv::CALIB_CB_SYMMETRIC_GRID, blobDetector)};
    if (!found) {
      spdlog::warn("Could not get circle centers from camera image.");
      continue;
    }

    // Get circle centers in Helios image
    cv::Mat heliosImg{calibration::scaleGrayscaleImage(
        cv::imread(heliosImagePath, cv::IMREAD_GRAYSCALE))};
    std::vector<cv::Point2f> heliosCircleCoords;
    found = cv::findCirclesGrid(heliosImg, cv::Size(5, 4), heliosCircleCoords,
                                cv::CALIB_CB_SYMMETRIC_GRID, blobDetector);
    if (!found) {
      spdlog::warn("Could not get circle centers from Helios intensity image.");
      continue;
    }

    // Round to integer pixel positions
    std::vector<cv::Point> heliosCircleCoordsInt;
    for (const auto& pt : heliosCircleCoords) {
      heliosCircleCoordsInt.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
    }

    // Parse XYZ data file
    cv::FileStorage xyzFileFs{heliosXyzFilePath, cv::FileStorage::READ};
    if (!xyzFileFs.isOpened() || xyzFileFs["xyz"].isNone()) {
      spdlog::error("Could not read XYZ file: {}", heliosXyzFilePath.string());
      return;
    }
    cv::Mat heliosXyz;
    xyzFileFs["xyz"] >> heliosXyz;
    xyzFileFs.release();

    // Get corresponding XYZ value from the XYZ data
    std::vector<cv::Point3f> circleXyzPositions;
    for (auto& pt : heliosCircleCoordsInt) {
      // Access XYZ at [y, x]
      cv::Vec3f xyz{heliosXyz.at<cv::Vec3f>(pt.y, pt.x)};
      circleXyzPositions.emplace_back(xyz[0], xyz[1], xyz[2]);
    }

    allCircleCoords.insert(allCircleCoords.end(), circleCoords.begin(),
                           circleCoords.end());
    allCircleXyzPositions.insert(allCircleXyzPositions.end(),
                                 circleXyzPositions.begin(),
                                 circleXyzPositions.end());
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

  // Calculate reprojection error
  std::vector<cv::Point2f> reprojected;
  cv::projectPoints(allCircleXyzPositions, rvec, tvec, intrinsicMatrix,
                    distCoeffs, reprojected);
  double totalError{0.0};
  for (size_t i = 0; i < allCircleCoords.size(); ++i) {
    totalError += cv::norm(allCircleCoords[i] - reprojected[i]);
  }
  double meanError{totalError / allCircleCoords.size()};
  spdlog::info("Reprojection error (mean): {}", meanError);

  // Construct extrinsic matrix and write to file
  cv::Mat extrinsicMatrix{calibration::constructExtrinsicMatrix(rvec, tvec)};

  std::filesystem::path outputDirExpandedPath{
      calibration::expandUser(outputDir)};
  std::filesystem::create_directories(outputDirExpandedPath);

  std::filesystem::path extrinsicsPath{
      std::filesystem::path(outputDirExpandedPath) / "extrinsics.yml"};
  cv::FileStorage fs{extrinsicsPath, cv::FileStorage::WRITE};
  fs << "extrinsicMatrix" << extrinsicMatrix;
  fs.release();
  spdlog::info("Saved extrinsics data to: {}", extrinsicsPath.string());
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
      calibration::expandUser(cameraImageFile)};
  cv::Mat cameraImg{calibration::scaleGrayscaleImage(
      cv::imread(cameraImageFileExpandedPath, cv::IMREAD_GRAYSCALE))};
  std::filesystem::path heliosIntensityimageFileExpandedPath{
      calibration::expandUser(heliosIntensityImageFile)};
  cv::Mat heliosIntensityImg{calibration::scaleGrayscaleImage(
      cv::imread(heliosIntensityimageFileExpandedPath, cv::IMREAD_GRAYSCALE))};

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
      if (cameraImg.at<uint8_t>(r, c) < 100) {
        projectionImg.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 128);
      }
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
      int thresh{(intensity < 30) ? 1 : 0};
      projectionImg.at<cv::Vec3b>(row, col)[1] = thresh * 255;
    }
  }

  std::filesystem::path outputFileExpandedPath{
      calibration::expandUser(outputFile)};
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

  auto calculateExtrinsicsXyzToHeliosCommand{app.add_subcommand(
      "calculate_extrinsics_xyz_to_helios",
      "Calculate extrinsics that describe the orientation of Helios relative "
      "to Helios XYZ from images of a calibration pattern")};
  calculateExtrinsicsXyzToHeliosCommand
      ->add_option("--helios_intrinsics_file", intrinsicsFile,
                   "yml file containing Helios camera intrinsics")
      ->required();
  calculateExtrinsicsXyzToHeliosCommand
      ->add_option("--helios_images_dir", heliosImagesDir,
                   "Path to directory containing Helios intensity images")
      ->required();
  calculateExtrinsicsXyzToHeliosCommand
      ->add_option("--helios_xyz_dir", heliosXyzDir,
                   "Path to directory containing Helios XYZ data")
      ->required();
  calculateExtrinsicsXyzToHeliosCommand
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
  } else if (*undistortImageCommand) {
    undistortImage(intrinsicsFile, imageFile, outputFile);
  } else if (*calculateExtrinsicsXyzToTritonCommand) {
    calculateExtrinsicsXyzToCamera(intrinsicsFile, tritonImagesDir,
                                   heliosImagesDir, heliosXyzDir, outputDir);
  } else if (*calculateExtrinsicsXyzToHeliosCommand) {
    calculateExtrinsicsXyzToCamera(intrinsicsFile, heliosImagesDir,
                                   heliosImagesDir, heliosXyzDir, outputDir);
  } else if (*visualizeExtrinsicsCommand) {
    visualizeExtrinsics(cameraImageFile, heliosIntensityImageFile,
                        heliosXyzFile, intrinsicsFile, extrinsicsFile,
                        outputFile);
  }

  return 0;
}