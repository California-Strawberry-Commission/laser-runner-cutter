#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>

#include "camera_control_cpp/camera/calibration.hpp"
#include "camera_control_cpp/camera/lucid_frame.hpp"
#include "spdlog/spdlog.h"

int main(int argc, char* argv[]) {
  CLI::App app{"Test LucidFrame's getCorrespondingDepthPixel"};

  std::string tritonIntrinsicsFile;
  std::string heliosIntrinsicsFile;
  std::string xyzToTritonExtrinsicsFile;
  std::string xyzToHeliosExtrinsicsFile;
  std::string tritonImageFile;
  std::string heliosImageFile;
  std::string xyzFile;
  std::tuple<int, int> coords{200, 200};
  std::string outputDir;
  app.add_option("--triton_intrinsics_file", tritonIntrinsicsFile,
                 "yml file containing Triton camera intrinsics")
      ->required();
  app.add_option("--helios_intrinsics_file", heliosIntrinsicsFile,
                 "yml file containing Helios camera intrinsics")
      ->required();
  app.add_option("--xyz_to_triton_extrinsics_file", xyzToTritonExtrinsicsFile,
                 "yml file containing XYZ to Triton extrinsics")
      ->required();
  app.add_option("--xyz_to_helios_extrinsics_file", xyzToHeliosExtrinsicsFile,
                 "yml file containing XYZ to Helios extrinsics")
      ->required();
  app.add_option("--triton_image_file", tritonImageFile,
                 "Path to Triton image file")
      ->required();
  app.add_option("--helios_image_file", heliosImageFile,
                 "Path to Helios intensity image file")
      ->required();
  app.add_option("--helios_xyz_file", xyzFile, "Path to Helios XYZ data file")
      ->required();
  app.add_option("--coords", coords,
                 "Two integers (x y) representing the color pixel coord");
  app.add_option("-o,--output_dir", outputDir,
                 "Path to the directory to write results to")
      ->required();

  CLI11_PARSE(app, argc, argv);

  // Parse intrinsics files
  auto tritonIntrinsicsOpt{
      calibration::readIntrinsicsFile(tritonIntrinsicsFile)};
  if (!tritonIntrinsicsOpt) {
    return 1;
  }
  auto [tritonIntrinsicMatrix, tritonDistCoeffs]{tritonIntrinsicsOpt.value()};
  auto heliosIntrinsicsOpt{
      calibration::readIntrinsicsFile(heliosIntrinsicsFile)};
  if (!heliosIntrinsicsOpt) {
    return 1;
  }
  auto [heliosIntrinsicMatrix, heliosDistCoeffs]{heliosIntrinsicsOpt.value()};

  // Parse extrinsics files
  auto xyzToTritonExtrinsicMatrixOpt{
      calibration::readExtrinsicsFile(xyzToTritonExtrinsicsFile)};
  if (!xyzToTritonExtrinsicMatrixOpt) {
    return 1;
  }
  cv::Mat xyzToTritonExtrinsicMatrix{xyzToTritonExtrinsicMatrixOpt.value()};
  auto xyzToHeliosExtrinsicMatrixOpt{
      calibration::readExtrinsicsFile(xyzToHeliosExtrinsicsFile)};
  if (!xyzToHeliosExtrinsicMatrixOpt) {
    return 1;
  }
  cv::Mat xyzToHeliosExtrinsicMatrix{xyzToHeliosExtrinsicMatrixOpt.value()};

  // Read image files
  std::filesystem::path tritonImageFileExpandedPath{
      calibration::expandUser(tritonImageFile)};
  cv::Mat tritonImg{cv::imread(tritonImageFileExpandedPath)};
  std::filesystem::path heliosImageFileExpandedPath{
      calibration::expandUser(heliosImageFile)};
  cv::Mat heliosIntensityImg{cv::imread(heliosImageFileExpandedPath)};

  // Read XYZ file
  auto xyzmmOpt{calibration::readXyzFile(xyzFile)};
  if (!xyzmmOpt) {
    return 1;
  }
  auto xyzmm{xyzmmOpt.value()};

  double timestampMillis{
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count()) /
      1000.0};
  LucidFrame frame{tritonImg,
                   xyzmm,
                   timestampMillis,
                   tritonIntrinsicMatrix,
                   tritonDistCoeffs,
                   heliosIntrinsicMatrix,
                   heliosDistCoeffs,
                   xyzToTritonExtrinsicMatrix,
                   xyzToHeliosExtrinsicMatrix};
  auto [x, y]{coords};
  cv::Point2i colorPixelCoord{x, y};
  auto depthPixelCoordOpt{frame.getCorrespondingDepthPixel(colorPixelCoord)};
  if (!depthPixelCoordOpt) {
    spdlog::error(
        "Could not find corresponding depth pixel for color pixel coord: ({}, "
        "{})",
        colorPixelCoord.x, colorPixelCoord.y);
    return 1;
  }
  cv::Point2i depthPixelCoord{depthPixelCoordOpt.value()};

  // Draw markers on pixel coords for each image
  cv::circle(tritonImg, colorPixelCoord, 5, cv::Scalar(0, 0, 255), 5);
  cv::circle(heliosIntensityImg, depthPixelCoord, 5, cv::Scalar(0, 0, 255), 5);

  std::filesystem::path outputDirExpandedPath{
      calibration::expandUser(outputDir)};
  std::filesystem::create_directories(outputDirExpandedPath);
  std::filesystem::path colorImagePath{
      std::filesystem::path(outputDirExpandedPath) / "triton.png"};
  cv::imwrite(colorImagePath, tritonImg);
  spdlog::info("Saved annotated color camera image to: {}",
               colorImagePath.string());
  std::filesystem::path depthIntensityImagePath{
      std::filesystem::path(outputDirExpandedPath) / "helios_intensity.png"};
  cv::imwrite(depthIntensityImagePath, heliosIntensityImg);
  spdlog::info("Saved annotated depth camera intensity image to: {}",
               depthIntensityImagePath.string());

  return 0;
}