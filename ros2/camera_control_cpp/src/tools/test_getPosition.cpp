#include "camera_control_cpp/tools/test_getPosition.hpp"

int main(int argc, char const* argv[]) {
  spdlog::info("Working");
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

  if (rgbImages.size() < 1 || intensityImages.size() < 1 ||
      xyzImages.size() < 1) {
    spdlog::error("Error: Directory must contain at least 1 image files.");
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

  std::vector<cv::Mat> boostedIntensityImages;
  for (cv::Mat img : intensityImages) {
    std::optional<cv::Mat> boosted = calibration::boostIntensity(img);
    if (boosted != std::nullopt) {
      boostedIntensityImages.push_back(*boosted);
    } else {
      spdlog::warn("Failed to boost an intensity image.");
    }
  }

  int imageIndex{0};
  cv::Mat tritonMonoImage{rgbImages[imageIndex]};
  cv::Mat heliosIntensityImage{boostedIntensityImages[imageIndex]};
  cv::Mat xyzImage{xyzImages[imageIndex]};

  double timestampMillis{
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count()) /
      1000.0};

  cv::Mat rgbIntrinsic, rgbDistortion, depthIntrinsic, depthDistortion,
      heliosToTritonExtrinsic, tritonToHeliosExtrinsic;

  std::string rgbIntrinsicPath =
      calibration::calibrationParamsDir + "triton_intrinsic_matrix.yml";
  std::string rgbDistortionPath =
      calibration::calibrationParamsDir + "triton_distortion_coeffs.yml";
  std::string depthIntrinsicPath =
      calibration::calibrationParamsDir + "helios_intrinsic_matrix.yml";
  std::string depthDistortionPath =
      calibration::calibrationParamsDir + "helios_distortion_coeffs.yml";
  std::string tritonToHeliosExtrinsicPath =
      calibration::calibrationParamsDir +
      "triton_to_helios_extrinsic_matrix.yml";
  std::string heliosToTritonExtrinsicPath =
      calibration::calibrationParamsDir +
      "helios_to_triton_extrinsic_matrix.yml";

  std::vector<std::tuple<std::string, cv::Mat*, std::string>> paths = {
      {rgbIntrinsicPath, &rgbIntrinsic, "Intrinsic Matrix"},
      {rgbDistortionPath, &rgbDistortion, "Distortion Coefficients"},
      {depthIntrinsicPath, &depthIntrinsic, "Intrinsic Matrix"},
      {depthDistortionPath, &depthDistortion, "Distortion Coefficients"},
      {tritonToHeliosExtrinsicPath, &tritonToHeliosExtrinsic,
       "Extrinsic Matrix"},
      {heliosToTritonExtrinsicPath, &heliosToTritonExtrinsic,
       "Extrinsic Matrix"}};

  for (const auto& [path, matPtr, key] : paths) {
    if (!std::filesystem::exists(path)) {
      spdlog::error("Calibration file not found: {}", path);
      return -1;
    }
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      spdlog::error("Failed to open calibration file: {}", path);
      return -1;
    }

    // Use the correct key name instead of "data"
    fs[key] >> *matPtr;
    fs.release();

    if (matPtr->empty()) {
      spdlog::error("Failed to load matrix from '{}' with key '{}'", path, key);
      return -1;
    }

    spdlog::info("Loaded calibration matrix from: {}", path);
    std::ostringstream oss;
    oss << *matPtr;
    spdlog::info("Loaded matrix from '{}':\n{}", key, oss.str());
  }

  LucidFrame frame(tritonMonoImage, xyzImage, timestampMillis,
                   rgbIntrinsic, rgbDistortion, depthIntrinsic, depthDistortion,
                   heliosToTritonExtrinsic, tritonToHeliosExtrinsic);

  std::pair<int, int> originalPoint(200,
                                    200);  // Example point, adjust as needed

  // Predict corresponding point in the other frame
  std::optional<std::tuple<double, double, double>> predictedPoint =
      frame.getPosition(originalPoint);
  if (!predictedPoint) {
    spdlog::error("Failed to predict corresponding point.");
    return -1;
  } else {
    auto& [x, y, z] = *predictedPoint;
    spdlog::info("Predicted corresponding point: ({}, {}, {})", x, y, z);
  }

  // Clone images for visualization
  cv::Mat tritonVis, heliosVis;
  cv::cvtColor(tritonMonoImage, tritonVis, cv::COLOR_GRAY2BGR);
  cv::cvtColor(heliosIntensityImage, heliosVis, cv::COLOR_GRAY2BGR);

  // Extract (x, y) from tuples for drawing
  auto [origX, origY] = originalPoint;
  auto [predX, predY, predZ] = *predictedPoint;

  // Draw original and predicted points
  cv::circle(tritonVis, cv::Point(origX, origY), 5, cv::Scalar(0, 0, 255),
             5);  // Red on Triton
  cv::circle(heliosVis, cv::Point(predX, predY), 5, cv::Scalar(0, 255, 0),
             2);  // Green on Helios

  // Optionally, mark the predicted point on Triton and original on Helios for
  // clarity
  cv::circle(tritonVis, cv::Point(predX, predY), 5, cv::Scalar(0, 255, 0),
             5);  // Green circle on Triton
  cv::circle(heliosVis, cv::Point(origX, origY), 5, cv::Scalar(0, 0, 255),
             2);  // Red circle on Helios

  // Show with Chafa
  spdlog::info(
      "Triton frame with original (red) and predicted (green) points:");
  calibration::showWithChafa(tritonVis, "tritonVis");

  spdlog::info(
      "Helios frame with predicted (green) and original (red) points:");
  calibration::showWithChafa(heliosVis, "heliosVis");

  return 0;
}
