#include <gtest/gtest.h>

#include <filesystem>

#include "camera_control/camera/calibration.hpp"

namespace {

cv::Mat makeIntrinsicMatrix(double fx = 500.0, double fy = 500.0,
                            double cx = 320.0, double cy = 240.0) {
  cv::Matx33d K{fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
  return cv::Mat(K);
}

cv::Mat makeZeroDistCoeffs() { return cv::Mat::zeros(1, 5, CV_64F); }

class CalibrationFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tempDir_ =
        std::filesystem::temp_directory_path() /
        ("camera_control_test_calibration_" +
         std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    std::filesystem::create_directories(tempDir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tempDir_, ec);
  }

  std::filesystem::path tempDir_;
};

}  // namespace

TEST(CalibrationTest, ConstructExtrinsicMatrixProducesExpectedTranslation) {
  cv::Mat rvec{0.0, 0.0, 0.0};
  cv::Mat tvec{1.0, 2.0, 3.0};

  cv::Mat extrinsic{calibration::constructExtrinsicMatrix(rvec, tvec)};

  ASSERT_EQ(extrinsic.rows, 4);
  ASSERT_EQ(extrinsic.cols, 4);
  // Zero rotation vector should produce an identity rotation block
  EXPECT_NEAR(extrinsic.at<double>(0, 0), 1.0, 1e-9);
  EXPECT_NEAR(extrinsic.at<double>(1, 1), 1.0, 1e-9);
  EXPECT_NEAR(extrinsic.at<double>(2, 2), 1.0, 1e-9);
  EXPECT_NEAR(extrinsic.at<double>(0, 3), 1.0, 1e-9);
  EXPECT_NEAR(extrinsic.at<double>(1, 3), 2.0, 1e-9);
  EXPECT_NEAR(extrinsic.at<double>(2, 3), 3.0, 1e-9);
  EXPECT_NEAR(extrinsic.at<double>(3, 3), 1.0, 1e-9);
}

TEST(CalibrationTest, ExtractPoseFromExtrinsicRoundTrip) {
  cv::Mat rvec{0.1, -0.2, 0.3};
  cv::Mat tvec{4.0, -5.0, 6.0};

  cv::Mat extrinsic{calibration::constructExtrinsicMatrix(rvec, tvec)};
  auto [rvecOut, tvecOut]{calibration::extractPoseFromExtrinsic(extrinsic)};

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(rvecOut.at<double>(i, 0), rvec.at<double>(i, 0), 1e-9);
    EXPECT_NEAR(tvecOut.at<double>(i, 0), tvec.at<double>(i, 0), 1e-9);
  }
}

TEST(CalibrationTest, DistortPixelCoordsWithZeroDistortionReturnsInput) {
  cv::Mat intrinsicMatrix{makeIntrinsicMatrix()};
  cv::Mat distCoeffs{makeZeroDistCoeffs()};
  cv::Point2f pixel{370.0f, 270.0f};  // (cx + 50, cy + 30)

  auto result{
      calibration::distortPixelCoords(pixel, intrinsicMatrix, distCoeffs)};

  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->x, pixel.x, 1e-3);
  EXPECT_NEAR(result->y, pixel.y, 1e-3);
}

TEST(CalibrationTest, DistortPixelCoordsInvalidIntrinsicTypeReturnsNullopt) {
  cv::Mat intrinsicMatrix32F;
  makeIntrinsicMatrix().convertTo(intrinsicMatrix32F, CV_32F);
  cv::Mat distCoeffs{makeZeroDistCoeffs()};

  auto result{calibration::distortPixelCoords({370.0f, 270.0f},
                                              intrinsicMatrix32F, distCoeffs)};

  EXPECT_FALSE(result.has_value());
}

TEST(CalibrationTest, DistortPixelCoordsInvalidDistCoeffsTypeReturnsNullopt) {
  cv::Mat intrinsicMatrix{makeIntrinsicMatrix()};
  cv::Mat distCoeffs32F;
  makeZeroDistCoeffs().convertTo(distCoeffs32F, CV_32F);

  auto result{calibration::distortPixelCoords({370.0f, 270.0f}, intrinsicMatrix,
                                              distCoeffs32F)};

  EXPECT_FALSE(result.has_value());
}

TEST(CalibrationTest, ScaleGrayscaleImageSpansFullRange) {
  cv::Mat image{cv::Mat::zeros(4, 4, CV_8UC1)};
  image.at<uint8_t>(0, 0) = 50;
  image.at<uint8_t>(1, 1) = 100;

  cv::Mat scaled{calibration::scaleGrayscaleImage(image)};

  double minVal, maxVal;
  cv::minMaxLoc(scaled, &minVal, &maxVal);
  EXPECT_NEAR(minVal, 0.0, 1e-6);
  EXPECT_NEAR(maxVal, 255.0, 1e-6);
  EXPECT_EQ(scaled.type(), CV_8U);
}

TEST(CalibrationTest, ScaleGrayscaleImageConstantImage) {
  cv::Mat image{cv::Mat(4, 4, CV_8UC1, cv::Scalar(42))};

  cv::Mat scaled{calibration::scaleGrayscaleImage(image)};

  double minVal, maxVal;
  cv::minMaxLoc(scaled, &minVal, &maxVal);
  EXPECT_NEAR(minVal, 0.0, 1e-6);
  EXPECT_NEAR(maxVal, 0.0, 1e-6);
}

TEST_F(CalibrationFileTest, ReadXyzFileMissingKeyReturnsNullopt) {
  std::filesystem::path filePath{tempDir_ / "no_xyz.yml"};
  {
    cv::FileStorage fs{filePath.string(), cv::FileStorage::WRITE};
    fs << "somethingElse" << 1;
    fs.release();
  }

  auto result{calibration::readXyzFile(filePath.string())};

  EXPECT_FALSE(result.has_value());
}

TEST_F(CalibrationFileTest, ReadXyzFileRoundTripsWrittenValues) {
  std::filesystem::path filePath{tempDir_ / "xyz.yml"};
  cv::Mat xyz{1.0, 2.0, 3.0};
  {
    cv::FileStorage fs{filePath.string(), cv::FileStorage::WRITE};
    fs << "xyz" << xyz;
    fs.release();
  }

  auto result{calibration::readXyzFile(filePath.string())};

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(cv::countNonZero(*result != xyz), 0);
}

TEST_F(CalibrationFileTest, ReadIntrinsicsFileMissingFileReturnsNullopt) {
  auto result{calibration::readIntrinsicsFile(
      (tempDir_ / "does_not_exist.yml").string())};

  EXPECT_FALSE(result.has_value());
}

TEST_F(CalibrationFileTest, ReadIntrinsicsFileRoundTripsWrittenValues) {
  std::filesystem::path filePath{tempDir_ / "intrinsics.yml"};
  cv::Mat intrinsicMatrix{makeIntrinsicMatrix()};
  cv::Mat distCoeffs{makeZeroDistCoeffs()};
  {
    cv::FileStorage fs{filePath.string(), cv::FileStorage::WRITE};
    fs << "intrinsicMatrix" << intrinsicMatrix;
    fs << "distCoeffs" << distCoeffs;
    fs.release();
  }

  auto result{calibration::readIntrinsicsFile(filePath.string())};

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(cv::countNonZero(result->intrinsicMatrix != intrinsicMatrix), 0);
  EXPECT_EQ(cv::countNonZero(result->distCoeffs != distCoeffs), 0);
}

TEST_F(CalibrationFileTest, ReadExtrinsicsFileMissingFileReturnsNullopt) {
  auto result{calibration::readExtrinsicsFile(
      (tempDir_ / "does_not_exist.yml").string())};

  EXPECT_FALSE(result.has_value());
}

TEST_F(CalibrationFileTest, ReadExtrinsicsFileRoundTripsWrittenValues) {
  std::filesystem::path filePath{tempDir_ / "extrinsics.yml"};
  cv::Mat extrinsicMatrix{cv::Mat::eye(4, 4, CV_64F)};
  extrinsicMatrix.at<double>(0, 3) = 7.0;
  {
    cv::FileStorage fs{filePath.string(), cv::FileStorage::WRITE};
    fs << "extrinsicMatrix" << extrinsicMatrix;
    fs.release();
  }

  auto result{calibration::readExtrinsicsFile(filePath.string())};

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(cv::countNonZero(*result != extrinsicMatrix), 0);
}
