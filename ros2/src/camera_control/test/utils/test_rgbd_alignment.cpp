#include <gtest/gtest.h>

#include "camera_control/utils/rgbd_alignment.hpp"

namespace {

constexpr double F_X{500.0};
constexpr double F_Y{500.0};
constexpr double C_X{320.0};
constexpr double C_Y{320.0};
constexpr int DEPTH_WIDTH{640};
constexpr int DEPTH_HEIGHT{480};

cv::Mat makeIntrinsicMatrix() {
  return (cv::Mat_<double>(3, 3) << F_X, 0.0, C_X, 0.0, F_Y, C_Y, 0.0, 0.0,
          1.0);
}

cv::Mat makeZeroDistCoeffs() { return cv::Mat::zeros(1, 5, CV_64F); }

cv::Mat makeIdentityExtrinsic() { return cv::Mat::eye(4, 4, CV_64F); }

cv::Mat makeTranslationExtrinsic(double tx, double ty, double tz) {
  cv::Mat extrinsic{cv::Mat::eye(4, 4, CV_64F)};
  extrinsic.at<double>(0, 3) = tx;
  extrinsic.at<double>(1, 3) = ty;
  extrinsic.at<double>(2, 3) = tz;
  return extrinsic;
}

// Builds a synthetic depth xyz image with a constant depth. Each pixel's xyz
// value is just the deprojection of its own pixel coordinate.
cv::Mat makePlaneDepthXyz(int width, int height, float planeDepthMm) {
  cv::Mat depthXyz(height, width, CV_32FC3);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      float x{static_cast<float>((u - C_X) / F_X * planeDepthMm)};
      float y{static_cast<float>((v - C_Y) / F_Y * planeDepthMm)};
      depthXyz.at<cv::Vec3f>(v, u) = cv::Vec3f{x, y, planeDepthMm};
    }
  }
  return depthXyz;
}

}  // namespace

TEST(RgbdAlignmentTest, CoLocatedCamerasReturnSameCorrespondingPixel) {
  RgbdAlignment alignment(makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIdentityExtrinsic(), makeIdentityExtrinsic());

  const cv::Point2i colorPixel{400, 350};
  const cv::Mat depthXyz{makePlaneDepthXyz(DEPTH_WIDTH, DEPTH_HEIGHT, 1000.0f)};

  auto depthPixelOpt{
      alignment.getCorrespondingDepthPixel(colorPixel, depthXyz)};

  ASSERT_TRUE(depthPixelOpt.has_value());
  EXPECT_EQ(*depthPixelOpt, colorPixel);
}

TEST(RgbdAlignmentTest, CoLocatedCamerasReturnExpectedPosition) {
  RgbdAlignment alignment(makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIdentityExtrinsic(), makeIdentityExtrinsic());

  const cv::Point2i colorPixel{400, 350};
  const cv::Mat depthXyz{makePlaneDepthXyz(DEPTH_WIDTH, DEPTH_HEIGHT, 1000.0f)};

  auto positionOpt{alignment.getPosition(colorPixel, depthXyz)};

  ASSERT_TRUE(positionOpt.has_value());
  EXPECT_NEAR((*positionOpt)[0], 160.0f, 1e-2);
  EXPECT_NEAR((*positionOpt)[1], 60.0f, 1e-2);
  EXPECT_NEAR((*positionOpt)[2], 1000.0f, 1e-2);
}

TEST(RgbdAlignmentTest, ColorFrameOffsetAppliesCorrectly) {
  // colorFrameOffset shifts the ROI-relative color pixel into full-frame
  // coordinates before searching, so a pixel at (300, 300) with an offset of
  // (100, 50) should resolve identically to full-frame pixel (400, 350).
  RgbdAlignment alignment(makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIdentityExtrinsic(), makeIdentityExtrinsic(),
                          /*colorFrameOffset=*/std::pair<int, int>{100, 50});

  const cv::Point2i colorPixel{300, 300};
  const cv::Mat depthXyz{makePlaneDepthXyz(DEPTH_WIDTH, DEPTH_HEIGHT, 1000.0f)};

  auto depthPixelOpt{
      alignment.getCorrespondingDepthPixel(colorPixel, depthXyz)};

  ASSERT_TRUE(depthPixelOpt.has_value());
  EXPECT_EQ(*depthPixelOpt, (cv::Point2i{400, 350}));
}

TEST(RgbdAlignmentTest, GetPositionRejectsInvalidDepth) {
  RgbdAlignment alignment(makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIdentityExtrinsic(), makeIdentityExtrinsic());

  const cv::Point2i colorPixel{400, 350};
  cv::Mat depthXyz{makePlaneDepthXyz(DEPTH_WIDTH, DEPTH_HEIGHT, 1000.0f)};
  // Mark the depth pixel that colorPixel corresponds to as invalid
  depthXyz.at<cv::Vec3f>(350, 400) = cv::Vec3f{-1.0f, -1.0f, -1.0f};

  auto positionOpt{alignment.getPosition(colorPixel, depthXyz)};
  EXPECT_FALSE(positionOpt.has_value());

  // Now, mark the same depth pixel as one above the max valid value
  // (depth greater than 2^14 - 1 indicates invalid position)
  depthXyz.at<cv::Vec3f>(350, 400)[2] = (1 << 14);

  positionOpt = alignment.getPosition(colorPixel, depthXyz);
  EXPECT_FALSE(positionOpt.has_value());
}

TEST(RgbdAlignmentTest, FindsCorrespondingPixelAcrossTranslatedDepthCamera) {
  // Offset the depth camera 50mm along the X axis from the color camera
  RgbdAlignment alignment(makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIntrinsicMatrix(), makeZeroDistCoeffs(),
                          makeIdentityExtrinsic(),
                          makeTranslationExtrinsic(50.0, 0.0, 0.0));

  cv::Mat depthXyz{makePlaneDepthXyz(DEPTH_WIDTH, DEPTH_HEIGHT, 1000.0f)};

  // A point at (100, 50, 1000) in the xyz coord system projects to color pixel
  // (370, 345), and lives in the depth camera frame at (150, 50, 1000) due to
  // the translation, which projects to depth pixel (395, 345).
  const cv::Vec3f truePosition{100.0f, 50.0f, 1000.0f};
  const cv::Point2i colorPixel{370, 345};
  const cv::Point2i expectedDepthPixel{395, 345};
  depthXyz.at<cv::Vec3f>(expectedDepthPixel.y, expectedDepthPixel.x) =
      truePosition;

  auto depthPixelOpt{
      alignment.getCorrespondingDepthPixel(colorPixel, depthXyz)};
  ASSERT_TRUE(depthPixelOpt.has_value());
  EXPECT_EQ(*depthPixelOpt, expectedDepthPixel);

  auto positionOpt{alignment.getPosition(colorPixel, depthXyz)};
  ASSERT_TRUE(positionOpt.has_value());
  EXPECT_NEAR((*positionOpt)[0], truePosition[0], 1e-2);
  EXPECT_NEAR((*positionOpt)[1], truePosition[1], 1e-2);
  EXPECT_NEAR((*positionOpt)[2], truePosition[2], 1e-2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
