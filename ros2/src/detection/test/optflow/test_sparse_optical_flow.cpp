// Tests for SparseOpticalFlow against real VPI PVA/CUDA backends. These
// require a Jetson with PVA hardware (e.g. Xavier or Orin).

#include <gtest/gtest.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "detection/optflow/sparse_optical_flow.hpp"

namespace {

constexpr int CROP_X{200};
constexpr int CROP_Y{200};
constexpr int CROP_WIDTH{1024};
constexpr int CROP_HEIGHT{768};
constexpr int SHIFT_DX{16};
constexpr int SHIFT_DY{10};
constexpr float FLOW_TOLERANCE_PX{1.0f};

cv::Mat loadTestImage() {
  std::string path{std::string(TEST_IMAGES_DIR) + "/20240703102531.png"};
  cv::Mat image{cv::imread(path)};
  if (image.empty()) {
    ADD_FAILURE() << "Failed to load test image at " << path;
  }
  return image;
}

cv::Mat cropAt(const cv::Mat& src, int x, int y, int width, int height) {
  return src(cv::Rect(x, y, width, height)).clone();
}

}  // namespace

class SparseOpticalFlowTest : public ::testing::Test {
 protected:
  cv::Mat sourceImage_{loadTestImage()};
};

TEST_F(SparseOpticalFlowTest, ThrowsOnEmptyFrames) {
  SparseOpticalFlow opticalFlow;
  cv::Mat frame{cropAt(sourceImage_, 0, 0, 100, 100)};

  EXPECT_THROW(opticalFlow.computeFlow(cv::Mat(), frame),
               std::invalid_argument);
  EXPECT_THROW(opticalFlow.computeFlow(frame, cv::Mat()),
               std::invalid_argument);
}

TEST_F(SparseOpticalFlowTest, ThrowsOnMismatchedFrameSize) {
  SparseOpticalFlow opticalFlow;
  cv::Mat prevFrame{cropAt(sourceImage_, 0, 0, 100, 100)};
  cv::Mat currFrame{cropAt(sourceImage_, 0, 0, 200, 100)};

  EXPECT_THROW(opticalFlow.computeFlow(prevFrame, currFrame),
               std::invalid_argument);
}

TEST_F(SparseOpticalFlowTest, ThrowsOnMismatchedFrameType) {
  SparseOpticalFlow opticalFlow;
  cv::Mat prevFrame{cropAt(sourceImage_, 0, 0, 100, 100)};
  cv::Mat currFrame;
  cv::cvtColor(prevFrame, currFrame, cv::COLOR_BGR2GRAY);

  EXPECT_THROW(opticalFlow.computeFlow(prevFrame, currFrame),
               std::invalid_argument);
}

TEST_F(SparseOpticalFlowTest, IdenticalFramesReturnZeroFlow) {
  SparseOpticalFlow opticalFlow;
  cv::Mat frame{cropAt(sourceImage_, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)};

  cv::Point2f flow{opticalFlow.computeFlow(frame, frame)};

  EXPECT_NEAR(flow.x, 0.0f, FLOW_TOLERANCE_PX);
  EXPECT_NEAR(flow.y, 0.0f, FLOW_TOLERANCE_PX);
}

TEST_F(SparseOpticalFlowTest, DetectsKnownTranslation) {
  SparseOpticalFlow opticalFlow;
  cv::Mat prevFrame{
      cropAt(sourceImage_, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)};
  cv::Mat currFrame{cropAt(sourceImage_, CROP_X + SHIFT_DX, CROP_Y + SHIFT_DY,
                           CROP_WIDTH, CROP_HEIGHT)};

  cv::Point2f flow{opticalFlow.computeFlow(prevFrame, currFrame)};

  // currFrame is cropped further right/down in the source image than
  // prevFrame, so content appears to have moved left/up by (SHIFT_DX,
  // SHIFT_DY) between prevFrame and currFrame.
  EXPECT_NEAR(flow.x, -SHIFT_DX, FLOW_TOLERANCE_PX);
  EXPECT_NEAR(flow.y, -SHIFT_DY, FLOW_TOLERANCE_PX);
}

TEST_F(SparseOpticalFlowTest, IncludeRegionWithNoFeaturesReturnsZeroFlow) {
  SparseOpticalFlow opticalFlow;
  cv::Mat prevFrame{
      cropAt(sourceImage_, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)};
  cv::Mat currFrame{cropAt(sourceImage_, CROP_X + SHIFT_DX, CROP_Y + SHIFT_DY,
                           CROP_WIDTH, CROP_HEIGHT)};
  // Paint a large flat, featureless patch over prevFrame, then query an
  // includeRegion strictly inside it so that the corners Harris finds along the
  // patch's edges fall outside the queried region, leaving no corners to track
  // within it.
  cv::Rect flatPatch{100, 100, 300, 300};
  cv::Rect includeRegion{175, 175, 150, 150};
  prevFrame(flatPatch).setTo(cv::Scalar(0, 0, 0));

  cv::Point2f flow{
      opticalFlow.computeFlow(prevFrame, currFrame, includeRegion)};

  EXPECT_FLOAT_EQ(flow.x, 0.0f);
  EXPECT_FLOAT_EQ(flow.y, 0.0f);
}

TEST_F(SparseOpticalFlowTest, ReallocatesBuffersWhenFrameSizeChanges) {
  SparseOpticalFlow opticalFlow;
  cv::Mat smallPrev{cropAt(sourceImage_, CROP_X, CROP_Y, 400, 300)};
  cv::Mat smallCurr{
      cropAt(sourceImage_, CROP_X + SHIFT_DX, CROP_Y + SHIFT_DY, 400, 300)};
  cv::Mat largePrev{
      cropAt(sourceImage_, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)};
  cv::Mat largeCurr{cropAt(sourceImage_, CROP_X + SHIFT_DX, CROP_Y + SHIFT_DY,
                           CROP_WIDTH, CROP_HEIGHT)};

  opticalFlow.computeFlow(smallPrev, smallCurr);
  cv::Point2f flow{opticalFlow.computeFlow(largePrev, largeCurr)};

  EXPECT_NEAR(flow.x, -SHIFT_DX, FLOW_TOLERANCE_PX);
  EXPECT_NEAR(flow.y, -SHIFT_DY, FLOW_TOLERANCE_PX);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
