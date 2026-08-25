// Tests for DenseOpticalFlow against real VPI OFA/CUDA/VIC backends. These
// require a Jetson with an Optical Flow Accelerator (e.g. Orin).

#include <gtest/gtest.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "detection/optflow/dense_optical_flow.hpp"

namespace {

constexpr int CROP_X{200};
constexpr int CROP_Y{200};
constexpr int CROP_WIDTH{1024};
constexpr int CROP_HEIGHT{768};
constexpr int SHIFT_DX{16};
constexpr int SHIFT_DY{10};
// Dense flow is averaged per grid cell rather than at individual feature
// points, and motion vectors are quantized to 1/32 px, so allow a slightly
// wide tolerance.
constexpr float FLOW_TOLERANCE_PX{3.0f};

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

class DenseOpticalFlowTest : public ::testing::Test {
 protected:
  cv::Mat sourceImage_{loadTestImage()};
};

TEST_F(DenseOpticalFlowTest, ThrowsOnEmptyFrames) {
  DenseOpticalFlow opticalFlow;
  cv::Mat frame{cropAt(sourceImage_, 0, 0, 100, 100)};

  EXPECT_THROW(opticalFlow.computeFlow(cv::Mat(), frame),
               std::invalid_argument);
  EXPECT_THROW(opticalFlow.computeFlow(frame, cv::Mat()),
               std::invalid_argument);
}

TEST_F(DenseOpticalFlowTest, ThrowsOnMismatchedFrameSize) {
  DenseOpticalFlow opticalFlow;
  cv::Mat prevFrame{cropAt(sourceImage_, 0, 0, 100, 100)};
  cv::Mat currFrame{cropAt(sourceImage_, 0, 0, 200, 100)};

  EXPECT_THROW(opticalFlow.computeFlow(prevFrame, currFrame),
               std::invalid_argument);
}

TEST_F(DenseOpticalFlowTest, ThrowsOnMismatchedFrameType) {
  DenseOpticalFlow opticalFlow;
  cv::Mat prevFrame{cropAt(sourceImage_, 0, 0, 100, 100)};
  cv::Mat currFrame;
  cv::cvtColor(prevFrame, currFrame, cv::COLOR_BGR2GRAY);

  EXPECT_THROW(opticalFlow.computeFlow(prevFrame, currFrame),
               std::invalid_argument);
}

TEST_F(DenseOpticalFlowTest, IdenticalFramesReturnZeroFlow) {
  DenseOpticalFlow opticalFlow;
  cv::Mat frame{cropAt(sourceImage_, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)};

  std::optional<cv::Point2f> flow{opticalFlow.computeFlow(frame, frame)};

  ASSERT_TRUE(flow.has_value());
  EXPECT_NEAR(flow->x, 0.0f, FLOW_TOLERANCE_PX);
  EXPECT_NEAR(flow->y, 0.0f, FLOW_TOLERANCE_PX);
}

TEST_F(DenseOpticalFlowTest, DetectsKnownTranslation) {
  DenseOpticalFlow opticalFlow;
  cv::Mat prevFrame{
      cropAt(sourceImage_, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)};
  cv::Mat currFrame{cropAt(sourceImage_, CROP_X + SHIFT_DX, CROP_Y + SHIFT_DY,
                           CROP_WIDTH, CROP_HEIGHT)};

  std::optional<cv::Point2f> flow{
      opticalFlow.computeFlow(prevFrame, currFrame)};

  // currFrame is cropped further right/down in the source image than
  // prevFrame, so content appears to have moved left/up by (SHIFT_DX,
  // SHIFT_DY) between prevFrame and currFrame.
  ASSERT_TRUE(flow.has_value());
  EXPECT_NEAR(flow->x, -SHIFT_DX, FLOW_TOLERANCE_PX);
  EXPECT_NEAR(flow->y, -SHIFT_DY, FLOW_TOLERANCE_PX);
}

TEST_F(DenseOpticalFlowTest, ReallocatesBuffersWhenFrameSizeChanges) {
  DenseOpticalFlow opticalFlow;
  cv::Mat smallPrev{cropAt(sourceImage_, CROP_X, CROP_Y, 400, 300)};
  cv::Mat smallCurr{
      cropAt(sourceImage_, CROP_X + SHIFT_DX, CROP_Y + SHIFT_DY, 400, 300)};
  cv::Mat largePrev{
      cropAt(sourceImage_, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)};
  cv::Mat largeCurr{cropAt(sourceImage_, CROP_X + SHIFT_DX, CROP_Y + SHIFT_DY,
                           CROP_WIDTH, CROP_HEIGHT)};

  opticalFlow.computeFlow(smallPrev, smallCurr);
  std::optional<cv::Point2f> flow{
      opticalFlow.computeFlow(largePrev, largeCurr)};

  ASSERT_TRUE(flow.has_value());
  EXPECT_NEAR(flow->x, -SHIFT_DX, FLOW_TOLERANCE_PX);
  EXPECT_NEAR(flow->y, -SHIFT_DY, FLOW_TOLERANCE_PX);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
