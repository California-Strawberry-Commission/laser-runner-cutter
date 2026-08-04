// Integration test for LucidCamera against real Arena SDK hardware. These tests
// require a LUCID Triton (color) + Helios (depth) camera pair to be connected.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "camera_control/camera/lucid_camera.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace {

constexpr int COLOR_FRAME_WIDTH{2048};
constexpr int COLOR_FRAME_HEIGHT{1536};
constexpr int DEPTH_FRAME_WIDTH{640};
constexpr int DEPTH_FRAME_HEIGHT{480};
constexpr int FRAMES_TO_COLLECT{3};
constexpr std::chrono::seconds FRAME_WAIT_TIMEOUT{20};

}  // namespace

class LucidCameraIntegrationTest : public ::testing::Test {
 protected:
  void TearDown() override { camera_.stop(); }

  // Fails the current test if the camera doesn't reach STREAMING within its
  // internal timeout. Callers must check HasFatalFailure() immediately
  // afterward and return early if so, since a fatal failure in a helper doesn't
  // unwind the calling test body.
  void waitForStreamingOrFail() {
    try {
      camera_.waitForStreaming();
    } catch (const std::exception& e) {
      FAIL() << "Camera did not reach STREAMING state. Make sure that a "
                "Triton + Helios camera pair is connected. ("
             << e.what() << ")";
    }
  }

  LucidCamera camera_{/*colorCameraSerialNumber=*/std::nullopt,
                      /*depthCameraSerialNumber=*/std::nullopt,
                      {COLOR_FRAME_WIDTH, COLOR_FRAME_HEIGHT}};
};

TEST_F(LucidCameraIntegrationTest, ConnectsAndStreamsContinuousFrames) {
  ASSERT_EQ(camera_.getState(), LucidCamera::State::DISCONNECTED);

  std::mutex frameMutex;
  std::condition_variable frameCv;
  int colorFramesReceived{0};
  int depthFramesReceived{0};
  sensor_msgs::msg::Image::SharedPtr lastColorImage;
  sensor_msgs::msg::Image::SharedPtr lastDepthXyz;
  sensor_msgs::msg::Image::SharedPtr lastDepthIntensity;

  LucidCamera::ColorCallback colorCallback{
      [&](sensor_msgs::msg::Image::UniquePtr image) {
        std::lock_guard<std::mutex> lock(frameMutex);
        lastColorImage = std::move(image);
        ++colorFramesReceived;
        frameCv.notify_all();
      }};
  LucidCamera::DepthCallback depthCallback{
      [&](sensor_msgs::msg::Image::UniquePtr xyz,
          sensor_msgs::msg::Image::UniquePtr intensity) {
        std::lock_guard<std::mutex> lock(frameMutex);
        lastDepthXyz = std::move(xyz);
        lastDepthIntensity = std::move(intensity);
        ++depthFramesReceived;
        frameCv.notify_all();
      }};

  camera_.start(LucidCamera::CaptureMode::CONTINUOUS, /*exposureUs=*/-1.0,
                /*gainDb=*/-1.0, colorCallback, depthCallback);
  waitForStreamingOrFail();
  if (HasFatalFailure()) {
    return;
  }

  EXPECT_EQ(camera_.getState(), LucidCamera::State::STREAMING);
  EXPECT_EQ(camera_.getColorFrameSize(),
            (std::pair<int, int>{COLOR_FRAME_WIDTH, COLOR_FRAME_HEIGHT}));
  auto depthFrameSize{camera_.getDepthFrameSize()};
  EXPECT_EQ(depthFrameSize,
            (std::pair<int, int>{DEPTH_FRAME_WIDTH, DEPTH_FRAME_HEIGHT}));

  {
    std::unique_lock<std::mutex> lock(frameMutex);
    bool gotFrames{frameCv.wait_for(lock, FRAME_WAIT_TIMEOUT, [&] {
      return colorFramesReceived >= FRAMES_TO_COLLECT &&
             depthFramesReceived >= FRAMES_TO_COLLECT;
    })};
    ASSERT_TRUE(gotFrames)
        << "Timed out waiting for frames from camera (received "
        << colorFramesReceived << " color, " << depthFramesReceived
        << " depth)";
  }

  ASSERT_TRUE(lastColorImage);
  EXPECT_EQ(lastColorImage->encoding,
            sensor_msgs::image_encodings::BAYER_RGGB8);
  EXPECT_EQ(static_cast<int>(lastColorImage->width), COLOR_FRAME_WIDTH);
  EXPECT_EQ(static_cast<int>(lastColorImage->height), COLOR_FRAME_HEIGHT);

  ASSERT_TRUE(lastDepthXyz);
  ASSERT_TRUE(lastDepthIntensity);
  EXPECT_EQ(lastDepthXyz->encoding, sensor_msgs::image_encodings::TYPE_32FC3);
  EXPECT_EQ(lastDepthIntensity->encoding, sensor_msgs::image_encodings::MONO16);
  EXPECT_EQ(lastDepthXyz->width, lastDepthIntensity->width);
  EXPECT_EQ(lastDepthXyz->height, lastDepthIntensity->height);
  EXPECT_EQ(static_cast<int>(lastDepthXyz->width), depthFrameSize.first);
  EXPECT_EQ(static_cast<int>(lastDepthXyz->height), depthFrameSize.second);

  camera_.stop();
  EXPECT_EQ(camera_.getState(), LucidCamera::State::DISCONNECTED);
}

TEST_F(LucidCameraIntegrationTest, ExposureAndGainRoundTrip) {
  camera_.start(LucidCamera::CaptureMode::CONTINUOUS);
  waitForStreamingOrFail();
  if (HasFatalFailure()) {
    return;
  }

  auto exposureRange{camera_.getExposureUsRange()};
  ASSERT_LT(exposureRange.first, exposureRange.second);
  double targetExposure{20000.0};
  camera_.setExposureUs(targetExposure);
  EXPECT_NEAR(camera_.getExposureUs(), targetExposure, 0.1);

  auto gainRange{camera_.getGainDbRange()};
  ASSERT_LE(gainRange.first, gainRange.second);
  double targetGain{4.0};
  camera_.setGainDb(targetGain);
  EXPECT_NEAR(camera_.getGainDb(), targetGain, 0.1);

  camera_.setExposureUs(-1.0);
  EXPECT_DOUBLE_EQ(camera_.getExposureUs(), -1.0);
  camera_.setGainDb(-1.0);
  EXPECT_DOUBLE_EQ(camera_.getGainDb(), -1.0);
}

TEST_F(LucidCameraIntegrationTest, SingleFrameCaptureReturnsFramesOnDemand) {
  camera_.start(LucidCamera::CaptureMode::SINGLE_FRAME);
  waitForStreamingOrFail();
  if (HasFatalFailure()) {
    return;
  }

  EXPECT_EQ(camera_.getCaptureMode(), LucidCamera::CaptureMode::SINGLE_FRAME);

  auto frameOpt{camera_.getNextFrame()};
  ASSERT_TRUE(frameOpt.has_value());
  EXPECT_TRUE(frameOpt->colorImage);
  EXPECT_TRUE(frameOpt->depthXyz);
  EXPECT_TRUE(frameOpt->depthIntensity);
  EXPECT_EQ(static_cast<int>(frameOpt->colorImage->width), COLOR_FRAME_WIDTH);
  EXPECT_EQ(static_cast<int>(frameOpt->colorImage->height), COLOR_FRAME_HEIGHT);

  auto secondFrameOpt{camera_.getNextFrame()};
  EXPECT_TRUE(secondFrameOpt.has_value());
  EXPECT_TRUE(secondFrameOpt->colorImage);
  EXPECT_TRUE(secondFrameOpt->depthXyz);
  EXPECT_TRUE(secondFrameOpt->depthIntensity);
  EXPECT_EQ(static_cast<int>(secondFrameOpt->colorImage->width),
            COLOR_FRAME_WIDTH);
  EXPECT_EQ(static_cast<int>(secondFrameOpt->colorImage->height),
            COLOR_FRAME_HEIGHT);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
