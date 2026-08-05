// Tests for RunnerDetector. drawDetections() is pure OpenCV logic and is
// tested with synthetic data. track() requires a real TensorRT engine and GPU
// matching the one the engine under test/../models was built on.

#include <gtest/gtest.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>

#include "detection/detector/runner_detector.hpp"

namespace {

cv::Mat loadRgbTestImage(const std::string& filename) {
  std::string path{std::string(TEST_IMAGES_DIR) + "/" + filename};
  cv::Mat bgr{cv::imread(path)};
  if (bgr.empty()) {
    ADD_FAILURE() << "Failed to load test image at " << path;
    return bgr;
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

float iou(const cv::Rect& a, const cv::Rect& b) {
  float interArea{static_cast<float>((a & b).area())};
  float unionArea{static_cast<float>(a.area() + b.area()) - interArea};
  return unionArea > 0.0f ? interArea / unionArea : 0.0f;
}

}  // namespace

class RunnerDetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Constructed fresh per test so each test gets an empty ByteTrack
    // tracking history.
    detector_ = std::make_unique<RunnerDetector>("RunnerSegYoloV8l.engine");
  }

  std::unique_ptr<RunnerDetector> detector_;
};

TEST_F(RunnerDetectorTest, ReturnsValidRunners) {
  cv::Mat image{loadRgbTestImage("20240703102906.png")};

  auto runners{detector_->track(image)};

  EXPECT_GT(runners.size(), 0);

  cv::Rect imageRect{0, 0, image.cols, image.rows};
  for (const auto& runner : runners) {
    EXPECT_GT(runner.conf, 0.0f);
    EXPECT_EQ(runner.rect, runner.rect & imageRect);
    if (runner.point.x >= 0 || runner.point.y >= 0) {
      EXPECT_TRUE(runner.rect.contains(runner.point));
    }
  }
}

TEST_F(RunnerDetectorTest, PreservesTrackIdAcrossConsecutiveFrames) {
  cv::Mat image{loadRgbTestImage("20240703102906.png")};

  // Run a warmup frame so ByteTrack has a chance to activate tracks before we
  // assert ID continuity between the next two frames.
  detector_->track(image);
  auto firstFrame{detector_->track(image)};
  auto secondFrame{detector_->track(image)};

  for (const auto& r2 : secondFrame) {
    if (r2.trackId < 0) {
      continue;
    }

    const RunnerDetector::Runner* bestMatch{nullptr};
    float bestIou{0.0f};
    for (const auto& r1 : firstFrame) {
      float overlap{iou(r2.rect, r1.rect)};
      if (overlap > bestIou) {
        bestIou = overlap;
        bestMatch = &r1;
      }
    }

    if (bestMatch != nullptr && bestIou > 0.9f && bestMatch->trackId >= 0) {
      EXPECT_EQ(r2.trackId, bestMatch->trackId);
    }
  }
}

TEST_F(RunnerDetectorTest, FullImageBoundsMatchesUnboundedTrack) {
  cv::Mat image{loadRgbTestImage("20240703102906.png")};
  auto unbounded{detector_->track(image)};
  RunnerDetector boundedDetector{"RunnerSegYoloV8l.engine"};

  cv::Rect fullImageBounds{0, 0, image.cols, image.rows};
  auto bounded{boundedDetector.track(image, fullImageBounds)};

  ASSERT_EQ(unbounded.size(), bounded.size());
  for (size_t i = 0; i < unbounded.size(); ++i) {
    EXPECT_EQ(unbounded[i].rect, bounded[i].rect);
    EXPECT_EQ(unbounded[i].point, bounded[i].point);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
