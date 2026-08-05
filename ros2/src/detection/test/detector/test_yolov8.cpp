// Tests for YoloV8 against a real TensorRT engine. These require a GPU whose
// TensorRT engine plan matches the target (e.g. the same Jetson the engine
// under test/../models was built on).

#include <gtest/gtest.h>

#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "detection/detector/yolov8.hpp"

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

}  // namespace

class YoloV8Test : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    model_ = std::make_unique<YoloV8>(std::string(MODELS_DIR) +
                                      "/RunnerSegYoloV8l.engine");
  }

  static void TearDownTestSuite() { model_.reset(); }

  static std::unique_ptr<YoloV8> model_;
};

std::unique_ptr<YoloV8> YoloV8Test::model_;

TEST_F(YoloV8Test, ThrowsOnEmptyImage) {
  EXPECT_THROW(model_->predict(cv::Mat()), std::invalid_argument);
}

TEST_F(YoloV8Test, ThrowsOnUnsupportedImageType) {
  cv::Mat grayImage{cv::Mat::zeros(480, 640, CV_8UC1)};
  EXPECT_THROW(model_->predict(grayImage), std::invalid_argument);
}

TEST_F(YoloV8Test, DetectionsAreValid) {
  cv::Mat image{loadRgbTestImage("20240703102906.png")};

  auto objects{model_->predict(image)};

  EXPECT_GT(objects.size(), 0);

  for (const auto& object : objects) {
    EXPECT_GE(object.label, 0);
    EXPECT_GT(object.conf, 0.25f);  // default confThreshold

    // Bounding box must lie within the image
    cv::Rect imageRect{0, 0, image.cols, image.rows};
    EXPECT_EQ(object.rect, object.rect & imageRect);

    if (object.rect.width > 0 && object.rect.height > 0) {
      ASSERT_FALSE(object.boxMask.empty());
      EXPECT_EQ(object.boxMask.cols, object.rect.width);
      EXPECT_EQ(object.boxMask.rows, object.rect.height);
      EXPECT_EQ(object.boxMask.type(), CV_8U);

      // Mask values must be binary (0 or 255)
      double minVal{0.0}, maxVal{0.0};
      cv::minMaxLoc(object.boxMask, &minVal, &maxVal);
      EXPECT_TRUE(minVal == 0.0 || minVal == 255.0);
      EXPECT_TRUE(maxVal == 0.0 || maxVal == 255.0);
    }
  }
}

TEST_F(YoloV8Test, MaxDetectionsCapsResultSize) {
  cv::Mat image{loadRgbTestImage("20240703102906.png")};

  auto objects{model_->predict(image, /*confThreshold=*/0.1f,
                               /*nmsThreshold=*/0.6f,
                               /*segmentationThreshold=*/0.5f,
                               /*maxDetections=*/1)};

  EXPECT_LE(objects.size(), 1u);
}

TEST_F(YoloV8Test, HandlesConsecutivePredictionsOnDifferentImageSizes) {
  // Checks that the internal buffer reuse paths correctly handle a change in
  // input image size between calls.
  cv::Mat fullImage{loadRgbTestImage("20240703102906.png")};
  cv::Mat croppedImage{fullImage(cv::Rect(0, 0, 640, 480)).clone()};

  EXPECT_NO_THROW(model_->predict(fullImage));
  EXPECT_NO_THROW(model_->predict(croppedImage));
  EXPECT_NO_THROW(model_->predict(fullImage));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
