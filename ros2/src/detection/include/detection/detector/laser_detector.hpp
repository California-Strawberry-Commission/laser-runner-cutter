#pragma once

#include <opencv2/opencv.hpp>

/**
 * Detects the center point of a red laser spot in an RGB image.
 *
 * Only pixels where red is the dominant channel are considered, so this is
 * intended for a red-colored laser and will not detect other laser colors.
 * Works best when the camera exposure is as low as possible.
 *
 * At most a single detection is returned.
 */
class LaserDetector {
 public:
  struct Laser {
    // The detection's confidence probability
    float conf{0.0f};
    // The representative point of the detected object
    cv::Point point{-1, -1};
  };

  static void drawDetections(cv::Mat& targetImage,
                             const std::vector<Laser>& lasers,
                             const cv::Size& originalImageSize,
                             cv::Scalar color = {255, 0, 255});

  explicit LaserDetector() = default;
  LaserDetector(const LaserDetector&) = delete;
  LaserDetector& operator=(const LaserDetector&) = delete;
  LaserDetector(LaserDetector&&) noexcept = default;
  LaserDetector& operator=(LaserDetector&&) noexcept = default;
  ~LaserDetector() = default;

  // Returns an empty vector if no laser is found, otherwise exactly one Laser
  std::vector<Laser> detect(const cv::Mat& imageRgb);
};