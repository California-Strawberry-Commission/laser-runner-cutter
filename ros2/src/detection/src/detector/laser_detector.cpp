#include "detection/detector/laser_detector.hpp"

namespace {

// Minimum peak red intensity for a spot to be considered a laser
constexpr double MIN_R{100.0};
// Blob threshold as a fraction of the peak intensity
constexpr double BLOB_RELATIVE_THRESHOLD{0.5};
// Allowed blob area range in pixels
constexpr int BLOB_MIN_AREA{4};
constexpr int BLOB_MAX_AREA{3000};

}  // namespace

void LaserDetector::drawDetections(
    cv::Mat& targetImage, const std::vector<LaserDetector::Laser>& lasers,
    const cv::Size& originalImageSize, cv::Scalar color) {
  // The image we are drawing on may not be the size of the image that the
  // runner detection was run on. Thus, we'll need to scale the bbox, mask, and
  // point of each runner to the image we are drawing on.
  double xScale{static_cast<double>(targetImage.cols) /
                originalImageSize.width};
  double yScale{static_cast<double>(targetImage.rows) /
                originalImageSize.height};

  for (const auto& laser : lasers) {
    if (laser.point.x >= 0 && laser.point.y >= 0) {
      int x{static_cast<int>(std::round(laser.point.x * xScale))};
      int y{static_cast<int>(std::round(laser.point.y * yScale))};
      cv::drawMarker(targetImage, cv::Point2i(x, y), color,
                     cv::MARKER_TILTED_CROSS, 20, 2);
    }
  }
}

std::vector<LaserDetector::Laser> LaserDetector::detect(
    const cv::Mat& imageRgb) {
  // Finds the single brightest red spot and returns its intensity-weighted
  // centroid. Works best when the camera exposure is as low as possible.
  std::vector<Laser> detections;

  // In order to eliminate potential false detections due to non-laser lighting
  // sources, we only look for the laser spot where the pixel is red-dominant.
  // Note that the saturated core of the spot is white so it also passes.
  std::vector<cv::Mat> channels;
  cv::split(imageRgb, channels);
  const cv::Mat& r{channels[0]};
  // Mask where red is the dominant channel
  cv::Mat redDominant{(r >= channels[1]) & (r >= channels[2])};
  // Copy the red channel values only where the red channel is dominant
  cv::Mat score{cv::Mat::zeros(r.size(), CV_8UC1)};
  r.copyTo(score, redDominant);

  // Apply a slight blur to suppress isolated hot pixels
  cv::GaussianBlur(score, score, cv::Size(3, 3), 0);

  double maxVal;
  cv::minMaxLoc(score, nullptr, &maxVal);
  if (maxVal < MIN_R) {
    return detections;
  }

  // Threshold relative to the peak and segment into blobs
  double thresh{maxVal * BLOB_RELATIVE_THRESHOLD};
  cv::Mat binary{score >= thresh};
  cv::Mat labels, stats, centroids;
  int numLabels{cv::connectedComponentsWithStats(binary, labels, stats,
                                                 centroids, 8, CV_32S)};

  // Pick the blob with the largest integrated intensity above threshold (the
  // brightest, largest one). Its intensity-weighted centroid is the laser
  // center.
  double bestMass{0.0};
  cv::Point2d bestCenter;
  for (int label = 1; label < numLabels; ++label) {
    int area{stats.at<int>(label, cv::CC_STAT_AREA)};
    if (area < BLOB_MIN_AREA || area > BLOB_MAX_AREA) {
      continue;
    }

    cv::Rect roi{stats.at<int>(label, cv::CC_STAT_LEFT),
                 stats.at<int>(label, cv::CC_STAT_TOP),
                 stats.at<int>(label, cv::CC_STAT_WIDTH),
                 stats.at<int>(label, cv::CC_STAT_HEIGHT)};
    cv::Mat weights;
    score(roi).convertTo(weights, CV_32F, 1.0, -thresh);
    weights.setTo(0.0f, labels(roi) != label);
    cv::Moments m{cv::moments(weights)};
    if (m.m00 > bestMass) {
      bestMass = m.m00;
      bestCenter = {roi.x + m.m10 / m.m00, roi.y + m.m01 / m.m00};
    }
  }

  if (bestMass <= 0.0) {
    return detections;
  }

  Laser laser;
  laser.point = cv::Point{static_cast<int>(std::lround(bestCenter.x)),
                          static_cast<int>(std::lround(bestCenter.y))};
  laser.conf = static_cast<float>(maxVal / 255.0);
  detections.push_back(laser);

  return detections;
}