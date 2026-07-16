#pragma once

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <opencv2/opencv.hpp>

class DenseOpticalFlow {
 public:
  explicit DenseOpticalFlow(
      int32_t gridSize = 4,
      VPIOpticalFlowQuality quality = VPI_OPTICAL_FLOW_QUALITY_MEDIUM);
  ~DenseOpticalFlow();
  DenseOpticalFlow(const DenseOpticalFlow&) = delete;
  DenseOpticalFlow& operator=(const DenseOpticalFlow&) = delete;

  // Computes dense optical flow between two same-sized frames and returns the
  // mean displacement vector (dx, dy) in pixels.
  cv::Point2f computeMeanFlow(const cv::Mat& prevFrame,
                              const cv::Mat& currFrame);

 private:
  void allocateBuffers(int32_t width, int32_t height);
  void destroyBuffers();

  int32_t gridSize_;
  VPIOpticalFlowQuality quality_;

  VPIStream stream_{nullptr};
  VPIPayload payload_{nullptr};
  VPIImage imgPrevPL_{nullptr};
  VPIImage imgCurrPL_{nullptr};
  VPIImage imgPrevTmp_{nullptr};
  VPIImage imgCurrTmp_{nullptr};
  VPIImage imgPrevBL_{nullptr};
  VPIImage imgCurrBL_{nullptr};
  VPIImage imgMotionVecBL_{nullptr};
  VPIImage imgMotionVecPL_{nullptr};

  cv::Size bufferedSize_{-1, -1};
};
