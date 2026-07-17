#pragma once

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Stream.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>

#include <opencv2/opencv.hpp>

class SparseOpticalFlow {
 public:
  explicit SparseOpticalFlow(int32_t maxCorners = 100,
                             int32_t pyramidLevels = 3,
                             cv::Rect includeRegion = cv::Rect());
  ~SparseOpticalFlow();
  SparseOpticalFlow(const SparseOpticalFlow&) = delete;
  SparseOpticalFlow& operator=(const SparseOpticalFlow&) = delete;

  // Detects feature points in prevFrame within includeRegion_, tracks them
  // into currFrame, and returns the median displacement vector (dx, dy) in
  // pixels over successfully tracked points.
  cv::Point2f computeFlow(const cv::Mat& prevFrame, const cv::Mat& currFrame);

 private:
  void allocateBuffers(int32_t width, int32_t height);
  void destroyBuffers();

  int32_t maxCorners_;
  int32_t pyramidLevels_;
  cv::Rect includeRegion_;

  VPIStream stream_{nullptr};
  VPIPayload harrisPayload_{nullptr};
  VPIPayload lkPayload_{nullptr};

  VPIImage imgPrevPL_{nullptr};
  VPIImage imgCurrPL_{nullptr};
  VPIImage imgPrevGray_{nullptr};
  VPIImage imgCurrGray_{nullptr};

  VPIPyramid pyrPrev_{nullptr};
  VPIPyramid pyrCurr_{nullptr};

  VPIArray keypointsPrev_{nullptr};
  VPIArray keypointsCurr_{nullptr};
  VPIArray scores_{nullptr};
  VPIArray status_{nullptr};

  cv::Size bufferedSize_{-1, -1};
};
