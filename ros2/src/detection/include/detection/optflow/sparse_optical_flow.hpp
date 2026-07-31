#pragma once

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Stream.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

class SparseOpticalFlow {
 public:
  explicit SparseOpticalFlow(int32_t pyramidLevels = 3,
                             int32_t maxCorners = 200,
                             float harrisStrengthThresh = 0.01f,
                             float harrisSensitivity = 0.04f);
  ~SparseOpticalFlow();
  SparseOpticalFlow(const SparseOpticalFlow&) = delete;
  SparseOpticalFlow& operator=(const SparseOpticalFlow&) = delete;

  // Detects feature points in prevFrame within includeRegion, tracks them
  // into currFrame, and returns the median displacement vector (dx, dy) in
  // pixels over successfully tracked points.
  cv::Point2f computeFlow(const cv::Mat& prevFrame, const cv::Mat& currFrame,
                          cv::Rect includeRegion = cv::Rect());

  // Detects feature points in prevFrame within includeRegion, tracks them
  // into currFrame, and returns the median displacement vector (dx, dy) in
  // pixels over successfully tracked points.
  cv::Point2f computeFlow(const cv::cuda::GpuMat& prevFrame,
                          const cv::cuda::GpuMat& currFrame,
                          cv::Rect includeRegion = cv::Rect());

 private:
  enum class InputMemory { NONE, HOST, CUDA };

  void allocateBuffers(int32_t width, int32_t height);

  void destroyBuffers();

  // Destroys imgPrevPL_ and imgCurrPL_ if they currently wrap a different kind
  // of memory than mode, so that they can be rebound to the current mode.
  void ensureInputMode(InputMemory mode);

  void wrapCudaMat(VPIImage& img, const cv::cuda::GpuMat& mat);

  cv::Point2f trackAndComputeMedianFlow(const cv::Rect& includeRegion);

  int32_t pyramidLevels_;
  int32_t maxCorners_;
  float harrisStrengthThresh_;
  float harrisSensitivity_;

  VPIStream stream_{nullptr};
  VPIPayload harrisPayload_{nullptr};
  VPIPayload lkPayload_{nullptr};

  // Tracks which kind of memory imgPrevPL_ and imgCurrPL_ currently wrap,
  // since a VPIImage wrapper created for one cannot be rebound to the other.
  InputMemory inputMemory_{InputMemory::NONE};

  VPIImage imgPrevPL_{nullptr};
  VPIImage imgCurrPL_{nullptr};
  VPIImage imgPrevGray_{nullptr};
  VPIImage imgCurrGray_{nullptr};
  VPIImage imgPrevGrayHarris_{nullptr};

  VPIPyramid pyrPrev_{nullptr};
  VPIPyramid pyrCurr_{nullptr};

  VPIArray keypointsPrev_{nullptr};
  VPIArray keypointsCurr_{nullptr};
  VPIArray scores_{nullptr};
  VPIArray status_{nullptr};

  cv::Size bufferedSize_{-1, -1};
};
