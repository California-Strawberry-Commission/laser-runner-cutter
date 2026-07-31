#pragma once

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <opencv2/core/cuda.hpp>
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
  // median displacement vector (dx, dy) in pixels.
  cv::Point2f computeFlow(const cv::Mat& prevFrame, const cv::Mat& currFrame);

  // Computes dense optical flow between two same-sized frames and returns the
  // median displacement vector (dx, dy) in pixels.
  cv::Point2f computeFlow(const cv::cuda::GpuMat& prevFrame,
                          const cv::cuda::GpuMat& currFrame);

 private:
  enum class InputMemory { NONE, HOST, CUDA };

  void allocateBuffers(int32_t width, int32_t height);
  void destroyBuffers();

  // Destroys imgPrevPL_ and imgCurrPL_ if they currently wrap a different kind
  // of memory than mode, so that they can be rebound to the current mode.
  void ensureInputMode(InputMemory mode);

  void wrapCudaMat(VPIImage& img, const cv::cuda::GpuMat& mat);

  cv::Point2f trackAndComputeMedianFlow();

  int32_t gridSize_;
  VPIOpticalFlowQuality quality_;

  VPIStream stream_{nullptr};
  VPIPayload payload_{nullptr};

  // Tracks which kind of memory imgPrevPL_ and imgCurrPL_ currently wrap,
  // since a VPIImage wrapper created for one cannot be rebound to the other.
  InputMemory inputMemory_{InputMemory::NONE};

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
