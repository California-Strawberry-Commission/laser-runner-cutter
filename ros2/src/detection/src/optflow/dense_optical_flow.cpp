#include "detection/optflow/dense_optical_flow.hpp"

#include <spdlog/spdlog.h>
#include <vpi/Status.h>
#include <vpi/algo/ConvertImageFormat.h>

#include <sstream>
#include <stdexcept>
#include <vpi/OpenCVInterop.hpp>

namespace {

void checkVpiStatus(VPIStatus status, int line) {
  if (status != VPI_SUCCESS) {
    char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(buffer, sizeof(buffer));
    std::ostringstream ss;
    ss << "DenseOpticalFlow: line " << line << ": " << vpiStatusGetName(status)
       << ": " << buffer;
    throw std::runtime_error(ss.str());
  }
}

}  // namespace

#define CHECK_STATUS(STMT) checkVpiStatus((STMT), __LINE__)

DenseOpticalFlow::DenseOpticalFlow(int32_t gridSize,
                                   VPIOpticalFlowQuality quality)
    : gridSize_{gridSize}, quality_{quality} {
  CHECK_STATUS(vpiStreamCreate(
      VPI_BACKEND_OFA | VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &stream_));
}

DenseOpticalFlow::~DenseOpticalFlow() {
  if (stream_ != nullptr) {
    vpiStreamSync(stream_);
  }
  destroyBuffers();
  vpiStreamDestroy(stream_);
}

void DenseOpticalFlow::destroyBuffers() {
  vpiPayloadDestroy(payload_);
  vpiImageDestroy(imgPrevPL_);
  vpiImageDestroy(imgCurrPL_);
  vpiImageDestroy(imgPrevTmp_);
  vpiImageDestroy(imgCurrTmp_);
  vpiImageDestroy(imgPrevBL_);
  vpiImageDestroy(imgCurrBL_);
  vpiImageDestroy(imgMotionVecBL_);
  vpiImageDestroy(imgMotionVecPL_);

  payload_ = nullptr;
  imgPrevPL_ = nullptr;
  imgCurrPL_ = nullptr;
  imgPrevTmp_ = nullptr;
  imgCurrTmp_ = nullptr;
  imgPrevBL_ = nullptr;
  imgCurrBL_ = nullptr;
  imgMotionVecBL_ = nullptr;
  imgMotionVecPL_ = nullptr;
  bufferedSize_ = cv::Size{-1, -1};
}

void DenseOpticalFlow::allocateBuffers(int32_t width, int32_t height) {
  if (payload_ != nullptr && bufferedSize_ == cv::Size(width, height)) {
    return;
  }

  if (payload_ != nullptr) {
    spdlog::warn(
        "DenseOpticalFlow: frame size changed from {}x{} to {}x{}; "
        "reallocating VPI buffers",
        bufferedSize_.width, bufferedSize_.height, width, height);
  }
  destroyBuffers();

  // Create Dense Optical Flow payload to be executed. Dense Optical Flow on
  // the OFA backend requires a single-level (numLevels=1) payload when
  // passing plain images rather than pyramids.
  CHECK_STATUS(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, width, height,
                                         VPI_IMAGE_FORMAT_Y8_ER_BL, &gridSize_,
                                         1, quality_, &payload_));

  // The Dense Optical Flow on OFA backend expects input to be in block-linear
  // (BL) format. Since Convert Image Format doesn't support direct BGR
  // pitch-linear (BGR/PL) to Y8 block-linear (Y8/BL) conversion, it must be
  // done in two passes: BGR/PL to Y8/PL using CUDA, then Y8/PL to Y8/BL using
  // VIC.
  CHECK_STATUS(
      vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y8_ER, 0, &imgPrevTmp_));
  CHECK_STATUS(
      vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y8_ER, 0, &imgCurrTmp_));
  CHECK_STATUS(
      vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &imgPrevBL_));
  CHECK_STATUS(
      vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &imgCurrBL_));

  // Motion vector image width and height, aligned to be a multiple of gridSize
  int32_t mvWidth{(width + gridSize_ - 1) / gridSize_};
  int32_t mvHeight{(height + gridSize_ - 1) / gridSize_};
  // Create the output motion vector buffers
  CHECK_STATUS(vpiImageCreate(mvWidth, mvHeight, VPI_IMAGE_FORMAT_2S16_BL, 0,
                              &imgMotionVecBL_));
  CHECK_STATUS(vpiImageCreate(mvWidth, mvHeight, VPI_IMAGE_FORMAT_2S16, 0,
                              &imgMotionVecPL_));

  bufferedSize_ = cv::Size(width, height);
}

cv::Point2f DenseOpticalFlow::computeMeanFlow(const cv::Mat& prevFrame,
                                              const cv::Mat& currFrame) {
  if (prevFrame.empty() || currFrame.empty()) {
    throw std::invalid_argument("Input frames must not be empty");
  }
  if (prevFrame.size() != currFrame.size() ||
      prevFrame.type() != currFrame.type()) {
    throw std::invalid_argument(
        "prevFrame and currFrame must have matching size and type");
  }

  allocateBuffers(prevFrame.cols, prevFrame.rows);

  // Wrap the input frames. If the wrappers already exist, just rebind them to
  // the new frame data instead of recreating the VPIImage wrapper objects.
  if (imgPrevPL_ == nullptr) {
    CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(prevFrame, 0, &imgPrevPL_));
    CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(currFrame, 0, &imgCurrPL_));
  } else {
    CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgPrevPL_, prevFrame));
    CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgCurrPL_, currFrame));
  }

  // BGR/PL -> Y8/PL on CUDA
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA,
                                           imgPrevPL_, imgPrevTmp_, nullptr));
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA,
                                           imgCurrPL_, imgCurrTmp_, nullptr));
  // Y8/PL -> Y8/BL on VIC
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC,
                                           imgPrevTmp_, imgPrevBL_, nullptr));
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC,
                                           imgCurrTmp_, imgCurrBL_, nullptr));

  // Dense Optical Flow on OFA
  CHECK_STATUS(vpiSubmitOpticalFlowDense(stream_, VPI_BACKEND_OFA, payload_,
                                         imgPrevBL_, imgCurrBL_,
                                         imgMotionVecBL_));
  // Convert motion vectors from BL to PL
  CHECK_STATUS(vpiSubmitConvertImageFormat(
      stream_, VPI_BACKEND_VIC, imgMotionVecBL_, imgMotionVecPL_, nullptr));

  // Wait for processing to finish
  CHECK_STATUS(vpiStreamSync(stream_));

  VPIImageData mvData;
  CHECK_STATUS(vpiImageLockData(imgMotionVecPL_, VPI_LOCK_READ,
                                VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));

  cv::Point2f meanFlow;
  try {
    cv::Mat mvImage;
    CHECK_STATUS(vpiImageDataExportOpenCVMat(mvData, &mvImage));

    // Motion vectors are in fixed-point S10.5 format, so divide by 32 to get
    // pixels
    cv::Mat flow;
    mvImage.convertTo(flow, CV_32FC2, 1.0 / (1 << 5));

    cv::Scalar mean{cv::mean(flow)};
    meanFlow =
        cv::Point2f(static_cast<float>(mean[0]), static_cast<float>(mean[1]));
  } catch (...) {
    vpiImageUnlock(imgMotionVecPL_);
    throw;
  }

  CHECK_STATUS(vpiImageUnlock(imgMotionVecPL_));

  return meanFlow;
}
