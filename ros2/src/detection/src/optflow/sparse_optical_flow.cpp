#include "detection/optflow/sparse_optical_flow.hpp"

#include <spdlog/spdlog.h>
#include <vpi/Status.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <vpi/OpenCVInterop.hpp>

namespace {

// Max number of corners detected by Harris corner algorithm
constexpr int32_t MAX_HARRIS_CORNERS{8192};

void checkVpiStatus(VPIStatus status, int line) {
  if (status != VPI_SUCCESS) {
    char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(buffer, sizeof(buffer));
    std::ostringstream ss;
    ss << "SparseOpticalFlow: line " << line << ": " << vpiStatusGetName(status)
       << ": " << buffer;
    throw std::runtime_error(ss.str());
  }
}

}  // namespace

#define CHECK_STATUS(STMT) checkVpiStatus((STMT), __LINE__)

namespace {

// Filters out keypoints falling outside includeRegion (if the Rect is not
// empty), sorts the remainder by descending score, and keeps only the top
// maxCount.
void filterSortAndTruncateKeypoints(VPIArray keypoints, VPIArray scores,
                                    const cv::Rect& includeRegion,
                                    int32_t maxCount) {
  VPIArrayData ptsData, scoresData;
  CHECK_STATUS(vpiArrayLockData(keypoints, VPI_LOCK_READ_WRITE,
                                VPI_ARRAY_BUFFER_HOST_AOS, &ptsData));
  CHECK_STATUS(vpiArrayLockData(scores, VPI_LOCK_READ_WRITE,
                                VPI_ARRAY_BUFFER_HOST_AOS, &scoresData));

  VPIArrayBufferAOS& aosKeypoints = ptsData.buffer.aos;
  VPIArrayBufferAOS& aosScores = scoresData.buffer.aos;

  const VPIKeypointF32* kptData =
      reinterpret_cast<const VPIKeypointF32*>(aosKeypoints.data);
  const uint32_t* scoreData = reinterpret_cast<const uint32_t*>(aosScores.data);
  int32_t totalCount = *aosKeypoints.sizePointer;

  std::vector<int32_t> indices;
  indices.reserve(totalCount);
  for (int32_t i = 0; i < totalCount; ++i) {
    cv::Point pt{static_cast<int>(kptData[i].x),
                 static_cast<int>(kptData[i].y)};
    if (includeRegion.empty() || includeRegion.contains(pt)) {
      indices.push_back(i);
    }
  }

  std::stable_sort(indices.begin(), indices.end(),
                   [scoreData](int32_t a, int32_t b) {
                     return scoreData[a] > scoreData[b];
                   });
  indices.resize(std::min<size_t>(indices.size(), maxCount));

  std::vector<VPIKeypointF32> sorted;
  sorted.reserve(indices.size());
  for (int32_t idx : indices) {
    sorted.push_back(kptData[idx]);
  }
  VPIKeypointF32* kptDataMutable =
      reinterpret_cast<VPIKeypointF32*>(aosKeypoints.data);
  std::copy(sorted.begin(), sorted.end(), kptDataMutable);

  *aosKeypoints.sizePointer = static_cast<int32_t>(sorted.size());

  vpiArrayUnlock(scores);
  vpiArrayUnlock(keypoints);
}

// Computes the median of v in place.
float median(std::vector<float>& v) {
  size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  float result = v[mid];
  if (v.size() % 2 == 0) {
    std::nth_element(v.begin(), v.begin() + mid - 1, v.begin() + mid);
    result = (result + v[mid - 1]) / 2.0f;
  }
  return result;
}

}  // namespace

SparseOpticalFlow::SparseOpticalFlow(int32_t maxCorners, int32_t pyramidLevels,
                                     cv::Rect includeRegion)
    : maxCorners_{maxCorners},
      pyramidLevels_{pyramidLevels},
      includeRegion_{includeRegion} {
  CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CPU | VPI_BACKEND_CUDA, &stream_));
}

SparseOpticalFlow::~SparseOpticalFlow() {
  if (stream_ != nullptr) {
    vpiStreamSync(stream_);
  }
  destroyBuffers();
  vpiImageDestroy(imgPrevPL_);
  vpiImageDestroy(imgCurrPL_);
  vpiStreamDestroy(stream_);
}

void SparseOpticalFlow::destroyBuffers() {
  vpiPayloadDestroy(harrisPayload_);
  vpiPayloadDestroy(lkPayload_);
  vpiImageDestroy(imgPrevGray_);
  vpiImageDestroy(imgCurrGray_);
  vpiPyramidDestroy(pyrPrev_);
  vpiPyramidDestroy(pyrCurr_);
  vpiArrayDestroy(keypointsPrev_);
  vpiArrayDestroy(keypointsCurr_);
  vpiArrayDestroy(scores_);
  vpiArrayDestroy(status_);

  harrisPayload_ = nullptr;
  lkPayload_ = nullptr;
  imgPrevGray_ = nullptr;
  imgCurrGray_ = nullptr;
  pyrPrev_ = nullptr;
  pyrCurr_ = nullptr;
  keypointsPrev_ = nullptr;
  keypointsCurr_ = nullptr;
  scores_ = nullptr;
  status_ = nullptr;
  bufferedSize_ = cv::Size{-1, -1};
}

void SparseOpticalFlow::allocateBuffers(int32_t width, int32_t height) {
  if (harrisPayload_ != nullptr && bufferedSize_ == cv::Size(width, height)) {
    return;
  }

  if (harrisPayload_ != nullptr) {
    spdlog::warn(
        "SparseOpticalFlow: frame size changed from {}x{} to {}x{}; "
        "reallocating VPI buffers",
        bufferedSize_.width, bufferedSize_.height, width, height);
  }
  destroyBuffers();

  CHECK_STATUS(
      vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, 0, &imgPrevGray_));
  CHECK_STATUS(
      vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, 0, &imgCurrGray_));

  CHECK_STATUS(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8,
                                pyramidLevels_, 0.5, 0, &pyrPrev_));
  CHECK_STATUS(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8,
                                pyramidLevels_, 0.5, 0, &pyrCurr_));

  // keypointsPrev_ and scores_ are Harris corner detector's raw output buffers,
  // and is sized to MAX_HARRIS_CORNERS since Harris can find far more
  // candidates than maxCorners_ on images. We will end up keeping only the
  // maxCorners_ strongest corner candidates, and thus keypointsCurr_ and
  // status_ only need a size of maxCorners_.
  CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32,
                              0, &keypointsPrev_));
  CHECK_STATUS(
      vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U32, 0, &scores_));
  CHECK_STATUS(vpiArrayCreate(maxCorners_, VPI_ARRAY_TYPE_KEYPOINT_F32, 0,
                              &keypointsCurr_));
  CHECK_STATUS(vpiArrayCreate(maxCorners_, VPI_ARRAY_TYPE_U8, 0, &status_));

  // Harris needs to run on CPU
  CHECK_STATUS(vpiCreateHarrisCornerDetector(VPI_BACKEND_CPU, width, height,
                                             &harrisPayload_));
  CHECK_STATUS(vpiCreateOpticalFlowPyrLK(VPI_BACKEND_CUDA, width, height,
                                         VPI_IMAGE_FORMAT_U8, pyramidLevels_,
                                         0.5, &lkPayload_));

  bufferedSize_ = cv::Size(width, height);
}

cv::Point2f SparseOpticalFlow::computeFlow(const cv::Mat& prevFrame,
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

  // Convert to grayscale on CUDA
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA,
                                           imgPrevPL_, imgPrevGray_, nullptr));
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA,
                                           imgCurrPL_, imgCurrGray_, nullptr));

  // Detect feature points to track in prevFrame using Harris corner detection
  VPIHarrisCornerDetectorParams harrisParams;
  CHECK_STATUS(vpiInitHarrisCornerDetectorParams(&harrisParams));
  harrisParams.strengthThresh = 0.01f;
  harrisParams.sensitivity = 0.04f;
  CHECK_STATUS(vpiSubmitHarrisCornerDetector(
      stream_, VPI_BACKEND_CPU, harrisPayload_, imgPrevGray_, keypointsPrev_,
      scores_, &harrisParams));

  // Wait for Harris corner detection to finish
  CHECK_STATUS(vpiStreamSync(stream_));
  // Harris can find far more than maxCorners_ candidates on images, so we
  // keep only points inside includeRegion_ and, of those, only the maxCorners_
  // strongest.
  filterSortAndTruncateKeypoints(keypointsPrev_, scores_, includeRegion_,
                                 maxCorners_);

  // Build pyramids for both frames
  CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(
      stream_, VPI_BACKEND_CUDA, imgPrevGray_, pyrPrev_, VPI_BORDER_CLAMP));
  CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(
      stream_, VPI_BACKEND_CUDA, imgCurrGray_, pyrCurr_, VPI_BORDER_CLAMP));

  // Track the feature points from prevFrame into currFrame
  VPIOpticalFlowPyrLKParams lkParams;
  CHECK_STATUS(vpiInitOpticalFlowPyrLKParams(VPI_BACKEND_CUDA, &lkParams));
  CHECK_STATUS(vpiSubmitOpticalFlowPyrLK(stream_, VPI_BACKEND_CUDA, lkPayload_,
                                         pyrPrev_, pyrCurr_, keypointsPrev_,
                                         keypointsCurr_, status_, &lkParams));

  // Wait for processing to finish
  CHECK_STATUS(vpiStreamSync(stream_));

  VPIArrayData prevData, currData, statusData;
  CHECK_STATUS(vpiArrayLockData(keypointsPrev_, VPI_LOCK_READ,
                                VPI_ARRAY_BUFFER_HOST_AOS, &prevData));
  CHECK_STATUS(vpiArrayLockData(keypointsCurr_, VPI_LOCK_READ,
                                VPI_ARRAY_BUFFER_HOST_AOS, &currData));
  CHECK_STATUS(vpiArrayLockData(status_, VPI_LOCK_READ,
                                VPI_ARRAY_BUFFER_HOST_AOS, &statusData));

  cv::Point2f medianFlow{0.0f, 0.0f};
  try {
    const VPIKeypointF32* prevPts =
        reinterpret_cast<const VPIKeypointF32*>(prevData.buffer.aos.data);
    const VPIKeypointF32* currPts =
        reinterpret_cast<const VPIKeypointF32*>(currData.buffer.aos.data);
    const uint8_t* trackingStatus =
        reinterpret_cast<const uint8_t*>(statusData.buffer.aos.data);
    int32_t numPoints = *prevData.buffer.aos.sizePointer;

    std::vector<float> dxs, dys;
    dxs.reserve(numPoints);
    dys.reserve(numPoints);
    for (int32_t i = 0; i < numPoints; ++i) {
      if (trackingStatus[i] == 0) {
        dxs.push_back(currPts[i].x - prevPts[i].x);
        dys.push_back(currPts[i].y - prevPts[i].y);
      }
    }

    if (!dxs.empty()) {
      medianFlow = cv::Point2f(median(dxs), median(dys));
    } else {
      spdlog::warn(
          "SparseOpticalFlow: no feature points were tracked ({} corners "
          "detected, 0 tracked)",
          numPoints);
    }
  } catch (...) {
    vpiArrayUnlock(keypointsPrev_);
    vpiArrayUnlock(keypointsCurr_);
    vpiArrayUnlock(status_);
    throw;
  }

  CHECK_STATUS(vpiArrayUnlock(keypointsPrev_));
  CHECK_STATUS(vpiArrayUnlock(keypointsCurr_));
  CHECK_STATUS(vpiArrayUnlock(status_));

  return medianFlow;
}
