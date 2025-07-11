#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <optional>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

cv::Mat constructExtrinsicMatrix(const cv::Mat& rvec, const cv::Mat& tvec);
cv::Mat invertExtrinsicMatrix(const cv::Mat& extrinsic);
cv::Point2f distortPixelCoords(const cv::Point2f& undistortedPixelCoords,
                               const cv::Mat& intrinsicMatrix,
                               const cv::Mat& distCoeffs);
cv::Ptr<cv::SimpleBlobDetector> createBlobDetector();
std::tuple<float, std::vector<float>> _calcReprojectionError(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);
std::optional<std::tuple<cv::Mat, cv::Mat>> calibrateCamera(
    const std::vector<cv::Mat> monoImages, const cv::Size& gridSize,
    int gridType = cv::CALIB_CB_SYMMETRIC_GRID,
    const cv::Ptr<cv::FeatureDetector> blobDetector = NULL);
void testUndistortion(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                      const cv::Mat& img);