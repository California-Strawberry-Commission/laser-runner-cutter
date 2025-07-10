#pragma once

#include <filesystem>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>


#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

Mat constructExtrinsicMatrix(Mat& rvec, Mat& tvec);
Mat invert_extrinsic_matrix(Mat& extrinsic);
Point2f distort_pixel_coords(
    Point2f& undistortedPixelCoords, 
    Mat& intrinsicMatrix,
    Mat& distCoeffs
);
cv::Ptr<cv::SimpleBlobDetector> createBlobDetector();
tuple<double, vector<double>> _calc_reprojection_error (
    vector<vector<Point3f>>& objectPoints,
    vector<vector<Point2f>>& imagePoints,
    vector<Mat>& rvecs,
    vector<Mat>& tvecs,
    Mat& cameraMatrix,
    Mat& distCoeffs
);
tuple<Mat, Mat> calibrate_camera(
    vector<Mat> monoImages,
    Size& gridSize,
    int gridType = CALIB_CB_SYMMETRIC_GRID,
    Ptr<FeatureDetector> blobDetector = NULL
);
void test_undistortion(
    Mat& cameraMatrix,
    Mat& distCoeffs,
    Mat& img
);