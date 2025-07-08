#include "camera_control_cpp/camera/calibrate.hpp"

using namespace std;
using namespace filesystem;
using namespace cv;

Ptr<SimpleBlobDetector> createBlobDetector() {
    /*
        Blob detector for white circles on black background
    */

    SimpleBlobDetector::Params params;

    // Filter By Color
    params.filterByColor = true;
    params.blobColor = 255;

    // Filter By Area
    params.filterByArea = true;
    params.minArea = 10.0;
    params.maxArea = 10000.0;

    return SimpleBlobDetector::create(params);
}

Mat constructExtrinsicMatrix(Mat& rvec, Mat& tvec) {
    /*
        Assemble 4x4 extrinsic matrix
    */

    Mat R;

    // Convert rotation vector to rotation matric using Rodrigues' formula
    Rodrigues(rvec, R);

    // Create extrinsic matrix
    Mat extrinsic = Mat::eye(4, 4, R.type());
    R.copyTo(extrinsic(Range(0, 3), Range(0, 3)));
    tvec.copyTo(extrinsic(Range(0, 3), Range(3, 4)));

    return extrinsic;
}

Mat invert_extrinsic_matrix(Mat& extrinsic) {
    /*
        Invert a 4x4 extrinsic matrix (rotation + translation)
    */

    // Extract the rotation matrix and the translation vetor
    Mat R = extrinsic(Range(0, 3), Range(0, 3));
    Mat t = extrinsic(Range(0, 3), Range(3, 4));

    // Compute the inverse rotation matrix
    Mat R_inv = R.t();

    // Compute the new translation vector
    Mat t_inv = -R_inv * t;

    // Construct the new extrinsic matrix
    Mat extrinsic_inv = Mat::eye(4, 4, R.type());
    R_inv.copyTo(extrinsic_inv(Range(0, 3), Range(0, 3)));
    t_inv.copyTo(extrinsic_inv(Range(0, 3), Range(3, 4)));

    return extrinsic_inv;
}

tuple<double, vector<double>> _calc_reprojection_error (
        vector<vector<Point3f>>& objectPoints,
        vector<vector<Point2f>>& imagePoints,
        vector<Mat>& rvecs,
        vector<Mat>& tvecs,
        Mat& cameraMatrix,
        Mat& distCoeffs
    ) {
    /*
    Compute the reprojection error.

    Args:
        object_points (List[np.ndarray]): List of object points in real-world space
        image_points (List[np.ndarray]): List of corresponding image points detected in images
        rvecs (np.ndarray): List of rotation vectors returned by cv2.calibrateCamera
        tvecs (np.ndarray): List of translation vectors returned by cv2.calibrateCamera
        camera_matrix (np.ndarray): Camera matrix
        dist_coeffs (np.ndarray): Distortion coefficients

    Returns:
        float: Tuple of (mean reprojection error, list of per-image errors)
    */
    vector<double> errors;
    double totalError = 0;
    double err;
    size_t i;
    for (i=0; i<objectPoints.size(); i++) {
        vector<Point2f> projected;
        projectPoints(
            objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projected
        );
        err = norm(
            imagePoints[i], projected, NORM_L2
        ) / projected.size();
        errors.push_back(err);
        totalError += err;
    }
    double meanErr = totalError / objectPoints.size();
    return make_tuple(meanErr, errors);
}


tuple<Mat, Mat> calibrate_camera(
        vector<Mat> monoImages,
        Size& gridSize,
        int gridType = CALIB_CB_SYMMETRIC_GRID,
        Ptr<FeatureDetector> blobDetector = NULL
    ) {
    /*
    Finds the camera intrinsic parameters and distortion coefficients from several views of a
    calibration pattern.

    Args:
        mono_images (List[np.ndarray]): Grayscale images each containing the calibration pattern.
        grid_size (Tuple[int, int]): (# cols, # rows) of the calibration pattern.
        grid_type (int): One of the following:
            cv2.CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
            cv2.CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
            cv2.CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter.
        blobDetector: Feature detector that finds blobs, like dark circalibrationPointscles on light background. If None then a default implementation is used.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple of (camera intrisic matrix, distortion coefficients), or (None, None) if calibration was unsuccessful.
    */

    // Prepare calibration pattern points,
    // These points are in the calibration pattern coordinate space. Since the calibration grid
    // is on a flat plane, we can set the Z coordinates as 0.

    vector<Point3f> calibrationPoints;
    int i, j;
    if(gridType == CALIB_CB_SYMMETRIC_GRID) {
        for (i=0; i < gridSize.height; i++) {
            for (j=0; j < gridSize.width; j++) {
                calibrationPoints.emplace_back(j, i, 0);
            }
        }
    } else if (gridType == CALIB_CB_ASYMMETRIC_GRID) {
        for (i=0; i < gridSize.height; i++) {
            for (j=0; j < gridSize.width; j++) {
                calibrationPoints.emplace_back((2 * j + i % 2), i, 0);
            }
        }
    } else {
        cerr << "Unsupported grid type." << endl;
        exit(-1);
    }

    vector<vector<Point3f>> objPoints;
    vector<vector<Point2f>> imgPoints;
    bool found;
    for (Mat image : monoImages) {
        vector<Point2f> centers;
        found = findCirclesGrid(image, gridSize, centers, gridType, blobDetector);
        if (found) {
            objPoints.push_back(calibrationPoints);
            imgPoints.push_back(centers);
        } else {
            cerr << "Could not get circle centers. Ignoring Image." << endl;
        }
    }

    try {
        vector<Mat> rvecs, tvecs;
        Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
        Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
        double retval = calibrateCamera(
            objPoints, imgPoints, monoImages[0].size(), cameraMatrix, distCoeffs, rvecs, tvecs
        );
        if (retval) {
            tuple<double, vector<double>> projErrors = _calc_reprojection_error(
                objPoints, imgPoints, rvecs, tvecs, cameraMatrix, distCoeffs
            );
            cout << "Calibration successful. Used " << objPoints.size() << " images. Mean reprojection error: " << get<0>(projErrors) << endl;
            return make_tuple(cameraMatrix, distCoeffs);
        }
    } catch(exception& e) {
        std::cerr << e.what() << endl;
    }

    cerr << "Calibration unsuccessful" << endl;
    exit(-1);
}



int main(int argc, char * argv[]) {

    if (argc != 2) {
        cout << "Imvalid call: Arguments must (exclusively) include calibration image dir" << endl;
        return 1;
    }

    path dir_path(argv[1]);
    if (!exists(dir_path) || !is_directory(dir_path)) {
        cout << "Error: Provided path is not a valid directory." << endl;
        return 1;
    }

    vector<path> image_paths;
    vector<Mat> images;
    size_t image_count = 0;
    for (const auto& entry : directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                Mat img = imread(entry.path().string());
                if (!img.empty()) {
                    image_paths.push_back(entry.path());
                    cvtColor(img, img, COLOR_BGR2GRAY);
                    images.push_back(img);
                    ++image_count;
                }
            }
        }
    }
    if (image_count < 9) {
        cout << "Error: Directory must contain at least 9 image (.png, .jpg, .jpeg) files." << endl;
        return 1;
    } else {
        cout << "Directory opened successfully: " << dir_path << endl;
    }

    Size gSize = Size(5,4);
    tuple<Mat, Mat> calibVals = calibrate_camera(
        images, gSize, CALIB_CB_SYMMETRIC_GRID, createBlobDetector()
    );
    
    Mat cameraMatrix = get<0>(calibVals);
    Mat distCoeffs = get<1>(calibVals);
    
    cout << "Calibrated intrins: " << cameraMatrix << endl;
    cout << "Distortion coeffs: " << distCoeffs << endl;

    // Undistortion Test: 
    Mat img = images[1];
    Mat newCameraMatrix, undistorted;
    Rect roi;
    newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, img.size(), 1, img.size(), &roi);
    undistort(img, undistorted, cameraMatrix, distCoeffs, newCameraMatrix);
    undistorted = undistorted(roi);
    string filename = "undistortion_test.png";
    bool success = imwrite(filename, img);
    if (success) {
        std::cout << "Image saved successfully as " << filename << std::endl;
    } else {
        std::cerr << "Error saving image to " << filename << std::endl;
    }

    return 0;
}