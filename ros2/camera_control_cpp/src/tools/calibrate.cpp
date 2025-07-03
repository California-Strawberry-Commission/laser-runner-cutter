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
    size_t image_count = 0;
    for (const auto& entry : directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                image_paths.push_back(entry.path());
                ++image_count;
            }
        }
    }
    if (image_count < 9) {
        cout << "Error: Directory must contain at least 9 image (.png, .jpg, .jpeg) files." << endl;
        return 1;
    } else {
        cout << "Directory opened successfully: " << dir_path << endl;
    }

    
    

    return 0;
}