import argparse
import os
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np


def calibrate_camera(
    mono_images: List[np.ndarray],
    grid_size: Tuple[int, int],
    grid_type: int = cv2.CALIB_CB_SYMMETRIC_GRID,
    blob_detector=None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Finds the camera intrinsic parameters and distortion coefficients from several views of a
    calibration pattern.

    Args:
        mono_images (List[np.ndarray]): Grayscale images each containing the calibration pattern.
        grid_size (Tuple[int, int]): (# cols, # rows) of the calibration pattern.
        grid_type (int): One of the following:
            cv2.CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
            cv2.CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
            cv2.CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter.
        blobDetector: Feature detector that finds blobs, like dark circles on light background. If None then a default implementation is used.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple of (camera intrisic matrix, distortion coefficients), or (None, None) if calibration was unsuccessful.
    """
    # Prepare calibration pattern points,
    # These points are in the calibration pattern coordinate space. Since the calibration grid
    # is on a flat plane, we can set the Z coordinates as 0.
    calibration_points = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)

    if grid_type == cv2.CALIB_CB_SYMMETRIC_GRID:
        calibration_points[:, :2] = np.mgrid[
            0 : grid_size[0], 0 : grid_size[1]
        ].T.reshape(-1, 2)
    elif grid_type == cv2.CALIB_CB_ASYMMETRIC_GRID:
        for i in range(0, grid_size[0] * grid_size[1]):
            row = i // grid_size[0]
            col = i % grid_size[0]
            row_offset = (row + 1) % 2
            x = col * 2 + row_offset
            y = row
            calibration_points[i] = (x, y, 0)
    else:
        raise Exception("Unsupported grid_type")

    obj_points = []
    img_points = []
    for image in mono_images:
        retval, centers = cv2.findCirclesGrid(
            image, grid_size, flags=grid_type, blobDetector=blob_detector
        )
        if retval:
            obj_points.append(calibration_points)
            img_points.append(centers)
        else:
            print(f"Could not get circle centers. Ignoring image.")

    try:
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, mono_images[0].shape[::-1], None, None
        )
        if retval:
            mean_error, errors = _calc_reprojection_error(
                obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs
            )
            print(
                f"Calibration successful. Used {len(obj_points)} images. Mean reprojection error: {mean_error}"
            )
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error during calibration: {e}")

    print("Calibration unsuccessful.")
    return None, None


def _calc_reprojection_error(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    rvecs: Tuple[np.ndarray, ...],
    tvecs: Tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[float, List[float]]:
    """
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
    """
    mean_error = 0
    total_points = 0
    errors = []
    for i in range(len(object_points)):
        projected_points, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(
            projected_points
        )
        errors.append(error)
        total_points += len(projected_points)
        mean_error += error

    mean_error /= len(object_points)
    return mean_error, errors


def create_blob_detector():
    """
    Blob detector for white circles on black background
    """
    params = cv2.SimpleBlobDetector_Params()

    # Filter by color
    params.filterByColor = True
    params.blobColor = 255

    # Filter by area
    params.filterByArea = True
    params.minArea = 10.0
    params.maxArea = 10000.0

    return cv2.SimpleBlobDetector_create(params)


def construct_extrinsic_matrix(rotation_vector, translation_vector):
    # Convert rotation vector to rotation matrix using Rodrigues' formula
    R, _ = cv2.Rodrigues(rotation_vector)

    # Create the extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = translation_vector

    return extrinsic_matrix


def invert_extrinsic_matrix(extrinsic_matrix: np.ndarray):
    # Extract the rotation matrix and translation vector
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]

    # Compute the inverse rotation matrix
    R_inv = R.T

    # Compute the new translation vector
    t_inv = -R_inv @ t

    # Construct the new extrinsic matrix
    extrinsic_matrix_inv = np.eye(4)
    extrinsic_matrix_inv[:3, :3] = R_inv
    extrinsic_matrix_inv[:3, 3] = t_inv

    return extrinsic_matrix_inv


def distort_pixel_coords(undistorted_pixel_coords, intrinsic_matrix, distortion_coeffs):
    # Convert the undistorted pixel coordinates to normalized camera coordinates
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    normalized_coords = (undistorted_pixel_coords - [cx, cy]) / [fx, fy]
    normalized_coords = np.append(normalized_coords, [1.0])

    # Reshape to (1, 1, 3) for projectPoints input
    normalized_coords = normalized_coords.reshape(1, 1, 3)

    # Project the normalized coordinates to distorted pixel coordinates using projectPoints
    distorted_pixel_coords, _ = cv2.projectPoints(
        normalized_coords,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        intrinsic_matrix,
        distortion_coeffs,
    )

    # Extract the distorted pixel coordinates
    distorted_pixel_coords = distorted_pixel_coords[0, 0, :2]

    return distorted_pixel_coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate camera intrinsics and distortion coefficients from images of a calibration pattern"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to the directory containing images of a calibration pattern",
    )

    args = parser.parse_args()
    image_paths = sorted(
        glob(os.path.join(args.input_dir, "*.jpg"))
        + glob(os.path.join(args.input_dir, "*.png"))
    )
    images = [
        cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2GRAY)
        for image_path in image_paths
    ]
    camera_matrix, dist_coeffs = calibrate_camera(
        images,
        (5, 4),
        grid_type=cv2.CALIB_CB_SYMMETRIC_GRID,
        blob_detector=create_blob_detector(),
    )
    print(f"Calibrated intrins: {camera_matrix}")
    print(f"Distortion coeffs: {dist_coeffs}")

    # Test calibration
    img = images[10]
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    undistorted_img = undistorted_img[y : y + h, x : x + w]
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
