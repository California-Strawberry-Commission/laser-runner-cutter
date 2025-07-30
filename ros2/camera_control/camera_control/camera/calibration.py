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

    if len(mono_images) < 3:
        raise Exception(
            "At least 3 images are required to calculate the camera intrinsics"
        )

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


def construct_extrinsic_matrix(rvec, tvec):
    # Convert rotation vector to rotation matrix using Rodrigues' formula
    R, _ = cv2.Rodrigues(rvec)

    # Create the extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = tvec

    return extrinsic_matrix


def extract_pose_from_extrinsic(extrinsic_matrix):
    R = extrinsic_matrix[:3, :3]
    tvec = extrinsic_matrix[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec


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


def scale_grayscale_image(mono_image: np.ndarray) -> np.ndarray:
    """
    Scale a grayscale image so that it uses the full 8-bit range.

    Args:
        mono_image: Grayscale image to scale.

    Returns:
        np.ndarray: Scaled uint8 grayscale image.
    """
    # Convert to float to avoid overflow or underflow issues
    mono_image = np.array(mono_image, dtype=np.float32)

    # Find the minimum and maximum values in the image
    min_val = np.min(mono_image)
    max_val = np.max(mono_image)

    # Normalize the image to the range 0 to 1
    if max_val > min_val:
        mono_image = (mono_image - min_val) / (max_val - min_val)
    else:
        mono_image = mono_image - min_val

    # Scale to 0-255 and convert to uint8
    mono_image = (mono_image * 255).astype(np.uint8)

    return mono_image


def _calc_intrinsics(
    images_dir: str,
    output_dir: str,
    grid_size: Tuple[int, int] = (5, 4),
    grid_type: int = cv2.CALIB_CB_SYMMETRIC_GRID,
    blob_detector=create_blob_detector(),
    show_undistorted=False,
):
    """
    Calculate camera intrinsics given images of a calibration grid.

    Args:
        images_dir: Directory containing images of a calibration grid.
        output_dir: Directory to write the intrinsic matrix and distortion coefficients.
        grid_size (Tuple[int, int]): (# cols, # rows) of the calibration pattern.
        grid_type (int): One of the following:
            cv2.CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
            cv2.CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
            cv2.CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter.
        blobDetector: Feature detector that finds blobs, like dark circles on light background. If None then a default implementation is used.
    """
    images_dir = os.path.expanduser(images_dir)
    output_dir = os.path.expanduser(output_dir)

    image_paths = sorted(
        glob(os.path.join(images_dir, "*.jpg"))
        + glob(os.path.join(images_dir, "*.png"))
    )
    images = [
        scale_grayscale_image(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        for image_path in image_paths
    ]
    intrinsic_matrix, distortion_coeffs = calibrate_camera(
        images,
        grid_size,
        grid_type,
        blob_detector,
    )

    print(f"Calibrated intrins: {intrinsic_matrix}")
    print(f"Distortion coeffs: {distortion_coeffs}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(
        os.path.join(output_dir, "intrinsic_matrix.npy"),
        intrinsic_matrix,
    )
    np.save(
        os.path.join(output_dir, "distortion_coeffs.npy"),
        distortion_coeffs,
    )

    if show_undistorted:
        # Visual check of the intrinsic matrix and distortion coefficients by showing the undistorted image
        for image in images:
            h, w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                intrinsic_matrix, distortion_coeffs, (w, h), 1, (w, h)
            )
            undistorted_img = cv2.undistort(
                image, intrinsic_matrix, distortion_coeffs, None, newcameramtx
            )
            # Crop the image
            x, y, w, h = roi
            undistorted_img = undistorted_img[y : y + h, x : x + w]
            cv2.imshow("undistorted", undistorted_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def _calc_extrinsics(
    camera_images_dir: str,
    helios_images_dir: str,
    helios_xyz_dir: str,
    camera_intrinsic_matrix_file: str,
    camera_distortion_coeffs_file: str,
    output_dir: str,
    grid_size: Tuple[int, int] = (5, 4),
    grid_type: int = cv2.CALIB_CB_SYMMETRIC_GRID,
    blob_detector=create_blob_detector(),
):
    """
    Calculate the extrinsic matrix given images of a calibration grid.
    `camera_images_dir`, `helios_images_dir`, and `helios_xyz_dir` must contain files with the same
    base name, which is how we form image correspondences. Images/files with the same base name
    should be of the calibration grid in the identical position.

    Args:
        camera_images_dir: Directory containing images of a calibration grid taken by the target camera.
        helios_images_dir: Directory containing Helios intensity images of a calibration grid.
        helios_xyz_dir: Directory containing Helios XYZ .npy files of a calibration grid.
        camera_intrinsic_matrix_file: File path to the target camera's intrinsic matrix (.npy file).
        camera_distortion_coeffs_file: File path to the target camera's distortion coefficients (.npy file).
        output_dir: Directory to write the intrinsic matrix and distortion coefficients.
        grid_size (Tuple[int, int]): (# cols, # rows) of the calibration pattern.
        grid_type (int): One of the following:
            cv2.CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
            cv2.CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
            cv2.CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter.
        blobDetector: Feature detector that finds blobs, like dark circles on light background. If None then a default implementation is used.
    """
    camera_images_dir = os.path.expanduser(camera_images_dir)
    helios_images_dir = os.path.expanduser(helios_images_dir)
    helios_xyz_dir = os.path.expanduser(helios_xyz_dir)
    camera_intrinsic_matrix_file = os.path.expanduser(camera_intrinsic_matrix_file)
    camera_distortion_coeffs_file = os.path.expanduser(camera_distortion_coeffs_file)
    output_dir = os.path.expanduser(output_dir)

    intrinsic_matrix = np.load(camera_intrinsic_matrix_file)
    distortion_coeffs = np.load(camera_distortion_coeffs_file)

    # Find all images in the target camera's images dir
    camera_image_paths = sorted(
        glob(os.path.join(os.path.expanduser(camera_images_dir), "*.jpg"))
        + glob(os.path.join(os.path.expanduser(camera_images_dir), "*.png"))
    )

    circle_coords = []
    circle_xyz_positions = []

    # For each image in the camera images dir, find the corresponding Helios intensity image and XYZ image
    for camera_image_path in camera_image_paths:
        base_name = os.path.splitext(os.path.basename(camera_image_path))[0]

        # Look for matching image in helios_images_dir
        helios_image_path = None
        for ext in [".jpg", ".png"]:
            candidate = os.path.join(helios_images_dir, base_name + ext)
            if os.path.exists(os.path.join(helios_images_dir, base_name + ext)):
                helios_image_path = candidate
                break

        # Look for matching xyz file in helios_xyz_dir
        helios_xyz_path = os.path.join(helios_xyz_dir, base_name + ".npy")

        if helios_image_path and os.path.exists(helios_xyz_path):
            print(f"Processing {base_name}:")
            print(f"  Camera image file: {camera_image_path}")
            print(f"  Helios image file: {helios_image_path}")
            print(f"  Helios xyz file:  {helios_xyz_path}")

            # Get circle centers in the camera image
            camera_image = scale_grayscale_image(
                cv2.imread(camera_image_path, cv2.IMREAD_GRAYSCALE)
            )
            retval, current_circle_coords = cv2.findCirclesGrid(
                camera_image, grid_size, flags=grid_type, blobDetector=blob_detector
            )
            if not retval:
                print("Could not get circle centers from the target camera's image.")
                continue
            current_circle_coords = np.squeeze(current_circle_coords)
            circle_coords.append(current_circle_coords)

            # Get Helios circle centers
            helios_image = scale_grayscale_image(
                cv2.imread(helios_image_path, cv2.IMREAD_GRAYSCALE)
            )
            retval, current_helios_circle_coords = cv2.findCirclesGrid(
                helios_image, grid_size, flags=grid_type, blobDetector=blob_detector
            )
            if not retval:
                print("Could not get circle centers from Helios intensity image.")
                continue
            current_helios_circle_coords = np.squeeze(
                np.round(current_helios_circle_coords).astype(np.int32)
            )

            # Get XYZ values from the Helios 3D image that match 2D locations on the target camera (corresponding points)
            helios_xyz = np.load(helios_xyz_path)
            current_circle_xyz_positions = helios_xyz[
                current_helios_circle_coords[:, 1], current_helios_circle_coords[:, 0]
            ]
            circle_xyz_positions.append(current_circle_xyz_positions)

    if len(circle_coords) == 0 or len(circle_xyz_positions) == 0:
        print("No suitable correspondences found")
        return

    circle_coords = np.concatenate(circle_coords, axis=0)
    circle_xyz_positions = np.concatenate(circle_xyz_positions, axis=0)

    retval, rvec, tvec = cv2.solvePnP(
        circle_xyz_positions,
        circle_coords,
        intrinsic_matrix,
        distortion_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    """
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        circle_xyz_positions,
        circle_coords,
        intrinsic_matrix,
        distortion_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=4.0,
        confidence=0.999,
    )
    """
    if not retval:
        print("Could not calculate extrinsic matrix.")
        return

    # Calculate reprojection error
    reprojected, _ = cv2.projectPoints(
        circle_xyz_positions,
        rvec,
        tvec,
        intrinsic_matrix,
        distortion_coeffs,
    )
    # Flatten projected points to (N, 2)
    reprojected = reprojected.reshape(-1, 2)
    errors = np.linalg.norm(circle_coords - reprojected, axis=1)

    # Mean reprojection error
    mean_error = np.mean(errors)
    print("Reprojection error (mean):", mean_error)

    rvec = rvec.flatten()
    tvec = tvec.flatten()
    extrinsic_matrix = construct_extrinsic_matrix(rvec, tvec)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(
        os.path.join(output_dir, "extrinsic_matrix.npy"),
        extrinsic_matrix,
    )


def _visualize_extrinsics(
    camera_mono_image_file: str,
    helios_intensity_image_file: str,
    helios_xyz_file: str,
    camera_intrinsic_matrix_file: str,
    camera_distortion_coeffs_file: str,
    xyz_to_camera_extrinsic_matrix_file: str,
):
    camera_mono_image_file = os.path.expanduser(camera_mono_image_file)
    helios_intensity_image_file = os.path.expanduser(helios_intensity_image_file)
    helios_xyz_file = os.path.expanduser(helios_xyz_file)
    camera_intrinsic_matrix_file = os.path.expanduser(camera_intrinsic_matrix_file)
    camera_distortion_coeffs_file = os.path.expanduser(camera_distortion_coeffs_file)
    xyz_to_camera_extrinsic_matrix_file = os.path.expanduser(
        xyz_to_camera_extrinsic_matrix_file
    )

    helios_xyz = np.load(helios_xyz_file)
    camera_intrinsic_matrix = np.load(camera_intrinsic_matrix_file)
    camera_distortion_coeffs = np.load(camera_distortion_coeffs_file)
    xyz_to_camera_extrinsic_matrix = np.load(xyz_to_camera_extrinsic_matrix_file)
    camera_mono_image = scale_grayscale_image(
        cv2.imread(camera_mono_image_file, cv2.IMREAD_GRAYSCALE)
    )
    helios_intensity_image = scale_grayscale_image(
        cv2.imread(helios_intensity_image_file, cv2.IMREAD_GRAYSCALE)
    )

    rvec, tvec = extract_pose_from_extrinsic(xyz_to_camera_extrinsic_matrix)
    h, w, c = helios_xyz.shape
    object_points = helios_xyz.reshape(h * w, c)
    projected_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_intrinsic_matrix, camera_distortion_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)

    # Render camera frame as red
    camera_h, camera_w = camera_mono_image.shape
    projection_img = np.zeros((camera_h, camera_w, 3), dtype=np.uint8)
    projection_img[camera_mono_image < 100] = [0, 0, 128]  # red

    # Render XYZ as green
    # For every pixel in the depth frame, render the xyz projected onto the camera image plane.
    # Only render if the intensity is lower than a certain threshold so that the grid circles are
    # visible.
    helios_intensity_image = helios_intensity_image.flatten()
    for point_idx in range(h * w):
        point = projected_points[point_idx]
        col = round(point[0])
        row = round(point[1])
        if 0 <= col and col < camera_w and 0 <= row and row < camera_h:
            intensity = helios_intensity_image[point_idx]
            thresh = 1 if intensity < 30 else 0
            projection_img[row][col][1] = thresh * 255  # green overlay

    cv2.namedWindow("projection", cv2.WINDOW_NORMAL)
    cv2.imshow("projection", projection_img)
    cv2.resizeWindow("projection", 800, 600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    calc_intrinsics_parser = subparsers.add_parser(
        "calc_intrinsics",
        help="Calculate camera intrinsics and distortion coefficients from images of a calibration pattern",
    )
    calc_intrinsics_parser.add_argument(
        "-i",
        "--images_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the directory containing grayscale images of a calibration pattern",
    )
    calc_intrinsics_parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the directory to write intrinsic parameters to",
    )
    calc_intrinsics_parser.add_argument(
        "-s",
        "--show_undistorted",
        action="store_true",
        help="Show undistorted versions of input images",
    )

    calc_extrinsics_xyz_to_triton_parser = subparsers.add_parser(
        "calc_extrinsics_xyz_to_triton",
        help="Calculate extrinsics that describe the orientation of Triton relative to Helios XYZ from images of a calibration pattern",
    )
    calc_extrinsics_xyz_to_triton_parser.add_argument(
        "--triton_images_dir", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_triton_parser.add_argument(
        "--helios_images_dir", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_triton_parser.add_argument(
        "--helios_xyz_dir", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_triton_parser.add_argument(
        "--triton_intrinsic_matrix_file", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_triton_parser.add_argument(
        "--triton_distortion_coeffs_file", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_triton_parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the directory to write extrinsic parameters to",
    )

    calc_extrinsics_xyz_to_helios_parser = subparsers.add_parser(
        "calc_extrinsics_xyz_to_helios",
        help="Calculate extrinsics that describe the orientation of Helios intensity relative to Helios XYZ from images of a calibration pattern",
    )
    calc_extrinsics_xyz_to_helios_parser.add_argument(
        "--helios_images_dir", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_helios_parser.add_argument(
        "--helios_xyz_dir", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_helios_parser.add_argument(
        "--helios_intrinsic_matrix_file", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_helios_parser.add_argument(
        "--helios_distortion_coeffs_file", type=str, default=None, required=True
    )
    calc_extrinsics_xyz_to_helios_parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the directory to write extrinsic parameters to",
    )

    visualize_extrinsics_xyz_to_triton_parser = subparsers.add_parser(
        "visualize_extrinsics_xyz_to_triton",
        help="Verify extrinsics between Triton and Helios XYZ by projecting XYZ onto Triton image",
    )
    visualize_extrinsics_xyz_to_triton_parser.add_argument(
        "--triton_mono_image", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_triton_parser.add_argument(
        "--helios_intensity_image", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_triton_parser.add_argument(
        "--helios_xyz", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_triton_parser.add_argument(
        "--triton_intrinsic_matrix_file", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_triton_parser.add_argument(
        "--triton_distortion_coeffs_file", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_triton_parser.add_argument(
        "--xyz_to_triton_extrinsic_matrix_file", type=str, default=None, required=True
    )

    visualize_extrinsics_xyz_to_helios_parser = subparsers.add_parser(
        "visualize_extrinsics_xyz_to_helios",
        help="Verify extrinsics between Helios intensity and Helios XYZ by projecting XYZ onto Helios intensity image",
    )
    visualize_extrinsics_xyz_to_helios_parser.add_argument(
        "--helios_intensity_image", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_helios_parser.add_argument(
        "--helios_xyz", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_helios_parser.add_argument(
        "--helios_intrinsic_matrix_file", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_helios_parser.add_argument(
        "--helios_distortion_coeffs_file", type=str, default=None, required=True
    )
    visualize_extrinsics_xyz_to_helios_parser.add_argument(
        "--xyz_to_helios_extrinsic_matrix_file", type=str, default=None, required=True
    )

    args = parser.parse_args()

    if args.command == "calc_intrinsics":
        _calc_intrinsics(
            args.images_dir, args.output_dir, show_undistorted=args.show_undistorted
        )
    elif args.command == "calc_extrinsics_xyz_to_triton":
        _calc_extrinsics(
            args.triton_images_dir,
            args.helios_images_dir,
            args.helios_xyz_dir,
            args.triton_intrinsic_matrix_file,
            args.triton_distortion_coeffs_file,
            args.output_dir,
        )
    elif args.command == "calc_extrinsics_xyz_to_helios":
        _calc_extrinsics(
            args.helios_images_dir,
            args.helios_images_dir,
            args.helios_xyz_dir,
            args.helios_intrinsic_matrix_file,
            args.helios_distortion_coeffs_file,
            args.output_dir,
        )
    elif args.command == "visualize_extrinsics_xyz_to_triton":
        _visualize_extrinsics(
            args.triton_mono_image,
            args.helios_intensity_image,
            args.helios_xyz,
            args.triton_intrinsic_matrix_file,
            args.triton_distortion_coeffs_file,
            args.xyz_to_triton_extrinsic_matrix_file,
        )
    elif args.command == "visualize_extrinsics_xyz_to_helios":
        _visualize_extrinsics(
            args.helios_intensity_image,
            args.helios_intensity_image,
            args.helios_xyz,
            args.helios_intrinsic_matrix_file,
            args.helios_distortion_coeffs_file,
            args.xyz_to_helios_extrinsic_matrix_file,
        )
    else:
        print("Invalid command.")
