import argparse
from glob import glob
import os
import cv2
import numpy as np
import bisect
from scipy import ndimage
from shapely import Polygon
from shapely.ops import nearest_points
from skimage.morphology import skeletonize
from . import segment_utils


def contour_closest_point_to_centroid(contour):
    polygon = Polygon(contour)
    closest_polygon_point, closest_point = nearest_points(polygon, polygon.centroid)
    return [closest_polygon_point.x, closest_polygon_point.y]


def mask_center(mask):
    # Apply morphological operations to find the skeleton
    skeleton = skeletonize(mask)

    # Find non-zero pixels in the skeleton and flip to (x, y)
    points = np.column_stack(np.where(skeleton > 0))[:, ::-1]

    if len(points) == 0:
        return None
    elif len(points) == 1:
        return points[0]

    # Find the point in the skeleton that is closest to the centroid
    centroid = ndimage.center_of_mass(mask)
    centroid = (centroid[1], centroid[0])
    distances = np.linalg.norm(points - np.array(centroid), axis=1)
    center = points[np.argmin(distances)]
    return [center[0], center[1]]


def contour_center(contour):
    mask = segment_utils.convert_contour_to_mask(contour)
    return mask_center(mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Input mask image or dir path"
    )

    args = parser.parse_args()

    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = sorted(
            glob(os.path.join(args.input, "*.jpg"))
            + glob(os.path.join(args.input, "*.png"))
        )

    for image_path in image_paths:
        binary_mask = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE,
        )
        _, binary_mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
        center = mask_center(binary_mask)

        image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(
            image,
            [int(center[0]), int(center[1])],
            (255, 0, 255),
            cv2.MARKER_STAR,
            thickness=1,
            markerSize=20,
        )

        # Show skeleton line for reference
        line = segment_utils.convert_mask_to_line_segments(binary_mask, epsilon=2.0)
        cv2.polylines(
            image,
            [np.array(line)],
            isClosed=False,
            color=(0, 255, 0),
            thickness=1,
        )

        cv2.imshow("Mask Center", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
