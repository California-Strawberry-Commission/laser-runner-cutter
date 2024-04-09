import argparse
from glob import glob
import os
import cv2
import numpy as np
import bisect
from shapely import Polygon
from shapely.ops import nearest_points
from . import segment_utils


def contour_closest_point_to_centroid(contour):
    polygon = Polygon(contour)
    closest_polygon_point, closest_point = nearest_points(polygon, polygon.centroid)
    return [closest_polygon_point.x, closest_polygon_point.y]


def mask_center(mask):
    line = segment_utils.convert_mask_to_line_segments(mask, epsilon=2.0)
    return line_center(line)


def contour_center(contour):
    line = segment_utils.convert_contour_to_line_segments(contour, epsilon=2.0)
    return line_center(line)


def line_center(line):
    """
    Compute the distance-wise center of a line.
    """
    if len(line) < 1:
        return None
    elif len(line) == 1:
        return line[0]

    cumulative_distances = [0]
    total_length = 0

    # Compute cumulative distances along the line
    for i in range(1, len(line)):
        total_length += distance(line[i - 1], line[i])
        cumulative_distances.append(total_length)

    # Find the midpoint and indices of the points that the midpoint lies between
    half_length = total_length / 2
    insertion_idx = bisect.bisect(cumulative_distances, half_length)

    # Interpolate between the points
    segment_length = (
        cumulative_distances[insertion_idx] - cumulative_distances[insertion_idx - 1]
    )
    ratio = (half_length - cumulative_distances[insertion_idx - 1]) / segment_length
    x = line[insertion_idx - 1][0] + ratio * (
        line[insertion_idx][0] - line[insertion_idx - 1][0]
    )
    y = line[insertion_idx - 1][1] + ratio * (
        line[insertion_idx][1] - line[insertion_idx - 1][1]
    )

    return [x, y]


def distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


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
