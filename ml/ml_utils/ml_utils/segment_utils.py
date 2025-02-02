import cv2
import numpy as np
from itertools import groupby
import math
from scipy import ndimage
from skimage.morphology import skeletonize


def convert_mask_to_yolo_segment(mask):
    """
    Convert a mask to a YOLO segment (a single polygon with normalized coordinates)

    Args:
        mask (numpy.ndarray)
    """
    height, width = mask.shape
    polygons = convert_mask_to_polygons(mask)
    if len(polygons) > 1:
        return (
            (
                np.concatenate(merge_multi_segment(polygons), axis=0)
                / np.array([width, height])
            )
            .reshape(-1)
            .tolist()
        )
    else:
        return (
            (np.array(polygons[0]).reshape(-1, 2) / np.array([width, height]))
            .reshape(-1)
            .tolist()
        )


def convert_mask_to_coco_rle_segment(mask):
    """
    Convert a mask to RLE (column-major), used in the COCO annotation format.

    Args:
        mask (numpy.ndarray)
    """
    size = list(mask.shape)
    counts = []
    for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):
        if i == 0 and value > 0:
            counts.append(0)
        counts.append(len(list(elements)))
    return counts, size


def convert_mask_to_coco_poly_segment(mask):
    """
    Convert a mask to a COCO polygon segment (a single polygon with raw coordinates)

    Args:
        mask (numpy.ndarray)
    """
    polygons = convert_mask_to_polygons(mask)
    if len(polygons) > 1:
        return (
            np.concatenate(merge_multi_segment(polygons), axis=0).reshape(-1).tolist()
        )

    else:
        return polygons[0]


def merge_multi_segment(segments):
    """
    Merge multiple segments to one list. Find the coordinates with min distance between each
    segment, then connect these coordinates with one thin line to merge all segments into one.

    Args:
        segments (List(List)): list of segments, where each segment is a list of coordinates [x1, y1, x2, y2, ..., xn, yn]
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])

    return s


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def convert_mask_to_polygons(mask):
    """
    Convert a mask to a list of polygons based on contours

    Args:
        mask (numpy.ndarray)
    """
    contours, hierarchies = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = _merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)

    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons


def convert_contour_to_mask(contour, mask_size=None):
    contour = np.array(contour, dtype=np.int32)
    contour = contour.reshape(-1, 1, 2)

    if contour.size <= 0:
        return None

    if mask_size is None:
        max_x = np.max(contour[:, 0, 0])
        max_y = np.max(contour[:, 0, 1])
        mask_size = (math.ceil(max_x), math.ceil(max_y))

    mask = np.zeros((mask_size[1], mask_size[0]), dtype=np.uint8)
    try:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    except Exception as exc:
        return None

    return mask


def angle_between_points(p1, p2, p3):
    """
    Calculate the angle in radians formed by three points.
    """
    vector1 = np.array(p1) - np.array(p2)
    vector2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(vector1, vector2)
    magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle_radians = np.arccos(np.clip(dot_product / magnitudes, -1.0, 1.0))
    return angle_radians


def convert_mask_to_line_segments(mask, epsilon=4.0):
    """
    Convert a mask to line segments that best resemble that mask

    Args:
        mask (numpy.ndarray)
        epsilon (float): tolerance parameter for Douglas-Peucker algorithm. A larger epsilon will result in less line segments. An epsilon <= 0 will bypass simplification.
    """
    # Apply morphological operations to find the skeleton
    skeleton = skeletonize(mask)

    # Find non-zero pixels in the skeleton and flip to (x, y)
    points = np.column_stack(np.where(skeleton > 0))[:, ::-1]

    if len(points) < 2:
        return []

    # Estimate endpoint
    centroid = ndimage.center_of_mass(mask)
    centroid = (centroid[1], centroid[0])
    distances = np.linalg.norm(points - np.array(centroid), axis=1)
    start_point_index = np.argmax(distances)

    # Order a list of points such that subsequent points are closest together,
    # using the nearest neighbor approach
    points = _order_points_nearest_neighbor(points, start_point_index)

    if epsilon > 0:
        # Apply Douglas-Peucker algorithm for simplification
        points = cv2.approxPolyDP(points, epsilon=epsilon, closed=False)
        points = np.squeeze(points, axis=1)

    points = points.tolist()
    if len(points) > 2:
        # Remove any points that form tight angles
        filtered_points = points[:2]
        for i in range(2, len(points)):
            radians = angle_between_points(
                filtered_points[len(filtered_points) - 2],
                filtered_points[len(filtered_points) - 1],
                points[i],
            )
            degrees = math.degrees(radians)
            if degrees >= 90 and degrees <= 270:
                filtered_points.append(points[i])
        points = filtered_points
    return np.array(points)


def convert_contour_to_line_segments(contour, epsilon=4.0):
    """
    Convert a contour to line segments that best resemble that contour

    Args:
        contour (List(List)): points corresponding to the contour outline
        epsilon (float): tolerance parameter for Douglas-Peucker algorithm. A larger epsilon will result in less line segments
    """
    mask = convert_contour_to_mask(contour)
    if mask is None:
        return []

    try:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    except Exception as exc:
        return []

    return convert_mask_to_line_segments(mask, epsilon)


def _merge_with_parent(contour_parent, contour):
    if not _is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if _is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = _get_merge_point_idx(contour_parent, contour)
    return _merge_contours(contour_parent, contour, idx1, idx2)


def _is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
    return value < 0


def _merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour


def _get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2)
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2


def _order_points_nearest_neighbor(points, start_index):
    if len(points) <= 1:
        return points

    current_point = points[start_index]
    ordered_points = [current_point]
    remaining_points = np.delete(points, start_index, axis=0)
    while len(remaining_points) > 0:
        distances = np.linalg.norm(remaining_points - current_point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_point = remaining_points[nearest_index]

        current_point = nearest_point
        ordered_points.append(nearest_point)
        remaining_points = np.delete(remaining_points, nearest_index, axis=0)

    return np.array(ordered_points)
