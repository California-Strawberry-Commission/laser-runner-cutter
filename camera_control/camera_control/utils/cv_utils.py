import cv2
import numpy as np
from shapely import Polygon


def draw_laser(debug_frame, laser_list):
    for laser in laser_list:
        pos = [int(laser[0]), int(laser[1])]
        debug_frame = cv2.drawMarker(
            debug_frame,
            pos,
            (0, 0, 0),
            cv2.MARKER_CROSS,
            thickness=1,
            markerSize=20,
        )
    return debug_frame


def draw_runners(debug_frame, runner_list):
    for runner in runner_list:
        pos = [int(runner[0]), int(runner[1])]
        debug_frame = cv2.drawMarker(
            debug_frame,
            pos,
            (255, 0, 0),
            cv2.MARKER_STAR,
            thickness=1,
            markerSize=20,
        )
    return debug_frame


def detect_runners(model, frames):
    image = np.asanyarray(frames["color"].get_data())
    res = model(image)
    point_list = []
    if res[0].masks:
        for cords in res[0].masks.xy:
            polygon = Polygon(cords)
            point_list.append((polygon.centroid.x, polygon.centroid.y))
    return point_list


def detect_laser(frames, background_image):
    curr_image = np.asanyarray(frames["color"].get_data())
    image = cv2.absdiff(curr_image, background_image)
    found_point_list = find_laser_point(image)
    return found_point_list


def find_laser_point(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    return check_circularity(mask)


def find_laser_point_red(frame):
    """Find red laser points using the red minus max(green, blue) frame."""
    rg = frame[:, :, 2].astype(np.float32) - np.max(
        np.dstack((frame[:, :, 0], frame[:, :, 1])), axis=2
    ).astype(np.float32)

    max_val = np.max(rg)
    min_val = np.min(rg)
    cur_range = max_val - min_val
    scaling_factor = 255 / cur_range
    norm_rg = (rg - min_val) * scaling_factor
    norm_rg = norm_rg.astype(np.uint8)

    # Create a mask for red regions
    ret, mask = cv2.threshold(norm_rg, 150, 255, cv2.THRESH_BINARY)

    return check_circularity(mask)


def check_circularity(mask):
    """Given a binary mask find contours and return the centroid of the more circular contour"""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate centers of the contours (green rings)
    max_circularity = 0
    if not contours:
        return []

    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 50:
            continue
        circularity = (4 * np.pi * area) / (perimeter**2)
        if circularity > max_circularity:
            best_contour = contour
            max_circularity = circularity

    if best_contour is None:
        return []

    # Compute the moments of the contour
    M = cv2.moments(best_contour)
    if M["m00"] != 0:
        # Calculate the center of the contour
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        return [(cX, cY)]
    else:
        return []
