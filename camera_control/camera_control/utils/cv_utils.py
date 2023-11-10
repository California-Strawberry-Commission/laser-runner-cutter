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


def detect_lasers(model, frames):
    point_list = []
    image = np.asanyarray(frames["color"].get_data())
    res = model(image)
    if len(res) <= 0 or not res[0].boxes or len(res[0].boxes.xywh) <= 0:
        return point_list

    for box in res[0].boxes.xywh:
        box_np = box.numpy().astype(float)
        point_list.append((box_np[0], box_np[1]))

    return point_list
