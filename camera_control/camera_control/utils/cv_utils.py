import cv2


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
