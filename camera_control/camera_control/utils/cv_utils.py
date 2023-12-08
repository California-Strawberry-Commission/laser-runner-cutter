import cv2


def draw_laser(debug_frame, laser_list, laser_scores, draw_scores=True):
    for score, laser in zip(laser_scores, laser_list):
        pos = [int(laser[0]), int(laser[1])]
        debug_frame = cv2.drawMarker(
            debug_frame,
            pos,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            thickness=1,
            markerSize=20,
        )
        pos = [int(laser[0]) - 15, int(laser[1]) - 15]
        font = cv2.FONT_HERSHEY_SIMPLEX
        debug_frame = cv2.putText(
            debug_frame, f"{score:.2f}", pos, font, 0.25, (255, 0, 255)
        )
    return debug_frame


def draw_runners(debug_frame, runner_list, runner_scores, draw_scores=True):
    for score, runner in zip(runner_scores, runner_list):
        pos = [int(runner[0]), int(runner[1])]
        debug_frame = cv2.drawMarker(
            debug_frame,
            pos,
            (255, 255, 255),
            cv2.MARKER_STAR,
            thickness=1,
            markerSize=20,
        )
        pos = [int(runner[0]) + 15, int(runner[1]) - 15]
        font = cv2.FONT_HERSHEY_SIMPLEX
        debug_frame = cv2.putText(
            debug_frame, f"{score:.2f}", pos, font, 0.25, (255, 255, 255)
        )
    return debug_frame
