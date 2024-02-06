from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Union
from natsort import natsorted

import uvicorn
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pyrealsense2 as rs
import numpy as np
import cv2
import base64
import asyncio
import queue
import threading


DEFAULT_SAVE_DIR = os.path.expanduser("~/Pictures/runners")
DEFAULT_FILE_PREFIX = "runner_"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ManualCaptureRequest(BaseModel):
    saveDir: Union[str, None] = None


class IntervalCaptureRequest(BaseModel):
    intervalSecs: float = 0.0
    saveDir: Union[str, None] = None


class OverlapCaptureRequest(BaseModel):
    overlap: float = 0.0
    saveDir: Union[str, None] = None


class FileManager:
    def save_frame(self, frame, directory=DEFAULT_SAVE_DIR, prefix=DEFAULT_FILE_PREFIX):
        directory = os.path.expanduser(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(
            directory, self._get_next_filename_with_prefix(directory, prefix)
        )
        cv2.imwrite(file_path, frame)
        return file_path

    def _get_next_filename_with_prefix(self, directory, prefix):
        last_file = self._find_last_filename_with_prefix(directory, prefix)
        next_id = (
            0
            if last_file is None
            else self._get_integer_in_filename(last_file, prefix) + 1
        )
        return f"{prefix}{next_id}.png"

    def _find_last_filename_with_prefix(self, directory, prefix):
        # TODO: cache files
        files = [
            f
            for f in os.listdir(directory)
            if f.startswith(prefix) and os.path.isfile(os.path.join(directory, f))
        ]

        if not files:
            return None

        sorted_files = natsorted(files)
        last_file = sorted_files[-1]
        return last_file

    def _get_integer_in_filename(self, filename, prefix):
        # Remove prefix
        filename = filename[len(prefix) :] if filename.startswith(prefix) else filename
        # Remove extension
        root, _ = os.path.splitext(filename)
        try:
            return int(root)
        except ValueError:
            return -1


class RealSense:
    def __init__(self, frame_size=(1280, 720), fps=6, camera_index=0):
        # Note: when connected via USB2, RealSense cameras are limited to:
        # - 1280x720 @ 6fps
        # - 640x480 @ 30fps
        # - 480x270 @ 60fps
        # See https://www.intelrealsense.com/usb2-support-for-intel-realsense-technology/

        self.frame_size = frame_size
        self.camera_index = camera_index
        self.fps = fps

    def initialize(self):
        # Setup code based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        self.config = rs.config()

        # Connect to specific camera
        context = rs.context()
        devices = context.query_devices()
        if self.camera_index < 0 or self.camera_index >= len(devices):
            raise Exception("camera_index is out of bounds")

        serial_number = devices[self.camera_index].get_info(
            rs.camera_info.serial_number
        )
        self.config.enable_device(serial_number)

        # Configure stream
        self.config.enable_stream(
            rs.stream.color,
            self.frame_size[0],
            self.frame_size[1],
            rs.format.rgb8,
            self.fps,
        )

        # Start pipeline
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        if not frames:
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        return color_frame


class LogQueue:
    def __init__(self, max_size=5):
        self.queue = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()

    def enqueue(self, item):
        with self.lock:
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self.dequeue()
                self.queue.put_nowait(item)

    def dequeue(self):
        with self.lock:
            try:
                return self.queue.get_nowait()
            except queue.Empty:
                return None

    def is_empty(self):
        with self.lock:
            return self.queue.empty()


file_manager = FileManager()
camera = RealSense()
camera.initialize()
log_queue = LogQueue()


def rs_to_cv_frame(rs_frame):
    frame = np.asanyarray(rs_frame.get_data())
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


async def send_frame(websocket: WebSocket):
    while True:
        frame = camera.get_frame()
        if frame:
            frame = rs_to_cv_frame(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            img_str = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(img_str)
        await asyncio.sleep(0.1)


async def send_log(websocket: WebSocket):
    while True:
        msg = log_queue.dequeue()
        if msg is not None:
            await websocket.send_text(msg)
        await asyncio.sleep(0.1)


async def interval_capture_task(
    interval_secs: float, save_dir: str, stop_event: asyncio.Event
):
    while not stop_event.is_set():
        await asyncio.sleep(interval_secs)
        frame = camera.get_frame()
        if frame:
            file_path = file_manager.save_frame(rs_to_cv_frame(frame), save_dir)
            log_queue.enqueue(f"Frame captured: {file_path}")


async def overlap_capture_task(
    overlap: float, save_dir: str, stop_event: asyncio.Event
):
    last_saved_frame = None
    while not stop_event.is_set():
        await asyncio.sleep(1.0 / 10)
        current_frame = camera.get_frame()
        if current_frame:
            current_frame = rs_to_cv_frame(current_frame)
            if last_saved_frame is None:
                file_path = file_manager.save_frame(current_frame, save_dir)
                log_queue.enqueue(f"Frame captured: {file_path}")
                last_saved_frame = current_frame
            else:
                # Calculate overlap with last saved frame
                overlap = calculate_overlap(last_saved_frame, current_frame)
                print(overlap)


def calculate_overlap(frame1, frame2):
    # TODO: can we use visual odometry on RealSense?

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Select the top N matches (you can adjust N based on your needs)
    N = min(len(matches), 50)
    selected_matches = matches[:N]

    # Extract corresponding keypoints
    src_points = np.float32(
        [keypoints1[m.queryIdx].pt for m in selected_matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [keypoints2[m.trainIdx].pt for m in selected_matches]
    ).reshape(-1, 1, 2)

    # Find the perspective transformation matrix
    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Get the dimensions of the input frames
    h, w = gray1.shape

    # Define the corners of the first frame
    corners1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
        -1, 1, 2
    )

    # Transform the corners using the perspective transformation matrix
    corners2 = cv2.perspectiveTransform(corners1, M)

    # Calculate the overlap percentage based on the transformed corners
    overlap_percentage = cv2.contourArea(corners2) / cv2.contourArea(corners1) * 100.0

    return overlap_percentage


@app.websocket("/camera_preview")
async def camera_preview(websocket: WebSocket):
    await websocket.accept()
    try:
        await send_frame(websocket)
    except asyncio.CancelledError:
        pass
    finally:
        await websocket.close()


@app.websocket("/log")
async def log(websocket: WebSocket):
    await websocket.accept()
    try:
        await send_log(websocket)
    except asyncio.CancelledError:
        pass
    finally:
        await websocket.close()


@app.post("/capture/manual")
async def manual_capture(request: ManualCaptureRequest) -> JSONResponse:
    frame = camera.get_frame()
    if frame:
        file_path = file_manager.save_frame(rs_to_cv_frame(frame), request.saveDir)
        log_queue.enqueue(f"Frame captured: {file_path}")
        return JSONResponse(content={"file": file_path}, status_code=200)
    else:
        log_queue.enqueue(f"Error: could not get frame")
        return JSONResponse(content={"error": "Could not get frame"}, status_code=500)


@app.post("/capture/interval")
async def interval_capture(request: IntervalCaptureRequest) -> JSONResponse:
    stop_event = asyncio.Event()

    async def interval_capture_task_wrapper():
        task = asyncio.create_task(
            interval_capture_task(request.intervalSecs, request.saveDir, stop_event)
        )
        camera.capture_task = task  # Store the task reference
        await task  # Wait for the task to complete or be canceled
        camera.capture_task = None  # Reset the task reference

    asyncio.create_task(interval_capture_task_wrapper())
    log_queue.enqueue(f"Interval capture started")
    return JSONResponse(
        content={
            "message": f"Interval capture scheduled for every {request.intervalSecs} seconds",
            "stop_url": f"/capture/stop",
        },
        status_code=200,
    )


@app.post("/capture/overlap")
async def overlap_capture(request: OverlapCaptureRequest) -> JSONResponse:
    stop_event = asyncio.Event()

    async def overlap_capture_task_wrapper():
        task = asyncio.create_task(
            overlap_capture_task(request.overlap, request.saveDir, stop_event)
        )
        camera.capture_task = task  # Store the task reference
        await task  # Wait for the task to complete or be canceled
        camera.capture_task = None  # Reset the task reference

    asyncio.create_task(overlap_capture_task_wrapper())
    log_queue.enqueue(f"Overlap capture started")
    return JSONResponse(
        content={
            "message": f"Overlap capture scheduled at {request.overlap}%",
            "stop_url": f"/capture/stop",
        },
        status_code=200,
    )


@app.get("/capture/stop")
async def stop_capture() -> JSONResponse:
    if camera.capture_task:
        camera.capture_task.cancel()
        message = "Capture task stopped"
        log_queue.enqueue(f"Capture stopped")
    else:
        message = "No capture tasks to stop"

    return JSONResponse(
        content={
            "message": message,
        },
        status_code=200,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8042, help="port to run the server")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    # Only serve the react app in debug mode
    if not args.debug:
        react_build_directory = Path(__file__).parent / "ts" / "dist"

        app.mount(
            "/",
            StaticFiles(directory=str(react_build_directory.resolve()), html=True),
        )

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
