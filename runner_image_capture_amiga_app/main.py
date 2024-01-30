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


class FileManager:
    def save_frame(self, frame, directory=DEFAULT_SAVE_DIR, prefix=DEFAULT_FILE_PREFIX):
        directory = os.path.expanduser(directory)
        file_path = os.path.join(
            directory, self._get_next_filename_with_prefix(directory, prefix)
        )
        cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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
    def __init__(self, frame_size=(1920, 1080), fps=30, camera_index=0):
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


file_manager = FileManager()
camera = RealSense()
camera.initialize()


async def send_frame(websocket: WebSocket):
    while True:
        frame = camera.get_frame()
        if frame:
            frame_data = np.asanyarray(frame.get_data())
            _, buffer = cv2.imencode(".jpg", frame_data)
            img_str = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(img_str)
        await asyncio.sleep(0.1)


class ManualCaptureRequest(BaseModel):
    saveDir: Union[str, None] = None


@app.post("/capture/manual")
async def manual_capture(request: ManualCaptureRequest) -> JSONResponse:
    frame = camera.get_frame()
    file_path = file_manager.save_frame(
        np.asanyarray(frame.get_data()), request.saveDir
    )
    return JSONResponse(content={"file": file_path}, status_code=200)


@app.websocket("/camera_preview")
async def camera_preview(websocket: WebSocket):
    await websocket.accept()
    try:
        await send_frame(websocket)
    except asyncio.CancelledError:
        pass
    finally:
        await websocket.close()


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
