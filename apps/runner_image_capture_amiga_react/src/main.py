from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Union
from natsort import natsorted
import numpy as np
import cv2
import base64
import asyncio
import queue
import threading
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))

from libs.camera.realsense import RealSense
from libs.file_manager.file_manager import FileManager


DEFAULT_IMAGE_DIR = os.path.expanduser("~/Pictures/runners")
DEFAULT_IMAGE_FILE_PREFIX = "runner_"


class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


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


class CameraExposureRequest(BaseModel):
    exposureMs: float = 0.2


class ManualCaptureRequest(BaseModel):
    saveDir: Union[str, None] = None
    filePrefix: Union[str, None] = None


class IntervalCaptureRequest(BaseModel):
    intervalSecs: float = 1.0
    saveDir: Union[str, None] = None
    filePrefix: Union[str, None] = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
connection_manager = ConnectionManager()
file_manager = FileManager()
camera = RealSense()
camera.initialize()
camera.set_exposure(0.2)
log_queue = LogQueue()


def rs_to_cv_frame(rs_frame):
    frame = np.asanyarray(rs_frame.get_data())
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


async def send_frame(websocket: WebSocket):
    try:
        while True:
            frame = camera.get_frame()
            if frame:
                frame = rs_to_cv_frame(frame)
                _, buffer = cv2.imencode(".jpg", frame)
                img_str = base64.b64encode(buffer).decode("utf-8")
                await websocket.send_text(img_str)
            await asyncio.sleep(1.0 / camera.fps)
    except Exception as e:
        print(f"Error in send_frame task: {e}")
    finally:
        connection_manager.disconnect(websocket)


async def send_log(websocket: WebSocket):
    try:
        websocket_connected = True
        while websocket_connected:
            msg = log_queue.dequeue()
            if msg is not None:
                await websocket.send_text(msg)
            else:
                try:
                    # Check if connection is active
                    await asyncio.wait_for(websocket.receive_bytes(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Connection is still alive
                    websocket_connected = True
                except WebSocketDisconnect:
                    # Connection closed by client
                    websocket_connected = False
                else:
                    # Received some data from the client, ignore it
                    websocket_connected = True
                await asyncio.sleep(1.0 / 10)
    except Exception as e:
        print(f"Error in send_log task: {e}")
    finally:
        connection_manager.disconnect(websocket)


async def interval_capture_task(
    interval_secs: float, save_dir: str, file_prefix: str, stop_event: asyncio.Event
):
    while not stop_event.is_set():
        frame = camera.get_frame()
        if frame:
            file_path = file_manager.save_frame(
                rs_to_cv_frame(frame), save_dir, file_prefix
            )
            log_queue.enqueue(f"Frame captured: {file_path}")
        await asyncio.sleep(interval_secs)


@app.websocket("/camera/preview")
async def camera_preview(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        await send_frame(websocket)
    except asyncio.CancelledError:
        pass
    finally:
        connection_manager.disconnect(websocket)


@app.post("/camera/exposure")
async def camera_exposure(request: CameraExposureRequest) -> JSONResponse:
    camera.set_exposure(request.exposureMs)
    return JSONResponse(content={"message": "Success"}, status_code=200)


@app.websocket("/log")
async def log(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        await send_log(websocket)
    except asyncio.CancelledError:
        pass
    finally:
        connection_manager.disconnect(websocket)


@app.post("/capture/manual")
async def manual_capture(request: ManualCaptureRequest) -> JSONResponse:
    frame = camera.get_frame()
    if frame:
        file_path = file_manager.save_frame(
            rs_to_cv_frame(frame), request.saveDir, request.filePrefix
        )
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
            interval_capture_task(
                request.intervalSecs, request.saveDir, request.filePrefix, stop_event
            )
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
