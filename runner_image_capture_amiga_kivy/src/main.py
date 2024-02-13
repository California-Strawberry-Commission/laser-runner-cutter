from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal
import cv2
import numpy as np
from datetime import datetime
from collections import deque

os.environ["KIVY_NO_ARGS"] = "1"

from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

from kivy.app import App  # noqa: E402
from kivy.clock import Clock  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.uix.image import Image  # noqa: E402

from file_manager import file_manager
from camera import realsense


# TODO: save/load from config file
DEFAULT_SAVE_DIR = "~/Pictures/runners"
DEFAULT_FILE_PREFIX = "runner_"
DEFAULT_EXPOSURE_MS = 0.2
DEFAULT_INTERVAL_S = 5


logger = logging.getLogger("amiga.apps.runnerimagecapture")


class RunnerImageCaptureApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.interval_capture_task: asyncio.Task = None

    def build(self):
        self.file_manager = file_manager.FileManager()
        self.log_queue = deque(maxlen=100)
        self.camera = realsense.RealSense()
        self.camera.initialize()
        self.camera.set_exposure(DEFAULT_EXPOSURE_MS)

        Window.clearcolor = (1, 1, 1, 1)
        return Builder.load_file("res/main.kv")

    def log(self, str) -> None:
        # Get the current timestamp
        current_time = datetime.now()
        # Format the timestamp
        str = f"{current_time.strftime('[%H:%M:%S]')} {str}"
        self.log_queue.append(str)
        self.root.ids["logs"].text = "\n".join(reversed(self.log_queue))

    def on_start(self) -> None:
        self.root.ids["save_dir"].text = DEFAULT_SAVE_DIR
        self.root.ids["file_prefix"].text = DEFAULT_FILE_PREFIX
        self.root.ids["exposure_ms"].text = str(DEFAULT_EXPOSURE_MS)
        self.root.ids["interval_s"].text = str(DEFAULT_INTERVAL_S)

    def capture_frame(self) -> None:
        frame = self.camera.get_frame()
        if frame:
            save_dir = self.root.ids["save_dir"].text
            file_prefix = self.root.ids["file_prefix"].text
            file_path = self.file_manager.save_frame(
                self._rs_to_cv_frame(frame), save_dir, file_prefix
            )
            self.log(f"Frame captured: {file_path}")

    def on_exposure_apply_click(self) -> None:
        exposure_ms = float(self.root.ids["exposure_ms"].text)
        self.camera.set_exposure(exposure_ms)

    def on_manual_capture_click(self) -> None:
        self.capture_frame()

    def on_interval_capture_click(self) -> None:
        if self.interval_capture_task is None:
            interval_s = float(self.root.ids["interval_s"].text)
            # Start async task
            self.interval_capture_task = asyncio.create_task(
                self.interval_capture(interval_s)
            )
            self.root.ids["interval_capture_button"].text = "Stop interval capture"
            self.log("Started interval capture")
        else:
            self.interval_capture_task.cancel()
            self.interval_capture_task = None
            self.root.ids["interval_capture_button"].text = "Start interval capture"
            self.log("Stopped interval capture")

    def on_back_click(self) -> None:
        """Kills the running kivy application."""
        for task in self.tasks:
            task.cancel()
        App.get_running_app().stop()

    async def app_func(self):
        async def run_wrapper():
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            if self.interval_capture_task is not None:
                self.interval_capture_task.cancel()

        self.tasks: list[asyncio.Task] = [asyncio.create_task(self.stream_camera())]

        return await asyncio.gather(run_wrapper(), *self.tasks)

    async def stream_camera(
        self,
    ) -> None:
        while self.root is None:
            await asyncio.sleep(0.01)

        while True:
            await asyncio.sleep(1.0 / self.camera.fps)

            frame = self.camera.get_frame()
            if frame:
                frame = self._rs_to_cv_frame(frame)
                texture = Texture.create(
                    size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
                )
                texture.flip_vertical()
                texture.blit_buffer(frame.tobytes(), colorfmt="bgr", bufferfmt="ubyte")
                self.root.ids["camera_preview"].texture = texture
                self.root.ids["camera_status"].text = (
                    f"{self.camera.frame_size[0]} x {self.camera.frame_size[1]} @ {self.camera.fps}fps"
                )

    async def interval_capture(self, interval_s) -> None:
        while self.root is None:
            await asyncio.sleep(0.01)

        while True:
            self.capture_frame()
            await asyncio.sleep(interval_s)

    def _rs_to_cv_frame(self, rs_frame):
        frame = np.asanyarray(rs_frame.get_data())
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(RunnerImageCaptureApp().app_func())
    except asyncio.CancelledError:
        pass
    loop.close()
