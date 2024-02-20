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

from kivymd.app import MDApp
from kivy.graphics.texture import Texture  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402

from config_manager.config_manager import ConfigManager
from file_manager.file_manager import FileManager
from camera.realsense import RealSense
from metadata_provider.metadata_provider import MetadataProvider
from amiga_client.amiga_client import AmigaClient


CONFIG_FILE = "~/.config/runner-image-capture/config.json"
CONFIG_KEY_SAVE_DIR = "saveDir"
CONFIG_KEY_FILE_PREFIX = "filePrefix"
CONFIG_KEY_EXPOSURE_MS = "exposureMs"
CONFIG_KEY_INTERVAL_S = "intervalS"
CONFIG_KEY_METADATA_SERVICES = "metaServices"
DEFAULT_CONFIG = {
    CONFIG_KEY_SAVE_DIR: "~/Pictures/runners",
    CONFIG_KEY_FILE_PREFIX: "runner_",
    CONFIG_KEY_EXPOSURE_MS: 0.2,
    CONFIG_KEY_INTERVAL_S: 5,
    CONFIG_KEY_METADATA_SERVICES: {
        "configs": [
            {
                "name": "gps",
                "port": 3001,
                "host": "localhost",
                "log_level": "INFO",
                "subscriptions": [
                    {
                        "uri": {
                            "path": "/pvt",
                            "query": "service_name=gps"
                        },
                        "every_n": 1
                    },
                ]
            },
            {
                "name": "filter",
                "port": 20001,
                "host": "localhost",
                "log_level": "INFO",
                "subscriptions": [
                    {
                        "uri": {
                            "path": "/state",
                            "query": "service_name=filter"
                        },
                        "every_n": 1
                    }
                ]
            }
        ]
    }
}


class RunnerImageCaptureApp(MDApp):
    def __init__(self) -> None:
        super().__init__()
        self.interval_capture_task: asyncio.Task = None
        self.logger = logging.getLogger("amiga.apps.runnerimagecapture")

    def build(self):
        self.config_manager = ConfigManager(CONFIG_FILE, DEFAULT_CONFIG)
        self.amiga_client = AmigaClient(self.config_manager.get(CONFIG_KEY_METADATA_SERVICES))
        self.metadata_provider = MetadataProvider(self.amiga_client, self.logger)
        self.file_manager = FileManager()
        self.log_queue = deque(maxlen=100)
        self.camera = RealSense()
        self.camera.initialize()
        self.camera.set_exposure(self.config_manager.get(CONFIG_KEY_EXPOSURE_MS))

        # KivyMD theme
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"

        return Builder.load_file("res/main.kv")

    def log(self, str) -> None:
        self.logger.info(str)

        # Get the current timestamp
        current_time = datetime.now()
        # Format the timestamp
        str = f"{current_time.strftime('[%H:%M:%S]')} {str}"
        self.log_queue.append(str)
        self.root.ids["logs"].text = "\n".join(reversed(self.log_queue))

    def on_start(self) -> None:
        # Set initial values on text fields
        self.root.ids["save_dir"].text = self.config_manager.get(CONFIG_KEY_SAVE_DIR)
        self.root.ids["file_prefix"].text = self.config_manager.get(
            CONFIG_KEY_FILE_PREFIX
        )
        self.root.ids["exposure_ms"].text = str(
            self.config_manager.get(CONFIG_KEY_EXPOSURE_MS)
        )
        self.root.ids["interval_s"].text = str(
            self.config_manager.get(CONFIG_KEY_INTERVAL_S)
        )
        
        self.amiga_client.init_clients()

    def on_stop(self) -> None:
        # Save config
        self.config_manager.set(CONFIG_KEY_SAVE_DIR, self.root.ids["save_dir"].text)
        self.config_manager.set(
            CONFIG_KEY_FILE_PREFIX, self.root.ids["file_prefix"].text
        )
        self.config_manager.set(
            CONFIG_KEY_EXPOSURE_MS, float(self.root.ids["exposure_ms"].text)
        )
        self.config_manager.set(
            CONFIG_KEY_INTERVAL_S, float(self.root.ids["interval_s"].text)
        )
        self.config_manager.write_config()

    def capture_frame(self) -> None:
        frame = self.camera.get_frame()
        if frame:
            save_dir = self.root.ids["save_dir"].text
            file_prefix = self.root.ids["file_prefix"].text
            file_path = self.file_manager.save_frame(
                self._rs_to_cv_frame(frame), save_dir, file_prefix
            )
            self.metadata_provider.add_exif(file_path)
            self.log(f"Frame captured: {file_path}")

    def on_exposure_apply_click(self) -> None:
        exposure_ms = float(self.root.ids["exposure_ms"].text)
        self.camera.set_exposure(exposure_ms)
        self.log(f"Exposure set to {exposure_ms}ms")

    def on_manual_capture_click(self) -> None:
        self.log(f"Manual capture")
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

    def on_quit_click(self) -> None:
        """Kills the running kivy application."""
        self.log(f"Quit button clicked")
        self.amiga_client.cleanup()
        for task in self.tasks:
            task.cancel()
        MDApp.get_running_app().stop()

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
