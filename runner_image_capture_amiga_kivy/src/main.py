from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Literal

os.environ["KIVY_NO_ARGS"] = "1"

from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

from kivy.app import App  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402


# TODO: save/load from config file
DEFAULT_SAVE_DIR = "~/Pictures/runners"
DEFAULT_FILE_PREFIX = "runner_"
DEFAULT_EXPOSURE_MS = 0.2
DEFAULT_INTERVAL_S = 5


logger = logging.getLogger("amiga.apps.runnerimagecapture")


class RunnerImageCaptureApp(App):

    STREAM_NAMES = ["rgb", "disparity", "left", "right"]

    def __init__(self) -> None:
        super().__init__()
        self.async_tasks: list[asyncio.Task] = []

    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        return Builder.load_file("res/main.kv")

    def on_start(self) -> None:
        self.root.ids.save_dir.text = DEFAULT_SAVE_DIR
        self.root.ids.file_prefix.text = DEFAULT_FILE_PREFIX
        self.root.ids.exposure_ms.text = str(DEFAULT_EXPOSURE_MS)
        self.root.ids.interval_s.text = str(DEFAULT_INTERVAL_S)

    def on_manual_capture_click(self) -> None:
        print("MANUAL")

    def on_interval_capture_click(self) -> None:
        print("INTERVAL")

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
            for task in self.async_tasks:
                task.cancel()

        self.tasks: list[asyncio.Task] = [
            asyncio.create_task(self.stream_camera(view_name))
            for view_name in self.STREAM_NAMES
        ]

        return await asyncio.gather(run_wrapper(), *self.tasks)

    async def stream_camera(
        self,
        view_name: Literal["rgb", "disparity", "left", "right"] = "rgb",
    ) -> None:
        while self.root is None:
            await asyncio.sleep(0.01)

        while True:
            await asyncio.sleep(1.0)

            print(f"HELLAAO: {view_name}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(RunnerImageCaptureApp().app_func())
    except asyncio.CancelledError:
        pass
    loop.close()
