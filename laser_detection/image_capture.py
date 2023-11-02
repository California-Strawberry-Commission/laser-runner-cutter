"""File: image_capture.py

Description: Script to capture laser images using Helios DAC and RealSense D435
"""

from laser_control.laser_dac import HeliosDAC
import pyrealsense2 as rs
from PIL import Image
import numpy as np
import time
import os

image_outdir = os.path.expanduser("~/test/")

helios_libfile = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../laser_control/include/libHeliosDacAPI.so",
)
helios_dac = HeliosDAC(helios_libfile)
num_helios_dacs = helios_dac.initialize()
helios_dac.connect(0)
helios_dac.set_color(1, 0, 0, 0.1)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

profile = pipeline.start(config)
color_sensor = profile.get_device().first_color_sensor()
# color_sensor.set_option(rs.option.enable_auto_exposure, 1)
color_sensor.set_option(
    rs.option.exposure, 1
)  # D435 has a minimum exposure time of 1us

laser_points = []
for i in range(100, 0, -20):
    scale = i / 100.0
    laser_points += helios_dac.get_bounds(scale)

helios_dac.play()
for index, laser_point in enumerate(laser_points):
    helios_dac.clear_points()
    helios_dac.add_point(laser_point[0], laser_point[1])
    time.sleep(0.5)

    # Capture color camera frame
    frames = pipeline.wait_for_frames()

    if frames:
        color_frame = frames.get_color_frame()
        color_frame_array = np.asanyarray(color_frame.get_data())
        color_image = Image.fromarray(color_frame_array)
        color_image.save(os.path.join(image_outdir, f"laser_{index:04}.jpg"))


pipeline.stop()
helios_dac.stop()
