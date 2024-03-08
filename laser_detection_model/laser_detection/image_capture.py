"""File: image_capture.py

Description: Script to capture laser images using Helios DAC and RealSense D435
"""

import argparse
from laser_control.laser_dac import HeliosDAC
import pyrealsense2 as rs
from PIL import Image
import numpy as np
import time
import os


def main(image_outdir, file_prefix="laser", file_start_index=0):
    if not os.path.exists(image_outdir):
        os.makedirs(image_outdir)

    # Initialize Helios laser DAC
    helios_libfile = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../laser_control/include/libHeliosDacAPI.so",
    )
    helios_dac = HeliosDAC(helios_libfile)
    num_helios_dacs = helios_dac.initialize()
    if num_helios_dacs <= 0:
        print("No Helios DAC detected")
        return

    helios_dac.connect(0)
    helios_dac.set_color(1, 0, 0, 0.1)

    # Initialize RealSense camera
    context = rs.context()
    devices = context.query_devices()
    if len(devices) <= 0:
        print("No RealSense camera detected")
        return

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
        time.sleep(0.2)

        # Capture color camera frame
        frames = pipeline.wait_for_frames()

        if frames:
            color_frame = frames.get_color_frame()
            color_frame_array = np.asanyarray(color_frame.get_data())
            color_image = Image.fromarray(color_frame_array)
            color_image.save(
                os.path.join(
                    image_outdir, f"{file_prefix}_{(file_start_index+index):04}.jpg"
                )
            )

    pipeline.stop()
    helios_dac.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture laser images using Helios DAC and RealSense D435"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/laser_images/"),
        help="Path to the directory where images will be written to",
    )
    parser.add_argument(
        "--file_prefix",
        default="laser",
    )
    parser.add_argument(
        "--file_start_index",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    main(args.output_dir, args.file_prefix, args.file_start_index)
