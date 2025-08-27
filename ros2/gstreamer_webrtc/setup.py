import os
from glob import glob

from setuptools import find_packages, setup

package_name = "gstreamer_webrtc"


def get_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as f:
            requirements = f.read().splitlines()
    return requirements


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include all launch files.
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=get_requirements(),
    zip_safe=True,
    maintainer="brendan",
    maintainer_email="bholt@calstrawberry.org",
    description="Low(er) latency streaming implemented with GStreamer and WebRTC",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gstreamer_node = gstreamer_webrtc.gstreamer_node:main",
        ],
    },
)
