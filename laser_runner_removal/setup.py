import os
from glob import glob
from setuptools import setup, find_packages

package_name = "laser_runner_removal"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "include"),
            glob("include/**/*", recursive=True),
        ),
        # Include all launch files.
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        # Include all config files.
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bobby",
    maintainer_email="rvinson@calstrawberry.org",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "realsense = laser_runner_removal.realsense:main",
            "main_node = laser_runner_removal.main_node:main",
            "test_node = laser_runner_removal.nodes.test_node:main",
        ],
    },
)
