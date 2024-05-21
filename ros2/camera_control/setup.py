import os
from setuptools import setup, find_packages
from glob import glob

package_name = "camera_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "models"),
            glob(os.path.join("models/**/*.pt"), recursive=True),
        ),
        (
            os.path.join("share", package_name, "calibration_params"),
            glob(os.path.join("calibration_params/**/*.npy"), recursive=True),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="genki",
    maintainer_email="kondo.genki@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_control_node = camera_control.camera_control_node:main",
        ],
    },
)
