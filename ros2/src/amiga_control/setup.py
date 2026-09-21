from glob import glob
import os
from setuptools import find_packages, setup

package_name = "amiga_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include all launch files.
        # Include all config files.
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Genki Kondo",
    maintainer_email="genki@kogentech.com",
    description="Node for controlling the Farm-ng Amiga.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "amiga_control_node = amiga_control.amiga_control_node:main",
        ],
    },
)
