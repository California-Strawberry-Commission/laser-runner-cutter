import os
from setuptools import setup, find_packages
from glob import glob

package_name = "laser_control"

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
            "laser_control_node = laser_control.nodes.laser_control_node:main"
        ],
    },
)
