from setuptools import find_packages, setup
from glob import glob

package_name = "furrow_perceiver"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("lib/" + package_name, glob(f"{package_name}/{package_name}/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ros",
    maintainer_email="45153623+DominicChm@users.noreply.github.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "furrow_perceiver_node = furrow_perceiver.furrow_perceiver_node:main"
        ],
    },
)
