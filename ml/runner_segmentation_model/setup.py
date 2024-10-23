from setuptools import setup, find_packages


package_name = "runner_segmentation"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "setuptools",
        "numpy",
        "Pillow",
        "ultralytics==8.3.2",
        "tensorrt==10.4.0",
        "natsort",
        "opencv-python",
        "albumentations",
        "torchmetrics",
        "tqdm",
    ],
    zip_safe=True,
    maintainer="genki",
    maintainer_email="kondo.genki@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=[],
    entry_points={},
)
