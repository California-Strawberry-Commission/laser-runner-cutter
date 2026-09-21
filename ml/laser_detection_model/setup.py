from setuptools import setup, find_packages


package_name = "laser_detection"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Genki Kondo",
    maintainer_email="genki@kogentech.com",
    description="ML pipeline and models for object detection of lasers.",
    license="MIT",
    tests_require=[],
    entry_points={},
)
