import os
from setuptools import setup, find_packages
from glob import glob


package_name = "ml_model"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[("data_store/weights", glob("data_store/weights/*"))],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bobby",
    maintainer_email="rvinson@calstrawberry.org",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=[],
    entry_points={},
)
