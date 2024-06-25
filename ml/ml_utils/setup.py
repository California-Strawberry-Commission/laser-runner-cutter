from setuptools import setup, find_packages
import os

package_name = "ml_utils"


def get_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as f:
            requirements = f.read().splitlines()
    requirements.append("setuptools")
    return requirements


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements(),
    zip_safe=True,
    maintainer="genki",
    maintainer_email="kondo.genki@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=[],
    entry_points={},
)
