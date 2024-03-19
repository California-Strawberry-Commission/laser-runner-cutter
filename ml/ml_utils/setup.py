from setuptools import setup, find_packages


package_name = "ml_utils"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="genki",
    maintainer_email="kondo.genki@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=[],
    entry_points={},
)
