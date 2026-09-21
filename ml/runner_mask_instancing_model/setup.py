from setuptools import setup, find_packages


package_name = "runner_mask_instancing"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Genki Kondo",
    maintainer_email="genki@kogentech.com",
    description="ML pipeline and models for instance segmentation of semantic segmented runner masks. Given a single binary mask that represents runners, segments it into separate instances of runners.",
    license="MIT",
    tests_require=[],
    entry_points={},
)
