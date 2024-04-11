from setuptools import find_packages, setup

package_name = 'amiga_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='45153623+DominicChm@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'amiga_control_node = amiga_control.amiga_control_node:main',
            'amiga_control_client = amiga_control.amiga_control_client:main',
            'test_cir = amiga_control.circular_node:main',
        ],
    },
)
