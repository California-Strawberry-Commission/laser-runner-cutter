from setuptools import setup

package_name = 'laser_runner_removal'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bobby',
    maintainer_email='rvinson@calstrawberry.org',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'realsense = laser_runner_removal.realsense:main',
            'main_node = laser_runner_removal.main_node:main',
        ],
    },
)
