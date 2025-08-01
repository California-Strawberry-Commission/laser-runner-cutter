# Laser Runner Cutter ROS2 Nodes

This project uses **Ubuntu 22.04 (Jammy Jellyfish)** and **ROS Humble**.

## Environment Setup

Installation assumes a fresh version of **Ubuntu 22.04 desktop/server** on a dedicated deployment or development PC.

1.  Install Git LFS

        $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        $ sudo apt-get install git-lfs

1.  Clone this repository into your home directory

        $ cd ~
        $ git clone https://github.com/California-Strawberry-Commission/laser-runner-cutter

1.  Run the install script in `laser-runner-cutter/ros2/scripts`. This will take a while.

        $ cd laser-runner-cutter/ros2/scripts
        $ ./install.sh

1.  Source the environment. If you cloned somewhere other than home, use that directory instead of `~`. Optionally, also add this line to the end of your `.bashrc` to automagically activate the environment on every login (useful for deployed/dev systems)

        $ source ~/laser-runner-cutter/ros2/scripts/setup.sh

### Using Helios DAC on Linux

Linux systems require udev rules to allow access to USB devices without root privileges. This is already set up as part of the auto-install process above. Make sure that the user account communicating with the DAC is in the _plugdev_ group.

### Using LUCID cameras (Triton and Helios2)

LUCID cameras require the Arena SDK and Arena Python Package, which can be found at https://thinklucid.com/downloads-hub/. This is already set up as part of the auto-install process above, in `/opt/ArenaSDK`.

## Development Environment

We use Visual Studio Code, and recommend the following extensions:

- [Robotics Developer Environment](https://marketplace.visualstudio.com/items?itemName=Ranch-Hand-Robotics.rde-pack)
- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
- [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
- [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
- [Tasks](https://marketplace.visualstudio.com/items?itemName=actboy168.tasks)
- [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)
- [Tailwind CSS IntelliSense](https://marketplace.visualstudio.com/items?itemName=bradlc.vscode-tailwindcss)

## Build and Run

### Local development

1.  To build all ROS 2 packages, as well as compile the TensorRT models, run `scripts/build.sh`. The TensorRT models may take some time to compile, but it will only need to be done once.

1.  To run all ROS 2 nodes, run `scripts/run_ros.sh`.

While making code changes locally, it may be convenient to build and run a single node. Note that if you've initially built by running `scripts/build.sh`, you will not need to rebuild Python nodes as they are symlinked. However, changes to C++ nodes will need to be rebuilt. Here is an example of building and running the laser_control node:

    $ colcon build --packages-select laser_control --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    $ ros2 launch laser_control launch.py

### Production deployment

On a production device, we can set up the machine to start the ROS2 nodes on startup:

1.  Modify `scripts/create_systemd_service.sh` with the correct username.

1.  Run `scripts/create_systemd_service.sh`.

1.  To view the logs, run:

        $ journalctl -f --unit laser-runner-cutter-ros.service

In addition, we can set up the machine to enable restarting the ROS2 nodes and rebooting the machine without a password, which allows the LifecycleManager node to trigger the relevant commands. This is useful for allowing other programs (such as a web-based app) to restart the nodes or reboot the machine in case of an irrecoverable issue.

1.  Run `sudo visudo`, then add the following lines:

        <username> ALL=(ALL) NOPASSWD: /bin/systemctl restart laser-runner-cutter-ros.service
        <username> ALL=(ALL) NOPASSWD: /sbin/reboot

## Native libraries

C/C++ libraries included were compiled for linux-x86_64 and linux-aarch64 from the following sources:

- Helios DAC: https://github.com/Grix/helios_dac
- Ether Dream 4 DAC: https://github.com/genkikondo/ether-dream-sdk

## LUCID camera calibration

We currently use the following LUCID cameras:

- Triton 3.2 MP Model ([TRI032S-CC](https://thinklucid.com/product/triton-32-mp-imx265/))
- Helios2 Ray Outdoor IP67 ToF 3D Camera ([HTR003S-001](https://thinklucid.com/product/helios2-ray-outdoor-tof-ip67-3d-camera/))

We need to calculate the intrinsic matrix and distortion coefficients for each camera, as well as the extrinic matrix that describes the rotation and translation between the two cameras. For more details, see https://support.thinklucid.com/app-note-helios-3d-point-cloud-with-rgb-color/

### Intrinsics

For each camera, do the following:

Grab at least 3 images (9 recommended) of the [calibration grid](https://arenasdk.s3-us-west-2.amazonaws.com/LUCID_target_whiteCircles.pdf) in varying locations in the camera's field of view, roughly 3 feet away (note: for the Helios camera, use the intensity channel of the frame to generate a grayscale image). Then, use `calibrate_camera` in [`calibration.py`](camera_control/camera_control/camera/calibration.py), which calculates the intrinsic matrix and distortion coefficients using the method based on https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html

### Extrinsics

Solving for orientation of the Helios relative to the Triton requires a single image of the calibration target from each camera. Place the calibration grid near the center of both cameras field of view at roughly 3 feet away. Make sure not to move the calibration target or cameras in between grabbing the Helios image and grabbing the Triton image, and make sure to grab both the xyz and intensity image from the Helios.

Then, use `get_extrinsics` in [`lucid_camera.py`](camera_control/camera_control/camera/lucid_camera.py) to get the rotation matrix and translation vector.

## Troubleshooting

### No space left on device

Make sure you have at least 32GB of available disk space. If installed using an LVM (Ubuntu server's default), expand your root (`/`) partition to at least 32GB.
