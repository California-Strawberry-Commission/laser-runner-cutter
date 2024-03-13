# Laser Runner Cutter ROS2 Nodes

## Setup

1.  Install [ROS 2](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html). Once installed, run:

        $ source /opt/ros/foxy/setup.zsh

1.  Create ROS workspace

        $ mkdir -p ~/ros2_ws/src
        $ cd ~/ros2_ws

1.  Create Python virtual environment

        $ python3 -m venv venv  --system-site-packages --symlinks
        $ touch venv/COLCON_IGNORE
        $ source venv/bin/activate

1.  If using VS Code, install the ROS extension for VS Code. Then, add the following to `.vscode/settings.json` in your project directory:

        "~/ros2_ws/venv/lib/python3.8/site-packages"

### Using Helios DAC on Linux

Linux systems require udev rules to allow access to USB devices without root privileges.

1.  Create a file _heliosdac.rules_ in /etc/udev with the contents:

        ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="e500", MODE="0660", GROUP="plugdev"

1.  Create a link in /etc/udev/rules.d to _heliosdac.rules_:

        cd /etc/udev/rules.d
        sudo ln -s /etc/udev/heliosdac.rules 011_heliosdac.rules

1.  Make sure the user account communicating with the DAC is in the _plugdev_ group. On a Raspberry Pi, the "pi" user is in the _plugdev_ group by default.

1.  Issue the command `sudo udevadm control --reload` (or restart the computer).

## Install

1.  Clone project and install dependencies

        $ cd ~/ros2_ws/src
        $ git clone https://github.com/California-Strawberry-Commission/laser-runner-cutter

        # Install dependencies
        $ laser-runner-cutter/ros2/install_requirements.sh

1.  Build packages

        $ cd ~/ros2_ws
        $ colcon build

## Run

1.  Launch ROS2 nodes using the launch file

        $ source laser-runner-cutter/ros2/setup.sh  # or setup.zsh in Z shell
        $ ros2 launch runner_cutter_control launch.py

## Native libraries

C/C++ libraries included were compiled for linux-x86_64 from the following sources:

- Helios DAC: https://github.com/Grix/helios_dac
- Ether Dream 4 DAC: https://github.com/genkikondo/ether-dream-sdk
