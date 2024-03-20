# Laser Runner Cutter ROS2 Nodes
This project uses **Ubuntu 20.04 (Focal Fossa)**! Newer versions of ROS are not currently supported!

## Auto-Install 
Auto-Installation assumes a fresh version of Ubuntu 20.04 on a dedicated deployment or development machine. 

1. Clone this repository into your home directory
        
        $ git clone https://github.com/California-Strawberry-Commission/laser-runner-cutter

2. Run the install script in `laser-runner-cutter/ros2`
        
        $ cd laser-runner-cutter/ros2
        $ bash install.sh

3. Set up your environment. If clone elsewhere, use that directory. Optionally, add this line to the end of your `.bashrc` to automagically activate the environment on every login (useful for deployed systems)

        $ source ~/laser-runner-cutter/setup.bash


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


This will install [ROS 2 - Foxy](https://docs.ros.org/en/foxy/index.html), [realsense-ros](https://github.com/IntelRealSense/realsense-ros), and all python dependencies. This guide assumes a clean install of Ubuntu 20.04. If you already have ROS installed, you can manually install components using the scripts
```sh
install_realsense_ros.sh
install_requirements.sh
install_ros.sh
```

## Troubleshooting
### No space left on device
Make sure you have at least 32GB of available disk space. If installed using an LVM (Ubuntu server's default), expand your root (`/`) partition to at least 32GB.