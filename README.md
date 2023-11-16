# Laser Runner Removal

Laser Runner Removal is a project by the California Strawberry Commission for automated detection and cutting of strawberry runners (also known as stolons).

## Environment setup

1.  Install [ROS 2](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html). Once installed, run:

        $ source /opt/ros/foxy/setup.zsh

1.  Create ROS workspace

        $ mkdir -p ~/ros2_ws/src
        $ cd ~/ros2_ws

1.  Create Python virtual environment

        $ python3 -m venv venv  --system-site-packages --symlinks
        $ touch venv/COLCON_IGNORE
        $ source venv/bin/activate

1.  Add the following to ~/.zshrc

        source /opt/ros/foxy/setup.zsh
        source ~/ros2_ws/install/local_setup.zsh
        export PYTHONPATH=$PYTHONPATH:~/ros2_ws/venv/lib/python3.8/site-packages

1.  Source the new zshrc

        source ~/.zshrc

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

### (Optional) Video stream folder

1.  If recording the video files locally create a video_stream and debug_video_stream directory
        $ sudo mkdir /opt/video_stream
        $ sudo mkdir /opt/debug_video_stream

## Install

1.  Install LRR

        $ cd ~/ros2_ws/src
        $ git clone https://github.com/CA-rvinson/LaserRunnerRemoval.git

        # Install dependencies
        $ cd LaserRunnerRemoval/laser_runner_removal/laser_runner_removal
        $ pip3 install -r requirements.txt

1.  Install [YASMIN](https://github.com/uleroboticsgroup/yasmin#installation)

1.  Build packages

        $ cd ~/ros2_ws
        $ colcon build
        $ source ~/ros2_ws/install/local_setup.zsh

## Run

    $ ros2 launch laser_runner_removal launch.py

## rqt plugin

A rqt plugin is used for visualization. If the "Laser Runner Removal" plugin does not appear in rqt, you may need to run rqt with `--force-discover`.

        $ rqt --force-discover

To develop the plugin UI, install QT Designer:

        $ pip install pyqt5-tools
        $ pyqt5-tools designer

## Libraries

C/C++ libraries included were compiled for linux-x86_64 from the following sources:

- Helios DAC: https://github.com/Grix/helios_dac
- Ether Dream 4 DAC: https://github.com/genkikondo/ether-dream-sdk
