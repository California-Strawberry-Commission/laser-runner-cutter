# Laser Runner Cutter

Laser Runner Cutter is a project by the California Strawberry Commission for automated detection and cutting of strawberry runners (also known as stolons).

## Packages

### ROS2

- `runner_cutter_control`: node for laser runner cutter control and automation.
- `laser_control`: node and client for controlling laser DACs. Currently supports Helios and Ether Dream DACs.
- `laser_control_interfaces`: custom interface definitions used in `laser_control`.
- `camera_control`: node and client for capturing camera frames and running runner and laser detection. Currently supports RealSense cameras.
- `camera_control_interfaces`: custom interface definitions used in `camera_control`.

### ML

- `runner_segmentation_model`: ML pipeline and models for instance segmentation of runners
- `laser_detection_model`: ML pipeline and models for object detection of lasers
- `runner_mask_instancing_model`: ML pipeline and models for instace segmentation of semantic segmented runner masks. Goal: given a single binary mask that represents runners, segment it into separate instances of runners.

### Apps

- `runner_image_capture_amiga_kivy`: An application deployable to the Amiga brain, written using Kivy, for capturing runner images in the field.
- `runner_image_capture_amiga_react`: An application deployable to the Amiga brain, written using React and FastAPI, for capturing runner images in the field.

### Viz

- `rqt_laser_runner_cutter`: rqt plugin for visualization and monitoring of relevant ROS2 topics.
- `simulator`: Unity-based simulator.

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

1.  Install Laser Runner Cutter

        $ cd ~/ros2_ws/src
        $ git clone https://github.com/California-Strawberry-Commission/laser-runner-cutter

        # Install dependencies
        $ cd laser-runner-cutter/runner_cutter_control/runner_cutter_control
        $ pip3 install -r requirements.txt

1.  Install [YASMIN](https://github.com/uleroboticsgroup/yasmin#installation)

1.  Build packages

        $ cd ~/ros2_ws
        $ colcon build
        $ source ~/ros2_ws/install/local_setup.zsh

## Run

    $ ros2 launch runner_cutter_control launch.py

## rqt plugin

A rqt plugin is used for visualization. Run `rqt`, under Plugins you should see Laser Runner Cutter. If the plugin does not appear in rqt, you may need to run `rqt --force-discover`.

To develop the plugin UI, install QT Designer:

        $ pip install pyqt5-tools
        $ pyqt5-tools designer

Then, open the UI file `rqt_laser_runner_cutter.ui`.

## Libraries

C/C++ libraries included were compiled for linux-x86_64 from the following sources:

- Helios DAC: https://github.com/Grix/helios_dac
- Ether Dream 4 DAC: https://github.com/genkikondo/ether-dream-sdk
