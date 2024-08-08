# Laser Runner Cutter ROS2 Nodes

This project uses **Ubuntu 20.04 (Focal Fossa)**! Newer versions of ROS are not currently supported!

## Auto-Install

Auto-Installation assumes a fresh version of **Ubuntu 20.04 desktop/server** on a dedicated deployment or development PC.

1.  Install Git LFS

        $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        $ sudo apt-get install git-lfs

1.  Clone this repository into your home directory

        $ cd ~
        $ git clone https://github.com/California-Strawberry-Commission/laser-runner-cutter

1.  Run the install script in `laser-runner-cutter/ros2/scripts`. This will take a while.

        $ cd laser-runner-cutter/ros2/scripts
        $ ./install.sh

1.  Set up your environment. If you cloned somewhere other than home, use that directory instead of `~`. Optionally, also add this line to the end of your `.bashrc` to automagically activate the environment on every login (useful for deployed/dev systems)

        $ source ~/laser-runner-cutter/ros2/scripts/setup.sh

### Using Helios DAC on Linux

Linux systems require udev rules to allow access to USB devices without root privileges.

1.  Create a file _heliosdac.rules_ in /etc/udev with the contents:

        ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="e500", MODE="0660", GROUP="plugdev"

1.  Create a link in /etc/udev/rules.d to _heliosdac.rules_:

        cd /etc/udev/rules.d
        sudo ln -s /etc/udev/heliosdac.rules 011_heliosdac.rules

1.  Make sure the user account communicating with the DAC is in the _plugdev_ group. On a Raspberry Pi, the "pi" user is in the _plugdev_ group by default.

1.  Issue the command `sudo udevadm control --reload` (or restart the computer).

### Using LUCID cameras (Triton and Helios2)

Download and install the Arena SDK and the Arena Python Package which can be found at https://thinklucid.com/downloads-hub/

## Run

### Local development

1.  Launch ROS2 nodes using the launch file

        $ source laser-runner-cutter/ros2/scripts/setup.sh  # or setup.zsh in Z shell
        $ ros2 launch runner_cutter_control launch.py

### Production device

1.  Copy systemd service file

        $ cp scripts/laser-runner-cutter-ros.service /etc/systemd/system/

1.  Edit the newly created `/etc/systemd/system/laser-runner-cutter-ros.service` to contain the correct username

1.  Enable the service to run on startup

        $ sudo systemctl daemon-reload
        $ sudo systemctl enable laser-runner-cutter-ros.service
        $ sudo systemctl start laser-runner-cutter-ros.service

1.  To view the logs, run:

        $ journalctl -f --unit laser-runner-cutter-ros.service

## Native libraries

C/C++ libraries included were compiled for linux-x86_64 and linux-aarch64 from the following sources:

- Helios DAC: https://github.com/Grix/helios_dac
- Ether Dream 4 DAC: https://github.com/genkikondo/ether-dream-sdk

## Troubleshooting

### No space left on device

Make sure you have at least 32GB of available disk space. If installed using an LVM (Ubuntu server's default), expand your root (`/`) partition to at least 32GB.
