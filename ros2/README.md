# Laser Runner Cutter ROS2 Nodes

This project uses **Ubuntu 22.04 (Jammy Jellyfish)** and **ROS Humble**.

## Auto-Install

Auto-installation assumes a fresh version of **Ubuntu 22.04 desktop/server** on a dedicated deployment or development PC.

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

Linux systems require udev rules to allow access to USB devices without root privileges. This is already set up as part of the auto-install process above. Make sure that the user account communicating with the DAC is in the _plugdev_ group.

### Using LUCID cameras (Triton and Helios2)

LUCID cameras require the Arena SDK and Arena Python Package, which can be found at https://thinklucid.com/downloads-hub/. This is already set up as part of the auto-install process above.

## Run

### Local development

1.  Run `scripts/run_ros.sh`

### Production device

On a production device, we can set up the machine to start the ROS2 nodes on startup:

1.  Modify `scripts/create_systemd_service.sh` with the correct username.

1.  Run `scripts/create_systemd_service.sh`.

1.  To view the logs, run:

        $ journalctl -f --unit laser-runner-cutter-ros.service

## Native libraries

C/C++ libraries included were compiled for linux-x86_64 and linux-aarch64 from the following sources:

- Helios DAC: https://github.com/Grix/helios_dac
- Ether Dream 4 DAC: https://github.com/genkikondo/ether-dream-sdk

## Troubleshooting

### No space left on device

Make sure you have at least 32GB of available disk space. If installed using an LVM (Ubuntu server's default), expand your root (`/`) partition to at least 32GB.
