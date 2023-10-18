# Laser Runner Removal

Laser Runner Removal is a project by the California Strawberry Commission for automated detection and cutting of strawberry runners (also known as stolons).

## Environment setup

1.  Install [ROS 2](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html). Once installed, run:

        $ source /opt/ros/foxy/setup.zsh

2.  Create ROS workspace

        $ mkdir -p ~/ros2_ws/src
        $ cd ~/ros2_ws

3.  Create Python virtual environment

        $ python3 -m venv venv  --system-site-packages --symlinks
        $ touch venv/COLCON_IGNORE
        $ source venv/bin/activate

4.  Add the following to ~/.zshrc

        source /opt/ros/foxy/setup.zsh
        source ~/ros2_ws/install/local_setup.zsh
        export PYTHONPATH=$PYTHONPATH:~/ros2_ws/venv/lib/python3.8/site-packages

5.  Source the new zshrc

        source ~/.zshrc

6.  If using VS Code, install the ROS extension for VS Code. Then, add the following to `.vscode/settings.json` in your project directory:

        "~/ros2_ws/venv/lib/python3.8/site-packages"

## Install LRR

1.  Install LRR

        $ cd ~/ros2_ws/src
        $ git clone https://github.com/CA-rvinson/LaserRunnerRemoval.git

        # Install dependencies
        $ cd LaserRunnerRemoval/laser_runner_removal/laser_runner_removal
        $ pip3 install -r requirements.txt

2.  Install [YASMIN](https://github.com/uleroboticsgroup/yasmin#installation)

3.  Build packages

        $ cd ~/ros2_ws
        $ colcon build
        $ source ~/ros2_ws/install/local_setup.zsh

## Run LRR

    $ ros2 launch laser_runner_removal launch.py
