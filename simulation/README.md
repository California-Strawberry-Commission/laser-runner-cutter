# Laser Runner Cutter Unity Simulator

Unity-based simulator for the Laser Runner Cutter. ROS2 communication from Unity is done via [ros2-for-unity](https://github.com/RobotecAI/ros2-for-unity).

## Updating generated custom msgs

The following instructions are based on [this guide](https://github.com/RobotecAI/ros2-for-unity/blob/develop/README-UBUNTU.md).

1. Install dependencies (note: we are using **ROS2 Galactic** as ros2-for-unity no longer supports Foxy. This is fine as Unity will run standalone, and communication happens over the same protocol across ROS2 versions):

   ```
   # Install ROS2 Galactic if you haven't already
   sudo apt install ros-galactic-desktop

   # Install rmw and tests-msgs for your ROS2 distribution
   apt install -y ros-galactic-test-msgs
   apt install -y ros-galactic-fastrtps ros-galactic-rmw-fastrtps-cpp
   apt install -y ros-galactic-cyclonedds ros-galactic-rmw-cyclonedds-cpp

   # Install vcstool package
   curl -s https://packagecloud.io/install/repositories/dirk-thomas/vcstool/script.deb.sh | sudo bash
   sudo apt-get update
   sudo apt-get install -y python3-vcstool

   # Install Microsoft packages (Ubuntu 20.04 only)
   wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
   sudo dpkg -i packages-microsoft-prod.deb
   rm packages-microsoft-prod.deb

   # Install .NET core 6.0 SDK
   sudo apt-get update; \
     sudo apt-get install -y apt-transport-https && \
     sudo apt-get update && \
     sudo apt-get install -y dotnet-sdk-6.0

   # Source ROS2 Galactic
   source /opt/ros/galactic/setup.zsh
   ```

1. Clone the ros2-for-unity project:

   ```
   git clone git@github.com:RobotecAI/ros2-for-unity.git
   cd ros2-for-unity
   ```

1. Run `./pull_repositories.sh`

1. Copy `camera_control_interfaces` and `laser_control_interfaces` to `src/ros2cs/custom_messages`

   ```
   mkdir src/ros2cs/custom_messages
   cp -r <path to this repo>/camera_control_interfaces src/ros2cs/custom_messages/camera_control_interfaces
   cp -r <path to this repo>/laser_control_interfaces src/ros2cs/custom_messages/laser_control_interfaces
   ```

1. Run `./build.sh`. You can add the `--clean-install` flag to make sure your installation directory is cleaned before building.

   1. You may run into an error when building ros2cs_examples. To work around this, just remove `ros2-for-unity/src/ros2cs/src/ros2cs/ros2cs_examples`.

1. Copy the new library files to the Unity project:

   ```
   cp -ru install/asset/Ros2ForUnity/Plugins <path to this repo>/simulation/LaserRunnerCutter/Assets/Ros2ForUnity/Plugins
   ```
