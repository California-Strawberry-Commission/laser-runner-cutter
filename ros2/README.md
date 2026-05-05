# Laser Runner Cutter ROS2 Nodes

This project uses **Ubuntu 22.04 (Jammy Jellyfish)** and **ROS Humble**, and is intended to be deployed on NVIDIA Jetson.

## Environment Setup

1. Clone this repository into your home directory

        $ cd ~
        $ git clone https://github.com/California-Strawberry-Commission/laser-runner-cutter

1. Run [`scripts/bootstrap.sh`](scripts/bootstrap.sh)

1. Create LiveKit API secret
    1. Create .env from template

            $ cd laser-runner-cutter
            $ cp .env.example .env

    2. Generate an API secret

            $ openssl rand -base64 32

    3. Edit `.env` and replace the API secret with the one generated in the previous step

### Using Helios DAC on Linux

Linux systems require udev rules to allow access to USB devices without root privileges. This is already set up as part of the auto-install process above. Make sure that the user account communicating with the DAC is in the _plugdev_ group.

### Using LUCID cameras (Triton and Helios2)

In order to connect to the cameras, you will need to configure the network interfaces and camera IPs. To do this, run [`scripts/configure_network.sh`](scripts/configure_network.sh).

## Development Environment

We use Visual Studio Code with a **Dev Container** for a consistent, pre-configured environment. The Dev Container is defined in [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) and is backed by [`docker-compose.yaml`](docker-compose.yaml).

### Getting started

1. Follow the [Environment Setup](#environment-setup) steps above.
2. Install the VS Code [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
3. Open the `ros2/` folder in VS Code.
4. When prompted, click **Reopen in Container** (or run `Dev Containers: Reopen in Container` from the command palette).

VS Code will build the container and install all extensions automatically. The following extensions are provisioned by the Dev Container:

- [Robotics Developer Environment](https://marketplace.visualstudio.com/items?itemName=Ranch-Hand-Robotics.rde-pack)
- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
- [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
- [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
- [Tasks](https://marketplace.visualstudio.com/items?itemName=actboy168.tasks)

### VS Code Tasks

Common workflows are defined in [.vscode/tasks.json](.vscode/tasks.json) and can be run via `Tasks: Run Task` in the command palette, or the Tasks status bar button.

### Production deployment

On a production device, we can set up the machine to start the ROS2 nodes on startup:

1.  TBD

In addition, we can set up the machine to enable restarting the ROS2 nodes and rebooting the machine without a password, which allows the LifecycleManager node to trigger the relevant commands. This is useful for allowing other programs (such as a web-based app) to restart the nodes or reboot the machine in case of an irrecoverable issue.

1.  Run `sudo visudo`, then add the following lines:

        <username> ALL=(ALL) NOPASSWD: /sbin/reboot

## LUCID camera calibration

We currently use the following LUCID cameras:

- Triton 3.2 MP Model ([TRI032S-CC](https://thinklucid.com/product/triton-32-mp-imx265/))
- Helios2 Ray Outdoor IP67 ToF 3D Camera ([HTR003S-001](https://thinklucid.com/product/helios2-ray-outdoor-tof-ip67-3d-camera/))

We need to calculate the intrinsic matrix and distortion coefficients for each camera, as well as the extrinic matrix that describes the rotation and translation between the two cameras. For more details, see https://support.thinklucid.com/app-note-helios-3d-point-cloud-with-rgb-color/

### Step 1: Capture frames

1.  Print the [calibration grid](https://arenasdk.s3-us-west-2.amazonaws.com/LUCID_target_whiteCircles.pdf)
2.  Imagine a 3x3 grid in the cameras' FOV. Do the following for each grid cell:
    1.  Place the calibration grid in the grid cell
    2.  Create a new directory where this set of images should be saved
    3.  Capture a frame (change param values as needed):

            ros2 run camera_control_cpp lucid_calibrate -- capture_frame --output_dir <output dir> --exposure_us 20000 --gain_db 1

In the end, you should have 9 sets of {Triton image (png), Helios intensity image (png), Helios xyz data (yml)}.

### Step 2: Calculate intrinsics

We calculate the intrinsic matrix and distortion coefficients using the method based on https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html

1.  Create a single directory with the 9 Triton images you captured.
2.  Run the following to calculate and save the Triton intrinsics:

        ros2 run camera_control_cpp lucid_calibrate -- calculate_intrinsics --images_dir <path to the dir containing the Triton images> --output_dir <where to write the intrinsics data file>

3.  Create a single directory with the 9 Helios intensity images you captured.
4.  Run the following to calculate and save the Helios intrinsics:

        ros2 run camera_control_cpp lucid_calibrate -- calculate_intrinsics --images_dir <path to the dir containing the Helios intensity images> --output_dir <where to write the intrinsics data file>

### Step 3: Calculate extrinsics

1.  Create 3 directories: one containing all of the Triton images, one containing all of the Helios intensity images, and one containing all of the Helios xyz data files. Make sure that the corresponding files in each directory share the same base name. For example, `triton_images/0.png`, `helios_intensity_images/0.png`, and `xyz_data/0.yml` should come from a single capture.

2.  Run the following to save the xyz-to-Triton extrinsics:

        ros2 run camera_control_cpp lucid_calibrate -- calculate_extrinsics_xyz_to_triton --triton_intrinsics_file <path to Triton intrinsics yml file> --triton_images_dir <dir containing all Triton images> --helios_images_dir <dir containing all Helios intensity images> --helios_xyz_dir <dir containing all xyz data files> --output_dir <where to write the extrinsics data file>

3.  Run the following to save the xyz-to-Helios extrinsics:

        ros2 run camera_control_cpp lucid_calibrate -- calculate_extrinsics_xyz_to_helios --helios_intrinsics_file <path to Helios intrinsics yml file> --helios_images_dir <dir containing all Helios intensity images> --helios_xyz_dir <dir containing all xyz data files> --output_dir <where to write the extrinsics data file>

## Troubleshooting

### No space left on device

Make sure you have at least 32GB of available disk space. If installed using an LVM (Ubuntu server's default), expand your root (`/`) partition to at least 32GB.
