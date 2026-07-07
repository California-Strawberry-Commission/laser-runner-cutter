# Laser Runner Cutter ROS2 Nodes

This project uses **Ubuntu 22.04 (Jammy Jellyfish)** and **ROS Humble**, and is intended to be deployed on NVIDIA Jetson.

The entire system runs via [`docker-compose.yaml`](docker-compose.yaml), which brings up four services: the ROS2 node container, a [LiveKit](https://livekit.io) media server (for streaming video to the web app), a LiveKit ingress service, and Redis (required by LiveKit).

## Environment Setup

```sh
# Clone this repository into the home directory
git clone https://github.com/California-Strawberry-Commission/laser-runner-cutter ~/laser-runner-cutter

# Run the bootstrap script
~/laser-runner-cutter/ros2/scripts/bootstrap.sh
```

### Tailscale

[Tailscale](https://tailscale.com) is a mesh VPN that gives each device a stable private IP, making it easy to reach the Jetson from another machine without port forwarding. It is installed by the bootstrap script above. After installation, authenticate and join your Tailscale network:

```sh
sudo tailscale up
# Follow the URL printed to log in and approve the device
```

### Using Helios DAC on Linux

Linux systems require udev rules to allow access to USB devices without root privileges. This is already set up as part of the auto-install process above. Make sure that the user account communicating with the DAC is in the `plugdev` group.

### Using LUCID cameras (Triton and Helios2)

In order to connect to the cameras, you will need to configure the network interfaces and camera IPs. To do this, run [`scripts/configure_network.sh`](scripts/configure_network.sh).

## Development Environment

We use Visual Studio Code with a **Dev Container** for a consistent, pre-configured environment. The Dev Container is defined in [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) and is backed by [`docker-compose.yaml`](docker-compose.yaml).

### Developing from the Jetson

1. Follow the [Environment Setup](#environment-setup) steps above.
2. Install the VS Code [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
3. Open the `ros2/` folder in VS Code.
4. When prompted, click **Reopen in Container** (or run `Dev Containers: Reopen in Container` from the command palette).

VS Code will build the container and install all extensions automatically (note: you may need to go into the Extensions tab and click "Install in Dev Container" for the extensions). The following extensions are provisioned by the Dev Container:

- [Robotics Developer Environment](https://marketplace.visualstudio.com/items?itemName=Ranch-Hand-Robotics.rde-pack)
- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
- [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
- [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
- [Tasks](https://marketplace.visualstudio.com/items?itemName=actboy168.tasks)

### Developing from another machine

You can develop remotely by connecting VS Code on your local machine to the Jetson over SSH.

1. Follow the [Environment Setup](#environment-setup) steps on the Jetson.
2. Install the VS Code [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension pack on your local machine.
3. Open the Command Palette and run **Remote-SSH: Connect to Host**, then enter `<username>@<jetson-ip>`. This will open a new VS Code window connected to the Jetson.
4. In that SSH-connected VS Code window, open the remote `~/laser-runner-cutter/ros2/` folder.
5. When prompted, click **Reopen in Container** (or run `Dev Containers: Reopen in Container` from the command palette). This will build and run the containers defined in `docker-compose.yaml` on the Jetson, and then attach to the Dev Container on the Jetson.

You may need to go into the Extensions tab and click "Install in Dev Container" for your extensions. All subsequent editing, building, and debugging happens on the Jetson while the UI runs locally.

### VS Code Tasks

Common workflows are defined in [.vscode/tasks.json](.vscode/tasks.json) and can be run via `Tasks: Run Task` in the command palette, or the Tasks status bar button.

### Commands

To build and run the system outside of VS Code:

```sh
# On the Jetson
cd ~/laser-runner-cutter/ros2

# Start containers
docker compose up --build --detach

# Open a shell in the ros2 container
docker exec -it ros2-ros2-1 bash

# Build and run all ROS nodes (inside container)
docker/build_and_run.sh

# Build a single package (inside container)
colcon build --packages-select <pkg>

# Stop containers
docker compose down
```

### Logs

If the ROS nodes are started using `docker/build_and_run.sh`, the logs will be written to `log/runner-cutter_*.log`.

## Production Deployment

Docker is enabled as a systemd service automatically when installed via [`scripts/install_docker.sh`](scripts/install_docker.sh). All services in [`docker-compose.yaml`](docker-compose.yaml) have `restart: unless-stopped`, so they come back up on reboot without any additional configuration.

For production, start the containers using the compose override (which runs `docker/build_and_run.sh` on container start):

```sh
# On the Jetson
cd ~/laser-runner-cutter/ros2
docker compose -f docker-compose.yaml -f docker-compose.prod.yaml up --build --detach
```

Note that if containers are already running at the time you run the above command, only those whose configuration has changed between the two invocations will be restarted.

When containers are running, when the Jetson is rebooted, all containers will automatically be started (with `docker/build_and_run.sh` called in the ros2 container) due to `restart: unless-stopped`.

### Passwordless reboot

To allow the LifecycleManager node to reboot the machine without a password (e.g. to recover from an irrecoverable error), run `sudo visudo` and add:

    <username> ALL=(ALL) NOPASSWD: /sbin/reboot

## LUCID Camera Calibration

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

## Updating the Runner Detection Model

1. Navigate to the model directory:

   ```sh
   cd laser-runner-cutter/ml/runner_segmentation_model
   ```

2. Follow the instructions in `runner_segmentation_model/README.md` to pull models using DVC.

3. Export the model to ONNX format:

   ```sh
   python -m runner_segmentation.yolo export_onnx --weights_file <path/to/pt/model/file>
   ```

4. Copy the generated `.onnx` file (which should be the same directory as the `.pt` file) to [src/detection/models/](src/detection/models/).

5. Update the `runner_model` parameter in [src/runner_cutter_control/config/parameters.yaml](src/runner_cutter_control/config/parameters.yaml) to reflect the new model name. The value should be the `.onnx` filename with `.onnx` replaced by `.engine`.

## Troubleshooting

### No space left on device

Make sure you have at least 32GB of available disk space. If installed using an LVM (Ubuntu server's default), expand your root (`/`) partition to at least 32GB.
