# Runner Cutter App

Web app for laser runner cutter control and automation, built with Next.js.

## Setup

1.  Follow setup steps in [ros2/README.md](../../ros2/README.md)

1.  Run the following:

        $ scripts/setup.sh

## Run locally

1.  Run the following to run all ROS2 nodes and the app (in dev mode) concurrently:

        $ scripts/local_run_dev.sh

1.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the app.

## Install and run on production device

1.  SSH into the device and clone the repo there.

1.  [Optional] Set environment variables

    By default, the Rosbridge websocket server URL will be set to `ws://<window.location.hostname>:9090`, and the video server URL will be set to `http://<window.location.hostname>:8080`. If you want to override those to custom URLs, you can do the following:

        $ cd laser-runner-cutter/apps/runner-cutter-app
        $ echo "NEXT_PUBLIC_ROSBRIDGE_URL=ws://localhost:9090" >> .env.local
        $ echo "NEXT_PUBLIC_VIDEO_SERVER_URL=http://localhost:8080" >> .env.local

    Change `ws://localhost:9090` to wherever Rosbridge will be running, and `http://localhost:8080` to wherever Web Video Server will be running.

1.  Run the setup script:

        $ scripts/setup.sh

### On Amiga Brain

1.  To register the app in the Amiga Brain's app launcher, run:

        $ scripts/amiga_register.sh

### On any device running Ubuntu, such as an NVIDIA Jetson

1.  Copy systemd service file

        $ cp scripts/laser-runner-cutter-app.service /etc/systemd/system/

1.  Edit the newly created `/etc/systemd/system/laser-runner-cutter-app.service` to contain the correct username

1.  Enable the service to run on startup

        $ sudo systemctl enable laser-runner-cutter-app.service
        $ sudo systemctl start laser-runner-cutter-app.service
