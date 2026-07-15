# Runner Cutter App

Web app for laser runner cutter control and automation, built with Next.js.

## Setup

1.  (Optional) Follow the setup steps in [ros2/README.md](../../ros2/README.md) to build and run the ROS 2 nodes.

1.  Run the following to fetch dependencies and build the app:

        $ scripts/setup.sh

## Run locally

1.  Run the following to run all ROS2 nodes and the app (in dev mode) concurrently:

        $ scripts/local_run_dev.sh

1.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the app.

## Install and run on production device

1.  SSH into the device and clone the repo there.

1.  [Optional] Set environment variables

    By default, the Rosbridge websocket server URL will be set to `ws://<window.location.hostname>:9090`, and the LiveKit server URL will be set to `ws://<window.location.hostname>:7880`. If will be running Rosbridge and/or LiveKit server on a remote machine, you will need to override those to custom URLs by doing the following:

        $ cd laser-runner-cutter/apps/runner-cutter-app
        $ echo "NEXT_PUBLIC_ROSBRIDGE_URL=ws://10.95.76.2:9090" >> .env.local
        $ echo "NEXT_PUBLIC_LIVEKIT_URL=ws://10.95.76.2:7880" >> .env.local

    Change `ws://10.95.76.2:9090` to wherever Rosbridge will be running, and `ws://10.95.76.2:7880` to wherever the LiveKit server will be running.

    In addition, if you are running LiveKit server on a remote machine, you will need to make sure that you have LIVEKIT_API_KEY and LIVEKIT_API_SECRET defined. These need to match exactly with the ones used by the LiveKit Ingress server and the LiveKitWhipNode (on whichever machine is running LiveKit, check laser-runner-cutter/ros2/.env).

        $ echo "LIVEKIT_API_KEY=devkey" >> .env.local
        $ echo 'LIVEKIT_API_SECRET="<your API secret>"' >> .env.local

1.  Run the setup script:

        $ scripts/setup.sh

### On Amiga Brain

1.  To register the app in the Amiga Brain's app launcher, run:

        $ scripts/amiga_register.sh

1.  To view the logs, run:

        $ journalctl -f --user-unit laser-runner-cutter-app.service

### On any device running Ubuntu, such as an NVIDIA Jetson

1.  To create and enable a systemd service (so that the app automatically starts on boot), run:

        $ scripts/amiga_register.sh

1.  To view the logs, run:

        $ journalctl -f --unit laser-runner-cutter-app.service
