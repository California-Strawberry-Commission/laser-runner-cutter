# GStreamer + WHIP + LiveKit

This node subscribes to a ROS2 Image topic, hardware encodes via GStreamer, and streams over LiveKit.

## Setup

1.  Install GStreamer (follow https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c)

1.  Install Docker (follow https://docs.docker.com/engine/install/ubuntu/)

    1.  Note: on Jetson, you may run into an issue with Docker Compose port mapping. If so, the following will need to be done:

            $ sudo apt-get install -y docker-ce=5:27.5* docker-ce-cli=5:27.5* --allow-downgrades

        See https://forums.developer.nvidia.com/t/issue-with-docker-compose-port-mapping-on-jetson-orin-nano-iptable-raw-module-missing/335728/4 for more details on the issue.

1.  Install Python deps

        $ cd livekit_ros2
        $ pip install -r requirements.txt

## Run

1.  Start Redis + LiveKit server + Ingress server via Docker

        $ cd livekit_ros2
        $ docker compose up

1.  Start livekit_whip_node

        $ ros2 launch livekit_ros2 launch.py
