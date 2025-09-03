# GStreamer + WHIP + LiveKit

This node subscribes to a ROS2 Image topic, hardware encodes via GStreamer, and streams over LiveKit.

## Setup

1.  Install GStreamer (follow https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c):

1.  Install Docker (follow https://docs.docker.com/engine/install/ubuntu/):

1.  Install LiveKit

        $ curl -sSL https://get.livekit.io | bash
        $ sudo apt-get install jq

## Run

1.  Start Redis + LiveKit server + Ingress server via Docker

        $ cd livekit_ros2
        $ docker compose up

1.  Start livekit_whip_node

        $ ros2 launch livekit_ros2 launch.py
