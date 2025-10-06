# GStreamer + WHIP + LiveKit

This node subscribes to a ROS2 Image topic, hardware encodes via GStreamer, and streams over LiveKit.

## Setup

1.  Run the install script:

        $ livekit_ros2/scripts/install.sh

1.  Create LiveKit API secret

    1.  Create .env from template

            $ cd laser-runner-cutter
            $ cp .env.example .env

    2.  Generate an API secret

            $ openssl rand -base64 32

    3.  Edit `.env` and replace the API secret with the one generated in the previous step

## Run

```
$ livekit_ros2/scripts/run.sh
```
