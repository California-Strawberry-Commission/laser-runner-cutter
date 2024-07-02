#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd script_dir
source setup.sh
ros2 launch guidance_brain launch.py