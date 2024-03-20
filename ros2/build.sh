#!/bin/bash

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
cd $script_dir

source setup.sh

# Build
colcon build --symlink-install