#!/bin/bash
set -e

# Install python deps of subpackages
# (Don't bother using ROS's dep management for py)
script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
source $script_dir/env.sh
source $VENV_DIR/bin/activate

# Find all requirement.txt files and iterate through them
find "$script_dir" -name 'requirements.txt' -type f | while read -r file; do
    # Extract directory path of the requirements.txt file
    dir_path=$(dirname "$file")

    # Navigate to the directory containing the requirements.txt file
    echo "Installing requirements from $file"
    pushd "$dir_path" || exit
    pip install -r "$(basename "$file")"
    popd || exit
done