#!/bin/bash
# -----------------------------------------------------------------------------
# Build script for ROS2 workspace
#
# Usage:
#   ./build.sh [--no-clean] [--packages <package1> <package2> ...]
#
# Options:
#   --no-clean             Skip removing build/, install/, log/ before building
#   --packages <list>      Only build the specified packages (otherwise builds all)
#
# Examples:
#   ./build.sh
#       -> Clean build of all packages
#
#   ./build.sh --packages camera_control laser_control
#       -> Clean build of only camera_control and laser_control
#
#   ./build.sh --no-clean
#       -> Incremental build of all packages (no rm -rf)
#
#   ./build.sh --no-clean --packages camera_control
#       -> Incremental build of only camera_control
# -----------------------------------------------------------------------------

set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source /opt/ros/$ROS_DISTRO/setup.sh

# Parse arguments
PACKAGES=()
CLEAN=true
while [[ $# -gt 0 ]]; do
  case $1 in
    --packages)
      shift
      while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
        PACKAGES+=("$1")
        shift
      done
      ;;
    --no-clean)
      CLEAN=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build TensorRT models
echo "Building TensorRT models"
bash "$script_dir/../src/camera_control/scripts/build_tensorrt_models.sh"

# Build ROS2 nodes
cd "$ROS_WS_DIR"

if $CLEAN; then
  echo "Cleaning workspace (removing build/, install/, log/)"
  rm -rf build install log
else
  echo "Skipping clean"
fi

if [[ ${#PACKAGES[@]} -gt 0 ]]; then
  echo "Building selected packages: ${PACKAGES[*]}"
  colcon build --symlink-install \
    --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    --packages-select "${PACKAGES[@]}"
else
  echo "Building all packages"
  colcon build --symlink-install \
    --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi
