#!/bin/bash

# Check if nvpmodel is installed
if ! command -v nvpmodel &> /dev/null; then
  echo "nvpmodel not found. This device may not support Jetson power modes."
  exit 0
fi

# 50W mode for Jetson AGX Orin 64GB
# Note: MAXN is not recommended for production use
sudo nvpmodel -m 3

# Disable dynamic scaling - pin the GPU and CPU clocks to the highest rate for the power mode
sudo jetson_clocks