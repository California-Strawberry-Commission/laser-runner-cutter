#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models_dir=$script_dir/../models
tools_dir=$script_dir/../src/tools
trtexec_bin="/usr/src/tensorrt/bin/trtexec"

if [ -z "$ROS_DISTRO" ]; then
  source "$script_dir/../scripts/setup.sh"
fi

if [ ! -x "$trtexec_bin" ]; then
  echo "trtexec not found at $trtexec_bin"
  exit 1
fi

if ! ls "$models_dir"/*.engine 1> /dev/null 2>&1; then
  pt_file=$(ls "$models_dir"/*.pt 2>/dev/null | head -n 1)
  if [ -n "$pt_file" ]; then
    python3 "$tools_dir/yolo_genOnnx.py" "$pt_file"
    onnx_file=$(ls "$models_dir"/*.onnx 2>/dev/null | head -n 1)
    if [ -n "$onnx_file" ]; then
      if [[ "$@" == *"--verbose"* ]]; then
      "$trtexec_bin" --onnx="$onnx_file" --saveEngine="$models_dir/RunnerSegYoloV8l.engine" --fp16 --verbose
      else
      "$trtexec_bin" --onnx="$onnx_file" --saveEngine="$models_dir/RunnerSegYoloV8l.engine" --fp16
      fi
    else
      echo "ONNX file was not created in $models_dir"
      exit 1
    fi
  else
    echo "No .pt file found in $models_dir"
    exit 1
  fi
fi