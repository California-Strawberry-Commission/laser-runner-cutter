#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models_dir=$script_dir/../models
trtexec_bin="/usr/src/tensorrt/bin/trtexec"

if [ ! -x "$trtexec_bin" ]; then
  echo "trtexec not found at $trtexec_bin"
  exit 1
fi

for onnx_file in "$models_dir"/*.onnx; do
  filename="$(basename "$onnx_file" .onnx)"
  engine_file="$models_dir/${filename}.engine"

  # Check if the engine file already exists
  if [ -f "$engine_file" ]; then
    echo "Skipping ${filename}: engine already exists at $engine_file"
    continue
  fi

  echo "Converting: $onnx_file"
  echo "Output:     $engine_file"
  echo "----------------------------------"
  "$trtexec_bin" --onnx="$onnx_file" --saveEngine="$engine_file" --fp16
done
