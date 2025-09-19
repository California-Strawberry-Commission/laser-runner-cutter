#!/bin/bash
script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/../../../scripts/env.sh
models_dir=$script_dir/../models

if [ ! -f "$models_dir/RunnerSegYoloV8l.engine" ]; then
    yolo export model=$models_dir/RunnerSegYoloV8l.pt format=engine imgsz=768,1024 half=True simplify=True device=0
fi
