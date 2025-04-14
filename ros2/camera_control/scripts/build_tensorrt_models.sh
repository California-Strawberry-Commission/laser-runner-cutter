#!/bin/bash
script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/../../scripts/env.sh

yolo export model=$script_dir/../models/RunnerSegYoloV8l.pt format=engine imgsz=768,1024 half=True simplify=True device=0