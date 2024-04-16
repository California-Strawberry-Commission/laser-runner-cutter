#!/bin/bash -ex

local=false
while getopts l opt; do
  case $opt in
    l )
      local=true
      ;;
    \? )
      echo "Usage: $0 [-l]"
      exit 1
      ;;
  esac
done

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..

$script_dir/bootstrap.sh $app_dir $app_dir/venv

if [ "$local" = true ]; then
  $app_dir/venv/bin/python $app_dir/src/main.py
else
  DISPLAY=:0 $app_dir/venv/bin/python $app_dir/src/main.py
fi

exit 0
