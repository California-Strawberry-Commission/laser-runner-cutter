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

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$DIR/bootstrap.sh $DIR $DIR/venv

if [ "$local" = true ]; then
  $DIR/venv/bin/python $DIR/src/main.py
else
  DISPLAY=:0 $DIR/venv/bin/python $DIR/src/main.py
fi

exit 0
