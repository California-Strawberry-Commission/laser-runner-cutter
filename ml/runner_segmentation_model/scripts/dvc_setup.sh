#!/bin/bash

usage() { echo "Usage: $0 [-a <AWS access key used by DVC>] [-s <AWS secret key used by DVC>]" 1>&2; exit 1; }

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir=$script_dir/../venv
source $venv_dir/bin/activate

aws_dvc_access_key=""
aws_dvc_secret_key=""
while getopts ":a:s:" opt; do
  case $opt in
    a)
      aws_dvc_access_key="$OPTARG"
      ;;
    s)
      aws_dvc_secret_key="$OPTARG"
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))
if [ -z "${aws_dvc_access_key}" ] || [ -z "${aws_dvc_secret_key}" ]; then
  usage
fi

dvc remote modify --local runner_segmentation access_key_id $aws_dvc_access_key
dvc remote modify --local runner_segmentation secret_access_key $aws_dvc_secret_key
dvc pull -r runner_segmentation
