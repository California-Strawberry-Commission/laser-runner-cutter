#!/bin/bash
# Be careful, note the shutdown command at the end

if [ $# -eq 0 ]; then
  echo "Usage: $0 <train command>"
  exit 1
fi

# Source venv
script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
source "$script_dir/../venv/bin/activate"

# Run train command
"$@"

# Zip and upload results
cd "$script_dir/.."
if [ -d "ultralytics" ]; then
  zip -r "ultralytics.zip" "ultralytics"
  aws s3 cp "ultralytics.zip" "s3://runner-segmentation-dvc/out/ultralytics.zip"
fi
if [ -d "maskrcnn" ]; then
  zip -r "maskrcnn.zip" "maskrcnn"
  aws s3 cp "maskrcnn.zip" "s3://runner-segmentation-dvc/out/maskrcnn.zip"
fi

# Send notification to SNS
aws sns publish --topic-arn "arn:aws:sns:us-west-2:197938073352:MlModelTrainingStatus" --message "Runner segmentation model training completed." > /dev/null 2>&1

# Shutdown EC2 instance
sudo shutdown -h now
