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

sns_topic_arn="arn:aws:sns:us-west-2:582853214798:CscMlModelTrainingStatus"
if [[ $? -eq 0 ]]; then
  # Zip and upload results
  cd "$script_dir/.."
  if [ -d "output" ]; then
    zip -r "output.zip" "output"
    aws s3 cp "output.zip" "s3://csc-runner-segmentation-dvc/out/output.zip"
  fi

  # Send notification to SNS
  aws sns publish --topic-arn "$sns_topic_arn" --message "Runner segmentation model training SUCCESSFUL." > /dev/null 2>&1
else
  # Send notification to SNS
  aws sns publish --topic-arn "$sns_topic_arn" --message "Runner segmentation model training FAILED." > /dev/null 2>&1
fi

# Shutdown EC2 instance
sudo shutdown -h now