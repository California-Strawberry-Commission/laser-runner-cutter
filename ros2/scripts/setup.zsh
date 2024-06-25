script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source /opt/ros/foxy/setup.zsh
source $ROS_WS_DIR/install/setup.zsh

export PYTHONPATH=$PYTHONPATH:$VENV_DIR/lib/python3.8/site-packages
# Our ROS2 nodes depend on local packages that are pip installed in editable mode
# and thus need to be added to PYTHONPATH so that ROS2 can find them
export PYTHONPATH=$PYTHONPATH:$ROS_WS_DIR/aioros2:$PROJECT_DIR/ml/ml_utils:$PROJECT_DIR/ml/runner_segmentation_model
# For some reason this is needed...
export PYTHONPATH=$PYTHONPATH:$VENV_DIR/lib/python3.8/site-packages/torchvision-0.16.1-py3.8-linux-aarch64.egg

alias build="$script_dir/build.sh"