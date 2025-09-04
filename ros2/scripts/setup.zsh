script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source /opt/ros/$ROS_DISTRO/setup.zsh
source $ROS_WS_DIR/install/setup.zsh
set -a
source $ROS_WS_DIR/../.env
set +a

export PYTHONPATH=$PYTHONPATH:$VENV_DIR/lib/python3.10/site-packages
# Our ROS2 nodes depend on local packages that are pip installed in editable mode
# and thus need to be added to PYTHONPATH so that ROS2 can find them
export PYTHONPATH=$PYTHONPATH:$ROS_WS_DIR/aioros2:$PROJECT_DIR/ml/ml_utils:$PROJECT_DIR/ml/runner_segmentation_model

alias build="$script_dir/build.sh"