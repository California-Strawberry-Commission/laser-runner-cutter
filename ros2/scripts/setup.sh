script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source $ROS_WS_DIR/install/setup.sh

export PYTHONPATH=$PYTHONPATH:$VENV_DIR/lib/python3.8/site-packages