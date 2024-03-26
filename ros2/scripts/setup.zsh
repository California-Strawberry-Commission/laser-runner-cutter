script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source $WS_DIR/install/setup.zsh

export PYTHONPATH=$PYTHONPATH:~/ros2_ws/venv/lib/python3.8/site-packages