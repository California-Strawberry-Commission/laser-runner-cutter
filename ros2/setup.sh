script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source /opt/ros/foxy/setup.sh
source $script_dir/install/local_setup.sh

export PYTHONPATH=$PYTHONPATH:$VENV_DIR/lib/python3.8/site-packages