set -e

source env.sh

source $VENV_DIR/bin/activate
source /opt/ros/foxy/setup.sh
source install/local_setup.sh

export PYTHONPATH=$PYTHONPATH:$VENV_DIR/lib/python3.8/site-packages