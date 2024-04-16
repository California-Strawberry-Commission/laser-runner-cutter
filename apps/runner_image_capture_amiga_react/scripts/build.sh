#!/bin/bash -x

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..

cd $app_dir

# create virtual environment for the backend
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# build the frontend
cd ts/
npm install
npm run build