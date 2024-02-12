#!/bin/bash -x

# create virtual environment for the backend
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# build the frontend
cd ts/
npm install
npm run build