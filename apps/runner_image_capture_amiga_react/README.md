# Runner Image Capture Amiga App

An application deployable to the Amiga brain, written using React and FastAPI, for capturing runner images in the field.

## Setup

1.  Prepare the backend

        cd runner_image_capture_amiga_react
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt

2.  Install NVM

        cd runner_image_capture_amiga_react/ts
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

3.  Prepare the frontend

        nvm install --lts
        npm install
        npm run build

## Run locally from the command line

1.  Run the backend

        cd runner_image_capture_amiga_react
        python main.py --debug  # add debug flag to use the local frontend, or remove the flag to use the prebuilt

2.  In a separate terminal, run the frontend (if the backend was started in debug mode)

        cd runner_image_capture_amiga_react/ts
        npm run dev

## Install and run on Amiga brain

SSH into the Amiga brain and clone the repo there. Then,

    cd runner_image_capture_amiga_react
    ./install.sh

You should now see the app on the brain screen. Click on it to launch it.
