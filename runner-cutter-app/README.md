# Runner Cutter App

Web app for laser runner cutter control and automation, built with Next.js.

## Setup

1.  Install Rosbridge

        sudo apt install ros-foxy-rosbridge-suite
        source /opt/ros/foxy/setup.zsh  # or setup.bash depending on your shell

1.  Install nvm

        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

1.  Install Node.js

        nvm install --lts

## Run

1.  Run Rosbridge server

        ros2 launch rosbridge_server rosbridge_websocket_launch.xml

1.  In a separate terminal, run the web server

        cd runner-cutter-app

        # Development mode (fast refresh, detailed error messages, debugging tools)
        npm run dev

        # Production mode
        npm run build
        npm run start

1.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the app.
