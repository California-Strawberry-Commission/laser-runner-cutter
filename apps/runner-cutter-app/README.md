# Runner Cutter App

Web app for laser runner cutter control and automation, built with Next.js.

## Setup

1.  Install Rosbridge

        $ sudo apt install ros-foxy-rosbridge-suite
        $ source laser-runner-cutter/ros2/setup.sh  # or setup.zsh in Z shell

1.  Install nvm

        $ curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

1.  Install Node.js

        $ nvm install --lts

## Run

1.  Run ROS2 nodes

        $ source laser-runner-cutter/ros2/setup.sh  # or setup.zsh in Z shell
        $ ros2 launch runner_cutter_control launch.py

1.  In another terminal, run Rosbridge server

        $ source laser-runner-cutter/ros2/setup.sh  # or setup.zsh in Z shell
        $ ros2 launch rosbridge_server rosbridge_websocket_launch.xml

1.  In yet another terminal, run the web server

        $ cd runner-cutter-app

        # Development mode (fast refresh, detailed error messages, debugging tools)
        $ npm run dev

        # Production mode
        $ npm run build
        $ npm run start

1.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the app.
