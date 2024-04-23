# Runner Cutter App

Web app for laser runner cutter control and automation, built with Next.js.

## Setup

1.  Follow setup steps in [ros2/README.md](../../ros2/README.md)

1.  Install nvm

        $ curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

1.  Install Node.js

        $ nvm install --lts

1.  Install npm packages

        $ cd laser-runner-cutter/apps/runner-cutter-app
        $ npm install

## Run locally

1.  Run the following:

        $ scripts/local_run_dev.sh

1.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the app.

## Install and run on Amiga brain

1.  SSH into the Amiga brain and clone the repo there.

1.  Run Setup steps above.

1.  Set environment variables

        $ cd laser-runner-cutter/apps/runner-cutter-app
        $ echo "NEXT_PUBLIC_ROSBRIDGE_URL=ws://localhost:9090" > .env.local

    Change `ws://localhost:9090` to wherever Rosbridge will be running.

1.  Build the prod app:

        $ cd laser-runner-cutter/apps/runner-cutter-app
        $ npm run build

1.  Finally, run:

        $ scripts/amiga_register.sh
