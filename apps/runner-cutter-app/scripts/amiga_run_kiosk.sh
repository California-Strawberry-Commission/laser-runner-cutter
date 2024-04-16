#!/bin/bash
# For use by systemd to launch the app and start Firefox in kiosk mode upon boot.

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $script_dir

./amiga_run_app.sh &

# Wait until the server is up and running
echo "Waiting for the server to start..."
while ! nc -z localhost 3000; do
  sleep 1
done

firefox --kiosk --display=:0.0 --private-window "localhost:3000"