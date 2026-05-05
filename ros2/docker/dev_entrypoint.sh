#!/bin/bash
# Entrypoint script that keeps the container alive indefinitely
# while allowing it to be stopped and restarted cleanly by Docker.
set -eo pipefail

# Trap SIGTERM and SIGINT signals and exit cleanly.
# This is necessary because PID 1 ignores signals by default, so we need to
# explicitly handle them to allow lifecycle_manager_node to stop the container
# (which Docker will then restart).
trap "exit" SIGTERM SIGINT

sleep infinity &
wait