#!/bin/bash
# For use by the Amiga Brain's launcher to launch the app.

if ! command -v node &> /dev/null; then
  # TODO: figure out a better way to run node when launched via systemd
  export NVM_DIR="/mnt/managed_home/farm-ng-user-genkikondo/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..

cd $app_dir

npm run start