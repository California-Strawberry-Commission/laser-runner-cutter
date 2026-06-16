#!/bin/bash
# For use by the Amiga Brain's launcher to launch the app.

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if ! command -v node &> /dev/null; then
  username=$(echo "$script_dir" | sed 's|/mnt/managed_home/\([^/]*\)/.*|\1|')
  export NVM_DIR="/mnt/managed_home/$username/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi

app_dir=$script_dir/..

cd $app_dir

npm run start