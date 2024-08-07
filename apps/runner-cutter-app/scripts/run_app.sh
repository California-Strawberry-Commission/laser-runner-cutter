#!/bin/bash
# For production. Starts the app.

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..

cd $app_dir

npm run start