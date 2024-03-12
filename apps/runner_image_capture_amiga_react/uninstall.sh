#! /bin/bash

set -uxeo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rm -f ~/manifest.json
rm -f ~/.config/systemd/user/*.service