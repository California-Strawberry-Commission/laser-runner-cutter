#! /bin/bash

set -uxeo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# TODO: this currently removes all other registered apps
rm -f ~/manifest.json
ln -s "$DIR/manifest.json" ~/manifest.json