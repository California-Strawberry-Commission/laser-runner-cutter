#!/bin/bash

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo cp $script_dir/laser-runner-cutter-app.service /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl start laser-runner-cutter-app.service
sudo systemctl enable laser-runner-cutter-app.service