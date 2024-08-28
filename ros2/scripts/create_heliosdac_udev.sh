#!/bin/bash

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo cp $script_dir/heliosdac.rules /etc/udev
sudo ln -s /etc/udev/heliosdac.rules /etc/udev/rules.d/011_heliosdac.rules
sudo udevadm control --reload