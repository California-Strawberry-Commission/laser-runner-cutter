#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir="$(dirname "$script_dir")"
service_name="laser-runner-cutter-app.service"

sudo tee "/etc/systemd/system/$service_name" > /dev/null <<EOF
[Unit]
Description="Laser Runner Cutter app"
After=network.target

[Service]
Environment="PATH=$PATH"
WorkingDirectory=$app_dir
ExecStart=$app_dir/scripts/run_app.sh
StandardOutput=inherit
StandardError=inherit
Restart=always
RestartSec=5
TimeoutStopSec=10
User=$USER

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl start "$service_name"
sudo systemctl enable "$service_name"
