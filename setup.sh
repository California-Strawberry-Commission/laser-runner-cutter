python3 -m venv ./venv
sudo apt install python3-tk
sudo cp ./heliosdac.rules /etc/udev/
sudo ln -s /etc/udev/heliosdac.rules /etc/udev/rules.d/011_heliosdac.rules
