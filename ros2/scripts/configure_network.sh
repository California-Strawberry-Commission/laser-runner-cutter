#!/bin/bash
# Configures network for LUCID Triton and Helios cameras.
# Usage:
#   sudo ./configure_network.sh \
#     --iface-triton enP5p1s0f0 \
#     --iface-helios enP5p1s0f1 \
#     --mac-triton 1C0FAF830702 \
#     --mac-helios 1C0FAF8C3EB1
#
# Optional (defaults shown):
#   --triton-ip 192.168.15.11 --helios-ip 192.168.16.11
#   --triton-host 192.168.15.1 --helios-host 192.168.16.1
#
# Notes:
# - Requires NetworkManager (nmcli) and LUCID Arena SDK installed under /opt/ArenaSDK/.
# - The camera MAC addresses can be listed using /opt/ArenaSDK/ArenaSDK_Linux_{x64,ARM64}/Utilities/IpConfigUtility /list

set -e

# Defaults
TRITON_HOST_IP="192.168.15.1"
TRITON_CAM_IP="192.168.15.11"
HELIOS_HOST_IP="192.168.16.1"
HELIOS_CAM_IP="192.168.16.11"
MTU="9000"

TRITON_IF=""
HELIOS_IF=""
TRITON_MAC=""
HELIOS_MAC=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface-triton) TRITON_IF="${2:?}"; shift 2 ;;
    --iface-helios) HELIOS_IF="${2:?}"; shift 2 ;;
    --mac-triton)   TRITON_MAC="${2:?}"; shift 2 ;;
    --mac-helios)   HELIOS_MAC="${2:?}"; shift 2 ;;
    --triton-ip)    TRITON_CAM_IP="${2:?}"; shift 2 ;;
    --helios-ip)    HELIOS_CAM_IP="${2:?}"; shift 2 ;;
    --triton-host)  TRITON_HOST_IP="${2:?}"; shift 2 ;;
    --helios-host)  HELIOS_HOST_IP="${2:?}"; shift 2 ;;
    --mtu)          MTU="${2:?}"; shift 2 ;;
    -h|--help)
      sed -n '1,16p' "$0"; exit 0
      ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Validate required args
if [[ -z "$TRITON_IF" || -z "$HELIOS_IF" || -z "$TRITON_MAC" || -z "$HELIOS_MAC" ]]; then
  echo "Missing required arguments. See --help."
  exit 1
fi

echo "Configuring network..."
echo "  Triton IF: $TRITON_IF  Host IP: $TRITON_HOST_IP  Cam IP: $TRITON_CAM_IP  MAC: $TRITON_MAC"
echo "  Helios IF: $HELIOS_IF  Host IP: $HELIOS_HOST_IP  Cam IP: $HELIOS_CAM_IP  MAC: $HELIOS_MAC"

# Pick ArenaSDK path by architecture
arch="$(uname -m)"
case "$arch" in
  x86_64)  ARENASDK_DIR="/opt/ArenaSDK/ArenaSDK_Linux_x64" ;;
  aarch64|arm64) ARENASDK_DIR="/opt/ArenaSDK/ArenaSDK_Linux_ARM64" ;;
  *)
    echo "Unsupported architecture: $arch"; exit 1 ;;
esac

IPCFG="$ARENASDK_DIR/Utilities/IpConfigUtility"
if [[ ! -x "$IPCFG" ]]; then
  echo "IpConfigUtility not found at $IPCFG"; exit 1
fi

# Ensure systemd-networkd.service is stopped and disabled, because it clashes with NetworkManager
sudo systemctl disable systemd-networkd.service || true
sudo systemctl stop systemd-networkd.service || true
sudo systemctl restart NetworkManager

# Configure NIC for cameras
if nmcli -t -f NAME c show | grep -qx "triton"; then
  sudo nmcli c delete triton || true
fi
if nmcli -t -f NAME c show | grep -qx "helios"; then
  sudo nmcli c delete helios || true
fi

sudo nmcli c add con-name triton ifname "$TRITON_IF" type ethernet
sudo nmcli c mod triton ipv4.addresses "${TRITON_HOST_IP}/24" ipv4.method manual ethernet.mtu 9000
sudo nmcli c up triton

sudo nmcli c add con-name helios ifname "$HELIOS_IF" type ethernet
sudo nmcli c mod helios ipv4.addresses "${HELIOS_HOST_IP}/24" ipv4.method manual ethernet.mtu 9000
sudo nmcli c up helios

# Set static IPs on cameras
# Note: /persist will fail if the camera's IP doesn't match subnet with the NIC. So, first force the IP.
sudo "$IPCFG" /force -m "$TRITON_MAC" -a "$TRITON_CAM_IP" -s 255.255.255.0 -g 0.0.0.0
sudo "$IPCFG" /persist -m "$TRITON_MAC" -a "$TRITON_CAM_IP" -s 255.255.255.0 -g 0.0.0.0
sudo "$IPCFG" /force -m "$HELIOS_MAC" -a "$HELIOS_CAM_IP" -s 255.255.255.0 -g 0.0.0.0
sudo "$IPCFG" /persist -m "$HELIOS_MAC" -a "$HELIOS_CAM_IP" -s 255.255.255.0 -g 0.0.0.0
