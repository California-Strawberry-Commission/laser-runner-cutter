#!/bin/bash
set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create .env file with generated LiveKit API secret
env_file="$script_dir/../.env"
if [ ! -f "$env_file" ]; then
    secret=$(openssl rand -base64 32)
    cat > "$env_file" <<EOF
LIVEKIT_API_KEY="devkey"
LIVEKIT_API_SECRET="$secret"

# Used by LiveKit Server. This should be the same key and secret as above.
LIVEKIT_KEYS="devkey: $secret"
EOF
    echo "[create_env_file] Created .env with generated LiveKit API secret"
else
    echo "[create_env_file] .env already exists, skipping generation of LiveKit API secret"
fi

# Detect Tailscale IP that LiveKit Server will advertise to WebRTC clients
# as an ICE host candidate, which is required for the media stream to reach
# clients connecting via Tailscale.
TAILSCALE_IP=$(ip -4 addr show tailscale0 2>/dev/null | sed 's|.*inet \([0-9.]*\)/.*|\1|;t;d' | head -1)
if grep -q '^TAILSCALE_IP=' "$env_file" 2>/dev/null; then
    # If TAILSCALE_IP already exists, replace in-place
    sed -i "s|^TAILSCALE_IP=.*|TAILSCALE_IP=$TAILSCALE_IP|" "$env_file"
else
    # If TAILSCALE_IP does note exist yet in the .env file, append it
    printf '\n# Tailscale IP advertised to WebRTC clients as ICE candidate\nTAILSCALE_IP=%s\n' "$TAILSCALE_IP" >> "$env_file"
fi
echo "[create_env_file] Set TAILSCALE_IP=${TAILSCALE_IP:-<not found, will auto-detect>}"
