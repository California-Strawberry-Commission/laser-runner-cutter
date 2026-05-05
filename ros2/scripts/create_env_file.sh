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
    echo "Created .env with generated LiveKit API secret"
else
    echo ".env already exists, skipping"
fi
