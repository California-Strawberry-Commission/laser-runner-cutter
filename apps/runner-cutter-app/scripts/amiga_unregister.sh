#! /bin/bash

set -uxeo pipefail

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js first."
    exit 1
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..
app_manifest_file=$app_dir/manifest.json
user_manifest_file=~/manifest.json

if [ ! -f "$user_manifest_file" ]; then
    echo "User manifest file not found."
    exit 1
else
    node <<EOF
const fs = require('fs');
const { execSync } = require('child_process');

const appManifestFile = '$app_manifest_file';
const userManifestFile = '$user_manifest_file';

try {
  const appManifestJson = JSON.parse(fs.readFileSync(appManifestFile, 'utf8'));
  const userManifestJson = JSON.parse(fs.readFileSync(userManifestFile, 'utf8'));

  // Remove services from appManifestJson that exist in userManifestJson
  for (const serviceName in appManifestJson.services) {
    delete userManifestJson.services[serviceName];
    execSync('rm -f ~/.config/systemd/user/' + serviceName + '.service');
  }

  // If no services remain, just remove the user manifest file
  if (Object.keys(userManifestJson.services).length === 0) {
    execSync('rm -f ' + userManifestFile);
    console.log('User manifest removed.');
  } else {
    fs.writeFileSync(userManifestFile, JSON.stringify(userManifestJson, null, 2));
    console.log('User manifest updated:', userManifestFile);
  }
} catch (error) {
  console.error('Error:', error.message);
  process.exit(1);
}
EOF
fi
