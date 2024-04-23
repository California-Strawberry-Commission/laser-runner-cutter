#! /bin/bash
# For the Amiga Brain. Registers the app on the Brain's launcher.

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

# If user manifest file does not exist, then just copy the app manifest file.
# Otherwise, insert the services in the app manifest file into the user manifest file.
if [ ! -f "$user_manifest_file" ]; then
  cp $app_manifest_file $user_manifest_file
else
  node <<EOF
const fs = require('fs');

const appManifestFile = '$app_manifest_file';
const userManifestFile = '$user_manifest_file';

try {
  const appManifestJson = JSON.parse(fs.readFileSync(appManifestFile, 'utf8'));
  const userManifestJson = JSON.parse(fs.readFileSync(userManifestFile, 'utf8'));

  // Merge "services" objects
  const mergedServices = { ...userManifestJson.services, ...appManifestJson.services };
  const newUserManifestJson = { services: mergedServices };

  fs.writeFileSync(userManifestFile, JSON.stringify(newUserManifestJson, null, 2));

  console.log('User manifest updated:', userManifestFile);
} catch (error) {
  console.error('Error:', error.message);
  process.exit(1);
}
EOF
fi
