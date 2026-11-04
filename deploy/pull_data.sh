#!/usr/bin/env bash
# Copy the logged data from the server to this Mac (only new files are transferred).
#   ./deploy/pull_data.sh <server-ip>
set -euo pipefail
IP="${1:?usage: ./deploy/pull_data.sh <server-ip>}"
cd "$(dirname "$0")/.."
mkdir -p data/live
rsync -az --exclude '.*.tmp' "root@$IP:/opt/train-app/data/live/" data/live/
