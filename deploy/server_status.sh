#!/usr/bin/env bash
# Is the logger running, what has it logged, how much disk is left?
#   ./deploy/server_status.sh <server-ip>
set -euo pipefail
IP="${1:?usage: ./deploy/server_status.sh <server-ip>}"
ssh root@"$IP" 'echo "service: $(systemctl is-active train-logger)  (since $(systemctl show train-logger -p ActiveEnterTimestamp --value))"
  journalctl -u train-logger -n 3 --no-pager
  cd /opt/train-app && sudo -u train .venv/bin/python -m pipeline.live status
  df -h / | tail -1 | awk "{print \"disk: \" \$4 \" free of \" \$2}"'
