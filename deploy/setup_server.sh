#!/usr/bin/env bash
# One-command setup of the live logger on a fresh Ubuntu server.
# Run from the project folder on your Mac:   ./deploy/setup_server.sh <server-ip>
# Safe to re-run: it updates the code and restarts the logger.
set -euo pipefail
IP="${1:?usage: ./deploy/setup_server.sh <server-ip>}"
SSH="ssh -o StrictHostKeyChecking=accept-new root@$IP"
cd "$(dirname "$0")/.."

echo "==> 1/5 Installing system packages and basic security (firewall, automatic security updates)"
$SSH 'export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3-venv rsync ufw unattended-upgrades >/dev/null
  ufw allow OpenSSH >/dev/null && ufw --force enable >/dev/null
  timedatectl set-timezone Europe/Amsterdam
  id train >/dev/null 2>&1 || useradd --system --create-home --home-dir /opt/train-app --shell /usr/sbin/nologin train
  mkdir -p /opt/train-app/data/live'

echo "==> 2/5 Copying the code (no data)"
rsync -az --delete --exclude '__pycache__' pipeline deploy requirements-logger.txt "root@$IP:/opt/train-app/"

echo "==> 3/5 Creating the Python environment"
$SSH 'cd /opt/train-app
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements-logger.txt
  chown -R train:train /opt/train-app'

echo "==> 4/5 Installing and (re)starting the logger service and its health-check timer"
$SSH 'cp /opt/train-app/deploy/train-logger.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable --quiet train-logger
  systemctl restart train-logger
  cp /opt/train-app/deploy/train-health.service /opt/train-app/deploy/train-health.timer /etc/systemd/system/
  [ -f /etc/default/train-health ] || echo "# HEALTH_NOTIFY_CMD=curl -s -d @- https://ntfy.sh/<private-topic>" > /etc/default/train-health
  systemctl daemon-reload
  systemctl enable --quiet --now train-health.timer'

echo "==> 5/5 Checking it runs (waiting 20 s)"
sleep 20
$SSH 'systemctl is-active train-logger && journalctl -u train-logger -n 5 --no-pager'
echo
echo "Done. The logger writes its first files after ~5 minutes."
echo "Check on it any time with:  ./deploy/server_status.sh $IP"
