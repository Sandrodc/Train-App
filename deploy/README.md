# Running the live logger on a server

The logger (`python -m pipeline.live run`) keeps one connection to the NDOV feed and
writes Parquet to `data/live/` every 5 minutes. It has to run 24/7, because gaps mean
missing NS predictions for those hours.

**Resources:** ~1% CPU, <150 MB RAM, ~0.2 GB/day disk (≈5–6 GB/month). The smallest
cloud server is plenty.

## Step 1: create the server (once, ~10 minutes, in your browser)

1. Sign up at [Hetzner Cloud](https://console.hetzner.cloud/) (German provider, EU
   data centres, pay per hour, cancel any time). They may ask for ID or a card check.
2. Create a project (any name, e.g. `train-app`), then **Add Server**:
   - **Location:** any EU one (Falkenstein, Nuremberg or Helsinki).
   - **Image:** Ubuntu 24.04.
   - **Type:** the smallest *shared vCPU* plan (x86 or Arm both work; the logger is tiny).
   - **Networking:** keep public IPv4 on (the feed needs it).
   - **SSH key:** click *Add SSH key* and paste the one-line contents of
     `~/.ssh/id_ed25519.pub` from this Mac. Run `cat ~/.ssh/id_ed25519.pub` to see it.
     This lets this Mac log in without a password; never share the file *without* `.pub`.
   - Leave volumes, firewalls, backups and cloud-config empty.
3. Click **Create & Buy now**. After ~30 s you'll see the server's **IPv4 address**.

## Step 2: install the logger (one command, from this folder on your Mac)

```bash
./deploy/setup_server.sh <server-ip>
```

It installs Python, turns on a firewall (only SSH allowed in) and automatic
security updates, copies the code (not the data), and starts the logger as a
service that restarts by itself after crashes and reboots. Re-run it after code
changes to update the server.

**Then stop the logger on your Mac.** The free feed allows one connection per
stream, so never run two loggers at once.

## Day to day

```bash
./deploy/server_status.sh <server-ip>   # running? what's logged? disk left?
./deploy/pull_data.sh <server-ip>       # copy new data to this Mac for analysis
ssh root@<server-ip>                    # log in (type `exit` to leave)
```

On the server: `journalctl -u train-logger -f` shows the live log (Ctrl-C to stop
watching), and `systemctl restart train-logger` restarts it.

## Feed terms

NDOV Loket best-effort feeds are CC0. Fair use: **max one connection per stream per
consumer**. Abuse gets the IP blocked. For a production service with guaranteed
availability, sign the NDOV user agreement for the premium (ACC ICT) network.
