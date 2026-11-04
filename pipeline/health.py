"""Health check for the live logger (task 2).

Runs on the server every 5 minutes (deploy/train-health.timer), so a stopped logger
alerts within 20 min: its last flush is <= 5 min before the stop, stale after 15.

Checks:
  - the newest board file is at most 15 min old (logger running and flushing)
  - at least 5 GB free on the data disk
  - the share of DVS rows with a change code over the last 6 hours is > 10% (catches a
    parser regression like the Element.iter() bug; normal is ~30%). Skipped at night,
    when traffic is thin.

Exit code 0 = healthy, 1 = at least one problem. Problems are printed one per line,
and, if HEALTH_NOTIFY_CMD is set, piped to that command (e.g. a curl to an alert
service). The alert channel is configured on the server in /etc/default/train-health.

Usage:
    python -m pipeline.health
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from .live import LIVE

MAX_AGE_MIN = 15
MIN_FREE_GB = 5
MIN_CHANGE_SHARE = 0.10
CHANGE_WINDOW_H = 6
DAY_HOURS = range(7, 23)        # local hours when the change-code check runs


def flushed_at(path) -> datetime:
    """board/date=YYYY-MM-DD/board-HHMMSS.parquet -> flush time (UTC). File mtimes are not
    used: reparse_board rewrites old files."""
    day = path.parent.name.removeprefix("date=")
    return datetime.strptime(day + path.stem.removeprefix("board-"), "%Y-%m-%d%H%M%S").replace(tzinfo=timezone.utc)


def board_files(live=LIVE) -> list:
    return sorted((live / "board").glob("date=*/board-*.parquet"), key=flushed_at)


def newest_board_age_min(live=LIVE, now=None) -> float | None:
    files = board_files(live)
    if not files:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - flushed_at(files[-1])).total_seconds() / 60


def change_share(live=LIVE, hours=CHANGE_WINDOW_H, now=None) -> tuple[float, int] | None:
    import duckdb
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=hours)
    files = [str(f) for f in board_files(live) if flushed_at(f) >= cutoff]
    if not files:
        return None
    share, n = duckdb.sql(f"""
        SELECT avg((len(change_codes) > 0)::int), count(*)
        FROM read_parquet({files!r}) WHERE kind = 'dvs'""").fetchone()
    return (share or 0.0), n


def check(live=LIVE, now=None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    problems = []
    age = newest_board_age_min(live, now)
    if age is None:
        problems.append("no board files at all")
    elif age > MAX_AGE_MIN:
        problems.append(f"newest board file is {age:.0f} min old (limit {MAX_AGE_MIN})")
    free_gb = shutil.disk_usage(live if live.exists() else "/").free / 1e9
    if free_gb < MIN_FREE_GB:
        problems.append(f"only {free_gb:.1f} GB free (limit {MIN_FREE_GB})")
    local_hour = (now + timedelta(hours=1)).hour     # CET is close enough for a day/night gate
    if local_hour in DAY_HOURS:
        cs = change_share(live, now=now)
        if cs and cs[1] >= 1000 and cs[0] <= MIN_CHANGE_SHARE:
            problems.append(f"only {100 * cs[0]:.1f}% of {cs[1]} DVS rows have a change code "
                            f"in the last {CHANGE_WINDOW_H} h (expect ~30%; parser broken?)")
    return problems


STATE = LIVE / ".health_state.json"
REMIND_HOURS = 6


def notify(msg: str) -> None:
    cmd = os.environ.get("HEALTH_NOTIFY_CMD")
    if cmd:
        subprocess.run(cmd, shell=True, input=msg, text=True, timeout=30)


def decide(problems: list, state: dict, now: datetime) -> tuple[str | None, dict]:
    """What to send, given the last state: one alert when it breaks, a reminder every
    REMIND_HOURS while it stays broken, one notice when it recovers. Never every run."""
    was_bad, last = state.get("bad", False), state.get("last_alert")
    if problems:
        due = not was_bad or last is None or \
            now - datetime.fromisoformat(last) >= timedelta(hours=REMIND_HOURS)
        msg = ("train-logger PROBLEM: " if not was_bad else "train-logger STILL BROKEN: ") + "; ".join(problems)
        return (msg if due else None), {"bad": True, "last_alert": now.isoformat() if due else last}
    if was_bad:
        return "train-logger recovered: all checks pass again", {"bad": False, "last_alert": None}
    return None, {"bad": False, "last_alert": None}


def main(argv=None):
    import json
    if argv is None:
        argv = sys.argv[1:]
    if "--test" in argv:
        notify("train-logger health: test alert. If you got this, alerts work.")
        print("test alert sent")
        return 0
    now = datetime.now(timezone.utc)
    problems = check(now=now)
    print("ok" if not problems else "problems: " + "; ".join(problems))
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        state = {}
    msg, state = decide(problems, state, now)
    if msg:
        notify(msg)
        print("notified:", msg)
    try:
        STATE.write_text(json.dumps(state))
    except OSError:
        pass
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
