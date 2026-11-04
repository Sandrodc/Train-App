"""Download Rijden de Treinen open data into data/raw/.

Usage:
    python -m pipeline.download                      # default window from config
    python -m pipeline.download --from 2025-01 --to 2025-03
    python -m pipeline.download --skip-services      # only stations + disruptions

Files that already exist are skipped, so the script is safe to re-run.
"""
import argparse
import sys
from pathlib import Path

import requests

from . import config


def months(first: str, last: str):
    y, m = map(int, first.split("-"))
    ly, lm = map(int, last.split("-"))
    while (y, m) <= (ly, lm):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip  {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code == 404:
            print(f"  404   {url}")
            return
        r.raise_for_status()
        size = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                size += len(chunk)
    tmp.rename(dest)
    print(f"  got   {dest.name} ({size / 1e6:.1f} MB)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="first", default=config.SERVICES_FIRST_MONTH)
    p.add_argument("--to", dest="last", default=config.SERVICES_LAST_MONTH)
    p.add_argument("--skip-services", action="store_true")
    args = p.parse_args(argv)

    print("stations")
    fetch(config.STATIONS_URL, config.RAW / "stations" / "stations.csv")

    print("disruptions")
    for year in range(config.DISRUPTIONS_FIRST_YEAR, config.DISRUPTIONS_LAST_YEAR + 1):
        fetch(config.DISRUPTIONS_URL.format(year=year),
              config.RAW / "disruptions" / f"disruptions-{year}.csv")

    if not args.skip_services:
        print("services")
        years_done = set()
        for ym in months(args.first, args.last):
            year = int(ym[:4])
            if year < config.SERVICES_MONTHLY_FROM:     # yearly file only; build_db splits it
                if year not in years_done:
                    fetch(config.SERVICES_YEAR_URL.format(year=year),
                          config.RAW / "services" / f"services-{year}.csv.gz")
                    years_done.add(year)
                continue
            fetch(config.SERVICES_URL.format(ym=ym),
                  config.RAW / "services" / f"services-{ym}.csv.gz")


if __name__ == "__main__":
    sys.exit(main())
