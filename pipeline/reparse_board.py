"""Rebuild data/live/board/ from data/live/raw/ with the current parser.

The raw layer is the only copy of the live feed and is never modified here. Every raw
file raw/date=D/raw-HHMMSS.parquet maps to board/date=D/board-HHMMSS.parquet (the
logger flushes both with the same stamp), so the rebuild is file-for-file:

  1. parse every raw file into a staging folder (data/live/.board-rebuild/)
  2. swap each staged file into board/ with os.replace (atomic per file)

Safe to run while the logger is writing (it only ever adds new stamps, and a stamp it
adds after the logger runs the fixed parser is already correct). Idempotent: running it
twice gives the same board/. Board files without a raw counterpart are left alone.

Usage:
    python -m pipeline.reparse_board              # all dates
    python -m pipeline.reparse_board --date 2026-11-04
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from .live import BOARD_SCHEMA, ENVELOPES, LIVE, parse_board


def parse_raw_file(path) -> tuple[pa.Table, int]:
    raw = pq.read_table(path).to_pylist()
    rows, errors = [], 0
    for r in raw:
        kind = ENVELOPES.get(r["envelope"])
        if kind not in ("dvs", "das"):
            continue
        try:
            rows.append(parse_board(kind, r["xml"], r["received_at"]))
        except Exception:
            errors += 1
    return pa.Table.from_pylist(rows, schema=BOARD_SCHEMA), errors


def rebuild(dates=None, live=LIVE) -> dict:
    raw_dir, board_dir, stage = live / "raw", live / "board", live / ".board-rebuild"
    shutil.rmtree(stage, ignore_errors=True)
    files = sorted(raw_dir.glob("date=*/raw-*.parquet"))
    if dates:
        files = [f for f in files if f.parent.name.removeprefix("date=") in dates]
    staged, rows, errors = [], 0, 0
    for f in files:
        table, err = parse_raw_file(f)
        rows, errors = rows + table.num_rows, errors + err
        if table.num_rows == 0:
            continue
        out = stage / f.parent.name / f.name.replace("raw-", "board-", 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out, compression="zstd", compression_level=9)
        staged.append(out)
    for out in staged:
        target = board_dir / out.parent.name / out.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(out, target)
    shutil.rmtree(stage, ignore_errors=True)
    return {"raw_files": len(files), "board_files": len(staged), "rows": rows, "parse_errors": errors}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", action="append", help="YYYY-MM-DD; repeat for several")
    args = ap.parse_args(argv)
    for d in args.date or []:
        datetime.strptime(d, "%Y-%m-%d")
    print(rebuild(set(args.date) if args.date else None))


if __name__ == "__main__":
    sys.exit(main())
