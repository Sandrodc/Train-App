"""NS benchmark from the live board log (PREDICTION_MODEL_PLAN.md, Steps 4 and 9).

For every corridor trip seen on the live feed, reconstructs what NS showed at
`as_of` (T-60 / T-15 before scheduled departure):
  ns_delay_min   NS's predicted arrival delay at the destination. Taken from the
                 destination's arrival board (DAS) if the train is listed there yet,
                 otherwise the departure delay at the origin (DVS): NS doesn't
                 predict recovery, so that is what its planner shows for arrival.
  ns_announced   NS already shows "Rijdt niet" (code 32) at origin or destination,
                 or "terminates early" (34) at the origin.
and compares it with the outcome. Until RdT publishes the month, the outcome is
taken from the feed itself: the last board message for that stop.

Usage:
    python -m pipeline.ns_benchmark                # all logged days
    python -m pipeline.ns_benchmark --min-age 3    # only trips that arrived >3 h ago
"""
import argparse
import sys

import duckdb
import numpy as np
import pandas as pd

from . import baselines, config
from .corridors import CORRIDORS

BOARD = config.DATA / "live" / "board"
HORIZONS = {"t60": 60, "t15": 15}

SQL = """
WITH b AS (
    SELECT kind, train_number, rit_date, station_code, delay_s, cancelled, terminates_early,
           timezone('Europe/Amsterdam', received_at)  AS seen,
           timezone('Europe/Amsterdam', planned_time) AS planned
    FROM read_parquet('{board}/*/*.parquet', hive_partitioning=true)
    WHERE operator IN ('NS', 'NS International') OR operator IS NULL
),
corr(corridor, orig, dest) AS (SELECT * FROM corridor_df),
final AS (   -- last word per stop = outcome (until RdT has the month)
    SELECT kind, train_number, rit_date, station_code,
           arg_max(delay_s, seen) AS delay_s, arg_max(cancelled, seen) AS cancelled,
           arg_max(terminates_early, seen) AS terminates_early,
           min(planned) AS planned, max(seen) AS last_seen
    FROM b GROUP BY ALL
),
trips AS (
    SELECT c.corridor, c.orig, c.dest, o.train_number, o.rit_date AS service_date,
           o.planned AS dep_planned, d.planned AS arr_planned, d.last_seen,
           o.cancelled OR d.cancelled OR o.terminates_early AS cancelled,
           CAST(floor(d.delay_s / 60) AS INTEGER) AS arr_delay
    FROM corr c
    JOIN final o ON o.kind = 'dvs' AND o.station_code = c.orig
    JOIN final d ON d.kind = 'das' AND d.station_code = c.dest
                AND d.train_number = o.train_number AND d.rit_date = o.rit_date
                AND d.planned > o.planned
),
q AS (SELECT *, dep_planned - to_minutes({h}) AS as_of FROM trips)
SELECT q.*,
       (od.seen IS NOT NULL OR dd.seen IS NOT NULL)                AS ns_seen,
       coalesce(dd.delay_s, od.delay_s) / 60.0                      AS ns_delay_min,
       coalesce(od.cancelled, false) OR coalesce(dd.cancelled, false)
           OR coalesce(od.terminates_early, false)                  AS ns_announced
FROM q
ASOF LEFT JOIN b od ON od.kind = 'dvs' AND od.station_code = q.orig AND od.train_number = q.train_number
                   AND od.rit_date = q.service_date AND od.seen <= q.as_of
ASOF LEFT JOIN b dd ON dd.kind = 'das' AND dd.station_code = q.dest AND dd.train_number = q.train_number
                   AND dd.rit_date = q.service_date AND dd.seen <= q.as_of
"""


def ns_trips(horizon_min: int, min_age_h: float) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("corridor_df", pd.DataFrame(CORRIDORS, columns=["corridor", "orig", "dest"]))
    df = con.execute(SQL.format(board=BOARD, h=horizon_min)).df()
    # Only trips whose final word is settled: we must have been logging from well
    # before as_of, and the train must have arrived a while ago.
    first_log = con.execute(f"SELECT min(timezone('Europe/Amsterdam', received_at)) "
                            f"FROM read_parquet('{BOARD}/*/*.parquet')").fetchone()[0]
    now_local = pd.Timestamp.now(tz="Europe/Amsterdam").tz_localize(None)
    df = df[(df["as_of"] >= first_log + pd.Timedelta(minutes=30)) &
            (df["arr_planned"] <= now_local - pd.Timedelta(hours=min_age_h))]
    df["service_date"] = pd.to_datetime(df["service_date"])
    return df.reset_index(drop=True)


def summarise(df: pd.DataFrame, hist: pd.DataFrame) -> dict:
    ran = ~df["cancelled"].values
    y = df["arr_delay"].values[ran].astype(float)
    ns = np.floor(df["ns_delay_min"].fillna(0).values[ran])
    out = {
        "trips": len(df),
        "ns_listed_at_as_of": df["ns_seen"].mean(),
        "cancelled": int(df["cancelled"].sum()),
        "cancel_announced_by_ns": (df.loc[df["cancelled"], "ns_announced"].mean()
                                   if df["cancelled"].any() else np.nan),
        "ns_false_alarms": int((df["ns_announced"] & ~df["cancelled"]).sum()),
        "mae_ns": np.mean(np.abs(y - ns)),
        "mae_timetable": np.mean(np.abs(y)),
        "ns_exact_within_1min": np.mean(np.abs(y - ns) <= 1),
        "late5_trips": int((y > 5).sum()),
        "late5_foreseen_by_ns": (np.mean(ns[y > 5] > 5) if (y > 5).any() else np.nan),
    }
    if hist is not None:
        out["mae_history_p50"] = np.mean(np.abs(y - hist["q50"].values[ran]))
        out["history_above_p90"] = np.mean(y > hist["q90"].values[ran])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-age", type=float, default=1.0, help="hours since scheduled arrival")
    args = ap.parse_args(argv)
    if not BOARD.exists():
        raise SystemExit("No live data yet. Run: python -m pipeline.live run")

    from .scoreboard import load_trips
    history = load_trips("day")
    rows = {}
    for name, h in HORIZONS.items():
        df = ns_trips(h, args.min_age)
        if df.empty:
            print(f"{name}: no settled trips yet")
            continue
        # History baseline (Step 5) for the same trips: uses only archive data before the day.
        t = df.assign(daytype=np.where(df["service_date"].dt.dayofweek >= 5,
                                       np.where(df["service_date"].dt.dayofweek == 5, "sat", "sun"), "wkd"),
                      dep_time=df["dep_planned"].dt.strftime("%H:%M"),
                      dep_hour=df["dep_planned"].dt.hour)
        hist = baselines.Hierarchical(k=30, window=365, half_life=60)(history, t)
        rows[name] = summarise(df, hist)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(out.to_string(float_format=lambda v: f"{v:.3f}"))
    path = config.DATA / "reports" / "ns_benchmark.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)


if __name__ == "__main__":
    sys.exit(main())
