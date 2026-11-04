"""First reliability scores (PREDICTION_MODEL_PLAN.md, Step 5.1).

Reads the corridor-trip labels (pipeline.labels: domestic trains only, replacement
train numbers collapsed) and summarises the arrival delay at the destination
over the last N days of data.

Buckets (arrival delay at destination, whole minutes as reported by RdT):
    on time   0-1 min
    minor     2-5 min
    major     >5 min
    cancelled departure at origin or arrival at destination cancelled

Writes data/reports/reliability_by_departure.csv and _by_corridor.csv.

Usage:
    python -m pipeline.reliability                 # last 90 days, weekdays
    python -m pipeline.reliability --days 180 --all-days
"""
import argparse
import sys

import duckdb
import pandas as pd

from . import config

TRIPS_SQL = """
SELECT * FROM read_parquet('{path}')
WHERE service_date > (SELECT max(service_date) FROM read_parquet('{path}')) - INTERVAL {days} DAY
  AND ({all_days} OR daytype = 'wkd')
"""

SUMMARY_COLS = """
    count(*)                                                        AS n,
    round(avg(CASE WHEN NOT cancelled AND arr_delay <= 1 THEN 1 ELSE 0 END) * 100, 1)  AS pct_on_time,
    round(avg(CASE WHEN NOT cancelled AND arr_delay BETWEEN 2 AND 5 THEN 1 ELSE 0 END) * 100, 1) AS pct_2_5,
    round(avg(CASE WHEN NOT cancelled AND arr_delay > 5 THEN 1 ELSE 0 END) * 100, 1)   AS pct_over_5,
    round(avg(CASE WHEN cancelled THEN 1 ELSE 0 END) * 100, 1)       AS pct_cancelled,
    quantile_cont(arr_delay, 0.5) FILTER (WHERE NOT cancelled)       AS p50_delay,
    quantile_cont(arr_delay, 0.9) FILTER (WHERE NOT cancelled)       AS p90_delay
"""


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--all-days", action="store_true", help="include weekends")
    p.add_argument("--min-n", type=int, default=20, help="min trips per departure row")
    args = p.parse_args(argv)

    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    con.execute("CREATE TEMP VIEW trips AS " + TRIPS_SQL.format(
        path=config.CLEAN / "corridor_trips.parquet", days=args.days, all_days="true" if args.all_days else "false"))

    by_departure = con.execute(f"""
        SELECT corridor, dep_time, service_type,
               string_agg(DISTINCT CAST(train_number AS VARCHAR), '/') AS train_numbers,
               {SUMMARY_COLS}
        FROM trips GROUP BY ALL HAVING count(*) >= {args.min_n}
        ORDER BY corridor, dep_time
    """).df()
    by_corridor = con.execute(f"""
        SELECT corridor, {SUMMARY_COLS} FROM trips GROUP BY corridor ORDER BY pct_on_time DESC
    """).df()
    first, last = con.execute("SELECT min(service_date), max(service_date) FROM trips").fetchone()

    out = config.DATA / "reports"
    out.mkdir(parents=True, exist_ok=True)
    by_departure.to_csv(out / "reliability_by_departure.csv", index=False)
    by_corridor.to_csv(out / "reliability_by_corridor.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"Window: {first} to {last} ({'all days' if args.all_days else 'weekdays'})\n")
    print(by_corridor.to_string(index=False))
    print(f"\nWrote {len(by_departure)} departure rows to {out}")


if __name__ == "__main__":
    sys.exit(main())
