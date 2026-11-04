"""Corridor-trip labels (PREDICTION_MODEL_PLAN.md, Step 1).

One row per (corridor, service_date, scheduled departure time at the origin).
This is the unit the app reasons about: "the 07:41 from Rotterdam to Amsterdam".

Rules:
- Only domestic train products (no Eurostar/ICE/night trains, no replacement buses).
  A train replaced by a bus counts as cancelled, which is what the commuter feels.
- Replacement train numbers (2216 -> 302216) are collapsed into one trip. The trip
  is cancelled only if every variant was cancelled; the delay is the best variant's.
- Delays are RdT's whole minutes, clipped at 0 (no early running in the data).

Writes data/clean/corridor_trips.parquet and the DuckDB view `corridor_trips`.

Usage:
    python -m pipeline.labels
"""
import sys

import duckdb
import pandas as pd

from . import config
from .corridors import CORRIDORS

DOMESTIC_TYPES = ("Sprinter", "Intercity", "Intercity direct", "Eurocity Direct")

SQL = """
COPY (
WITH corridors(corridor, orig, dest) AS (SELECT * FROM corridor_df),
s AS (
    SELECT * FROM stops
    WHERE company IN ('NS', 'NS Int') AND service_type IN {types}
),
variants AS (
    SELECT
        c.corridor, c.orig, c.dest,
        o.service_id, o.service_date, o.train_number, o.service_type,
        o.dep_planned,
        d.arr_planned,
        o.dep_delay,
        d.arr_delay,
        coalesce(o.dep_cancelled, false) OR coalesce(d.arr_cancelled, false) AS cancelled
    FROM corridors c
    JOIN s o ON o.station_code = c.orig AND o.dep_planned IS NOT NULL
    JOIN s d ON d.service_id   = o.service_id
            AND d.station_code = c.dest
            AND d.stop_seq     > o.stop_seq
            AND d.arr_planned IS NOT NULL
)
SELECT
    corridor, any_value(orig) AS orig, any_value(dest) AS dest,
    service_date,
    dep_planned,
    strftime(dep_planned, '%H:%M')                      AS dep_time,
    min(arr_planned)                                    AS arr_planned,
    min(train_number)                                   AS train_number,
    arg_min(service_id, train_number)                   AS service_id,
    arg_min(service_type, train_number)                 AS service_type,
    count(*)                                            AS n_variants,
    bool_and(cancelled)                                 AS cancelled,
    min(dep_delay) FILTER (WHERE NOT cancelled)         AS dep_delay,
    min(arr_delay) FILTER (WHERE NOT cancelled)         AS arr_delay,
    CASE isodow(service_date) WHEN 6 THEN 'sat' WHEN 7 THEN 'sun' ELSE 'wkd' END AS daytype,
    hour(dep_planned)                                   AS dep_hour
FROM variants
GROUP BY corridor, service_date, dep_planned
ORDER BY corridor, dep_planned
) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
"""


def main(argv=None):
    con = duckdb.connect(str(config.DB_PATH))
    corridor_df = pd.DataFrame(CORRIDORS, columns=["corridor", "orig", "dest"])
    con.register("corridor_df", corridor_df)
    out = config.CLEAN / "corridor_trips.parquet"
    con.execute(SQL.format(types=DOMESTIC_TYPES, out=out))
    con.execute(f"CREATE OR REPLACE VIEW corridor_trips AS SELECT * FROM read_parquet('{out}')")
    print(con.sql("""
        SELECT corridor, count(*) trips, min(service_date) AS first_day, max(service_date) AS last_day,
               round(avg(cancelled::int) * 100, 1) pct_cancelled,
               round(avg(arr_delay), 2) mean_delay
        FROM corridor_trips GROUP BY corridor ORDER BY corridor
    """).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    sys.exit(main())
