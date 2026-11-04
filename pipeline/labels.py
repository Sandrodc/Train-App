"""Corridor-trip labels (PREDICTION_MODEL_PLAN.md, Step 1).

One row per (corridor, service_date, scheduled departure time at the origin).
This is the unit the app reasons about: "the 07:41 from Rotterdam to Amsterdam".

Rules:
- Only domestic train products (no Eurostar/ICE/night trains, no replacement buses).
  A train replaced by a bus counts as cancelled, which is what the commuter feels.
- Replacement train numbers (2216 -> 302216) are collapsed into one trip. The trip
  is cancelled only if every variant was cancelled; the delay is the best variant's.
- Delays are RdT's whole minutes, clipped at 0 (no early running in the data).

Passenger delay (what the commuter feels, PREDICTION_MODEL_PLAN review, task 3):
- pax_arr_actual: the earliest actual arrival among trips that ran on the same corridor
  and day, that actually departed in [this trip's planned departure, +180 min] and are
  planned to arrive no earlier than this trip. For a trip that ran this is usually its
  own arrival; for a cancelled trip it is the next train you could actually take.
- pax_delay = pax_arr_actual - arr_planned, in minutes. NULL when nothing ran within
  180 min.

"Not running today": a base timetable per corridor, timetable type (wkd/sat/sun, public
holidays separate) and dep_time = the slots present on at least 5 of the previous 8
days of that type. A base slot with no trip within +-3 min on a day (planned works, reduced timetable) gets a
synthetic row with not_running = true, cancelled = true, and pax_delay computed the same
way. Synthetic rows go to a separate file so existing features and models keep the label
set they were tuned on; `corridor_trips_all` is the union. Public holidays get no
synthetic rows.

Writes data/clean/corridor_trips.parquet (+ pax columns, not_running = false),
data/clean/corridor_not_running.parquet and the DuckDB views `corridor_trips`,
`corridor_not_running`, `corridor_trips_all`.

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


BASE_WINDOW, BASE_MIN = 8, 5          # slot is "base" if present on >= 5 of previous 8 days
MATCH_MIN = 3                         # a trip within +-3 min of a base slot counts as that slot
PAX_WINDOW_MIN = 180

NOT_RUNNING_SQL = """
COPY (
WITH t AS (SELECT * FROM read_parquet('{trips}')),
cal AS (SELECT date::DATE AS service_date, public_holiday FROM read_parquet('{calendar}')),
dates AS (
    SELECT DISTINCT t.service_date,
           CASE WHEN coalesce(c.public_holiday, 0) = 1 THEN 'hol' ELSE t.daytype END AS tt_type
    FROM t LEFT JOIN cal c USING (service_date)
),
days AS (   -- every corridor on every date with data, so whole-day closures count too
    SELECT c.corridor, c.orig, c.dest, d.service_date, d.tt_type,
           row_number() OVER (PARTITION BY c.corridor, d.tt_type ORDER BY d.service_date) AS k
    FROM (SELECT DISTINCT corridor, orig, dest FROM t) c CROSS JOIN dates d
),
present AS (
    SELECT t.corridor, d.tt_type, d.k, t.dep_time,
           t.dep_planned - t.service_date::TIMESTAMP AS dep_offset,
           t.arr_planned - t.dep_planned             AS run_time
    FROM t JOIN days d USING (corridor, service_date)
),
base AS (
    SELECT d.corridor, any_value(d.orig) AS orig, any_value(d.dest) AS dest,
           d.service_date, d.tt_type, p.dep_time,
           min(p.dep_offset) AS dep_offset, median(p.run_time) AS run_time
    FROM days d JOIN present p
      ON p.corridor = d.corridor AND p.tt_type = d.tt_type
     AND p.k BETWEEN d.k - {window} AND d.k - 1
    WHERE d.tt_type <> 'hol'
    GROUP BY d.corridor, d.service_date, d.tt_type, p.dep_time
    HAVING count(DISTINCT p.k) >= {min_days}
)
SELECT b.corridor, b.orig, b.dest, b.service_date,
       b.service_date::TIMESTAMP + b.dep_offset              AS dep_planned,
       b.dep_time,
       b.service_date::TIMESTAMP + b.dep_offset + b.run_time AS arr_planned,
       NULL::INTEGER AS train_number, NULL::BIGINT AS service_id, NULL::VARCHAR AS service_type,
       0::BIGINT AS n_variants, true AS cancelled,
       NULL::INTEGER AS dep_delay, NULL::INTEGER AS arr_delay,
       CASE isodow(b.service_date) WHEN 6 THEN 'sat' WHEN 7 THEN 'sun' ELSE 'wkd' END AS daytype,
       hour(b.service_date::TIMESTAMP + b.dep_offset)          AS dep_hour,
       true AS not_running
FROM base b
ANTI JOIN t ON t.corridor = b.corridor AND t.service_date = b.service_date
           AND abs(date_diff('minute', b.service_date::TIMESTAMP + b.dep_offset, t.dep_planned)) <= {match_min}
) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
"""

PAX_SQL = """
CREATE OR REPLACE TEMP TABLE pax AS
WITH ran AS (
    SELECT corridor, service_date,
           dep_planned + to_minutes(dep_delay::BIGINT) AS dep_actual,
           arr_planned,
           arr_planned + to_minutes(arr_delay::BIGINT) AS arr_actual
    FROM read_parquet('{trips}')
    WHERE NOT cancelled AND dep_delay IS NOT NULL AND arr_delay IS NOT NULL
),
targets AS (
    SELECT corridor, service_date, dep_planned, arr_planned FROM read_parquet('{trips}')
    UNION ALL
    SELECT corridor, service_date, dep_planned, arr_planned FROM read_parquet('{not_running}')
)
SELECT x.corridor, x.service_date, x.dep_planned,
       min(r.arr_actual) AS pax_arr_actual,
       date_diff('minute', x.arr_planned, min(r.arr_actual)) AS pax_delay
FROM targets x
LEFT JOIN ran r
  ON r.corridor = x.corridor AND r.service_date = x.service_date
 AND r.dep_actual >= x.dep_planned
 AND r.dep_actual <= x.dep_planned + INTERVAL {window} MINUTE
 AND r.arr_planned >= x.arr_planned
GROUP BY x.corridor, x.service_date, x.dep_planned, x.arr_planned
"""


def passenger_labels(con, out):
    """Add pax_arr_actual / pax_delay / not_running to `out`; write the not-running rows."""
    nr = config.CLEAN / "corridor_not_running.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    con.execute(NOT_RUNNING_SQL.format(trips=out, calendar=config.CLEAN / "calendar.parquet",
                                       window=BASE_WINDOW, min_days=BASE_MIN, match_min=MATCH_MIN, out=nr))
    con.execute(PAX_SQL.format(trips=out, not_running=nr, window=PAX_WINDOW_MIN))
    keys = "USING (corridor, service_date, dep_planned)"
    con.execute(f"""COPY (SELECT t.*, false AS not_running, p.pax_arr_actual, p.pax_delay
                          FROM read_parquet('{out}') t LEFT JOIN pax p {keys}
                          ORDER BY corridor, dep_planned)
                    TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)""")
    tmp.replace(out)
    tmp_nr = nr.with_suffix(".tmp.parquet")
    con.execute(f"""COPY (SELECT n.*, p.pax_arr_actual, p.pax_delay
                          FROM read_parquet('{nr}') n LEFT JOIN pax p {keys}
                          ORDER BY corridor, dep_planned)
                    TO '{tmp_nr}' (FORMAT parquet, COMPRESSION zstd)""")
    tmp_nr.replace(nr)
    con.execute(f"CREATE OR REPLACE VIEW corridor_not_running AS SELECT * FROM read_parquet('{nr}')")
    con.execute(f"""CREATE OR REPLACE VIEW corridor_trips_all AS
                    SELECT * FROM read_parquet('{out}') UNION ALL BY NAME SELECT * FROM read_parquet('{nr}')""")


def main(argv=None):
    con = duckdb.connect(str(config.DB_PATH))
    corridor_df = pd.DataFrame(CORRIDORS, columns=["corridor", "orig", "dest"])
    con.register("corridor_df", corridor_df)
    out = config.CLEAN / "corridor_trips.parquet"
    con.execute(SQL.format(types=DOMESTIC_TYPES, out=out))
    passenger_labels(con, out)
    con.execute(f"CREATE OR REPLACE VIEW corridor_trips AS SELECT * FROM read_parquet('{out}')")
    print(con.sql("""
        SELECT corridor, count(*) trips, min(service_date) AS first_day, max(service_date) AS last_day,
               round(avg(cancelled::int) * 100, 1) pct_cancelled,
               round(avg(arr_delay), 2) mean_delay,
               round(avg((pax_delay > 5)::int) * 100, 1) pct_pax_over_5,
               (SELECT count(*) FROM corridor_not_running n WHERE n.corridor = t.corridor) AS not_running
        FROM corridor_trips t GROUP BY corridor ORDER BY corridor
    """).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    sys.exit(main())
