"""Connection miss rate per transfer (PREDICTION_MODEL_PLAN.md, Step 5.4).

For each feeder train orig -> via, the planned connection is the first onward train
via -> dest that departs at least `walk` minutes after the feeder's scheduled
arrival (what a journey planner would offer). Feeders that continue to `dest`
themselves are direct trips and skipped.

A connection is missed when both trains ran and
    feeder actual arrival + walk > onward actual departure.
Held connections show up as onward departure delay, so they count as made.

A journey is broken when the connection is missed or either leg is cancelled, which is
what the commuter feels. Reported per planned connection (all of them):
  pct_missed         missed, among connections where both legs ran
  pct_cancelled_leg  feeder or onward cancelled
  pct_broken         missed or a cancelled leg
  extra_min_*        when broken: minutes from the planned onward departure to the actual
                     departure of the next onward train that ran and that you could reach
                     (if the feeder was cancelled: via the next feeder that ran). Floored at
                     0: a later but faster feeder can reach an earlier onward train.

Writes data/reports/transfers_by_buffer.csv and transfers_by_slot.csv.

Usage:
    python -m pipeline.transfers --days 180
"""
import argparse
import sys

import duckdb
import pandas as pd

from . import config
from .labels import DOMESTIC_TYPES

# Placeholder transfers until interviews; walk = minutes needed to change platforms.
TRANSFERS = [
    ("Den Haag C > Utrecht C > Amersfoort C", "GVC", "UT", "AMF", 3),
    ("Rotterdam C > Utrecht C > Arnhem C", "RTD", "UT", "AH", 3),
    ("Eindhoven C > Utrecht C > Amersfoort C", "EHV", "UT", "AMF", 3),
]

SQL = """
WITH t(transfer, orig, via, dest, walk) AS (SELECT * FROM transfer_df),
s AS (
    SELECT * FROM stops
    WHERE company IN ('NS', 'NS Int') AND service_type IN {types}
      AND service_date > (SELECT max(service_date) FROM stops) - INTERVAL {days} DAY
),
feeder_v AS (   -- orig -> via, not continuing to dest
    SELECT t.transfer, t.walk, o.service_date, o.dep_planned AS f_dep, v.arr_planned AS f_arr,
           v.arr_delay, coalesce(o.dep_cancelled, false) OR coalesce(v.arr_cancelled, false) AS cancelled
    FROM t
    JOIN s o ON o.station_code = t.orig AND o.dep_planned IS NOT NULL
    JOIN s v ON v.service_id = o.service_id AND v.station_code = t.via AND v.stop_seq > o.stop_seq
            AND v.arr_planned IS NOT NULL
    WHERE NOT EXISTS (SELECT 1 FROM s d WHERE d.service_id = o.service_id
                      AND d.station_code = t.dest AND d.stop_seq > v.stop_seq)
),
feeder AS (
    SELECT transfer, any_value(walk) AS walk, service_date, f_dep, min(f_arr) AS f_arr,
           bool_and(cancelled) AS f_cancelled, min(arr_delay) FILTER (WHERE NOT cancelled) AS f_delay
    FROM feeder_v GROUP BY transfer, service_date, f_dep
),
onward_v AS (   -- via -> dest
    SELECT t.transfer, v.service_date, v.dep_planned AS o_dep, v.dep_delay,
           coalesce(v.dep_cancelled, false) OR coalesce(d.arr_cancelled, false) AS cancelled
    FROM t
    JOIN s v ON v.station_code = t.via AND v.dep_planned IS NOT NULL
    JOIN s d ON d.service_id = v.service_id AND d.station_code = t.dest AND d.stop_seq > v.stop_seq
),
onward AS (
    SELECT transfer, service_date, o_dep, bool_and(cancelled) AS o_cancelled,
           min(dep_delay) FILTER (WHERE NOT cancelled) AS o_delay
    FROM onward_v GROUP BY transfer, service_date, o_dep
),
pairs AS (
    SELECT f.*, o.o_dep, o.o_cancelled, o.o_delay,
           date_diff('minute', f.f_arr, o.o_dep) AS buffer
    FROM feeder f
    ASOF JOIN onward o
      ON o.transfer = f.transfer AND o.service_date = f.service_date
     AND o.o_dep >= f.f_arr + to_minutes(f.walk)
),
judged AS (
    SELECT *,
        CASE WHEN f_cancelled OR o_cancelled THEN NULL
             ELSE f_delay + walk > buffer + o_delay END AS missed,
        isodow(service_date) <= 5 AS weekday
    FROM pairs
    WHERE buffer <= 30            -- ignore late-evening gaps with no sensible connection
),
ready AS (      -- when you are on the platform at `via`, ready to board
    SELECT j.transfer, j.service_date, j.f_dep,
           CASE WHEN NOT j.f_cancelled THEN j.f_arr + to_minutes(j.f_delay + j.walk)
                ELSE min(f2.f_arr + to_minutes(f2.f_delay + j.walk)) END AS ready_at
    FROM judged j
    LEFT JOIN feeder f2 ON j.f_cancelled AND f2.transfer = j.transfer
         AND f2.service_date = j.service_date AND f2.f_dep >= j.f_dep AND NOT f2.f_cancelled
    WHERE coalesce(j.missed, true)
    GROUP BY j.transfer, j.service_date, j.f_dep, j.f_cancelled, j.f_arr, j.f_delay, j.walk
),
fallback AS (
    SELECT r.transfer, r.service_date, r.f_dep,
           min(o.o_dep + to_minutes(o.o_delay)) AS next_onward_actual
    FROM ready r
    JOIN onward o ON o.transfer = r.transfer AND o.service_date = r.service_date
     AND NOT o.o_cancelled AND o.o_dep + to_minutes(o.o_delay) >= r.ready_at
    GROUP BY ALL
)
SELECT j.*,
    coalesce(j.missed, false) OR j.f_cancelled OR j.o_cancelled AS broken,
    CASE WHEN coalesce(j.missed, false) OR j.f_cancelled OR j.o_cancelled
         THEN greatest(0, date_diff('minute', j.o_dep, fb.next_onward_actual)) END AS extra_min
FROM judged j
LEFT JOIN fallback fb USING (transfer, service_date, f_dep)
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args(argv)

    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    con.register("transfer_df", pd.DataFrame(TRANSFERS, columns=["transfer", "orig", "via", "dest", "walk"]))
    con.execute("CREATE TEMP TABLE pairs AS " + SQL.format(types=DOMESTIC_TYPES, days=args.days))

    by_buffer = con.sql("""
        SELECT transfer, buffer AS planned_buffer_min, count(*) AS n,
               round(avg(missed::int) * 100, 1)        AS pct_missed,
               round(avg((f_cancelled OR o_cancelled)::int) * 100, 1) AS pct_cancelled_leg,
               round(avg(broken::int) * 100, 1) AS pct_broken,
               median(extra_min) AS extra_min_median, round(avg(extra_min), 1) AS extra_min_mean
        FROM pairs WHERE weekday GROUP BY ALL HAVING count(*) >= 30 ORDER BY transfer, planned_buffer_min
    """).df()
    by_slot = con.sql("""
        SELECT transfer, strftime(f_dep, '%H:%M') AS feeder_dep, strftime(o_dep, '%H:%M') AS onward_dep,
               mode(buffer) AS buffer, count(*) AS n,
               round(avg(missed::int) * 100, 1) AS pct_missed,
               round(avg((f_cancelled OR o_cancelled)::int) * 100, 1) AS pct_cancelled_leg,
               round(avg(broken::int) * 100, 1) AS pct_broken,
               median(extra_min) AS extra_min_median
        FROM pairs WHERE weekday GROUP BY ALL HAVING count(*) >= 20 ORDER BY transfer, feeder_dep
    """).df()
    overall = con.sql("""
        SELECT transfer, count(*) AS n, round(avg(missed::int) * 100, 1) AS pct_missed,
               round(avg((f_cancelled OR o_cancelled)::int) * 100, 1) AS pct_cancelled_leg,
               round(avg(broken::int) * 100, 1) AS pct_broken,
               median(extra_min) AS extra_min_median,
               median(buffer) AS median_buffer
        FROM pairs WHERE weekday GROUP BY transfer ORDER BY transfer
    """).df()

    out = config.DATA / "reports"
    out.mkdir(parents=True, exist_ok=True)
    by_buffer.to_csv(out / "transfers_by_buffer.csv", index=False)
    by_slot.to_csv(out / "transfers_by_slot.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"last {args.days} days, weekdays\n")
    print(overall.to_string(index=False), "\n")
    print(by_buffer.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
