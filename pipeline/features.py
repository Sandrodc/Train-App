"""Point-in-time features per horizon (PREDICTION_MODEL_PLAN.md, Step 6).

One row per corridor trip and horizon. `as_of` is the moment of prediction:
    day     00:00 on the service day           (schedule, history, calendar, weather, planned-works proxy)
    t60     60 min before scheduled departure  (+ live network and own-train state)
    t15     15 min before scheduled departure

Live state is rebuilt from the RdT stop archive. A past stop event is knowable at
`as_of` when its *actual* time (planned + delay) is <= as_of. Cancellations are only
treated as known 5 minutes after the cancelled event's planned time, because the
archive doesn't say when NS decided to cancel. These two rules are the whole
leakage contract; `--check` re-derives a sample of rows by brute force to verify them.

Writes data/clean/features_<horizon>.parquet.

Usage:
    python -m pipeline.features            # build all horizons
    python -m pipeline.features --check    # brute-force leakage check on a sample
"""
import argparse
import sys

import duckdb
import numpy as np
import pandas as pd

from . import baselines, config
from .labels import DOMESTIC_TYPES
from .weather import WEATHER_FEATURES, weather_features

HORIZONS = {"day": None, "t60": 60, "t15": 15}
LIVE_STATIONS = ("RTD", "ASD", "UT", "GVC", "EHV")
CANCEL_KNOWN_AFTER = 5     # minutes after planned time
WINDOW = 30                # minutes of recent departures for delay state
CANCEL_WINDOW = 60

DAY_FEATURES = [
    "corridor_code", "service_type_code", "dep_hour", "dep_minute", "isodow", "travel_min",
    "hist_q50", "hist_q90", "hist_p_cancel",
    "prev_day_cancel_share", "prev_day_mean_delay",
    "sched_ratio",
    "public_holiday", "school_holiday_regions", "days_since_dst", "leaves_season", "month",
]
# Old daily weather (observed, same day: a "perfect forecast" that wasn't knowable day-ahead).
# Kept in the feature files for comparison only.
WX_DAILY = ["wx_gust_max", "wx_wind_max", "wx_rain_max", "wx_rain_hours_max", "wx_tmin", "wx_tmax"]
LIVE_FEATURES = [
    "own_known", "own_last_delay", "own_minutes_since",
    "orig_n", "orig_mean_delay", "orig_late5_share", "orig_cancel_share",
    "dest_n", "dest_mean_delay", "dest_late5_share", "dest_cancel_share",
    "net_n", "net_mean_delay", "net_late5_share", "net_cancel_share",
    "corr_prev_n", "corr_prev_mean_delay", "corr_cancel_share",
]


def features_for(horizon: str, weather: str = "hourly"):
    """weather: 'hourly' (KNMI obs before as_of + day-ahead forecasts), 'daily' (old), or 'none'."""
    wx = {"hourly": WEATHER_FEATURES, "daily": WX_DAILY, "none": []}[weather]
    return DAY_FEATURES + wx + (LIVE_FEATURES if HORIZONS[horizon] else [])


def load(con, name):
    return con.execute(f"SELECT * FROM read_parquet('{config.CLEAN / name}')").df()


# ---------------------------------------------------------------- day-level

def day_features(con) -> pd.DataFrame:
    trips = load(con, "corridor_trips.parquet")
    trips["service_date"] = pd.to_datetime(trips["service_date"])
    trips = trips.sort_values(["corridor", "dep_planned"]).reset_index(drop=True)

    f = trips[["corridor", "service_date", "dep_planned"]].copy()
    f["corridor_code"] = trips["corridor"].astype("category").cat.codes
    f["service_type_code"] = trips["service_type"].map(
        {t: i for i, t in enumerate(DOMESTIC_TYPES)}).fillna(-1).astype(int)
    f["dep_hour"] = trips["dep_hour"]
    f["dep_minute"] = trips["dep_planned"].dt.hour * 60 + trips["dep_planned"].dt.minute
    f["isodow"] = trips["service_date"].dt.dayofweek + 1
    f["month"] = trips["service_date"].dt.month
    f["travel_min"] = (trips["arr_planned"] - trips["dep_planned"]).dt.total_seconds() / 60

    # History: the tuned Step 5 baseline, computed walk-forward (only days < service_date).
    hist = baselines.Hierarchical(k=30, window=365, half_life=60)(trips, trips)
    f["hist_q50"], f["hist_q90"], f["hist_p_cancel"] = hist["q50"], hist["q90"], hist["p_cancel"]

    # Yesterday on this corridor.
    daily = trips.groupby(["corridor", "service_date"]).agg(
        cs=("cancelled", "mean"), md=("arr_delay", "mean")).reset_index()
    daily["service_date"] += pd.Timedelta(days=1)
    f = f.merge(daily.rename(columns={"cs": "prev_day_cancel_share", "md": "prev_day_mean_delay"}),
                on=["corridor", "service_date"], how="left")

    # Planned-works proxy: today's scheduled regular trips vs the usual count for this
    # corridor and day type over the previous 8 weeks. Timetables (incl. planned works)
    # are published in advance; cancelled trips still count as scheduled.
    reg = trips[trips["train_number"] < 100000]
    cnt = reg.groupby(["corridor", "daytype", "service_date"]).size().rename("n").reset_index()
    cnt = cnt.sort_values("service_date")
    cnt["usual"] = (cnt.groupby(["corridor", "daytype"])["n"]
                    .transform(lambda s: s.shift(1).rolling(8, min_periods=3).median()))
    cnt["sched_ratio"] = cnt["n"] / cnt["usual"]
    f = f.merge(cnt[["corridor", "service_date", "sched_ratio"]], on=["corridor", "service_date"], how="left")

    cal = load(con, "calendar.parquet").rename(columns={"date": "service_date"})
    wx = load(con, "weather_daily.parquet").rename(columns={"date": "service_date"})
    for t in (cal, wx):
        t["service_date"] = pd.to_datetime(t["service_date"])
        f = f.merge(t, on="service_date", how="left")
    return f, trips


# ---------------------------------------------------------------- live state

EVENT_SQL = f"""
CREATE OR REPLACE TEMP TABLE ev AS
SELECT station_code, service_id,
       dep_planned AS planned,
       dep_planned + to_minutes(coalesce(dep_delay, 0)) AS actual,
       dep_delay AS delay,
       coalesce(dep_cancelled, false) AS cancelled
FROM stops
WHERE company IN ('NS', 'NS Int') AND service_type IN {DOMESTIC_TYPES} AND dep_planned IS NOT NULL;

-- Delay state: cumulative sums over actual departure minute, per live station and network-wide ('*').
CREATE OR REPLACE TEMP TABLE dcum AS
WITH m AS (
    SELECT station_code AS st, actual AS t, count(*) n, sum(delay) d, sum((delay > 5)::int) l
    FROM ev WHERE NOT cancelled AND station_code IN {LIVE_STATIONS} GROUP BY ALL
    UNION ALL
    SELECT '*', actual, count(*), sum(delay), sum((delay > 5)::int)
    FROM ev WHERE NOT cancelled GROUP BY ALL
)
SELECT st, t, sum(n) OVER w AS cn, sum(d) OVER w AS cd, sum(l) OVER w AS cl
FROM m WINDOW w AS (PARTITION BY st ORDER BY t ROWS UNBOUNDED PRECEDING);

-- Cancellation state: cumulative over planned minute.
CREATE OR REPLACE TEMP TABLE ccum AS
WITH m AS (
    SELECT station_code AS st, planned AS t, count(*) n, sum(cancelled::int) c
    FROM ev WHERE station_code IN {LIVE_STATIONS} GROUP BY ALL
    UNION ALL
    SELECT '*', planned, count(*), sum(cancelled::int) FROM ev GROUP BY ALL
)
SELECT st, t, sum(n) OVER w AS cn, sum(c) OVER w AS cc
FROM m WINDOW w AS (PARTITION BY st ORDER BY t ROWS UNBOUNDED PRECEDING);

-- Earlier trips on the same corridor: arrivals at the destination by actual time.
CREATE OR REPLACE TEMP TABLE acum AS
WITH m AS (
    SELECT corridor, arr_planned + to_minutes(arr_delay) AS t, count(*) n, sum(arr_delay) d
    FROM read_parquet('{config.CLEAN / "corridor_trips.parquet"}') WHERE NOT cancelled GROUP BY ALL
)
SELECT corridor, t, sum(n) OVER w AS cn, sum(d) OVER w AS cd
FROM m WINDOW w AS (PARTITION BY corridor ORDER BY t ROWS UNBOUNDED PRECEDING);

CREATE OR REPLACE TEMP TABLE kcum AS
WITH m AS (
    SELECT corridor, dep_planned AS t, count(*) n, sum(cancelled::int) c
    FROM read_parquet('{config.CLEAN / "corridor_trips.parquet"}') GROUP BY ALL
)
SELECT corridor, t, sum(n) OVER w AS cn, sum(c) OVER w AS cc
FROM m WINDOW w AS (PARTITION BY corridor ORDER BY t ROWS UNBOUNDED PRECEDING);

-- Own train: departures of this service that already happened.
CREATE OR REPLACE TEMP TABLE own AS
SELECT service_id, actual AS t, delay FROM ev WHERE NOT cancelled;
"""


def window_delay(alias, st_expr, lo, hi):
    """Delay stats for events with actual time in (lo, hi] via two ASOF lookups."""
    return f"""
    ASOF LEFT JOIN dcum {alias}_a ON {alias}_a.st = {st_expr} AND {alias}_a.t <= {hi}
    ASOF LEFT JOIN dcum {alias}_b ON {alias}_b.st = {st_expr} AND {alias}_b.t <= {lo}"""


def live_features(con, horizon_min: int) -> pd.DataFrame:
    w, cw, ck = WINDOW, CANCEL_WINDOW, CANCEL_KNOWN_AFTER

    def delay_cols(a):
        n = f"(coalesce({a}_a.cn,0) - coalesce({a}_b.cn,0))"
        return (f"{n} AS {a}_n, "
                f"(coalesce({a}_a.cd,0) - coalesce({a}_b.cd,0)) / nullif({n},0) AS {a}_mean_delay, "
                f"(coalesce({a}_a.cl,0) - coalesce({a}_b.cl,0)) / nullif({n},0) AS {a}_late5_share")

    def cancel_join(a, st):
        return f"""
    ASOF LEFT JOIN ccum {a}_ca ON {a}_ca.st = {st} AND {a}_ca.t <= t.as_of - to_minutes({ck})
    ASOF LEFT JOIN ccum {a}_cb ON {a}_cb.st = {st} AND {a}_cb.t <= t.as_of - to_minutes({cw})"""

    def cancel_col(a):
        return (f"(coalesce({a}_ca.cc,0) - coalesce({a}_cb.cc,0)) / "
                f"nullif(coalesce({a}_ca.cn,0) - coalesce({a}_cb.cn,0), 0) AS {a}_cancel_share")

    sql = f"""
    WITH t AS (
        SELECT corridor, orig, dest, service_date, dep_planned, service_id,
               dep_planned - to_minutes({horizon_min}) AS as_of
        FROM read_parquet('{config.CLEAN / "corridor_trips.parquet"}')
    )
    SELECT t.corridor, t.service_date, t.dep_planned,
        (o.t IS NOT NULL)::int AS own_known, o.delay AS own_last_delay,
        date_diff('minute', o.t, t.as_of) AS own_minutes_since,
        {delay_cols('orig')}, {cancel_col('orig')},
        {delay_cols('dest')}, {cancel_col('dest')},
        {delay_cols('net')},  {cancel_col('net')},
        coalesce(cp_a.cn,0) - coalesce(cp_b.cn,0) AS corr_prev_n,
        (coalesce(cp_a.cd,0) - coalesce(cp_b.cd,0)) / nullif(coalesce(cp_a.cn,0) - coalesce(cp_b.cn,0), 0) AS corr_prev_mean_delay,
        (coalesce(ck_a.cc,0) - coalesce(ck_b.cc,0)) / nullif(coalesce(ck_a.cn,0) - coalesce(ck_b.cn,0), 0) AS corr_cancel_share
    FROM t
    ASOF LEFT JOIN own o ON o.service_id = t.service_id AND o.t <= t.as_of
    {window_delay('orig', 't.orig', f't.as_of - to_minutes({w})', 't.as_of')}
    {cancel_join('orig', 't.orig')}
    {window_delay('dest', 't.dest', f't.as_of - to_minutes({w})', 't.as_of')}
    {cancel_join('dest', 't.dest')}
    {window_delay('net', "'*'", f't.as_of - to_minutes({w})', 't.as_of')}
    {cancel_join('net', "'*'")}
    ASOF LEFT JOIN acum cp_a ON cp_a.corridor = t.corridor AND cp_a.t <= t.as_of
    ASOF LEFT JOIN acum cp_b ON cp_b.corridor = t.corridor AND cp_b.t <= t.as_of - to_minutes(60)
    ASOF LEFT JOIN kcum ck_a ON ck_a.corridor = t.corridor AND ck_a.t <= t.as_of - to_minutes({ck})
    ASOF LEFT JOIN kcum ck_b ON ck_b.corridor = t.corridor AND ck_b.t <= t.as_of - to_minutes(90)
    """
    df = con.execute(sql).df()
    df["service_date"] = pd.to_datetime(df["service_date"])
    return df


# ---------------------------------------------------------------- leakage check

def check(con, n: int = 200, seed: int = 0) -> None:
    """Recompute live features for random rows by direct filtering and compare."""
    rng = np.random.default_rng(seed)
    for hz, mins in HORIZONS.items():
        if not mins:
            continue
        f = pd.read_parquet(config.CLEAN / f"features_{hz}.parquet")
        trips = load(con, "corridor_trips.parquet")
        sample = f.iloc[rng.choice(len(f), n, replace=False)].merge(
            trips[["corridor", "dep_planned", "orig", "service_id"]], on=["corridor", "dep_planned"])
        bad = 0
        for _, r in sample.iterrows():
            as_of = r["dep_planned"] - pd.Timedelta(minutes=mins)
            day = con.execute("""
                SELECT station_code, service_id, dep_planned + to_minutes(coalesce(dep_delay,0)) AS actual,
                       dep_delay AS delay, coalesce(dep_cancelled,false) AS cancelled
                FROM stops WHERE dep_planned BETWEEN ? - INTERVAL 1 DAY AND ?
                  AND month IN (strftime(? - INTERVAL 1 DAY, '%Y-%m'), strftime(?, '%Y-%m'),
                                strftime(? + INTERVAL 1 MONTH, '%Y-%m'))
                  AND company IN ('NS','NS Int') AND service_type IN """ + str(DOMESTIC_TYPES) + """
                  AND dep_planned IS NOT NULL""", [as_of, as_of, as_of, as_of, as_of]).df()
            # Filter on planned/actual time, not service_date: the archive has a few
            # services whose service_date is a day off from their stop times.
            ran = day[~day["cancelled"]]
            o = ran[(ran["station_code"] == r["orig"]) & (ran["actual"] <= as_of)
                    & (ran["actual"] > as_of - pd.Timedelta(minutes=WINDOW))]
            own = ran[(ran["service_id"] == r["service_id"]) & (ran["actual"] <= as_of)]
            exp_n, exp_own = len(o), (own.sort_values("actual")["delay"].iloc[-1] if len(own) else np.nan)
            got_own = r["own_last_delay"]
            same_own = (np.isnan(exp_own) and pd.isna(got_own)) or exp_own == got_own
            if exp_n != r["orig_n"] or not same_own:
                bad += 1
                if bad <= 3:
                    print(f"  mismatch {hz} {r['corridor']} {r['dep_planned']}: orig_n {r['orig_n']} vs {exp_n}, "
                          f"own {got_own} vs {exp_own}")
        print(f"  {hz}: {n - bad}/{n} sampled rows match the brute-force recomputation")
        if bad:
            raise SystemExit("leakage check failed")
    check_weather(con, n, seed)


def check_weather(con, n: int = 200, seed: int = 0) -> None:
    """Observed weather may only use hours that ended by as_of; forecasts only for hours
    whose day-1 run was issued by as_of (hour <= as_of + 24 h)."""
    from .weather import OBS_PATH
    obs = pd.read_parquet(OBS_PATH)
    trips = load(con, "corridor_trips.parquet")
    rng = np.random.default_rng(seed)
    for hz, mins in HORIZONS.items():
        f = pd.read_parquet(config.CLEAN / f"features_{hz}.parquet")
        sample = f.iloc[rng.choice(len(f), n, replace=False)].merge(
            trips[["corridor", "dep_planned", "orig", "dest", "arr_planned"]], on=["corridor", "dep_planned"])
        bad = 0
        for _, r in sample.iterrows():
            as_of = (pd.Timestamp(r["service_date"]) if mins is None
                     else r["dep_planned"] - pd.Timedelta(minutes=mins))
            o = obs[obs["station_code"].isin([r["orig"], r["dest"]])
                    & (obs["hour_end"] <= as_of) & (obs["hour_end"] > as_of - pd.Timedelta(hours=24))]
            exp_snow = o.groupby("station_code")["snow"].sum().max() if len(o) else np.nan
            o12 = o[o["hour_end"] > as_of - pd.Timedelta(hours=12)]
            exp_tmin = o12["t"].min() if len(o12) else np.nan
            fc_allowed = r["arr_planned"].floor("h") <= as_of + pd.Timedelta(hours=24)
            ok = (np.isclose(exp_snow, r["obs_snow_hours_24"], equal_nan=True)
                  and np.isclose(exp_tmin, r["obs_tmin_12"], equal_nan=True)
                  and (fc_allowed or pd.isna(r["fc_gust_max"])))
            if not ok:
                bad += 1
                if bad <= 3:
                    print(f"  weather mismatch {hz} {r['corridor']} {r['dep_planned']}: snow {r['obs_snow_hours_24']} vs "
                          f"{exp_snow}, tmin {r['obs_tmin_12']} vs {exp_tmin}, fc allowed {fc_allowed}")
        print(f"  {hz} weather: {n - bad}/{n} sampled rows match")
        if bad:
            raise SystemExit("weather leakage check failed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    if args.check:
        check(con)
        return

    print("day-level features")
    day, _ = day_features(con)
    print("event tables")
    for stmt in filter(str.strip, EVENT_SQL.split(";")):
        con.execute(stmt)
    keys = ["corridor", "service_date", "dep_planned"]
    for hz, mins in HORIZONS.items():
        f = day if mins is None else day.merge(live_features(con, mins), on=keys, how="left")
        f = f.merge(weather_features(con, mins), on=keys, how="left")
        f.to_parquet(config.CLEAN / f"features_{hz}.parquet", index=False)
        print(f"  {hz}: {len(f):,} rows, {len(features_for(hz))} features")


if __name__ == "__main__":
    sys.exit(main())
