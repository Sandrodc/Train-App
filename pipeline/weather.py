"""Hourly weather: KNMI observations (incl. snow and ice) + archived day-ahead forecasts.

Replaces the daily weather in pipeline.external for modelling (Step 6). Two sources,
each tied to the trip's own stations and hours:

  Observed  KNMI hourly validated data (POST to daggegevens.knmi.nl/klimatologie/uurgegevens),
            nearest KNMI station to each corridor endpoint. Has per-hour flags for
            snow (S) and ice formation (Y), plus temperature, gusts, precipitation.
            KNMI hours are in UT and label the hour that *ends* at HH:00; an hour
            is only usable once it has ended (hour_end <= as_of).

  Forecast  Open-Meteo "previous runs" API, `*_previous_day1` = the value forecast
            ~24 h before that hour, at the station's coordinates. Available from
            early 2024; earlier trips get NULL (unknown). A day-1 forecast for hour H
            was issued around H - 24h, so it is knowable at as_of whenever
            H <= as_of + 24h; later hours are dropped.

Licences: KNMI open data (CC BY 4.0). Open-Meteo data is CC BY 4.0, but the free
API is for non-commercial use: a commercial launch needs an Open-Meteo API plan
(or KNMI's own forecast products).

Writes data/clean/weather_obs_hourly.parquet and weather_fc_hourly.parquet.

Usage:
    python -m pipeline.weather
"""
import sys

import duckdb
import numpy as np
import pandas as pd
import requests

from . import config

KNMI_HOURLY = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
OPEN_METEO = "https://previous-runs-api.open-meteo.com/v1/forecast"

# NS station -> (nearest KNMI station, lat, lon of the NS station)
POINTS = {
    "RTD": (344, 51.9250, 4.4689),   # Rotterdam The Hague Airport
    "ASD": (240, 52.3789, 4.9003),   # Schiphol
    "UT":  (260, 52.0889, 5.1103),   # De Bilt
    "GVC": (215, 52.0803, 4.3250),   # Voorschoten
    "EHV": (370, 51.4433, 5.4814),   # Eindhoven Airport
}
KNMI_VARS = "T:FH:FX:RH:DR:VV:S:Y:M:O"
FC_VARS = ["temperature_2m", "precipitation", "snowfall", "wind_gusts_10m", "weather_code"]
FREEZING_CODES = (56, 57, 66, 67)                      # freezing drizzle / rain
SNOW_CODES = (71, 73, 75, 77, 85, 86)
THUNDER_CODES = (95, 96, 99)

OBS_PATH = config.CLEAN / "weather_obs_hourly.parquet"
FC_PATH = config.CLEAN / "weather_fc_hourly.parquet"


# ------------------------------------------------------------------ fetch

def fetch_obs(start="2022-12-01", end="2026-10-31") -> pd.DataFrame:
    knmi_to_ns = {v[0]: k for k, v in POINTS.items()}
    frames = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        s, e = max(start, f"{y}-01-01"), min(end, f"{y}-12-31")
        r = requests.post(KNMI_HOURLY, timeout=300, data={
            "stns": ":".join(str(k) for k in knmi_to_ns), "vars": KNMI_VARS,
            "start": s.replace("-", "") + "01", "end": e.replace("-", "") + "24", "fmt": "json"})
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()))
        print(f"  KNMI {y}: {len(frames[-1]):,} station-hours")
    df = pd.concat(frames, ignore_index=True)
    # hour h (1..24, UT) covers (h-1):00-h:00 UT; keep the moment it ends, in local time.
    end_utc = pd.to_datetime(df["date"]).dt.tz_convert("UTC") + pd.to_timedelta(df["hour"], unit="h")
    out = pd.DataFrame({
        "station_code": df["station_code"].map(knmi_to_ns),
        "hour_end": end_utc.dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None),
        "t": df["T"] / 10,                                  # C
        "wind": df["FH"] / 10,                              # m/s, hourly mean
        "gust": df["FX"] / 10,                              # m/s
        "precip": df["RH"].clip(lower=0) / 10,              # mm (-1 = <0.05 mm)
        "visibility_code": df["VV"],
        "snow": df["S"], "ice": df["Y"], "fog": df["M"], "thunder": df["O"],
    })
    return out.sort_values(["station_code", "hour_end"]).reset_index(drop=True)


def _get_json(url, params, tries=4):
    import time
    for i in range(tries):
        r = requests.get(url, timeout=300, params=params)
        if r.ok:
            try:
                return r.json()
            except ValueError:
                pass
        time.sleep(10 * (i + 1))
    raise RuntimeError(f"{url} failed after {tries} tries: HTTP {r.status_code} {r.text[:200]}")


def fetch_forecast(start="2024-01-01", end="2026-10-31") -> pd.DataFrame:
    # Requested in 6-month chunks: one request for the whole range gets rejected.
    chunks = pd.date_range(start, end, freq="6MS").tolist() + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    frames = []
    for code, (_, lat, lon) in POINTS.items():
        for a, b in zip(chunks[:-1], chunks[1:]):
            h = _get_json(OPEN_METEO, {
                "latitude": lat, "longitude": lon, "timezone": "Europe/Amsterdam",
                "start_date": a.strftime("%Y-%m-%d"), "end_date": (b - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "wind_speed_unit": "ms", "hourly": ",".join(f"{v}_previous_day1" for v in FC_VARS)})["hourly"]
            df = pd.DataFrame({"station_code": code, "hour": pd.to_datetime(h["time"])})
            for v in FC_VARS:
                df[v] = h[f"{v}_previous_day1"]
            frames.append(df)
        print(f"  forecast {code}: done")
    df = pd.concat(frames, ignore_index=True)
    code = df["weather_code"]
    return pd.DataFrame({
        "station_code": df["station_code"], "hour": df["hour"],
        "fc_t": df["temperature_2m"], "fc_precip": df["precipitation"],
        "fc_snow": df["snowfall"], "fc_gust": df["wind_gusts_10m"],
        "fc_freezing": code.isin(FREEZING_CODES).astype(float).where(code.notna()),
        "fc_snow_code": code.isin(SNOW_CODES).astype(float).where(code.notna()),
        "fc_thunder": code.isin(THUNDER_CODES).astype(float).where(code.notna()),
    })


# ------------------------------------------------------------------ features

OBS_FEATURES = ["obs_snow_hours_24", "obs_ice_hours_24", "obs_tmin_12", "obs_precip_6",
                "obs_gust_max_3", "obs_fog_hours_6"]
FC_FEATURES = ["fc_gust_max", "fc_precip_sum", "fc_snow_sum", "fc_tmin", "fc_tmax",
               "fc_freezing", "fc_snow_code", "fc_thunder", "fc_snow_prev24", "fc_frost_hours_prev12"]
WEATHER_FEATURES = OBS_FEATURES + FC_FEATURES

FEATURE_SQL = """
WITH obs AS (   -- trailing windows per station, labelled by the hour they end
    SELECT station_code, hour_end,
        sum(snow)   OVER (PARTITION BY station_code ORDER BY hour_end RANGE BETWEEN INTERVAL 23 HOUR PRECEDING AND CURRENT ROW) AS snow24,
        sum(ice)    OVER (PARTITION BY station_code ORDER BY hour_end RANGE BETWEEN INTERVAL 23 HOUR PRECEDING AND CURRENT ROW) AS ice24,
        min(t)      OVER (PARTITION BY station_code ORDER BY hour_end RANGE BETWEEN INTERVAL 11 HOUR PRECEDING AND CURRENT ROW) AS tmin12,
        sum(precip) OVER (PARTITION BY station_code ORDER BY hour_end RANGE BETWEEN INTERVAL 5 HOUR PRECEDING AND CURRENT ROW) AS precip6,
        max(gust)   OVER (PARTITION BY station_code ORDER BY hour_end RANGE BETWEEN INTERVAL 2 HOUR PRECEDING AND CURRENT ROW) AS gust3,
        sum(fog)    OVER (PARTITION BY station_code ORDER BY hour_end RANGE BETWEEN INTERVAL 5 HOUR PRECEDING AND CURRENT ROW) AS fog6
    FROM read_parquet('{obs}')
),
fc AS (          -- +-1 h around the hour, and the 24 h / 12 h before it
    SELECT station_code, hour,
        max(fc_gust)   OVER w3 AS gust, sum(fc_precip) OVER w3 AS precip, sum(fc_snow) OVER w3 AS snow,
        min(fc_t)      OVER w3 AS tmin, max(fc_t) OVER w3 AS tmax,
        max(fc_freezing) OVER w3 AS freezing, max(fc_snow_code) OVER w3 AS snowcode, max(fc_thunder) OVER w3 AS thunder,
        sum(fc_snow)   OVER (PARTITION BY station_code ORDER BY hour RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW) AS snow_prev24,
        sum((fc_t < 0)::int) OVER (PARTITION BY station_code ORDER BY hour RANGE BETWEEN INTERVAL 12 HOUR PRECEDING AND CURRENT ROW) AS frost_prev12
    FROM read_parquet('{fc}')
    WINDOW w3 AS (PARTITION BY station_code ORDER BY hour RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND INTERVAL 1 HOUR FOLLOWING)
),
t AS (
    SELECT corridor, orig, dest, service_date, dep_planned, arr_planned,
           {as_of} AS as_of,
           date_trunc('hour', dep_planned) AS dep_h, date_trunc('hour', arr_planned) AS arr_h
    FROM read_parquet('{trips}')
)
SELECT t.corridor, t.service_date, t.dep_planned,
    greatest(oo.snow24, od.snow24)  AS obs_snow_hours_24,
    greatest(oo.ice24, od.ice24)    AS obs_ice_hours_24,
    least(oo.tmin12, od.tmin12)     AS obs_tmin_12,
    greatest(oo.precip6, od.precip6) AS obs_precip_6,
    greatest(oo.gust3, od.gust3)    AS obs_gust_max_3,
    greatest(oo.fog6, od.fog6)      AS obs_fog_hours_6,
    -- forecast at the origin around departure and at the destination around arrival;
    -- dropped when that forecast would have been issued after as_of
    CASE WHEN t.arr_h <= t.as_of + INTERVAL 24 HOUR THEN greatest(fo.gust, fd.gust) END      AS fc_gust_max,
    CASE WHEN t.arr_h <= t.as_of + INTERVAL 24 HOUR THEN fo.precip + fd.precip END           AS fc_precip_sum,
    CASE WHEN t.arr_h <= t.as_of + INTERVAL 24 HOUR THEN fo.snow + fd.snow END               AS fc_snow_sum,
    CASE WHEN t.arr_h <= t.as_of + INTERVAL 24 HOUR THEN least(fo.tmin, fd.tmin) END         AS fc_tmin,
    CASE WHEN t.arr_h <= t.as_of + INTERVAL 24 HOUR THEN greatest(fo.tmax, fd.tmax) END      AS fc_tmax,
    CASE WHEN t.arr_h <= t.as_of + INTERVAL 24 HOUR THEN greatest(fo.freezing, fd.freezing) END AS fc_freezing,
    CASE WHEN t.arr_h <= t.as_of + INTERVAL 24 HOUR THEN greatest(fo.snowcode, fd.snowcode) END AS fc_snow_code,
    CASE WHEN t.arr_h <= t.as_of + INTERVAL 24 HOUR THEN greatest(fo.thunder, fd.thunder) END   AS fc_thunder,
    CASE WHEN t.dep_h <= t.as_of + INTERVAL 24 HOUR THEN fo.snow_prev24 END                  AS fc_snow_prev24,
    CASE WHEN t.dep_h <= t.as_of + INTERVAL 24 HOUR THEN fo.frost_prev12 END                 AS fc_frost_hours_prev12
FROM t
ASOF LEFT JOIN obs oo ON oo.station_code = t.orig AND oo.hour_end <= t.as_of
ASOF LEFT JOIN obs od ON od.station_code = t.dest AND od.hour_end <= t.as_of
LEFT JOIN fc fo ON fo.station_code = t.orig AND fo.hour = t.dep_h
LEFT JOIN fc fd ON fd.station_code = t.dest AND fd.hour = t.arr_h
"""


def weather_features(con: duckdb.DuckDBPyConnection, horizon_min=None) -> pd.DataFrame:
    """Weather features per corridor trip at as_of (00:00 for day-ahead, else dep - horizon)."""
    as_of = ("CAST(service_date AS TIMESTAMP)" if horizon_min is None
             else f"dep_planned - to_minutes({horizon_min})")
    df = con.execute(FEATURE_SQL.format(obs=OBS_PATH, fc=FC_PATH, as_of=as_of,
                                        trips=config.CLEAN / "corridor_trips.parquet")).df()
    df["service_date"] = pd.to_datetime(df["service_date"])
    return df


def main(argv=None):
    print("KNMI hourly observations")
    if OBS_PATH.exists() and "--refresh" not in (argv or sys.argv):
        obs = pd.read_parquet(OBS_PATH)
        print("  (cached; pass --refresh to refetch)")
    else:
        obs = fetch_obs()
        obs.to_parquet(OBS_PATH, index=False)
    print(f"  {len(obs):,} rows; snow hours {int(obs['snow'].sum()):,}, ice hours {int(obs['ice'].sum()):,}")
    print("Open-Meteo day-ahead forecasts")
    fc = fetch_forecast()
    fc.to_parquet(FC_PATH, index=False)
    first = fc.dropna(subset=["fc_t"])["hour"].min()
    print(f"  {len(fc):,} rows; forecasts available from {first}")


if __name__ == "__main__":
    sys.exit(main())
