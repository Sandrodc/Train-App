"""External data: KNMI daily weather and the Dutch holiday calendar (Step 2 / Step 6).

Writes:
  data/clean/weather_daily.parquet    one row per date, network-wide aggregates
  data/clean/calendar.parquet         one row per date

Weather: KNMI daily observations at De Bilt (260), Rotterdam (344), Schiphol (240),
Eindhoven (370). KNMI's hourly endpoint returned no data, so V1 uses daily values.
For day-ahead predictions observed weather stands in for the forecast ("perfect
forecast"), which flatters weather features a little; swap in archived forecasts later.

School holidays: rijksoverheid open data, available from school year 2024-2025 on.
Earlier dates get NULL (unknown), never a guessed 0.

Usage:
    python -m pipeline.external
"""
import sys

import holidays
import pandas as pd
import requests

from . import config

KNMI_DAILY = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
STATIONS = {260: "de_bilt", 344: "rotterdam", 240: "schiphol", 370: "eindhoven"}
# FHX max hourly mean wind, FXX max gust (0.1 m/s); TN/TX min/max temp (0.1 C);
# RH precipitation (0.1 mm, -1 = <0.05 mm); DR precipitation duration (0.1 h).
VARS = "FHX:FXX:TN:TX:RH:DR"
SCHOOL_API = "https://opendata.rijksoverheid.nl/v1/infotypes/schoolholidays?output=json"
SCHOOL_KNOWN_FROM = "2024-09-02"   # summer 2024 belongs to the uncovered 2023-24 school year
# COVID-19: first lockdown (16 Mar 2020) until the last measures were dropped (end Mar 2022).
# A feature for the models, and a filter the scoreboard can apply (--exclude-covid).
COVID_START, COVID_END = "2020-03-16", "2022-03-31"
CALENDAR_START = "2018-12-01"      # services start 2019-01; one month of warm-up


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    r = requests.get(KNMI_DAILY, timeout=120, params={
        "stns": ":".join(map(str, STATIONS)), "vars": VARS,
        "start": start.replace("-", ""), "end": end.replace("-", ""), "fmt": "json"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["RH"] = df["RH"].clip(lower=0)
    g = df.groupby("date")
    out = pd.DataFrame({
        "wx_gust_max": g["FXX"].max() / 10,          # m/s
        "wx_wind_max": g["FHX"].max() / 10,
        "wx_rain_max": g["RH"].max() / 10,           # mm
        "wx_rain_hours_max": g["DR"].max() / 10,
        "wx_tmin": g["TN"].min() / 10,               # C
        "wx_tmax": g["TX"].max() / 10,
    }).reset_index()
    return out


def fetch_school_holidays() -> pd.DataFrame:
    """Rows of (region, start, end) for every school holiday the API knows."""
    rows = []
    for year in requests.get(SCHOOL_API, timeout=60).json():
        for content in year["content"]:
            for v in content["vacations"]:
                for reg in v["regions"]:
                    rows.append((v["type"].strip(), reg["region"].strip(),
                                 pd.Timestamp(reg["startdate"]).tz_convert("Europe/Amsterdam").normalize().tz_localize(None),
                                 pd.Timestamp(reg["enddate"]).tz_convert("Europe/Amsterdam").normalize().tz_localize(None)))
    return pd.DataFrame(rows, columns=["type", "region", "start", "end"])


def build_calendar(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"date": dates})
    nl = holidays.Netherlands(years=range(dates.min().year, dates.max().year + 1))
    cal["public_holiday"] = cal["date"].dt.date.map(lambda d: d in nl).astype(int)

    sh = fetch_school_holidays()
    for region in ("noord", "midden", "zuid"):
        r = sh[sh["region"].isin([region, "heel Nederland"])]
        flag = pd.Series(0.0, index=cal.index)
        for _, v in r.iterrows():
            flag[(cal["date"] >= v["start"]) & (cal["date"] <= v["end"])] = 1.0
        flag[cal["date"] < SCHOOL_KNOWN_FROM] = float("nan")
        cal[f"school_holiday_{region}"] = flag
    cal["school_holiday_regions"] = cal[[f"school_holiday_{r}" for r in ("noord", "midden", "zuid")]].sum(axis=1, min_count=3)

    # Days since the last DST switch (last Sunday of March / October).
    def last_sunday(y, m):
        d = pd.Timestamp(y, m + 1, 1) - pd.Timedelta(days=1)
        return d - pd.Timedelta(days=(d.dayofweek + 1) % 7)
    switches = sorted(last_sunday(y, m) for y in range(dates.min().year - 1, dates.max().year + 1) for m in (3, 10))
    s = pd.Series(switches)
    cal["days_since_dst"] = cal["date"].map(lambda d: (d - s[s <= d].max()).days)
    # Leaves on the line season (autumn adhesion problems).
    doy = cal["date"].dt.dayofyear
    cal["leaves_season"] = ((doy >= 274) & (doy <= 349)).astype(int)   # ~1 Oct - 15 Dec
    cal["covid_period"] = cal["date"].between(COVID_START, COVID_END).astype(int)
    return cal


def main(argv=None):
    start, end = CALENDAR_START, "2026-10-31"
    config.CLEAN.mkdir(parents=True, exist_ok=True)
    wx = fetch_weather(start, end)
    wx.to_parquet(config.CLEAN / "weather_daily.parquet", index=False)
    print(f"weather: {len(wx)} days, gust max {wx['wx_gust_max'].max()} m/s on "
          f"{wx.loc[wx['wx_gust_max'].idxmax(), 'date'].date()}")
    cal = build_calendar(start, end)
    cal.to_parquet(config.CLEAN / "calendar.parquet", index=False)
    print(f"calendar: {len(cal)} days, {int(cal['public_holiday'].sum())} public holidays, "
          f"school holiday days known from {SCHOOL_KNOWN_FROM}, "
          f"{int(cal['covid_period'].sum())} COVID-period days")


if __name__ == "__main__":
    sys.exit(main())
