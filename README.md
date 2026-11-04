# Train app

Working title: **Flighty for Dutch public transport** — a commute confidence layer on top of NS trains.

| Doc | What's in it |
| --- | --- |
| [APP_IDEA.md](APP_IDEA.md) | Product brief: problem, features, architecture, data licensing status, validation plan, risks |
| [PREDICTION_MODEL_PLAN.md](PREDICTION_MODEL_PLAN.md) | Step-by-step plan for the data pipeline and delay/connection prediction model |

## Data pipeline

Python 3.9+, everything runs locally on DuckDB + Parquet. `data/` is git-ignored (~2.3 GB once built).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pipeline.download      # RdT stations, disruptions 2011-2025, services 2023-01..2026-10 (~1.4 GB)
.venv/bin/python -m pipeline.build_db      # clean layer: data/clean/*.parquet + data/train.duckdb views
.venv/bin/python -m pipeline.labels        # corridor-trip labels (domestic trains, variants collapsed)
.venv/bin/python -m pipeline.external      # holiday calendar (+ old daily weather, kept for comparison)
.venv/bin/python -m pipeline.weather       # KNMI hourly obs (snow, ice) + Open-Meteo day-ahead forecasts
.venv/bin/python -m pipeline.features      # point-in-time features per horizon (day, t60, t15); --check for leakage
.venv/bin/python -m pipeline.reliability   # reliability per corridor / departure -> data/reports/*.csv
.venv/bin/python -m pipeline.scoreboard --horizon t15   # baselines + GBM (--split test, --sweep, --horizon day|t60|t15)
.venv/bin/python -m pipeline.model --horizon t15        # permutation feature importance
.venv/bin/python -m pipeline.scoreboard --horizon t15 --folds monthly  # rolling-origin, Jan 2024 - Oct 2026
.venv/bin/python -m pipeline.live run                   # live NS logger (NDOV ZeroMQ); `status` to inspect
.venv/bin/python -m pipeline.reparse_board              # rebuild data/live/board/ from raw/ with the current parser
.venv/bin/python -m pipeline.health                     # logger health check (runs on the server every 5 min)
.venv/bin/python -m pipeline.ns_benchmark               # what NS showed at T-60/T-15 vs what happened
.venv/bin/python -m pipeline.transfers     # missed / cancelled-leg / broken journeys per transfer and buffer
.venv/bin/python -m pytest -q tests        # parser fixtures, passenger label, scoreboard helpers
```

| File | Role |
| --- | --- |
| [pipeline/config.py](pipeline/config.py) | Paths, RdT URLs, default download window |
| [pipeline/download.py](pipeline/download.py) | Idempotent downloader (skips existing files) |
| [pipeline/build_db.py](pipeline/build_db.py) | Raw CSV → clean Parquet (one row per train × station × service day), DuckDB views `stops`, `stations`, `disruptions` |
| [pipeline/corridors.py](pipeline/corridors.py) | V1 corridors (placeholder until interviews) |
| [pipeline/labels.py](pipeline/labels.py) | Step 1: one row per corridor trip, the shared label table; passenger delay (`pax_delay`) and base-timetable "not running" rows |
| [pipeline/reliability.py](pipeline/reliability.py) | Step 5.1: delay distribution per corridor and per departure slot |
| [pipeline/scoreboard.py](pipeline/scoreboard.py) | Step 4: walk-forward splits, metrics, leakage check, day-block bootstrap, regimes, monthly folds, `experiments/log.csv` |
| [pipeline/baselines.py](pipeline/baselines.py) | Step 5: naive + hierarchical histogram baselines with recency weighting |
| [pipeline/transfers.py](pipeline/transfers.py) | Step 5.4: connection miss rate, cancelled legs and broken journeys per transfer |
| [pipeline/external.py](pipeline/external.py) | Public + school holidays, DST, leaves season (and the old daily weather) |
| [pipeline/weather.py](pipeline/weather.py) | Hourly KNMI observations incl. snow/ice flags, archived day-ahead forecasts, trip-window weather features |
| [pipeline/features.py](pipeline/features.py) | Step 6: point-in-time features per horizon, brute-force leakage check |
| [pipeline/live.py](pipeline/live.py) | Live logger: raw archive + parsed NS board snapshots (change codes, shown delay, rolling stock). Deploy notes in [deploy/](deploy/README.md) |
| [pipeline/reparse_board.py](pipeline/reparse_board.py) | Rebuild `board/` from `raw/` after a parser change (raw is never modified) |
| [pipeline/health.py](pipeline/health.py) | Logger health check: stale board files, low disk, change-code share |
| [pipeline/ns_benchmark.py](pipeline/ns_benchmark.py) | NS's own predictions and cancellation announcements vs outcome |
| [pipeline/model.py](pipeline/model.py) | Step 7: quantile + cancellation GBMs with conformal calibration; persistence benchmark |

Query the data directly:

```bash
.venv/bin/python -c "import duckdb; print(duckdb.connect('data/train.duckdb', read_only=True).sql('SELECT count(*) FROM stops'))"
```

Data: [Rijden de Treinen](https://www.rijdendetreinen.nl/en/open-data) (CC BY 4.0), [KNMI](https://www.daggegevens.knmi.nl/) hourly observations, [Open-Meteo](https://open-meteo.com/) archived forecasts (CC BY 4.0; free API is non-commercial), [Rijksoverheid](https://opendata.rijksoverheid.nl/) school holidays.
