"""Paths and dataset locations shared by the pipeline scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
DB_PATH = DATA / "train.duckdb"

# Rijden de Treinen open data (CC BY 4.0, attribution: "Rijden de Treinen").
RDT_BASE = "https://opendata.rijdendetreinen.nl/public"
SERVICES_URL = RDT_BASE + "/services/services-{ym}.csv.gz"   # ym = "2024-03"
DISRUPTIONS_URL = RDT_BASE + "/disruptions/disruptions-{year}.csv"
STATIONS_URL = RDT_BASE + "/stations/stations-2023-09.csv"

# Default history window: Jan 2023 up to the latest published month.
SERVICES_FIRST_MONTH = "2023-01"
SERVICES_LAST_MONTH = "2026-10"
DISRUPTIONS_FIRST_YEAR = 2011
DISRUPTIONS_LAST_YEAR = 2025
