"""Build the clean layer from the raw RdT downloads.

Writes:
  data/clean/stops/month=YYYY-MM/*.parquet   one row per train x station x service day
  data/clean/disruptions.parquet
  data/clean/stations.parquet
  data/train.duckdb                           views over the parquet files

Usage:
    python -m pipeline.build_db
"""
import gzip
import shutil
import sys

import duckdb

from . import config

TZ = "Europe/Amsterdam"

SERVICES_COLUMNS = {
    "Service:RDT-ID": "BIGINT",
    "Service:Date": "DATE",
    "Service:Type": "VARCHAR",
    "Service:Company": "VARCHAR",
    "Service:Train number": "INTEGER",
    "Service:Completely cancelled": "BOOLEAN",
    "Service:Partly cancelled": "BOOLEAN",
    "Service:Maximum delay": "INTEGER",
    "Stop:RDT-ID": "BIGINT",
    "Stop:Station code": "VARCHAR",
    "Stop:Station name": "VARCHAR",
    "Stop:Arrival time": "TIMESTAMPTZ",
    "Stop:Arrival delay": "INTEGER",
    "Stop:Arrival cancelled": "BOOLEAN",
    "Stop:Departure time": "TIMESTAMPTZ",
    "Stop:Departure delay": "INTEGER",
    "Stop:Departure cancelled": "BOOLEAN",
    "Stop:Platform change": "BOOLEAN",
    "Stop:Planned platform": "VARCHAR",
    "Stop:Actual platform": "VARCHAR",
}


def stops_select(f, cols: str) -> str:
    return f"""
              WITH s AS (
                SELECT * FROM read_csv('{f}', header=true, columns={{{cols}}})
              )
              SELECT
                "Service:RDT-ID"               AS service_id,
                "Service:Date"                 AS service_date,
                "Service:Type"                 AS service_type,
                "Service:Company"              AS company,
                "Service:Train number"         AS train_number,
                "Service:Completely cancelled" AS service_cancelled,
                "Service:Partly cancelled"     AS service_partly_cancelled,
                "Service:Maximum delay"        AS service_max_delay,
                "Stop:RDT-ID"                  AS stop_id,
                row_number() OVER (PARTITION BY "Service:RDT-ID" ORDER BY "Stop:RDT-ID") AS stop_seq,
                count(*)     OVER (PARTITION BY "Service:RDT-ID")                        AS stop_count,
                upper("Stop:Station code")     AS station_code,
                "Stop:Station name"            AS station_name,
                timezone('{TZ}', "Stop:Arrival time")   AS arr_planned,   -- local wall-clock time
                "Stop:Arrival delay"           AS arr_delay,
                "Stop:Arrival cancelled"       AS arr_cancelled,
                timezone('{TZ}', "Stop:Departure time") AS dep_planned,
                "Stop:Departure delay"         AS dep_delay,
                "Stop:Departure cancelled"     AS dep_cancelled,
                "Stop:Platform change"         AS platform_changed,
                "Stop:Planned platform"        AS planned_platform,
                "Stop:Actual platform"         AS actual_platform
              FROM s"""


def check_header(f) -> None:
    """Stop if a file's columns differ from SERVICES_COLUMNS (older years could differ)."""
    with gzip.open(f, "rt") as fh:
        header = fh.readline().strip().split(",")
    if header != list(SERVICES_COLUMNS):
        raise SystemExit(f"{f.name}: columns differ from the expected schema:\n  got  {header}\n"
                         f"  want {list(SERVICES_COLUMNS)}")


def build_stops(con: duckdb.DuckDBPyConnection) -> None:
    """One parquet per month. Monthly files map 1:1; yearly files (2019-2022) are split."""
    files = sorted((config.RAW / "services").glob("services-*.csv.gz"))
    if not files:
        raise SystemExit("No services files in data/raw/services. Run pipeline.download first.")
    out = config.CLEAN / "stops"
    out.mkdir(parents=True, exist_ok=True)
    cols = ", ".join(f"'{k}': '{v}'" for k, v in SERVICES_COLUMNS.items())
    for f in files:
        check_header(f)

    for f in files:
        period = f.name.removeprefix("services-").removesuffix(".csv.gz")
        if len(period) == 4:                                  # yearly file
            wanted = [f"month={period}-{m:02d}" for m in range(1, 13)]
            if all((out / w).exists() for w in wanted):
                print(f"  skip  {period}")
                continue
            tmp = out / f".year-{period}"
            shutil.rmtree(tmp, ignore_errors=True)
            con.execute(f"""
                COPY (SELECT *, strftime(service_date, '%Y-%m') AS month FROM ({stops_select(f, cols)}))
                TO '{tmp}' (FORMAT parquet, COMPRESSION zstd, PARTITION_BY (month))
            """)
            for part in sorted(tmp.glob("month=*")):
                target = out / part.name
                if target.exists():
                    print(f"  skip  {part.name.removeprefix('month=')} (exists)")
                    continue
                target.mkdir()
                for i, pq_file in enumerate(sorted(part.glob("*.parquet"))):
                    pq_file.rename(target / ("stops.parquet" if i == 0 else f"stops-{i}.parquet"))
                print(f"  built {part.name.removeprefix('month=')} (from {f.name})")
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        part = out / f"month={period}"
        if part.exists():
            print(f"  skip  {period}")
            continue
        part.mkdir()
        con.execute(f"COPY ({stops_select(f, cols)}) TO '{part}/stops.parquet' (FORMAT parquet, COMPRESSION zstd)")
        print(f"  built {period}")


def build_small_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        COPY (
          SELECT id AS rdt_id, upper(code) AS code, uic, name_short, name_medium, name_long,
                 slug, country, type, geo_lat, geo_lng
          FROM read_csv('{config.RAW / "stations" / "stations.csv"}', header=true)
        ) TO '{config.CLEAN / "stations.parquet"}' (FORMAT parquet)
    """)
    print("  built stations")

    con.execute(f"""
        COPY (
          SELECT rdt_id, ns_lines, rdt_lines, rdt_lines_id, rdt_station_names,
                 list_transform(string_split(upper(rdt_station_codes), ','), x -> trim(x)) AS station_codes,
                 cause_nl, cause_en, statistical_cause_nl, statistical_cause_en, cause_group,
                 CAST(start_time AS TIMESTAMP) AS start_time,
                 CAST(end_time AS TIMESTAMP)   AS end_time,
                 CAST(duration_minutes AS INTEGER) AS duration_minutes
          FROM read_csv('{config.RAW / "disruptions"}/disruptions-*.csv',
                        header=true, union_by_name=true, all_varchar=true)
        ) TO '{config.CLEAN / "disruptions.parquet"}' (FORMAT parquet)
    """)
    print("  built disruptions")


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    clean = config.CLEAN
    con.execute(f"""
        CREATE OR REPLACE VIEW stops AS
        SELECT * FROM read_parquet('{clean}/stops/*/*.parquet', hive_partitioning=true);
        CREATE OR REPLACE VIEW stations AS SELECT * FROM read_parquet('{clean}/stations.parquet');
        CREATE OR REPLACE VIEW disruptions AS SELECT * FROM read_parquet('{clean}/disruptions.parquet');
    """)


def main(argv=None):
    config.CLEAN.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH))
    print("stations + disruptions")
    build_small_tables(con)
    print("stops")
    build_stops(con)
    create_views(con)
    n, first, last = con.execute(
        "SELECT count(*), min(service_date), max(service_date) FROM stops").fetchone()
    print(f"stops: {n:,} rows, {first} to {last}")
    con.close()


if __name__ == "__main__":
    sys.exit(main())
