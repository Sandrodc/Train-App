"""Live logger for NS InfoPlus via NDOV Loket (PREDICTION_MODEL_PLAN.md, Step 2-3).

Subscribes to the NDOV best-effort ZeroMQ feed (CC0, fair use: one connection per
stream) and writes two layers every few minutes:

  data/live/raw/date=YYYY-MM-DD/<envelope>-<HHMMSS>.parquet
      received_at (UTC), envelope, xml (the message, decompressed). Immutable.
      Stored decompressed on purpose: zstd across a 5-minute batch of near-identical
      XML is ~10x smaller than keeping each message's own gzip (~0.2 GB/day vs ~1.5).
  data/live/board/date=YYYY-MM-DD/<kind>-<HHMMSS>.parquet
      one row per DVS (departure) or DAS (arrival) message: what the station board
      said at `received_at` — NS's predicted delay, platform, cancellation codes.

The board layer is what makes NS's own predictions replayable: "what did NS show
at 07:12?" = last board row per (train, date, station) with received_at <= 07:12.

Change codes seen on the feed (WijzigingType):
  10 delay · 20 platform changed · 22 platform assigned
  32 does not run from this station ("Rijdt niet") · 34 terminates early ("Rijdt niet verder dan ...")

Usage:
    python -m pipeline.live run                 # run until stopped (Ctrl-C / SIGTERM flushes)
    python -m pipeline.live run --minutes 30
    python -m pipeline.live status              # what has been logged so far
"""
import argparse
import gzip
import logging
import signal
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from . import config

ENDPOINT = "tcp://pubsub.besteffort.ndovloket.nl:7664"
ENVELOPES = {
    "/RIG/InfoPlusDVSInterface4": "dvs",   # departures per station
    "/RIG/InfoPlusDASInterface4": "das",   # arrivals per station
    "/RIG/InfoPlusRITInterface5": "rit",   # full trip, archived raw only for now
}
LIVE = config.DATA / "live"
FLUSH_SECONDS = 300
STALL_SECONDS = 180      # no message for this long -> reconnect
TZ_LOCAL = "Europe/Amsterdam"

log = logging.getLogger("live")

BOARD_SCHEMA = pa.schema([
    ("received_at", pa.timestamp("ms", tz="UTC")),
    ("msg_timestamp", pa.timestamp("ms", tz="UTC")),
    ("kind", pa.string()),                 # dvs | das
    ("train_number", pa.int32()),
    ("rit_date", pa.date32()),
    ("station_code", pa.string()),
    ("planned_time", pa.timestamp("ms", tz="UTC")),
    ("actual_time", pa.timestamp("ms", tz="UTC")),
    ("delay_s", pa.int32()),               # ExacteVertrek/AankomstVertraging
    ("planned_platform", pa.string()),
    ("actual_platform", pa.string()),
    ("train_status", pa.int8()),
    ("train_type", pa.string()),
    ("operator", pa.string()),
    ("change_codes", pa.list_(pa.int16())),
    ("cancelled", pa.bool_()),             # code 32
    ("terminates_early", pa.bool_()),      # code 34
    ("cause", pa.string()),                # WijzigingOorzaakKort, if any
    ("cause_code", pa.string()),           # WijzigingOorzaakCode, if any (not seen on the feed so far)
    ("shown_delay_s", pa.int32()),         # GedempteVertrek/AankomstVertraging: the delay NS displays
    ("material_numbers", pa.list_(pa.string())),  # MaterieelNummer per unit (DVS only)
    ("material_types", pa.list_(pa.string())),    # MaterieelSoort-MaterieelAanduiding per unit, e.g. SLT-6
    ("train_length_m", pa.float32()),      # sum of MaterieelLengte (feed unit is cm)
    ("n_units", pa.int16()),               # number of MaterieelDeel elements
])
RAW_SCHEMA = pa.schema([
    ("received_at", pa.timestamp("ms", tz="UTC")),
    ("envelope", pa.string()),
    ("xml", pa.binary()),
])


# ------------------------------------------------------------------ parsing

def _iso_duration_s(text):
    """'PT3M30S' / '-PT1M' / 'PT0S' -> seconds."""
    if not text:
        return None
    sign = -1 if text.startswith("-") else 1
    t = text.lstrip("-").removeprefix("PT")
    secs, num = 0.0, ""
    for ch in t:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            secs += float(num or 0) * {"H": 3600, "M": 60, "S": 1}[ch]
            num = ""
    return int(sign * secs)


def _ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None


def _status(parent, tag, status):
    for el in parent.findall(f"{{*}}{tag}"):
        if el.get("InfoStatus") == status:
            return el
    return None


def _text(el, path):
    if el is None:
        return None
    x = el.find(path)
    return x.text.strip() if x is not None and x.text else None


def parse_board(kind: str, xml: bytes, received_at: datetime) -> dict:
    root = ET.fromstring(xml)
    product = root.find("{*}ReisInformatieProductDVS" if kind == "dvs" else "{*}ReisInformatieProductDAS")
    state = product.find("{*}DynamischeVertrekStaat" if kind == "dvs" else "{*}DynamischeAankomstStaat")
    train = state.find("{*}Trein" if kind == "dvs" else "{*}TreinAankomst")
    time_tag, delay_tag, plat_tag = (
        ("VertrekTijd", "ExacteVertrekVertraging", "TreinVertrekSpoor") if kind == "dvs"
        else ("AankomstTijd", "ExacteAankomstVertraging", "TreinAankomstSpoor"))
    # Element.iter() ignores the {*} wildcard; findall() supports it. DAS messages carry
    # the change as WijzigingHerkomst (a change on the way in) instead of Wijziging.
    changes = train.findall(".//{*}Wijziging") + train.findall(".//{*}WijzigingHerkomst")
    codes, cause, cause_code = [], None, None
    for w in changes:
        c = _text(w, "{*}WijzigingType")
        if c:
            codes.append(int(c))
        cause = cause or _text(w, "{*}WijzigingOorzaakKort")
        cause_code = cause_code or _text(w, "{*}WijzigingOorzaakCode")
    units = train.findall(".//{*}MaterieelDeelDVS")
    lengths = [_text(u, "{*}MaterieelLengte") for u in units]
    lengths = [int(x) for x in lengths if x]
    material_types = []
    for u in units:
        kind_, size = _text(u, "{*}MaterieelSoort"), _text(u, "{*}MaterieelAanduiding")
        if kind_:
            material_types.append(f"{kind_}-{size}" if size else kind_)
    shown_tag = "GedempteVertrekVertraging" if kind == "dvs" else "GedempteAankomstVertraging"
    status = _text(train, "{*}TreinStatus")
    rit_date = _text(state, "{*}RitDatum")
    return {
        "received_at": received_at,
        "msg_timestamp": _ts(product.get("TimeStamp")),
        "kind": kind,
        "train_number": int(_text(train, "{*}TreinNummer")),
        "rit_date": datetime.strptime(rit_date, "%Y-%m-%d").date() if rit_date else None,
        "station_code": _text(state, "{*}RitStation/{*}StationCode"),
        "planned_time": _ts(_text(_status(train, time_tag, "Gepland"), ".")),
        "actual_time": _ts(_text(_status(train, time_tag, "Actueel"), ".")),
        "delay_s": _iso_duration_s(_text(train, f"{{*}}{delay_tag}")),
        "planned_platform": _text(_status(train, plat_tag, "Gepland"), "{*}SpoorNummer"),
        "actual_platform": _text(_status(train, plat_tag, "Actueel"), "{*}SpoorNummer"),
        "train_status": int(status) if status else None,
        "train_type": (train.find("{*}TreinSoort").get("Code") if train.find("{*}TreinSoort") is not None else None),
        "operator": _text(train, "{*}Vervoerder"),
        "change_codes": codes,
        "cancelled": 32 in codes,
        "terminates_early": 34 in codes,
        "cause": cause,
        "cause_code": cause_code,
        "shown_delay_s": _iso_duration_s(_text(train, f"{{*}}{shown_tag}")),
        "material_numbers": [m for m in (_text(u, "{*}MaterieelNummer") for u in units) if m],
        "material_types": material_types,
        "train_length_m": sum(lengths) / 100 if lengths else None,
        "n_units": len(units) if units else None,
    }


# ------------------------------------------------------------------ writer

class Buffer:
    def __init__(self):
        self.raw, self.board = [], []
        self.parse_errors = 0
        self.last_flush = time.time()

    def add(self, envelope: str, body: bytes, received_at: datetime):
        try:
            xml = gzip.decompress(body)
        except OSError:
            xml = body
        self.raw.append({"received_at": received_at, "envelope": envelope, "xml": xml})
        kind = ENVELOPES.get(envelope)
        if kind in ("dvs", "das"):
            try:
                self.board.append(parse_board(kind, xml, received_at))
            except Exception as e:      # never lose raw data because of a parse bug
                self.parse_errors += 1
                if self.parse_errors <= 5:
                    log.warning("parse error on %s: %s", envelope, e)

    def flush(self):
        if not self.raw:
            return
        now = datetime.now(timezone.utc)
        stamp, day = now.strftime("%H%M%S"), now.strftime("%Y-%m-%d")
        for layer, rows, schema in (("raw", self.raw, RAW_SCHEMA), ("board", self.board, BOARD_SCHEMA)):
            if not rows:
                continue
            d = LIVE / layer / f"date={day}"
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{layer}-{stamp}.parquet.tmp"
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp,
                           compression="zstd", compression_level=9)
            tmp.rename(d / f"{layer}-{stamp}.parquet")       # atomic: readers never see half files
        log.info("flushed %d raw / %d board rows (%d parse errors so far)",
                 len(self.raw), len(self.board), self.parse_errors)
        self.raw, self.board = [], []
        self.last_flush = time.time()


def run(minutes=None):
    import zmq
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    buf, stop = Buffer(), {"flag": False}

    def _stop(*_):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    ctx = zmq.Context()
    deadline = time.time() + minutes * 60 if minutes else None
    while not stop["flag"]:
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(ENDPOINT)
        for env in ENVELOPES:
            sock.setsockopt(zmq.SUBSCRIBE, env.encode())
        log.info("connected to %s", ENDPOINT)
        last_msg, count = time.time(), 0
        while not stop["flag"]:
            if deadline and time.time() > deadline:
                stop["flag"] = True
                break
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                if time.time() - last_msg > STALL_SECONDS:
                    log.warning("no messages for %ds, reconnecting", STALL_SECONDS)
                    break
            else:
                last_msg = time.time()
                count += 1
                buf.add(parts[0].decode(), parts[1], datetime.now(timezone.utc))
            if time.time() - buf.last_flush > FLUSH_SECONDS:
                buf.flush()
        sock.close()
        if not stop["flag"]:
            time.sleep(10)
    buf.flush()
    ctx.term()
    log.info("stopped")


def status():
    import duckdb
    board = LIVE / "board"
    if not board.exists():
        print("nothing logged yet")
        return
    con = duckdb.connect()
    print(con.sql(f"""
        SELECT kind, count(*) AS messages, count(DISTINCT (train_number, rit_date)) AS trains,
               min(received_at) AS first, max(received_at) AS last,
               sum(cancelled::int) AS cancel_msgs, round(avg(delay_s) / 60, 2) AS mean_delay_min
        FROM read_parquet('{board}/*/*.parquet', hive_partitioning=true) GROUP BY kind
    """))
    raw = sum(f.stat().st_size for f in (LIVE / "raw").rglob("*.parquet"))
    brd = sum(f.stat().st_size for f in board.rglob("*.parquet"))
    print(f"disk: raw {raw / 1e6:.1f} MB, board {brd / 1e6:.1f} MB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--minutes", type=float)
    sub.add_parser("status")
    args = ap.parse_args(argv)
    run(args.minutes) if args.cmd == "run" else status()


if __name__ == "__main__":
    sys.exit(main())
