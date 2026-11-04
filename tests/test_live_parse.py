"""parse_board on real DVS/DAS messages saved from data/live/raw/ (4 Nov 2026)."""
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.live import BOARD_SCHEMA, RAW_SCHEMA, parse_board
from pipeline.reparse_board import rebuild

FIX = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def parse(name):
    return parse_board(name[:3], (FIX / name).read_bytes(), NOW)


def test_cancelled_code_32():
    r = parse("dvs_cancelled_32.xml")
    assert r["train_number"] == 3931 and r["station_code"] == "STD"
    assert r["change_codes"] == [32]
    assert r["cancelled"] is True and r["terminates_early"] is False
    assert r["cause"] == "seinstoring"


def test_terminates_early_code_34():
    r = parse("dvs_terminates_34.xml")
    assert 34 in r["change_codes"] and r["terminates_early"] is True
    assert r["cancelled"] is False
    assert r["delay_s"] == 654 and r["shown_delay_s"] == 600


def test_material_and_composition():
    r = parse("dvs_material.xml")
    assert r["material_numbers"] == ["000000-02656-0", "000000-02424-0"]
    assert r["material_types"] == ["SLT-6", "SLT-4"]
    assert r["n_units"] == 2
    assert r["train_length_m"] == pytest.approx(169.0)   # 10000 + 6900 cm
    assert r["change_codes"] == [10]


def test_no_changes():
    r = parse("dvs_plain.xml")
    assert r["change_codes"] == [] and not r["cancelled"]
    assert r["material_types"] == ["VIRM-6"]


def test_das_change_herkomst():
    r = parse("das_herkomst.xml")
    assert r["kind"] == "das"
    assert r["change_codes"] == [11, 21]
    assert r["cause"] == "defecte trein"
    assert r["shown_delay_s"] == 1500
    assert r["material_numbers"] == [] and r["n_units"] is None


def test_rows_fit_schema():
    rows = [parse(f.name) for f in sorted(FIX.glob("*.xml"))]
    assert pa.Table.from_pylist(rows, schema=BOARD_SCHEMA).num_rows == len(rows)


def test_reparse_is_idempotent_and_leaves_raw(tmp_path):
    raw = tmp_path / "raw" / "date=2026-11-04"
    raw.mkdir(parents=True)
    rows = [{"received_at": NOW, "envelope": "/RIG/InfoPlusDVSInterface4" if f.name.startswith("dvs")
             else "/RIG/InfoPlusDASInterface4", "xml": f.read_bytes()} for f in sorted(FIX.glob("*.xml"))]
    rows.append({"received_at": NOW, "envelope": "/RIG/InfoPlusRITInterface5", "xml": b"<x/>"})
    pq.write_table(pa.Table.from_pylist(rows, schema=RAW_SCHEMA), raw / "raw-120000.parquet")
    before = (raw / "raw-120000.parquet").read_bytes()

    first = rebuild(live=tmp_path)
    out = tmp_path / "board" / "date=2026-11-04" / "board-120000.parquet"
    t1 = pq.read_table(out)
    second = rebuild(live=tmp_path)
    assert first == second == {"raw_files": 1, "board_files": 1, "rows": 5, "parse_errors": 0}
    assert pq.read_table(out).equals(t1)
    assert sum(t1.column("cancelled").to_pylist()) == 1
    assert (raw / "raw-120000.parquet").read_bytes() == before
    assert not (tmp_path / ".board-rebuild").exists()
