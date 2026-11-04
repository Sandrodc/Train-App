"""Passenger-delay label (labels.PAX_SQL) and scoreboard helpers on small synthetic data."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from pipeline import labels, scoreboard

D = pd.Timestamp("2026-06-01")


def trip(dep, arr, cancelled=False, dep_delay=0, arr_delay=0):
    return {"corridor": "A -> B", "service_date": D.date(), "dep_planned": D + pd.Timedelta(dep + ":00"),
            "arr_planned": D + pd.Timedelta(arr + ":00"), "cancelled": cancelled,
            "dep_delay": None if cancelled else dep_delay, "arr_delay": None if cancelled else arr_delay}


def pax(tmp_path, trips, not_running=()):
    t, n = tmp_path / "t.parquet", tmp_path / "n.parquet"
    pd.DataFrame(trips).to_parquet(t)
    cols = ["corridor", "service_date", "dep_planned", "arr_planned"]
    pd.DataFrame(list(not_running), columns=cols).astype({"dep_planned": "datetime64[ns]",
                                                          "arr_planned": "datetime64[ns]"}).to_parquet(n)
    con = duckdb.connect()
    con.execute(labels.PAX_SQL.format(trips=t, not_running=n, window=labels.PAX_WINDOW_MIN))
    return con.sql("SELECT dep_planned, pax_delay FROM pax ORDER BY dep_planned").df()["pax_delay"].tolist()


def test_pax_delay(tmp_path):
    out = pax(tmp_path, [
        trip("07:00", "07:40", arr_delay=2),          # ran, 2 min late -> 2
        trip("07:15", "07:55", cancelled=True),       # cancelled -> next train 07:30 (+4) -> 07:74 - 07:55
        trip("07:30", "08:10", dep_delay=3, arr_delay=4),
        trip("23:00", "23:40", cancelled=True),       # nothing after -> NULL
    ])
    assert out[0] == 2
    assert out[1] == (8 * 60 + 14) - (7 * 60 + 55)    # 19 min
    assert out[2] == 4
    assert pd.isna(out[3])


def test_pax_delay_never_negative_and_not_running(tmp_path):
    # a later, faster train that is planned to arrive before ours does not count
    rows = [trip("07:00", "08:00", arr_delay=10), trip("07:05", "07:50", arr_delay=0)]
    nr = [{"corridor": "A -> B", "service_date": D.date(), "dep_planned": D + pd.Timedelta("06:50:00"),
           "arr_planned": D + pd.Timedelta("07:45:00")}]
    out = pax(tmp_path, rows, nr)
    assert out == [5, 10, 0]      # not-running 06:50 -> the 07:05 arrives 07:50; 07:00 -> its own +10


def preds(n, q=(0, 1, 4), p=0.05):
    return pd.DataFrame({"q10": [q[0]] * n, "q50": [q[1]] * n, "q90": [q[2]] * n, "p_cancel": [p] * n})


def test_p_le_monotone_and_bounded():
    p = preds(1)
    vals = [scoreboard.p_le(p, x)[0] for x in range(-1, 30)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert 0 <= vals[0] and vals[-1] <= 0.95 + 1e-9
    assert scoreboard.p_le(p, 1)[0] == pytest.approx(0.95 * 0.5)


def test_bootstrap_identical_models_is_zero():
    rng = np.random.default_rng(0)
    n = 400
    y = pd.DataFrame({"service_date": np.repeat(pd.date_range("2026-01-01", periods=40), 10),
                      "cancelled": rng.random(n) < 0.05, "arr_delay": rng.integers(0, 10, n),
                      "pax_delay": rng.integers(0, 15, n)})
    l = scoreboard.trip_losses(y, preds(n))
    out = scoreboard.bootstrap(y, {"ref": l, "same": l.copy()}, "ref", n=200)
    assert out.loc["same", "pinball"].startswith("+0.0% [+0.0, +0.0]")


def test_first_cancel_recall():
    y = pd.DataFrame({"corridor": ["c"] * 4, "service_date": [D] * 4,
                      "dep_planned": [D + pd.Timedelta(minutes=m) for m in (0, 10, 20, 30)],
                      "cancelled": [False, True, True, True]})
    p = pd.DataFrame({"p_cancel": [0.0, 0.1, 0.5, 0.9]})
    r = scoreboard.first_cancel_recall(y, p)
    assert r["first_cancels"] == 1 and r["first_recall"] == 0.0
    assert r["later_cancels"] == 2 and r["later_recall"] == 1.0


def test_health_alerts_once_reminds_and_recovers():
    from datetime import datetime, timedelta, timezone
    from pipeline.health import decide
    t0 = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    msg, s = decide(["stale"], {}, t0)
    assert msg.startswith("train-logger PROBLEM")
    msg, s = decide(["stale"], s, t0 + timedelta(minutes=5))
    assert msg is None
    msg, s = decide(["stale"], s, t0 + timedelta(hours=6))
    assert msg.startswith("train-logger STILL BROKEN")
    msg, s = decide([], s, t0 + timedelta(hours=7))
    assert "recovered" in msg
    msg, s = decide([], s, t0 + timedelta(hours=8))
    assert msg is None


def test_drop_covid_keeps_edges_out():
    days = pd.to_datetime(["2020-03-15", "2020-03-16", "2022-03-31", "2022-04-01"])
    kept = scoreboard.drop_covid(pd.DataFrame({"service_date": days}))
    assert list(kept["service_date"].dt.strftime("%Y-%m-%d")) == ["2020-03-15", "2022-04-01"]
