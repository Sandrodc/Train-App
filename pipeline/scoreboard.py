"""Scoreboard (PREDICTION_MODEL_PLAN.md, Step 4).

Scores every predictor the same way on a time-based split, overall, per corridor
and for the weekday morning peak. Before scoring it checks each predictor for
look-ahead leakage by re-predicting a few days with history truncated at that day.

Usage:
    python -m pipeline.scoreboard                         # all baselines on validation
    python -m pipeline.scoreboard --split test
    python -m pipeline.scoreboard --split validation --sweep   # tune k / half-life
    python -m pipeline.scoreboard --split test --horizon t15   # baselines + GBM at T-15 min

Writes data/reports/scoreboard_<split>.md and appends to experiments/log.csv.
"""
import argparse
import datetime as dt
import sys
import time

import duckdb
import numpy as np
import pandas as pd

from . import baselines, config

# Walk-forward evaluation windows. Everything before a window's start is history.
SPLITS = {
    "validation": ("2025-11-01", "2026-04-30"),
    "test": ("2026-05-01", "2026-10-31"),
}
LOG = config.ROOT / "experiments" / "log.csv"


HORIZON_LABEL = {"day": "day-ahead (as_of 00:00)", "t60": "T-60 min", "t15": "T-15 min"}


def load_trips(horizon: str = "day") -> pd.DataFrame:
    """Corridor trips with labels, plus the point-in-time features for `horizon`."""
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    df = con.execute(f"SELECT * FROM read_parquet('{config.CLEAN / 'corridor_trips.parquet'}')").df()
    con.close()
    df["service_date"] = pd.to_datetime(df["service_date"])
    feats = config.CLEAN / f"features_{horizon}.parquet"
    if feats.exists():
        f = pd.read_parquet(feats)
        f["service_date"] = pd.to_datetime(f["service_date"])
        keys = ["corridor", "service_date", "dep_planned"]
        f = f[keys + [c for c in f.columns if c not in df.columns]]
        df = df.merge(f, on=["corridor", "service_date", "dep_planned"], how="left")
    return df.reset_index(drop=True)


def metrics(y: pd.DataFrame, p: pd.DataFrame) -> dict:
    ran = ~y["cancelled"].values
    yd = y["arr_delay"].values[ran].astype(float)
    q10, q50, q90 = (p[c].values[ran] for c in ("q10", "q50", "q90"))
    pinball = np.mean([
        np.mean(np.maximum(a * (yd - q), (a - 1) * (yd - q)))
        for a, q in ((0.1, q10), (0.5, q50), (0.9, q90))
    ])
    c = y["cancelled"].values.astype(float)
    return {
        "trips": len(y),
        "mae_p50": np.mean(np.abs(yd - q50)),
        "pinball": pinball,
        "coverage_10_90": np.mean((yd >= q10) & (yd <= q90)),
        "above_p90": np.mean(yd > q90),
        "surprise_5": np.mean(yd > q90 + 5),
        "brier_cancel": np.mean((p["p_cancel"].values - c) ** 2),
        "mean_p_cancel": p["p_cancel"].mean(),
        "cancel_rate": c.mean(),
    }


def leakage_check(predictor, history: pd.DataFrame, targets: pd.DataFrame,
                  bulk: pd.DataFrame, n_days: int = 3) -> None:
    """Predictions for day D must not change when everything from D onward is removed."""
    days = np.sort(targets["service_date"].unique())
    for d in days[np.linspace(0, len(days) - 1, n_days).astype(int)]:
        t = targets[targets["service_date"] == d]
        p = predictor(history[history["service_date"] < d], t)
        if not np.allclose(p.values, bulk.loc[t.index].values, atol=1e-5):
            raise AssertionError(f"{predictor.name}: look-ahead leakage detected on {pd.Timestamp(d).date()}")


def is_peak(t: pd.DataFrame) -> np.ndarray:
    return ((t["daytype"] == "wkd") & t["dep_hour"].between(6, 9)).values


def score(predictors, split: str, horizon: str = "day", check: bool = True):
    trips = load_trips(horizon)
    start, end = map(pd.Timestamp, SPLITS[split])
    history = trips[trips["service_date"] <= end]          # predictors only look back
    targets = trips[(trips["service_date"] >= start) & (trips["service_date"] <= end)]

    rows, per_corridor, preds = [], {}, {}
    for pr in predictors:
        t0 = time.time()
        p = pr(history, targets)
        secs = time.time() - t0
        if check:
            leakage_check(pr, history, targets, p)
        preds[pr.name] = p
        m = metrics(targets, p)
        pk = is_peak(targets)
        m_peak = metrics(targets[pk], p[pk])
        rows.append({"model": pr.name, **m,
                     "peak_mae_p50": m_peak["mae_p50"], "peak_coverage": m_peak["coverage_10_90"],
                     "peak_surprise_5": m_peak["surprise_5"], "seconds": round(secs, 1)})
        per_corridor[pr.name] = pd.DataFrame(
            {c: metrics(targets[g], p[g]) for c, g in
             ((c, (targets["corridor"] == c).values) for c in sorted(targets["corridor"].unique()))}
        ).T
        print(f"  {pr.name:<28} pinball {m['pinball']:.3f}  cov {m['coverage_10_90']:.3f}  "
              f"brier {m['brier_cancel']:.4f}  ({secs:.0f}s)")
    return pd.DataFrame(rows), per_corridor, preds, targets


def reliability_curve(targets: pd.DataFrame, p: pd.DataFrame, bins=(0, .02, .05, .1, .2, .4, 1.0)):
    b = pd.cut(p["p_cancel"], bins, include_lowest=True)
    return (pd.DataFrame({"bin": b, "pred": p["p_cancel"], "obs": targets["cancelled"].astype(float)})
            .groupby("bin", observed=True).agg(trips=("obs", "size"), predicted=("pred", "mean"),
                                               observed=("obs", "mean")))


def fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".3f") if hasattr(df, "to_markdown") else df.to_string()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, default="validation")
    ap.add_argument("--sweep", action="store_true", help="grid over k and half-life")
    ap.add_argument("--horizon", choices=HORIZON_LABEL, default="day")
    ap.add_argument("--no-check", action="store_true", help="skip the leakage check")
    args = ap.parse_args(argv)

    if args.sweep:
        predictors = [baselines.Hierarchical(k=k, window=w, half_life=hl)
                      for k in (3, 10, 30)
                      for w, hl in ((90, None), (365, 14), (365, 30), (365, 60))]
    else:
        from .model import GBM, Persistence
        predictors = baselines.registry()
        if args.horizon != "day":
            predictors.append(Persistence(next(p for p in predictors if p.name == "hier_tuned")))
        predictors.append(GBM(args.horizon, train_end=SPLITS[args.split][0], calibrate=False))
        predictors.append(GBM(args.horizon, train_end=SPLITS[args.split][0]))

    horizon = HORIZON_LABEL[args.horizon]
    print(f"split {args.split} {SPLITS[args.split]}, horizon {horizon}")
    table, per_corridor, preds, targets = score(predictors, args.split, args.horizon,
                                                check=not args.no_check)
    table = table.sort_values("pinball").reset_index(drop=True)
    best = table.loc[0, "model"]

    out = config.DATA / "reports"
    out.mkdir(parents=True, exist_ok=True)
    report = [
        f"# Scoreboard: {args.split} {SPLITS[args.split][0]} to {SPLITS[args.split][1]}",
        f"Horizon: {horizon}. Delay metrics on trips that ran; Brier on all trips. "
        "NS's own prediction column needs the live logger (not yet available).",
        "## All corridors", fmt(table.set_index("model")),
        f"## Per corridor: {best}", fmt(per_corridor[best]),
        f"## Cancellation reliability curve: {best}", fmt(reliability_curve(targets, preds[best])),
    ]
    path = out / f"scoreboard_{args.split}_{args.horizon}{'_sweep' if args.sweep else ''}.md"
    path.write_text("\n\n".join(report) + "\n")

    LOG.parent.mkdir(exist_ok=True)
    log = table.assign(run_at=dt.datetime.now().isoformat(timespec="seconds"),
                       split=args.split, horizon=args.horizon)
    log.to_csv(LOG, mode="a", header=not LOG.exists(), index=False, float_format="%.5f")

    pd.set_option("display.width", 220)
    print("\n" + table.drop(columns=["seconds"]).to_string(index=False, float_format="%.3f"))
    print(f"\nbest: {best}\nreport: {path}")


if __name__ == "__main__":
    sys.exit(main())
