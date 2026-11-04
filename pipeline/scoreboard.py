"""Scoreboard (PREDICTION_MODEL_PLAN.md, Step 4).

Scores every predictor the same way on a time-based split, overall, per corridor
and for the weekday morning peak. Before scoring it checks each predictor for
look-ahead leakage by re-predicting a few days with history truncated at that day.

Usage:
    python -m pipeline.scoreboard                         # all baselines on validation
    python -m pipeline.scoreboard --split test
    python -m pipeline.scoreboard --split validation --sweep   # tune k / half-life
    python -m pipeline.scoreboard --split test --horizon t15   # baselines + GBM at T-15 min
    python -m pipeline.scoreboard --horizon t15 --folds monthly # rolling-origin, Jan 2024 - Oct 2026
    python -m pipeline.scoreboard --horizon day --folds monthly --folds-start 2020-01 --exclude-covid

Every run also reports, against --reference (default hier_tuned):
  - a day-block bootstrap (2,000 resamples of days) of each loss difference, 95% interval
  - a regime breakdown (corridor-day cancel share, frost days, own-train signal)
  - first-cancellation recall (first cancelled trip of a corridor-day with p >= 0.4)
  - passenger-delay metrics (pax_surprise_5, calibration of P(pax_delay <= X))

Writes data/reports/scoreboard_<split>_<horizon>.md and appends to experiments/log.csv,
stamped with the git commit and the feature-set name.
"""
import argparse
import datetime as dt
import subprocess
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
PAX_X = (2, 5, 10)                 # minutes: calibration of P(pax_delay <= X)
FIRST_CANCEL_P = 0.4
BOOT_N = 2000
FOLDS_START, FOLDS_END = "2024-01", "2026-10"


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
    pax = pax_metrics(y, p)
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
        **pax,
    }


def p_le(p: pd.DataFrame, x: float) -> np.ndarray:
    """P(passenger delay <= x) from a predictor's (q10, q50, q90, p_cancel).

    Delay CDF if the train runs: piecewise linear through (q10, .1), (q50, .5), (q90, .9),
    linear from (-1, 0) up to q10, exponential tail above q90. A cancelled train is
    counted as delay > x (the commuter waits for the next one): an approximation until
    the binned model (task 13) predicts pax_delay directly.
    """
    q = np.sort(p[["q10", "q50", "q90"]].values.astype(float), axis=1)
    q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
    f = np.where(x < q10, 0.1 * (x + 1) / (q10 + 1),
        np.where(x < q50, 0.1 + 0.4 * (x - q10) / np.maximum(q50 - q10, 1e-9),
        np.where(x < q90, 0.5 + 0.4 * (x - q50) / np.maximum(q90 - q50, 1e-9),
                 1 - 0.1 * np.exp(-(x - q90) / np.maximum(q90 - q50, 1.0)))))
    f = np.clip(f, 0, 1)
    return (1 - p["p_cancel"].values) * f


def pax_metrics(y: pd.DataFrame, p: pd.DataFrame) -> dict:
    """Commuter-level metrics on pax_delay (all trips, cancelled ones included)."""
    if "pax_delay" not in y:
        return {}
    has = y["pax_delay"].notna().values
    pd_ = y["pax_delay"].values[has].astype(float)
    out = {"pax_surprise_5": np.mean(pd_ > p["q90"].values[has] + 5) if has.any() else np.nan}
    for x in PAX_X:
        pr = p_le(p[has], x)
        obs = (pd_ <= x).astype(float)
        out[f"pax_le{x}_pred"] = pr.mean() if has.any() else np.nan
        out[f"pax_le{x}_obs"] = obs.mean() if has.any() else np.nan
        out[f"brier_pax_le{x}"] = np.mean((pr - obs) ** 2) if has.any() else np.nan
    return out


# ---------------------------------------------------------------- per-trip losses + bootstrap

LOSSES = ("pinball", "mae_p50", "brier_cancel", "brier_pax_le5")


def trip_losses(y: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Per-trip losses whose sums define the headline metrics (NaN where undefined)."""
    ran = ~y["cancelled"].values
    yd = np.where(ran, y["arr_delay"].values.astype(float), np.nan)
    pin = np.mean([np.maximum(a * (yd - p[c].values), (a - 1) * (yd - p[c].values))
                   for a, c in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90"))], axis=0)
    out = pd.DataFrame({"pinball": pin, "mae_p50": np.abs(yd - p["q50"].values),
                        "brier_cancel": (p["p_cancel"].values - y["cancelled"].values.astype(float)) ** 2},
                       index=y.index)
    if "pax_delay" in y:
        obs = np.where(y["pax_delay"].notna(), (y["pax_delay"].values <= 5).astype(float), np.nan)
        out["brier_pax_le5"] = (p_le(p, 5) - obs) ** 2
    return out


def bootstrap(targets: pd.DataFrame, losses: dict, reference: str, n: int = BOOT_N, seed: int = 0):
    """Day-block bootstrap of the relative difference in summed loss vs `reference`.

    Days (all corridors together) are resampled with replacement; the statistic is
    sum(loss_model) / sum(loss_ref) - 1 over the resampled days. Returns a table with
    the point estimate and the 2.5 / 97.5 percentiles, in %."""
    days, day_idx = np.unique(targets["service_date"].values, return_inverse=True)
    counts = np.random.default_rng(seed).multinomial(len(days), np.full(len(days), 1 / len(days)), size=n)

    def day_sums(df, col):
        return np.bincount(day_idx, weights=np.nan_to_num(df[col].values), minlength=len(days))

    rows = []
    ref = losses[reference]
    for name, l in losses.items():
        if name == reference:
            continue
        row = {"model": name, "vs": reference}
        for col in LOSSES:
            if col not in l or col not in ref:
                continue
            m, r = day_sums(l, col), day_sums(ref, col)
            point = m.sum() / r.sum() - 1
            boot = (counts @ m) / (counts @ r) - 1
            lo, hi = np.percentile(boot, [2.5, 97.5])
            sig = "*" if lo > 0 or hi < 0 else ""
            row[col] = f"{100 * point:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]{sig}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------- regimes + first-cancel recall

def regimes(targets: pd.DataFrame) -> dict:
    """Evaluation strata. Built from outcomes and outcome-day weather: never a feature."""
    share = targets.groupby(["corridor", "service_date"])["cancelled"].transform("mean").values
    r = {
        "cancel share 0": share == 0,
        "cancel share <10%": (share > 0) & (share < 0.10),
        "cancel share 10-30%": (share >= 0.10) & (share <= 0.30),
        "cancel share >30%": share > 0.30,
    }
    if "wx_tmin" in targets:
        r["frost day (Tmin < -2 C)"] = (targets["wx_tmin"] < -2).values
        r["no frost"] = (targets["wx_tmin"] >= -2).values
    if "own_known" in targets:
        r["own-train signal"] = (targets["own_known"] == 1).values
        r["no own-train signal"] = (targets["own_known"] != 1).values
    return r


REGIME_COLS = ("trips", "pinball", "coverage_10_90", "above_p90", "brier_cancel", "cancel_rate", "pax_surprise_5")


def regime_table(targets: pd.DataFrame, preds: dict) -> pd.DataFrame:
    rows = []
    for rname, mask in regimes(targets).items():
        if mask.sum() == 0 or (~targets["cancelled"].values[mask]).sum() == 0:
            continue
        for name, p in preds.items():
            m = metrics(targets[mask], p[mask])
            rows.append({"regime": rname, "model": name, **{c: m.get(c) for c in REGIME_COLS}})
    return pd.DataFrame(rows).set_index(["regime", "model"])


def first_cancel_recall(targets: pd.DataFrame, p: pd.DataFrame, threshold: float = FIRST_CANCEL_P) -> dict:
    """Of cancellations that are the first on their corridor that day, the share with p >= threshold."""
    c = targets[targets["cancelled"]].sort_values("dep_planned")
    first = c.groupby(["corridor", "service_date"]).head(1).index
    later = c.index.difference(first)
    pc = p["p_cancel"]
    return {"first_cancels": len(first), "first_recall": float((pc[first] >= threshold).mean()) if len(first) else np.nan,
            "later_cancels": len(later), "later_recall": float((pc[later] >= threshold).mean()) if len(later) else np.nan}


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


def drop_covid(targets: pd.DataFrame) -> pd.DataFrame:
    """Scoring filter only: COVID days stay in the history predictors learn from."""
    from .external import COVID_END, COVID_START
    return targets[~targets["service_date"].between(COVID_START, COVID_END)]


def score(predictors, split: str, horizon: str = "day", check: bool = True, exclude_covid: bool = False):
    trips = load_trips(horizon)
    start, end = map(pd.Timestamp, SPLITS[split])
    history = trips[trips["service_date"] <= end]          # predictors only look back
    targets = trips[(trips["service_date"] >= start) & (trips["service_date"] <= end)]
    if exclude_covid:
        targets = drop_covid(targets)

    rows, per_corridor, preds = [], {}, {}
    for pr in predictors:
        t0 = time.time()
        p = pr(history, targets)
        secs = time.time() - t0
        if check:
            leakage_check(pr, history, targets, p)
        preds[pr.name] = p
        m = metrics(targets, p)
        m.update({f"fc_{k}": v for k, v in first_cancel_recall(targets, p).items() if k.endswith("recall")})
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


def git_stamp() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "pipeline"], cwd=config.ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def feature_set(pr) -> str:
    return f"{pr.horizon}/{pr.weather}" if hasattr(pr, "weather") else "history-only"


def append_log(rows: pd.DataFrame) -> None:
    """Append to experiments/log.csv, widening the header when new columns appear."""
    LOG.parent.mkdir(exist_ok=True)
    if LOG.exists():
        old = pd.read_csv(LOG)
        if set(rows.columns) - set(old.columns) or list(old.columns) != list(rows.columns):
            pd.concat([old, rows], ignore_index=True).to_csv(LOG, index=False, float_format="%.5f")
            return
    rows.to_csv(LOG, mode="a", header=not LOG.exists(), index=False, float_format="%.5f")


def monthly_folds(make_predictors, horizon: str, start: str = FOLDS_START, end: str = FOLDS_END,
                  exclude_covid: bool = False):
    """Rolling origin: each month is scored by predictors trained on everything before it."""
    trips = load_trips(horizon)
    rows = []
    for month in pd.period_range(start, end, freq="M"):
        m0, m1 = month.start_time, month.end_time.normalize()
        history = trips[trips["service_date"] <= m1]
        targets = trips[(trips["service_date"] >= m0) & (trips["service_date"] <= m1)]
        if exclude_covid:
            targets = drop_covid(targets)
        if targets.empty:
            continue
        for pr in make_predictors(str(m0.date())):
            p = pr(history, targets)
            m = metrics(targets, p)
            rows.append({"month": str(month), "model": pr.name, **m})
        print(f"  {month}: {len(targets)} trips")
    df = pd.DataFrame(rows)
    cols = ["pinball", "coverage_10_90", "surprise_5", "brier_cancel", "pax_surprise_5", "brier_pax_le5"]
    cols = [c for c in cols if c in df]
    summary = df.groupby("model")[cols].agg(["median", "max"])
    worst = df.loc[df.groupby("model")["pinball"].idxmax(), ["model", "month", "pinball"]].set_index("model")
    return df, summary, worst


def fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".3f") if hasattr(df, "to_markdown") else df.to_string()


def make_predictors(horizon: str, train_end: str):
    from .model import GBM, Persistence
    predictors = baselines.registry()
    if horizon != "day":
        predictors.append(Persistence(next(p for p in predictors if p.name == "hier_tuned")))
    predictors.append(GBM(horizon, train_end=train_end, calibrate=False))
    predictors.append(GBM(horizon, train_end=train_end))
    return predictors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, default="validation")
    ap.add_argument("--sweep", action="store_true", help="grid over k and half-life")
    ap.add_argument("--horizon", choices=HORIZON_LABEL, default="day")
    ap.add_argument("--no-check", action="store_true", help="skip the leakage check")
    ap.add_argument("--reference", default="hier_tuned", help="model the bootstrap compares against")
    ap.add_argument("--folds", choices=["monthly"], help="rolling-origin evaluation instead of a split")
    ap.add_argument("--folds-start", default=FOLDS_START, help="first fold month, YYYY-MM")
    ap.add_argument("--exclude-covid", action="store_true", help="do not score COVID-period days")
    args = ap.parse_args(argv)
    out = config.DATA / "reports"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    horizon = HORIZON_LABEL[args.horizon]

    if args.folds:
        print(f"monthly folds {args.folds_start} to {FOLDS_END}, horizon {horizon}"
              + (", COVID days excluded" if args.exclude_covid else ""))
        df, summary, worst = monthly_folds(lambda end: make_predictors(args.horizon, end), args.horizon,
                                           start=args.folds_start, exclude_covid=args.exclude_covid)
        path = out / f"scoreboard_folds_{args.horizon}.md"
        path.write_text("\n\n".join([
            f"# Rolling-origin folds {args.folds_start} to {FOLDS_END}: {horizon}"
            + (" (COVID days excluded)" if args.exclude_covid else ""),
            "Each month scored by predictors trained on everything before it. median / max over months.",
            "## Summary", fmt(summary), "## Worst month (pinball)", fmt(worst),
            "## Per month", fmt(df.set_index(["month", "model"]))]) + "\n")
        print(summary.to_string(float_format="%.3f"))
        print(f"\nreport: {path}")
        return

    if args.sweep:
        predictors = [baselines.Hierarchical(k=k, window=w, half_life=hl)
                      for k in (3, 10, 30)
                      for w, hl in ((90, None), (365, 14), (365, 30), (365, 60))]
    else:
        predictors = make_predictors(args.horizon, SPLITS[args.split][0])

    print(f"split {args.split} {SPLITS[args.split]}, horizon {horizon}")
    table, per_corridor, preds, targets = score(predictors, args.split, args.horizon,
                                                check=not args.no_check, exclude_covid=args.exclude_covid)
    table = table.sort_values("pinball").reset_index(drop=True)
    best = table.loc[0, "model"]
    ref = args.reference if args.reference in preds else best
    losses = {name: trip_losses(targets, p) for name, p in preds.items()}
    boot = bootstrap(targets, losses, ref)
    reg = regime_table(targets, preds)

    report = [
        f"# Scoreboard: {args.split} {SPLITS[args.split][0]} to {SPLITS[args.split][1]}",
        f"Horizon: {horizon}. Delay metrics on trips that ran; Brier and pax metrics on all trips. "
        "NS's own prediction column needs the live logger (not yet available).",
        "## All corridors", fmt(table.set_index("model")),
        f"## Difference vs {ref}: day-block bootstrap",
        f"Relative difference in summed loss, {BOOT_N} resamples of {targets['service_date'].nunique()} days, "
        "95% interval. Negative = better than the reference. * = interval excludes zero.",
        fmt(boot),
        "## Regimes", "Strata from outcomes (corridor-day cancel share) and outcome-day weather; never features.",
        fmt(reg),
        f"## Per corridor: {best}", fmt(per_corridor[best]),
        f"## Cancellation reliability curve: {best}", fmt(reliability_curve(targets, preds[best])),
    ]
    path = out / f"scoreboard_{args.split}_{args.horizon}{'_sweep' if args.sweep else ''}.md"
    path.write_text("\n\n".join(report) + "\n")

    stamp = git_stamp()
    fsets = {pr.name: feature_set(pr) for pr in predictors}
    append_log(table.assign(run_at=dt.datetime.now().isoformat(timespec="seconds"),
                            split=args.split, horizon=args.horizon, commit=stamp,
                            feature_set=table["model"].map(fsets)))

    cols = ["model", "pinball", "coverage_10_90", "surprise_5", "brier_cancel", "fc_first_recall",
            "pax_surprise_5", "brier_pax_le5"]
    print("\n" + table[[c for c in cols if c in table]].to_string(index=False, float_format="%.3f"))
    print(f"\nvs {ref}:\n{boot.to_string()}")
    print(f"\nbest: {best}\nreport: {path}")


if __name__ == "__main__":
    sys.exit(main())
