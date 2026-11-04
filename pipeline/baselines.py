"""Statistical baselines (PREDICTION_MODEL_PLAN.md, Step 5).

Every predictor follows the same contract used by the scoreboard:

    predictor(history, targets) -> DataFrame[q10, q50, q90, p_cancel], indexed like targets

`history` holds every known corridor trip (labels included). A predictor may only use
history rows with service_date < the target's service_date: the "day-ahead" horizon
(as_of = 00:00 on the service day). The scoreboard checks this with a truncation test.

The histogram predictors represent a delay distribution as counts over bins
0..59 min, 60+ min, plus one extra bin for "cancelled". Recency weighting and the
look-back window are applied as a day-by-day weight matrix, so walk-forward over
every test day costs one matrix product per group.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

N_DELAY_BINS = 61            # 0..59 and 60 = "60+"
CANCEL_BIN = N_DELAY_BINS    # index of the cancelled bin
N_BINS = N_DELAY_BINS + 1
QUANTILES = (0.1, 0.5, 0.9)

SLOT = ("corridor", "daytype", "dep_time")
HOUR = ("corridor", "daytype", "dep_hour")
CORRIDOR = ("corridor", "daytype")


def day_index(dates: pd.Series) -> np.ndarray:
    return (pd.to_datetime(dates).values.astype("datetime64[D]").astype(np.int64))


def label_bins(df: pd.DataFrame) -> np.ndarray:
    delay = df["arr_delay"].fillna(0).clip(0, N_DELAY_BINS - 1).astype(int).values
    return np.where(df["cancelled"].values, CANCEL_BIN, delay)


def weighted_hist(sample_days, sample_bins, query_days, window: int,
                  half_life: Optional[float]) -> np.ndarray:
    """Histogram per query day over samples from the `window` days strictly before it."""
    if len(sample_days) == 0:
        return np.zeros((len(query_days), N_BINS), dtype=np.float32)
    days, inv = np.unique(sample_days, return_inverse=True)
    daily = np.zeros((len(days), N_BINS), dtype=np.float32)
    np.add.at(daily, (inv, sample_bins), 1.0)
    delta = query_days[:, None] - days[None, :]
    w = ((delta >= 1) & (delta <= window)).astype(np.float32)
    if half_life:
        w *= np.power(0.5, delta / half_life, dtype=np.float32)
    return w @ daily


def level_hist(history: pd.DataFrame, targets: pd.DataFrame, keys: Sequence[str],
               window: int, half_life: Optional[float]) -> np.ndarray:
    """Weighted histogram for each target row, from history rows in the same group."""
    out = np.zeros((len(targets), N_BINS), dtype=np.float32)
    hist_groups = history.groupby(list(keys), sort=False).indices
    for key, t_idx in targets.groupby(list(keys), sort=False).indices.items():
        h_idx = hist_groups.get(key)
        if h_idx is None:
            continue
        t_days = targets["_day"].values[t_idx]
        q_days, q_inv = np.unique(t_days, return_inverse=True)
        s = weighted_hist(history["_day"].values[h_idx], history["_bin"].values[h_idx],
                          q_days, window, half_life)
        out[t_idx] = s[q_inv]
    return out


def to_prediction(p: np.ndarray, index) -> pd.DataFrame:
    """Turn a (n, N_BINS) probability array into quantiles + cancel probability."""
    p_cancel = p[:, CANCEL_BIN]
    delay = p[:, :N_DELAY_BINS]
    total = delay.sum(axis=1, keepdims=True)
    cdf = np.cumsum(delay / np.where(total > 0, total, 1), axis=1)
    cols = {}
    for a in QUANTILES:
        q = (cdf >= a - 1e-9).argmax(axis=1).astype(float)
        q[total[:, 0] == 0] = 0.0
        cols[f"q{int(a * 100)}"] = q
    cols["p_cancel"] = p_cancel
    return pd.DataFrame(cols, index=index)


@dataclass
class Hierarchical:
    """Empirical distribution with hierarchical shrinkage (Step 5.1-5.3).

    Each level is blended with its parent like a Dirichlet prior with `k` pseudo-trips:
        P_level = (counts_level + k * P_parent) / (n_level + k)
    Small k ~ trust the specific departure slot; large k ~ lean on the parent.
    """
    levels: Tuple[Tuple[str, ...], ...] = (SLOT, HOUR, CORRIDOR)
    k: float = 10.0
    window: int = 90
    half_life: Optional[float] = None
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            hl = f"hl{self.half_life:g}" if self.half_life else "flat"
            self.name = f"hier_L{len(self.levels)}_k{self.k:g}_w{self.window}_{hl}"

    def __call__(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        history = history.assign(_day=day_index(history["service_date"]), _bin=label_bins(history))
        targets = targets.assign(_day=day_index(targets["service_date"]))
        preds = []
        # One corridor at a time keeps memory small.
        for corridor, t in targets.groupby("corridor", sort=False):
            h = history[history["corridor"] == corridor]
            p = None
            for keys in reversed(self.levels):          # coarsest level first
                s = level_hist(h, t, keys, self.window, self.half_life)
                n = s.sum(axis=1, keepdims=True)
                if p is None:
                    p = s / np.where(n > 0, n, 1)
                else:
                    p = (s + self.k * p) / (n + self.k)
            preds.append(to_prediction(p, t.index))
        return pd.concat(preds).loc[targets.index]


def naive(history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """"The timetable is right": zero delay, never cancelled."""
    del history
    return pd.DataFrame({"q10": 0.0, "q50": 0.0, "q90": 0.0, "p_cancel": 0.0}, index=targets.index)
naive.name = "naive_timetable"


def registry():
    """Named baselines the scoreboard runs by default."""
    return [
        naive,
        Hierarchical(levels=(CORRIDOR,), name="corridor_90d"),
        Hierarchical(k=1e-6, name="slot_90d"),            # slot only; parent used when slot is empty
        Hierarchical(k=10, name="hier_90d"),
        Hierarchical(k=10, window=365, half_life=30, name="hier_decay30"),
        # Picked by `scoreboard --split validation --sweep` (Nov 2025 - Apr 2026).
        Hierarchical(k=30, window=365, half_life=60, name="hier_tuned"),
    ]
