"""Gradient-boosted models (PREDICTION_MODEL_PLAN.md, Step 7).

Per horizon: three quantile regressors (p10/p50/p90 of arrival delay, trained on
trips that ran) and one cancellation classifier. scikit-learn's
HistGradientBoosting is used instead of LightGBM: same algorithm family, and it
installs without a system OpenMP library.

Training data is every trip from TRAIN_START up to (not including) `train_end`,
which the scoreboard sets to the start of the evaluation window. Features come
from pipeline.features and are point-in-time by construction.

Usage (permutation importance for one horizon):
    python -m pipeline.model --horizon t15
"""
import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from . import config
from .baselines import QUANTILES
from .features import features_for

TRAIN_START = "2023-04-01"     # first 3 months only warm up the history features
CATEGORICAL = ("corridor_code", "service_type_code", "isodow")
DELAY_CAP = 60


def _params(**kw):
    base = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=0)
    base.update(kw)
    return base


@dataclass
class GBM:
    """Quantile + cancellation models for one horizon.

    With `calibrate` (Step 7.3) the models are first fit on data up to 8 weeks before
    train_end. On those last 8 weeks we learn (a) a Platt correction of the cancel
    probability (off by default, see `platt`) and (b) conformal offsets for p10/p90 (CQR). The final models are then
    refit on everything before train_end and the corrections applied on top.
    """
    horizon: str
    train_end: str
    weather: str = "hourly"
    calibrate: bool = True
    platt: bool = False        # tried on validation: no Brier gain, and it flattened day-ahead scores
    cal_weeks: int = 8
    name: str = ""
    _fitted: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.features = features_for(self.horizon, self.weather)
        self.name = self.name or f"gbm_{self.horizon}" + ("" if self.calibrate else "_raw")

    def _cat_mask(self):
        return [f in CATEGORICAL for f in self.features]

    def _fit_models(self, h: pd.DataFrame) -> dict:
        X = h[self.features].astype(float)
        ran = ~h["cancelled"].values
        y = h.loc[ran, "arr_delay"].clip(0, DELAY_CAP).astype(float)
        models = {}
        for a in QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=a,
                                              categorical_features=self._cat_mask(), **_params())
            models[a] = m.fit(X[ran], y)
        clf = HistGradientBoostingClassifier(categorical_features=self._cat_mask(), **_params())
        models["cancel"] = clf.fit(X, h["cancelled"].astype(int))
        return models

    def _raw(self, models: dict, df: pd.DataFrame):
        X = df[self.features].astype(float)
        q = np.column_stack([models[a].predict(X) for a in QUANTILES])
        return q, models["cancel"].predict_proba(X)[:, 1]

    def fit(self, history: pd.DataFrame):
        h = history[(history["service_date"] >= TRAIN_START) & (history["service_date"] < self.train_end)]
        cal = {"q10": 0.0, "q90": 0.0, "platt": None}
        if self.calibrate:
            cal_start = pd.Timestamp(self.train_end) - pd.Timedelta(weeks=self.cal_weeks)
            early, window = h[h["service_date"] < cal_start], h[h["service_date"] >= cal_start]
            q, p = self._raw(self._fit_models(early), window)
            ran = ~window["cancelled"].values
            y = window["arr_delay"].values[ran].astype(float)
            cal["q10"] = float(np.quantile(y - q[ran, 0], 0.1))
            cal["q90"] = float(np.quantile(y - q[ran, 2], 0.9))
            logit = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            if self.platt:
                cal["platt"] = LogisticRegression(C=1e6).fit(logit[:, None], window["cancelled"].astype(int))
        self._fitted = {"train_end": self.train_end, "models": self._fit_models(h), "cal": cal,
                        "rows": len(h)}
        return self

    def __call__(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        # The model only ever sees rows before train_end, so a fit is reusable across calls
        # (including the scoreboard's truncated-history leakage check).
        if self._fitted.get("train_end") != self.train_end:
            self.fit(history)
        cal = self._fitted["cal"]
        q, p = self._raw(self._fitted["models"], targets)
        q[:, 0] += cal["q10"]
        q[:, 2] += cal["q90"]
        if cal["platt"] is not None:
            logit = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            p = cal["platt"].predict_proba(logit[:, None])[:, 1]
        q = np.sort(np.clip(np.round(q), 0, DELAY_CAP), axis=1)   # whole minutes, no crossing
        return pd.DataFrame({"q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2], "p_cancel": p},
                            index=targets.index)


@dataclass
class Persistence:
    """Stand-in for NS's own propagation: the history distribution shifted by the
    train's last known delay. Real NS predictions need the live logger."""
    base: object
    name: str = "persist_last_delay"

    def __call__(self, history, targets):
        p = self.base(history, targets)
        shift = targets["own_last_delay"].astype(float).fillna(0).values
        for c in ("q10", "q50", "q90"):
            p[c] = p[c] + shift
        return p


def importance(horizon: str, train_end: str = "2026-05-01", n: int = 30000, seed: int = 0):
    """Permutation importance of each feature for p90 pinball loss and cancel Brier."""
    from .scoreboard import load_trips
    trips = load_trips(horizon)
    model = GBM(horizon, train_end)
    model(trips, trips.head(1))
    test = trips[trips["service_date"] >= train_end].sample(n, random_state=seed)
    rng = np.random.default_rng(seed)

    def losses(df):
        p = model(trips, df)
        ran = ~df["cancelled"].values
        y = df["arr_delay"].values[ran]
        e = y - p["q90"].values[ran]
        return np.mean(np.maximum(0.9 * e, -0.1 * e)), np.mean((p["p_cancel"] - df["cancelled"]) ** 2)

    base90, basec = losses(test)
    rows = []
    for f in model.features:
        t = test.copy()
        t[f] = rng.permutation(t[f].values)
        l90, lc = losses(t)
        rows.append((f, l90 - base90, lc - basec))
    out = pd.DataFrame(rows, columns=["feature", "d_pinball_p90", "d_brier_cancel"])
    return out.sort_values("d_pinball_p90", ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="t15")
    args = ap.parse_args(argv)
    imp = importance(args.horizon)
    path = config.DATA / "reports" / f"importance_{args.horizon}.csv"
    imp.to_csv(path, index=False)
    print(imp.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    sys.exit(main())
