"""Late-aircraft delay map used by our allocator and by incremental DCB search."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import run_close_july as cj


class LateAircraftHGB:
    def fit(self, may, wx, te):
        _, hgb = cj.fit_models(may, wx, te)
        self.hgb, self.wx, self.te = hgb, wx, te
        self.quantile = None
        return self

    def pred(self, day):
        return cj.pred_model(self.hgb, day, self.wx, self.te)[0]


class QuantileDelayHGB:
    """Confirmation-day quantile of inbound delay for predict-then-assign."""

    def __init__(self, quantile=0.5):
        self.quantile = quantile
        self.hgb = None
        self.wx = None
        self.te = None

    def fit(self, train, wx, te):
        Xtr, ytr = [], []
        for day in train:
            X, y = cj.rich_xy(day, wx, te)
            Xtr.append(X)
            ytr.append(y)
        self.hgb = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(self.quantile),
            max_depth=6,
            max_iter=200,
            learning_rate=0.08,
            random_state=0,
        ).fit(np.vstack(Xtr), np.concatenate(ytr))
        self.wx, self.te = wx, te
        return self

    def pred(self, day):
        return cj.pred_model(self.hgb, day, self.wx, self.te)[0]
