"""Quantile-forest stochastic predict-then-optimize AAR.

Analogue of the 2025 stochastic PTO GDP (quantile capacity, then a rate
programme). HistGradientBoosting pinball 1/3 on 15-min landed counts
given hour, visibility, wind and IMC. Open BTS+METAR, no ADS-B.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from run_bakeoff_2025 import add_lags


class QuantilePTOAAR:
    def fit(self, train_bins):
        tr = []
        for _, g in train_bins.groupby("day"):
            tr.append(add_lags(g))
        import pandas as pd

        tr = pd.concat(tr, ignore_index=True)
        X = np.column_stack(
            [
                tr["hod"], tr["vsby"], tr["sknt"], tr["imc"], tr["demand"],
                np.sin(2 * np.pi * tr["hod"] / 24.0),
                np.cos(2 * np.pi * tr["hod"] / 24.0),
            ]
        )
        y = tr["landed"].to_numpy(float)
        self.model = HistGradientBoostingRegressor(
            loss="quantile", quantile=1.0 / 3.0, max_depth=4, max_iter=120, random_state=0
        ).fit(X, y)
        return self

    def aar(self, gbin):
        g = add_lags(gbin)
        X = np.column_stack(
            [
                g["hod"], g["vsby"], g["sknt"], g["imc"], g["demand"],
                np.sin(2 * np.pi * g["hod"] / 24.0),
                np.cos(2 * np.pi * g["hod"] / 24.0),
            ]
        )
        return np.clip(self.model.predict(X), 0.5, 20.0)
