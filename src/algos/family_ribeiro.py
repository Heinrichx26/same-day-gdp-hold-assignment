"""Delay-predictive AAR family.

Ribeiro, Tay, Ng, Birolini, TR-C 171 (2025) 104947.
Hybrid waiting-line feature + GBM of interval delay and landed count.
AAR = 0.5 * demand * 15/(15+delay) + 0.5 * predicted landed.
Open-data analogue on BTS 15-min bins + METAR (no purchased ADS-B).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from run_bakeoff_2025 import add_lags, queue_feature
from run_loop_gdp import lookup, rate_aar, rib_X


class RibeiroAAR:
    def fit(self, train_bins):
        hod_mean = train_bins.groupby("hod")["landed"].mean()
        tr = pd_concat_lags(train_bins)
        X = rib_X(tr, hod_mean)
        self.hod_mean = hod_mean
        self.gbm_d = GradientBoostingRegressor(max_depth=3, n_estimators=140, random_state=0).fit(
            X, tr["arr"].to_numpy(float)
        )
        self.gbm_c = GradientBoostingRegressor(max_depth=3, n_estimators=140, random_state=1).fit(
            X, tr["landed"].to_numpy(float)
        )
        return self

    def aar(self, gbin):
        g = add_lags(gbin)
        X = rib_X(g, self.hod_mean)
        delay = np.clip(self.gbm_d.predict(X), 0, 90)
        land = np.clip(self.gbm_c.predict(X), 0.5, 20)
        dem = g["demand"].to_numpy(float)
        return np.clip(0.5 * rate_aar(dem, delay, 1.0) + 0.5 * land, 0.5, 20.0)


def pd_concat_lags(train_bins):
    import pandas as pd

    return pd.concat([add_lags(g) for _, g in train_bins.groupby("day")], ignore_index=True)
