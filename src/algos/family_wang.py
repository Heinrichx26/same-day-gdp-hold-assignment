"""Declared-capacity AAR family.

Wang et al., TR-C 171 (2025) 105012.
k-medoids + DTW on daily landing profiles, then a one-third newsvendor
mix of the medoids as the declared 15-min rate.
"""
from __future__ import annotations

import numpy as np

from run_bakeoff_2025 import hod_series, kmedoids_dtw, nv_quantile
from run_loop_gdp import lookup


class WangAAR:
    def fit(self, train_bins):
        series = [hod_series(gb) for _, gb in train_bins.groupby("day")]
        pi, medoids, _ = kmedoids_dtw(series, k=min(4, len(series)))
        self.table = {
            hod: max(nv_quantile(np.array([medoids[k][hi] for k in range(len(medoids))]), pi, 1.0 / 3.0), 0.5)
            for hi, hod in enumerate(range(6, 23))
        }
        return self

    def aar(self, gbin):
        return lookup(self.table, gbin["hod"])
