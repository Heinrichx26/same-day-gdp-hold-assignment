"""Distributionally robust AAR family.

Wu, Estes, Li (arXiv 2306.09836); Wu et al. DR-MAGHP (arXiv 2509.18492).
One-airport analogue: hour-of-day one-third quantile of 15-min landed
counts, minus a 0.5-aircraft Wasserstein-1 shift.
"""
from __future__ import annotations

import numpy as np

from run_loop_gdp import lookup


class WuAAR:
    def fit(self, train_bins):
        self.p33 = train_bins.groupby("hod")["landed"].quantile(1.0 / 3.0)
        return self

    def aar(self, gbin):
        return np.clip(lookup(self.p33, gbin["hod"]) - 0.5, 0.5, 20.0)
