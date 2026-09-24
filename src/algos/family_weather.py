"""Weather-responsive family: Hamdan et al., TRE 212 (2026) 104916.

Their ATFM model clusters meteorological conditions into scenarios and
assigns ground holding under that scenario tree. The reconstruction here
clusters hourly METAR vectors with k-medoids, builds a delay residual for
each cluster on the estimation days, and assigns the extra-hold total by
expected two-leg cascade costs across those weather scenarios.
"""
from __future__ import annotations

import numpy as np

from algos.instance import arrays
from algos.transport_hold import allocate_from_costs, incremental_costs

K = 4


def _stack_wx(wx, keys):
    rows = []
    for key in keys:
        arr = np.asarray(wx[key], float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows.append(arr)
    return np.concatenate(rows, axis=0)


def _kmedoids(x, k=K, n_iter=8, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    k = min(k, n)
    med = rng.choice(n, size=k, replace=False)
    assign = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist = np.linalg.norm(x[:, None, :] - x[med][None, :, :], axis=2)
        assign = dist.argmin(axis=1)
        new = []
        for j in range(k):
            idx = np.flatnonzero(assign == j)
            if len(idx) == 0:
                new.append(med[j])
                continue
            sub = x[idx]
            d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2).sum(axis=1)
            new.append(int(idx[int(d.argmin())]))
        new = np.array(new, int)
        if np.array_equal(new, med):
            break
        med = new
    pi = np.array([(assign == j).mean() for j in range(k)], float)
    pi = pi / max(pi.sum(), 1e-9)
    return pi, x[med], assign


class WeatherGH:
    def fit(self, train, wx, delay):
        keys = [d["day"].iloc[0] for d in train]
        x = _stack_wx(wx, keys)
        self.pi, self.medoids, _ = _kmedoids(x)
        resid = {j: [] for j in range(len(self.medoids))}
        for day in train:
            key = day["day"].iloc[0]
            pred = delay.pred(day)
            y = day["arr_delay"].to_numpy(float)
            hod = np.clip(day["hod"].to_numpy(int) - 6, 0, 16)
            w = np.asarray(wx[key], float)
            if w.ndim == 1:
                w = np.repeat(w.reshape(1, -1), 17, axis=0)
            for i, h in enumerate(hod):
                j = int(np.linalg.norm(w[h] - self.medoids, axis=1).argmin())
                resid[j].append(float(y[i] - pred[i]))
        self.shift = np.array(
            [float(np.mean(resid[j])) if resid[j] else 0.0 for j in range(len(self.medoids))],
            float,
        )
        return self

    def _scenario_preds(self, day, base_pred):
        out = []
        for j, sh in enumerate(self.shift):
            out.append(np.clip(base_pred + sh, -30.0, 180.0))
        return out

    def assign(self, day, budget, base_pred):
        a = arrays(day)
        preds = self._scenario_preds(day, base_pred)
        acc = None
        for pi, p in zip(self.pi, preds):
            costs = incremental_costs(p, a["s1"], a["s2"], a["h1"], a["h2"])
            if acc is None:
                acc = [float(pi) * c for c in costs]
            else:
                acc = [u + float(pi) * c for u, c in zip(acc, costs)]
        return allocate_from_costs(acc, a["h1"], budget)
