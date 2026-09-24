"""Operational-priority ranking: Zeng et al., JATM 124 (2025) 102751.

Their slot model searches by IATA operational priority in hierarchical
classes. On extra-hold assignment the classes are terminating flights,
then connecting flights in remaining-predicted-buffer quartiles. Each
class is filled to its cap before the next class; a short intra-class
search then moves 15-minute quanta when predicted cascade falls.
This ranking is the first ranking of TLNEHA and the leftover baseline.
"""
from __future__ import annotations

import numpy as np

from algos.hold_ops import QUANT, caps


def _leg(hold, slack, s2, h1, h2):
    if not h1:
        return 0.0
    leg2 = max(0.0, hold - slack)
    return leg2 + (max(0.0, leg2 - s2) if h2 else 0.0)


def hierarchical_priority(pred, s1, has1, s2, h2, budget, hmax=90.0, hmax_term=240.0):
    n = len(s1)
    rem = float(np.asarray(budget, float).sum())
    h = np.zeros(n)
    cap = np.where(has1, hmax, hmax_term).astype(float)
    resid = np.where(has1, np.clip(s1 - np.clip(pred, -30.0, 180.0), 0.0, None), hmax_term)
    term = np.flatnonzero(~np.asarray(has1, bool))
    conn = np.flatnonzero(np.asarray(has1, bool))
    classes = [term]
    if len(conn):
        q = np.quantile(resid[conn], [0.75, 0.5, 0.25])
        for lo, hi in ((np.inf, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], -np.inf)):
            classes.append(conn[(resid[conn] <= lo) & (resid[conn] > hi)])
    for cls in classes:
        if rem <= 1e-9 or len(cls) == 0:
            continue
        order = cls[np.argsort(-resid[cls])]
        for j in order:
            take = min(cap[j] - h[j], resid[j] - h[j] if has1[j] else cap[j] - h[j], rem)
            take = max(take, 0.0)
            h[j] += take
            rem -= take
            if rem <= 1e-9:
                break
    if rem > 1e-9:
        room = cap - h
        for j in np.argsort(-resid):
            take = min(max(room[j], 0.0), rem)
            h[j] += take
            rem -= take
            if rem <= 1e-9:
                break
    slack = s1 - np.clip(pred, -30.0, 180.0)
    cap_arr = caps(has1)
    for _ in range(3):
        moved = False
        for cls in classes:
            if len(cls) < 2:
                continue
            for a, i in enumerate(cls):
                for j in cls[a + 1 :]:
                    for src, dst in ((i, j), (j, i)):
                        if h[src] < QUANT - 1e-9 or h[dst] + QUANT > cap_arr[dst] + 1e-9:
                            continue
                        before = _leg(h[src], slack[src], s2[src], has1[src], h2[src]) + _leg(
                            h[dst], slack[dst], s2[dst], has1[dst], h2[dst]
                        )
                        after = _leg(h[src] - QUANT, slack[src], s2[src], has1[src], h2[src]) + _leg(
                            h[dst] + QUANT, slack[dst], s2[dst], has1[dst], h2[dst]
                        )
                        if after + 1e-9 < before:
                            h[src] -= QUANT
                            h[dst] += QUANT
                            moved = True
        if not moved:
            break
    return h
