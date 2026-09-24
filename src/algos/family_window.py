"""Window-slot family: Wang and Ni, TRE 196 (2025) 104009.

Their low-carbon slot model reallocates flights with flight-based variable
neighbourhood search. Displacement stays inside the coordinated slot system.
On extra-hold assignment the slot is the 15-minute interval that created the
extra hold; neighbourhoods move 15-minute quanta inside that interval or to
an adjacent interval.
"""
from __future__ import annotations

import numpy as np

from algos.hold_ops import QUANT, caps


def _leg(hold, slack, s2, h1, h2):
    if not h1:
        return 0.0
    leg2 = max(0.0, hold - slack)
    return leg2 + (max(0.0, leg2 - s2) if h2 else 0.0)


def _delta(h, i, j, slack, s2, h1, h2, q=QUANT):
    return (
        _leg(h[i] - q, slack[i], s2[i], h1[i], h2[i])
        + _leg(h[j] + q, slack[j], s2[j], h1[j], h2[j])
        - _leg(h[i], slack[i], s2[i], h1[i], h2[i])
        - _leg(h[j], slack[j], s2[j], h1[j], h2[j])
    )


def _try_move(h, cap, i, j, slack, s2, h1, h2):
    if h[i] < QUANT - 1e-9 or h[j] + QUANT > cap[j] + 1e-9:
        return 0.0
    d = _delta(h, i, j, slack, s2, h1, h2)
    if d >= -1e-9:
        return 0.0
    h[i] -= QUANT
    h[j] += QUANT
    return d


def _local(h, pairs, cap, slack, s2, h1, h2, passes=4):
    for _ in range(passes):
        moved = False
        for i, j in pairs:
            if _try_move(h, cap, i, j, slack, s2, h1, h2) < 0:
                moved = True
            elif _try_move(h, cap, j, i, slack, s2, h1, h2) < 0:
                moved = True
        if not moved:
            break


def _pairs(sched, adjacent):
    groups = {}
    for i, b in enumerate(np.asarray(sched, int)):
        groups.setdefault(int(b), []).append(i)
    out = []
    for b, idx in groups.items():
        for a, i in enumerate(idx):
            for j in idx[a + 1 :]:
                out.append((i, j))
        if adjacent:
            for j in groups.get(b + 1, ()):
                for i in idx:
                    out.append((i, j))
    return out


def _shake(h, pairs, cap, rng, k):
    if not pairs:
        return
    for _ in range(k):
        i, j = pairs[int(rng.integers(len(pairs)))]
        if rng.random() < 0.5:
            i, j = j, i
        if h[i] >= QUANT - 1e-9 and h[j] + QUANT <= cap[j] + 1e-9:
            h[i] -= QUANT
            h[j] += QUANT


def window_vns(hold0, sched, pred, s1, s2, h1, h2, seed=0, k_max=3, rounds=20):
    cap = caps(h1)
    h = np.minimum(np.asarray(hold0, float).copy(), cap)
    slack = s1 - np.clip(np.asarray(pred, float), -30.0, 180.0)
    intra = _pairs(sched, adjacent=False)
    both = _pairs(sched, adjacent=True)
    rng = np.random.default_rng(seed)
    best_h = h.copy()
    best = sum(_leg(h[i], slack[i], s2[i], h1[i], h2[i]) for i in range(len(h)))
    _local(h, intra, cap, slack, s2, h1, h2)
    _local(h, both, cap, slack, s2, h1, h2)
    cur = sum(_leg(h[i], slack[i], s2[i], h1[i], h2[i]) for i in range(len(h)))
    if cur < best:
        best, best_h = cur, h.copy()
    for r in range(rounds):
        cand = best_h.copy()
        _shake(cand, both, cap, rng, 1 + (r % k_max))
        _local(cand, intra, cap, slack, s2, h1, h2)
        _local(cand, both, cap, slack, s2, h1, h2)
        val = sum(_leg(cand[i], slack[i], s2[i], h1[i], h2[i]) for i in range(len(cand)))
        if val + 1e-9 < best:
            best, best_h = val, cand
    return best_h
