"""Metaheuristic DCB family: Melgosa, Vidosavljevic and Prats, TR-C 180 (2025) 105306.

Their hybrid method uses simulated annealing on the demand-side delay vector
and a constructive repair on the capacity side. Paper settings used here:
cooling factor 0.995, 2000 transitions per run, T_min = T_init * 1e-4,
heating when the accepted share falls below 0.8 on a block of transitions.
On extra-hold assignment the demand move is a 15-minute transfer between
flights; the repair clips connecting/terminating caps and restores the daily
total in schedule order. Predicted cascade is the acceptance metric.
"""
from __future__ import annotations

import numpy as np

from algos.hold_ops import QUANT, caps, pred_cascade

ALPHA = 0.995
N_TRANS = 2000
TMIN_RATIO = 1e-4
HEAT_FRAC = 0.8
BLOCK = 200


def _repair(h, cap, total):
    h = np.minimum(np.maximum(h, 0.0), cap)
    gap = float(total) - float(h.sum())
    if abs(gap) <= 1e-6:
        return h
    if gap > 0:
        room = cap - h
        for i in np.argsort(-room):
            take = min(max(room[i], 0.0), gap)
            h[i] += take
            gap -= take
            if gap <= 1e-6:
                break
    else:
        for i in np.argsort(-h):
            take = min(h[i], -gap)
            h[i] -= take
            gap += take
            if gap >= -1e-6:
                break
    return h


def _t_init(hold0, cap, pred, s1, s2, h1, h2, rng, trials=40):
    h = np.asarray(hold0, float)
    n = len(h)
    diffs = []
    cur = pred_cascade(h, pred, s1, s2, h1, h2)
    for _ in range(trials):
        i, j = rng.integers(n, size=2)
        if i == j or h[i] < QUANT or h[j] + QUANT > cap[j] + 1e-9:
            continue
        h[i] -= QUANT
        h[j] += QUANT
        nxt = pred_cascade(h, pred, s1, s2, h1, h2)
        diffs.append(abs(nxt - cur))
        h[i] += QUANT
        h[j] -= QUANT
    mean = float(np.mean(diffs)) if diffs else 25.0
    return max(mean / max(-np.log(0.8), 1e-6), 1.0)


def anneal_budget(hold0, pred, s1, s2, h1, h2, seed=0, steps=N_TRANS):
    rng = np.random.default_rng(seed)
    cap = caps(h1)
    total = float(np.asarray(hold0, float).sum())
    h = _repair(np.asarray(hold0, float).copy(), cap, total)
    cur = pred_cascade(h, pred, s1, s2, h1, h2)
    best, best_h = cur, h.copy()
    n = len(h)
    temp = _t_init(h, cap, pred, s1, s2, h1, h2, rng)
    tmin = max(temp * TMIN_RATIO, 1e-6)
    accepted = 0
    block_acc = 0
    for t in range(1, steps + 1):
        i, j = rng.integers(n, size=2)
        if i == j or h[i] < QUANT - 1e-9 or h[j] + QUANT > cap[j] + 1e-9:
            temp = max(temp * ALPHA, tmin)
            continue
        cand = h.copy()
        cand[i] -= QUANT
        cand[j] += QUANT
        cand = _repair(cand, cap, total)
        nxt = pred_cascade(cand, pred, s1, s2, h1, h2)
        if nxt <= cur or rng.random() < np.exp(min(20.0, (cur - nxt) / max(temp, 1e-6))):
            h, cur = cand, nxt
            accepted += 1
            block_acc += 1
            if cur < best:
                best, best_h = cur, h.copy()
        temp = max(temp * ALPHA, tmin)
        if t % BLOCK == 0:
            if block_acc < HEAT_FRAC * BLOCK:
                temp = max(temp / (ALPHA ** 25), tmin * 10.0)
            block_acc = 0
    return best_h
