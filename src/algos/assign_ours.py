"""Our allocator: late-aircraft delay map + ex-ante time-expanded fill.

The daily extra-hold total is taken from the first-stage AAR (same budget
as RBS). Predicted slack uses previous-leg delay; realized ArrDelay is
not used in the assignment. Terminators may absorb hold up to 240 min.
Connecting tails are capped at 90 min. This is the Chapter 2 assignment
step after the 2025 rate families.
"""
from __future__ import annotations

import numpy as np

import run_trc_gate as g


def rbs(budget):
    return np.asarray(budget, float).copy()


def rbd(budget, dist, has1, hmax=90.0, hmax_term=240.0):
    """Ration-by-distance analogue: short-haul flights sit extra hold first."""
    cap = np.where(has1, hmax, hmax_term).astype(float)
    h = np.zeros(len(budget))
    rem = float(np.asarray(budget, float).sum())
    for i in np.argsort(np.asarray(dist, float)):
        take = min(cap[i], rem)
        h[i] = take
        rem -= take
        if rem <= 1e-9:
            break
    return h


def greedy15(sched, s1, budget):
    n = len(sched)
    h = np.zeros(n)
    for b in np.unique(sched):
        idx = np.flatnonzero(sched == b)
        rem = float(budget[idx].sum())
        order = idx[np.argsort(-s1[idx])]
        for j in order:
            take = min(max(s1[j], 0.0), rem, 90.0)
            h[j] = take
            rem -= take
            if rem <= 1e-9:
                break
        if rem > 1e-9:
            h[idx] += rem / max(len(idx), 1)
    return h


def pred_fill(pred, s1, has1, budget, hmax=90.0, hmax_term=240.0):
    """Time-expanded fill of predicted residual slack. No realized delay."""
    n = len(budget)
    rem = float(np.asarray(budget, float).sum())
    h = np.zeros(n)
    cap = np.where(has1, hmax, hmax_term).astype(float)
    residual = np.where(has1, np.clip(s1 - np.clip(pred, -30, 180), 0.0, None), hmax_term)
    for j in np.argsort(-residual):
        take = min(cap[j], rem, residual[j])
        h[j] = take
        rem -= take
        if rem <= 1e-9:
            return h
    room = cap - h
    for j in np.argsort(-s1):
        take = min(max(room[j], 0.0), rem)
        h[j] += take
        rem -= take
        if rem <= 1e-9:
            return h
    if rem > 1e-9:
        h += rem / n
    return h


def mcf_delay(sched, s1, s2, h1, h2, pred, table):
    s1p = s1.copy()
    s1p[h1] = s1p[h1] - np.clip(pred[h1], -30, 180)
    return g.mcf(sched, s1p, s2, h1, h2, table)


def fluid_budget(sched, hod, aar_by_bin):
    """RBS analogue: fluid-queue wait in each 15-min bin, copied to flights."""
    bins = np.unique(sched)
    dem = np.array([np.sum(sched == b) for b in bins], float)
    aar = np.array([float(aar_by_bin.get(int(b), 4.0)) for b in bins], float)
    q = 0.0
    wbin = {}
    for b, d, u in zip(bins, dem, aar):
        q = q + d
        send = min(q, u)
        q = q - send
        wbin[int(b)] = float(np.clip(15.0 * q / max(d, 1.0), 0.0, 90.0))
    return np.array([wbin[int(b)] for b in sched], float)
