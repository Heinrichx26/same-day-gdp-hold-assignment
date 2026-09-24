"""Integer transport of a hold budget onto flights.

Each connecting flight may take at most 90 minutes (6 quanta of 15 min);
terminators may take 240 minutes. The cost of the k-th quantum is the
increment in two-leg cascade under predicted slack. Incremental costs are
convex, so a heap that always takes the cheapest next quantum solves the
same integer program as a hold-transport min-cost flow. Realized delay
does not enter the costs.
"""
from __future__ import annotations

import heapq

import numpy as np

QUANT = 15.0
K_CONN = 6
K_TERM = 16


def incremental_costs(pred, s1, s2, h1, h2):
    n = len(s1)
    slack = s1 - np.clip(np.asarray(pred, float), -30.0, 180.0)
    out = []
    for i in range(n):
        kmax = K_CONN if h1[i] else K_TERM
        inc = np.zeros(kmax)
        prev = 0.0
        for k in range(1, kmax + 1):
            w = QUANT * k
            leg2 = max(0.0, w - slack[i]) if h1[i] else 0.0
            leg3 = max(0.0, leg2 - s2[i]) if h2[i] else 0.0
            inc[k - 1] = (leg2 + leg3) - prev
            prev = leg2 + leg3
        out.append(inc)
    return out


def allocate_from_costs(incs, h1, budget):
    n = len(h1)
    total = float(np.asarray(budget, float).sum())
    qtot = int(round(total / QUANT))
    if qtot <= 0:
        return np.zeros(n)
    taken = np.zeros(n, dtype=int)
    heap = [(float(incs[i][0]), i) for i in range(n) if len(incs[i])]
    heapq.heapify(heap)
    left = qtot
    while left > 0 and heap:
        _, i = heapq.heappop(heap)
        taken[i] += 1
        left -= 1
        k = taken[i]
        if k < len(incs[i]):
            heapq.heappush(heap, (float(incs[i][k]), i))
    hold = taken.astype(float) * QUANT
    gap = total - hold.sum()
    if abs(gap) > 1e-6:
        room = np.where(h1, 90.0, 240.0) - hold
        if gap > 0:
            for i in np.argsort(-room):
                take = min(max(room[i], 0.0), gap)
                hold[i] += take
                gap -= take
                if gap <= 1e-6:
                    break
        else:
            for i in np.argsort(-hold):
                take = min(hold[i], -gap)
                hold[i] -= take
                gap += take
                if gap >= -1e-6:
                    break
    return hold


def allocate(pred, s1, s2, h1, h2, budget, bias=None):
    """Return a hold vector whose sum matches the extra-hold total.

    Optional per-flight bias is subtracted from every quantum cost of that
    flight (higher bias = cheaper to hold).
    """
    incs = incremental_costs(pred, s1, s2, h1, h2)
    if bias is not None:
        b = np.asarray(bias, float)
        incs = [inc - float(b[i]) for i, inc in enumerate(incs)]
    return allocate_from_costs(incs, h1, budget)
