"""15-minute window engines: CASA and incremental DCB search.

CASA: EUROCONTROL ATFCM slot engine; request-order fill of each 15-min
count window (Melgosa et al., TR-C 180 (2025) 105306 baseline).

Incremental search: Chen, Dalmau, Alam, TR-C (2025) tactical DCB analogue.
Start from CASA, then move connecting flights one bin at a time when the
predicted two-leg cascade falls and the count cap still holds.
"""
from __future__ import annotations

import numpy as np


def _cap_of(bin_id, table, n):
    hod = (int(bin_id) * 15) // 60
    return max(int(np.floor(table.get(hod, 4.0))), 1) if hod <= 22 else n


def casa(sched, table):
    n = len(sched)
    order = np.argsort(sched, kind="mergesort")
    used = {}
    wait = np.zeros(n)
    for i in order:
        b = int(sched[i])
        while used.get(b, 0) >= _cap_of(b, table, n):
            b += 1
        used[b] = used.get(b, 0) + 1
        wait[i] = 15.0 * max(0, b - int(sched[i]))
    return wait


def incremental_search(sched, table, s1, s2, h1, h2, pred, passes=3):
    """Local search on the CASA endowment. Predicted slack, not realized delay."""
    n = len(sched)
    wait = casa(sched, table)
    release = sched + (wait / 15.0).astype(int)
    used = {}
    for b in release:
        used[int(b)] = used.get(int(b), 0) + 1
    slack = s1 - np.clip(pred, -30, 180)

    def cost_of(w):
        leg2 = np.where(h1, np.maximum(0.0, w - slack), 0.0)
        leg3 = np.where(h2, np.maximum(0.0, leg2 - s2), 0.0)
        return leg2 + leg3

    conn = np.where(h1)[0]
    term = np.where(~h1)[0]
    for _ in range(passes):
        cur = cost_of(wait)
        improved = False
        for i in conn[np.argsort(-cur[conn])]:
            b0 = int(release[i])
            best = None
            best_drop = 0.0
            for db in (-2, -1, 1, 2, 3):
                b = b0 + db
                if b < int(sched[i]):
                    continue
                w2 = wait.copy()
                w2[i] = 15.0 * max(0, b - int(sched[i]))
                partner = None
                if used.get(b, 0) >= _cap_of(b, table, n) and b != b0:
                    cand = [j for j in term if int(release[j]) == b and int(sched[j]) <= b0]
                    if not cand:
                        continue
                    partner = cand[0]
                    w2[partner] = 15.0 * max(0, b0 - int(sched[partner]))
                drop = float(cost_of(wait)[i] - cost_of(w2)[i])
                if partner is not None:
                    drop -= float(cost_of(w2)[partner] - cost_of(wait)[partner])
                if drop > best_drop + 1e-6:
                    best_drop, best = drop, (b, partner, w2[i], None if partner is None else w2[partner])
            if best is None:
                continue
            b, partner, wi, wp = best
            used[b0] = used.get(b0, 1) - 1
            used[b] = used.get(b, 0) + 1
            if partner is not None:
                used[b] -= 1
                used[b0] = used.get(b0, 0) + 1
                release[partner] = b0
                wait[partner] = wp
            release[i] = b
            wait[i] = wi
            improved = True
        if not improved:
            break
    return wait
