"""Second-stage GDP assignment: interval conservation, CDM rules, cascade LP.

Closed-loop arrival delay is predicted unimpeded delay plus assigned hold.
Cascade is piecewise-linear convex in hold, so a one-knapsack assignment
is solved by marginal-cost fill and a two-margin assignment is a linear
program.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algos.assign_ours import pred_fill
from algos.instance import arrays
from algos.transport_hold import QUANT, allocate

CAP_CONN = 90.0
CAP_TERM = 240.0


def caps(h1):
    return np.where(h1, CAP_CONN, CAP_TERM).astype(np.float64)


def pack(day):
    a = arrays(day)
    n = len(a["sched"])
    if "Reporting_Airline" in day.columns:
        a["carrier"] = day["Reporting_Airline"].fillna("UNK").astype(str).to_numpy()
    else:
        a["carrier"] = np.array(["UNK"] * n)
    a["origin"] = (
        day["Origin"].astype(str).to_numpy() if "Origin" in day.columns else np.array(["UNK"] * n)
    )
    late = (
        pd.to_numeric(day["LateAircraftDelay"], errors="coerce").fillna(0.0).to_numpy(float)
        if "LateAircraftDelay" in day.columns
        else np.zeros(n)
    )
    a["late"] = late
    a["base_delay"] = np.clip(a["delay"] - late, -30.0, 180.0)
    if "sched" in day.columns:
        st = pd.to_datetime(day["sched"], utc=False)
        try:
            arr_min = st.dt.hour.to_numpy(int) * 60 + st.dt.minute.to_numpy(int)
        except Exception:
            arr_min = a["hod"] * 60
    else:
        arr_min = a["hod"] * 60
    a["arr_clock"] = arr_min.astype(float)
    a["dep_clock"] = a["arr_clock"] - np.asarray(a["dist"], float)
    return a


def slack0(pred, s1, h1, cap):
    """Minutes of extra hold that still have zero closed-loop cascade."""
    p = np.clip(np.asarray(pred, float), -30.0, 180.0)
    return np.where(h1, np.clip(s1 - p, 0.0, cap), cap)


def closed_cascade(hold, pred, s1, s2, h1, h2):
    h = np.asarray(hold, float)
    sl = slack0(pred, s1, h1, caps(h1))
    # remaining buffer already accounts for predicted unimpeded delay
    g2 = np.where(h1, np.maximum(0.0, h - sl), 0.0)
    g3 = np.where(h2, np.maximum(0.0, g2 - np.clip(s2, 0.0, None)), 0.0)
    tot = g2 + g3
    return float(tot.sum()), tot, g2, g3


def origin_hold(hold, h1):
    h = np.asarray(hold, float)
    return float(h[~np.asarray(h1, bool)].sum())


def origin_hour_peak(hold, h1, origin, hod):
    h = np.asarray(hold, float)
    term = ~np.asarray(h1, bool)
    if not term.any():
        return 0.0
    keys = origin[term] + "|" + hod[term].astype(int).astype(str)
    load = pd.Series(h[term]).groupby(keys, sort=False).sum()
    return float(load.max()) if len(load) else 0.0


def airline_mad(hold, budget, carrier):
    h = np.asarray(hold, float)
    b = np.asarray(budget, float)
    rec = []
    for a in np.unique(carrier):
        m = carrier == a
        rb = float(b[m].sum())
        rh = float(h[m].sum())
        rec.append(abs(rh - rb) / max(rb, 1.0))
    return float(np.mean(rec)) if rec else 0.0


def worst_airline_cascade(hold, pred, a):
    tot, _, _, _ = closed_cascade(hold, pred, a["s1"], a["s2"], a["h1"], a["h2"])
    # per-carrier cascade share
    h = np.asarray(hold, float)
    sl = slack0(pred, a["s1"], a["h1"], caps(a["h1"]))
    g2 = np.where(a["h1"], np.maximum(0.0, h - sl), 0.0)
    g3 = np.where(a["h2"], np.maximum(0.0, g2 - np.clip(a["s2"], 0.0, None)), 0.0)
    c = g2 + g3
    worst = 0.0
    for al in np.unique(a["carrier"]):
        m = (a["carrier"] == al) & a["h1"]
        if m.any():
            worst = max(worst, float(c[m].sum()))
    return worst, tot


def match_sum(h, target, cap):
    h = np.asarray(h, float).copy()
    gap = float(target) - float(h.sum())
    if abs(gap) <= 1e-6:
        return h
    if gap > 0:
        room = np.maximum(cap - h, 0.0)
        for i in np.argsort(-room):
            take = min(room[i], gap)
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


def convex_fill(pred, s1, s2, h1, h2, total, cap=None):
    """Marginal-cost fill of a single knapsack. Optimal for closed-loop cascade."""
    n = len(s1)
    cap = caps(h1) if cap is None else np.asarray(cap, float)
    h = np.zeros(n, dtype=np.float64)
    rem = float(total)
    if rem <= 1e-9:
        return h
    sl = slack0(pred, s1, h1, cap)
    for j in np.argsort(-sl):
        take = min(cap[j] - h[j], sl[j], rem)
        h[j] += take
        rem -= take
        if rem <= 1e-9:
            return h
    room1 = np.zeros(n)
    conn = np.asarray(h1, bool)
    room1[conn] = np.minimum(np.clip(s2[conn], 0.0, None), cap[conn] - h[conn])
    for j in np.argsort(-room1):
        take = min(max(room1[j], 0.0), rem)
        h[j] += take
        rem -= take
        if rem <= 1e-9:
            return h
    room = cap - h
    for j in np.argsort(-room):
        take = min(max(room[j], 0.0), rem)
        h[j] += take
        rem -= take
        if rem <= 1e-9:
            return h
    return match_sum(h, total, cap)


def group_fill(pred, s1, s2, h1, h2, budget, keys):
    """Solve one knapsack per group key, keeping each group's extra-hold total."""
    n = len(s1)
    h = np.zeros(n, dtype=np.float64)
    cap = caps(h1)
    keys = np.asarray(keys)
    bud = np.asarray(budget, float)
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        h[idx] = convex_fill(
            pred[idx], s1[idx], s2[idx], h1[idx], h2[idx], float(bud[idx].sum()), cap[idx]
        )
    return h


def rbs(budget):
    return np.asarray(budget, float).copy()


def interval_keys(sched):
    return np.asarray(sched).astype(str)


def airline_interval_keys(sched, carrier):
    return np.asarray(carrier).astype(str) + "|" + np.asarray(sched).astype(str)


def remainder_after_zero_cost(pred, s1, h1, budget, keys):
    cap = caps(h1)
    sl = slack0(pred, s1, h1, cap)
    rem = 0.0
    absorb = 0.0
    tot = 0.0
    keys = np.asarray(keys)
    bud = np.asarray(budget, float)
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        B = float(bud[idx].sum())
        room = float(np.minimum(cap[idx], sl[idx]).sum())
        tot += B
        absorb += min(B, room)
        rem += max(0.0, B - room)
    return rem, absorb, tot


def leftover_rank(pred, s1, s2, h1, h2, budget, keys, score):
    """Zero-cost fill, then spend remainder by a score inside each group."""
    n = len(s1)
    cap = caps(h1)
    h = np.zeros(n, dtype=np.float64)
    keys = np.asarray(keys)
    bud = np.asarray(budget, float)
    sc = np.asarray(score, float)
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        B = float(bud[idx].sum())
        sl = slack0(pred[idx], s1[idx], h1[idx], cap[idx])
        h0 = np.zeros(len(idx))
        rem = B
        for j in np.argsort(-sl):
            take = min(cap[idx][j] - h0[j], sl[j], rem)
            h0[j] += take
            rem -= take
            if rem <= 1e-9:
                break
        if rem > 1e-9:
            room = cap[idx] - h0
            extra = np.zeros(len(idx))
            for j in np.argsort(-sc[idx]):
                take = min(max(room[j], 0.0), rem)
                extra[j] = take
                rem -= take
                if rem <= 1e-9:
                    break
            h0 = h0 + extra
        h[idx] = match_sum(h0, B, cap[idx])
    return h


def s0_score(s2, h1, h2):
    return (1.0 - np.asarray(h1, bool).astype(float)) * 400.0 + np.where(
        np.asarray(h2, bool), np.clip(s2, 0.0, None), 0.0
    )


def equity_lp(pred, s1, s2, h1, h2, budget, sched, carrier, origin_cap=None, time_limit=8.0):
    """Min closed-loop cascade with interval totals and airline RBS totals.

    Optional origin_cap bounds total terminator hold (minutes).
    """
    import gurobipy as gp
    from gurobipy import GRB

    n = len(s1)
    cap = caps(h1)
    sl = slack0(pred, s1, h1, cap)
    s2c = np.clip(s2, 0.0, None)
    bud = np.asarray(budget, float)
    m = gp.Model("equity")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.Method = 2
    h = m.addVars(n, lb=0.0, ub=cap.tolist(), name="h")
    g2 = m.addVars(n, lb=0.0, name="g2")
    g3 = m.addVars(n, lb=0.0, name="g3")
    for i in range(n):
        if h1[i]:
            m.addConstr(g2[i] >= h[i] - float(sl[i]))
            if h2[i]:
                m.addConstr(g3[i] >= g2[i] - float(s2c[i]))
            else:
                m.addConstr(g3[i] == 0.0)
        else:
            m.addConstr(g2[i] == 0.0)
            m.addConstr(g3[i] == 0.0)
    for t in np.unique(sched):
        idx = np.flatnonzero(sched == t)
        m.addConstr(gp.quicksum(h[int(i)] for i in idx) == float(bud[idx].sum()))
    for al in np.unique(carrier):
        idx = np.flatnonzero(carrier == al)
        m.addConstr(gp.quicksum(h[int(i)] for i in idx) == float(bud[idx].sum()))
    if origin_cap is not None:
        term = np.flatnonzero(~np.asarray(h1, bool))
        if len(term):
            m.addConstr(gp.quicksum(h[int(i)] for i in term) <= float(origin_cap))
    m.setObjective(gp.quicksum(g2[i] + g3[i] for i in range(n)), GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount < 1:
        return None, dict(status=int(m.Status), gap=None, obj=None)
    x = np.array([h[i].X for i in range(n)], dtype=np.float64)
    info = dict(status=int(m.Status), gap=float(m.MIPGap) if m.IsMIP else 0.0, obj=float(m.ObjVal))
    return match_sum(x, float(bud.sum()), cap), info


def origin_qp(pred, s1, s2, h1, h2, budget, sched, origin, hod, lam=1e-4, time_limit=8.0):
    """Min cascade plus a quadratic origin-hour occupancy penalty."""
    import gurobipy as gp
    from gurobipy import GRB

    n = len(s1)
    cap = caps(h1)
    sl = slack0(pred, s1, h1, cap)
    s2c = np.clip(s2, 0.0, None)
    bud = np.asarray(budget, float)
    m = gp.Model("origin")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    h = m.addVars(n, lb=0.0, ub=cap.tolist(), name="h")
    g2 = m.addVars(n, lb=0.0, name="g2")
    g3 = m.addVars(n, lb=0.0, name="g3")
    for i in range(n):
        if h1[i]:
            m.addConstr(g2[i] >= h[i] - float(sl[i]))
            if h2[i]:
                m.addConstr(g3[i] >= g2[i] - float(s2c[i]))
            else:
                m.addConstr(g3[i] == 0.0)
        else:
            m.addConstr(g2[i] == 0.0)
            m.addConstr(g3[i] == 0.0)
    for t in np.unique(sched):
        idx = np.flatnonzero(sched == t)
        m.addConstr(gp.quicksum(h[int(i)] for i in idx) == float(bud[idx].sum()))
    term = np.flatnonzero(~np.asarray(h1, bool))
    obj = gp.quicksum(g2[i] + g3[i] for i in range(n))
    if len(term):
        keys = origin[term] + "|" + np.asarray(hod[term], int).astype(str)
        for k in np.unique(keys):
            idx = term[np.flatnonzero(keys == k)]
            L = gp.quicksum(h[int(i)] for i in idx)
            obj = obj + float(lam) * L * L
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) or m.SolCount < 1:
        return None, dict(status=int(m.Status), obj=None)
    x = np.array([h[i].X for i in range(n)], dtype=np.float64)
    return match_sum(x, float(bud.sum()), cap), dict(status=int(m.Status), obj=float(m.ObjVal))


PEAK_HOD = (7, 8, 9, 16, 17, 18, 19)


def rbd_interval(budget, dist, h1, sched):
    """Short-haul flights sit extra hold first, keeping each interval total."""
    n = len(sched)
    cap = caps(h1)
    h = np.zeros(n, dtype=np.float64)
    bud = np.asarray(budget, float)
    d = np.asarray(dist, float)
    for t in np.unique(sched):
        idx = np.flatnonzero(sched == t)
        rem = float(bud[idx].sum())
        for j in idx[np.argsort(d[idx])]:
            take = min(cap[j], rem)
            h[j] = take
            rem -= take
            if rem <= 1e-9:
                break
        h[idx] = match_sum(h[idx], float(bud[idx].sum()), cap[idx])
    return h


def fill_active(pred, s1, s2, h1, h2, budget, sched, frozen, h_frozen):
    """Reassign extra hold on flights that have not yet departed."""
    n = len(sched)
    cap = caps(h1)
    h = np.asarray(h_frozen, float).copy()
    frozen = np.asarray(frozen, bool)
    bud = np.asarray(budget, float)
    for t in np.unique(sched):
        idx = np.flatnonzero(sched == t)
        B = float(bud[idx].sum())
        fr = frozen[idx]
        act = idx[~fr]
        if len(act) == 0:
            continue
        taken = float(h[idx[fr]].sum()) if fr.any() else 0.0
        left = max(0.0, B - taken)
        h[act] = convex_fill(
            pred[act], s1[act], s2[act], h1[act], h2[act], left, cap[act]
        )
    return h


def expected_lp(pred, shifts, probs, s1, s2, h1, h2, budget, sched, carrier=None, time_limit=8.0):
    """Min expected closed-loop cascade over residual shifts of predicted delay."""
    import gurobipy as gp
    from gurobipy import GRB

    n = len(s1)
    cap = caps(h1)
    s2c = np.clip(s2, 0.0, None)
    bud = np.asarray(budget, float)
    K = len(shifts)
    m = gp.Model("expect")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    hv = m.addVars(n, lb=0.0, ub=cap.tolist(), name="h")
    g2 = m.addVars(n, K, lb=0.0, name="g2")
    g3 = m.addVars(n, K, lb=0.0, name="g3")
    for k, q in enumerate(shifts):
        sl = slack0(np.asarray(pred, float) + float(q), s1, h1, cap)
        for i in range(n):
            if h1[i]:
                m.addConstr(g2[i, k] >= hv[i] - float(sl[i]))
                if h2[i]:
                    m.addConstr(g3[i, k] >= g2[i, k] - float(s2c[i]))
                else:
                    m.addConstr(g3[i, k] == 0.0)
            else:
                m.addConstr(g2[i, k] == 0.0)
                m.addConstr(g3[i, k] == 0.0)
    for t in np.unique(sched):
        idx = np.flatnonzero(sched == t)
        m.addConstr(gp.quicksum(hv[int(i)] for i in idx) == float(bud[idx].sum()))
    if carrier is not None:
        for al in np.unique(carrier):
            idx = np.flatnonzero(carrier == al)
            m.addConstr(gp.quicksum(hv[int(i)] for i in idx) == float(bud[idx].sum()))
    m.setObjective(
        gp.quicksum(float(probs[k]) * (g2[i, k] + g3[i, k]) for i in range(n) for k in range(K)),
        GRB.MINIMIZE,
    )
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount < 1:
        return None, dict(status=int(m.Status), obj=None)
    x = np.array([hv[i].X for i in range(n)], dtype=np.float64)
    return match_sum(x, float(bud.sum()), cap), dict(status=int(m.Status), obj=float(m.ObjVal))


def score_bundle(hold, pred, a, budget):
    C, tot, g2, g3 = closed_cascade(hold, pred, a["s1"], a["s2"], a["h1"], a["h2"])
    Cr, _, _, _ = closed_cascade(hold, a["base_delay"], a["s1"], a["s2"], a["h1"], a["h2"])
    worst, _ = worst_airline_cascade(hold, pred, a)
    peak = np.isin(a["hod"], PEAK_HOD)
    c_peak = float(tot[peak].sum()) if peak.any() else 0.0
    c_off = float(tot[~peak].sum()) if (~peak).any() else 0.0
    return {
        "cascade": C,
        "cascade_base": Cr,
        "cascade_peak": c_peak,
        "cascade_off": c_off,
        "worst": worst,
        "leg2": float(g2.sum()),
        "leg3": float(g3.sum()),
        "wait": float(np.asarray(hold).sum()),
        "origin": origin_hold(hold, a["h1"]),
        "origin_peak": origin_hour_peak(hold, a["h1"], a["origin"], a["hod"]),
        "mad": airline_mad(hold, budget, a["carrier"]),
        "n": int(len(hold)),
        "n_rot": int(np.asarray(a["h1"]).sum()),
        "n_peak": int(peak.sum()),
    }


def add(acc, rec):
    for k in (
        "cascade",
        "cascade_base",
        "cascade_peak",
        "cascade_off",
        "cascade_real",
        "cascade_nas",
        "worst",
        "leg2",
        "leg3",
        "wait",
        "origin",
        "origin_peak",
        "mad",
    ):
        acc[k] = acc.get(k, 0.0) + rec.get(k, 0.0)
    acc["n"] = acc.get("n", 0) + rec["n"]
    acc["n_rot"] = acc.get("n_rot", 0) + rec["n_rot"]
    acc["n_peak"] = acc.get("n_peak", 0) + rec.get("n_peak", 0)
    acc["days"] = acc.get("days", 0) + 1
    return acc


def mean_pack(acc):
    d = max(acc.get("days", 1), 1)
    keys = (
        "cascade",
        "cascade_base",
        "cascade_peak",
        "cascade_off",
        "cascade_real",
        "cascade_nas",
        "worst",
        "leg2",
        "leg3",
        "wait",
        "origin",
        "n",
        "n_rot",
        "n_peak",
        "days",
    )
    out = {k: acc.get(k, 0.0) for k in keys}
    out["origin_peak"] = acc.get("origin_peak", 0.0) / d
    out["mad"] = acc.get("mad", 0.0) / d
    out["worst"] = acc.get("worst", 0.0) / d
    return out


def vs(base, ours, key="cascade"):
    b, o = base[key], ours[key]
    return float((b - o) / b) if b else None


def origin_hour_lp(pred, s1, s2, h1, h2, budget, sched, origin, hod, time_limit=8.0):
    """Min cascade with interval totals and per origin-hour terminator mass at RBS."""
    import gurobipy as gp
    from gurobipy import GRB

    n = len(s1)
    cap = caps(h1)
    sl = slack0(pred, s1, h1, cap)
    s2c = np.clip(s2, 0.0, None)
    bud = np.asarray(budget, float)
    m = gp.Model("ohour")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    hv = m.addVars(n, lb=0.0, ub=cap.tolist(), name="h")
    g2 = m.addVars(n, lb=0.0, name="g2")
    g3 = m.addVars(n, lb=0.0, name="g3")
    for i in range(n):
        if h1[i]:
            m.addConstr(g2[i] >= hv[i] - float(sl[i]))
            if h2[i]:
                m.addConstr(g3[i] >= g2[i] - float(s2c[i]))
            else:
                m.addConstr(g3[i] == 0.0)
        else:
            m.addConstr(g2[i] == 0.0)
            m.addConstr(g3[i] == 0.0)
    for t in np.unique(sched):
        idx = np.flatnonzero(sched == t)
        m.addConstr(gp.quicksum(hv[int(i)] for i in idx) == float(bud[idx].sum()))
    term = np.flatnonzero(~np.asarray(h1, bool))
    if len(term):
        keys = np.asarray(origin)[term] + "|" + np.asarray(hod[term], int).astype(str)
        for k in np.unique(keys):
            idx = term[np.flatnonzero(keys == k)]
            m.addConstr(gp.quicksum(hv[int(i)] for i in idx) <= float(bud[idx].sum()))
    m.setObjective(gp.quicksum(g2[i] + g3[i] for i in range(n)), GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount < 1:
        return None, dict(status=int(m.Status), obj=None)
    x = np.array([hv[i].X for i in range(n)], dtype=np.float64)
    return match_sum(x, float(bud.sum()), cap), dict(status=int(m.Status), obj=float(m.ObjVal))


def bank_pairs(a, banks, connect=30.0):
    """List (inbound index, outbound dep clock, connect minutes) for same-carrier banks."""
    pairs = []
    if not banks:
        return pairs
    for i in range(len(a["sched"])):
        rows = banks.get(a["carrier"][i])
        if rows is None:
            continue
        g = rows - a["arr_clock"][i]
        hit = np.flatnonzero((g > 20.0) & (g < 90.0))
        for j in hit:
            pairs.append((i, float(rows[j]), connect))
    return pairs


def bank_lp(pred, s1, s2, h1, h2, budget, sched, a, banks, pax_cap, connect=30.0, time_limit=8.0):
    """Min cascade subject to interval totals and same-carrier outbound miss cap."""
    import gurobipy as gp
    from gurobipy import GRB

    n = len(s1)
    cap = caps(h1)
    sl = slack0(pred, s1, h1, cap)
    s2c = np.clip(s2, 0.0, None)
    bud = np.asarray(budget, float)
    pairs = bank_pairs(a, banks, connect)
    m = gp.Model("bank")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    hv = m.addVars(n, lb=0.0, ub=cap.tolist(), name="h")
    g2 = m.addVars(n, lb=0.0, name="g2")
    g3 = m.addVars(n, lb=0.0, name="g3")
    for i in range(n):
        if h1[i]:
            m.addConstr(g2[i] >= hv[i] - float(sl[i]))
            if h2[i]:
                m.addConstr(g3[i] >= g2[i] - float(s2c[i]))
            else:
                m.addConstr(g3[i] == 0.0)
        else:
            m.addConstr(g2[i] == 0.0)
            m.addConstr(g3[i] == 0.0)
    for t in np.unique(sched):
        idx = np.flatnonzero(sched == t)
        m.addConstr(gp.quicksum(hv[int(i)] for i in idx) == float(bud[idx].sum()))
    miss = m.addVars(len(pairs), lb=0.0, name="m")
    for k, (i, dep, conn) in enumerate(pairs):
        base = float(a["arr_clock"][i] + np.clip(pred[i], -30.0, 180.0) + conn - dep)
        m.addConstr(miss[k] >= hv[i] + base)
    if pairs:
        m.addConstr(gp.quicksum(miss[k] for k in range(len(pairs))) <= float(pax_cap))
    m.setObjective(gp.quicksum(g2[i] + g3[i] for i in range(n)), GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INFEASIBLE) or m.SolCount < 1:
        return None, dict(status=int(m.Status), obj=None, pairs=len(pairs))
    x = np.array([hv[i].X for i in range(n)], dtype=np.float64)
    return match_sum(x, float(bud.sum()), cap), dict(status=int(m.Status), obj=float(m.ObjVal), pairs=len(pairs))


def carrier_bank_minutes(hold, pred, a, banks, connect=30.0):
    """Minutes by which assigned arrival misses same-carrier outbounds."""
    if banks is None or len(banks) == 0:
        return 0.0
    arr = np.asarray(a["arr_clock"], float) + np.clip(np.asarray(pred, float), -30.0, 180.0) + np.asarray(hold, float)
    tot = 0.0
    for i in range(len(arr)):
        car = a["carrier"][i]
        rows = banks.get(car)
        if rows is None:
            continue
        dep = rows
        g = dep - a["arr_clock"][i]
        hit = (g > 20.0) & (g < 90.0)
        if hit.any():
            tot += float(np.maximum(0.0, arr[i] + connect - dep[hit]).sum())
    return tot
