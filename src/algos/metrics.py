"""Cascade and connecting spill on a hold vector. Score uses realized delay."""
from __future__ import annotations

import numpy as np

import run_trc_gate as g


def score_hold(hold, a):
    wait = np.asarray(hold, float)
    s1 = a["s1"].copy()
    s1[a["h1"]] = s1[a["h1"]] - a["delay"][a["h1"]]
    tot, l2, l3 = g.cascade(wait, s1, a["s2"], a["h1"], a["h2"])
    has = a["h1"]
    spill = float(np.maximum(0.0, a["delay"][has] + wait[has] - a["s1"][has]).mean()) if has.any() else 0.0
    return {
        "cascade": float(tot.sum()),
        "leg2": float(l2.sum()),
        "leg3": float(l3.sum()),
        "wait": float(wait.sum()),
        "spill": spill,
        "n": int(len(wait)),
        "n_rot": int(has.sum()),
    }


def add(acc, rec):
    for k in ("cascade", "leg2", "leg3", "wait", "spill"):
        acc[k] = acc.get(k, 0.0) + rec[k]
    acc["n"] = acc.get("n", 0) + rec["n"]
    acc["n_rot"] = acc.get("n_rot", 0) + rec["n_rot"]
    acc["days"] = acc.get("days", 0) + 1
    return acc


def mean_pack(acc):
    d = max(acc.get("days", 1), 1)
    return {
        "cascade": acc["cascade"],
        "leg2": acc["leg2"],
        "leg3": acc["leg3"],
        "wait": acc["wait"],
        "spill": acc["spill"] / d,
        "n": acc["n"],
        "n_rot": acc["n_rot"],
        "days": acc["days"],
    }
