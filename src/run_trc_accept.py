"""Closed-loop second-stage GDP assignment for the TR-C revision.

Usage:
  python run_trc_accept.py smoke
  python run_trc_accept.py DFW summer
  python run_trc_accept.py all
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_trc_gate as g

g.COLS = list(dict.fromkeys(list(g.COLS) + ["Reporting_Airline", "LateAircraftDelay", "NASDelay"]))

from algos.assign_ours import fluid_budget
from algos.delay_map import LateAircraftHGB
from algos.family_ribeiro import RibeiroAAR
from algos.family_wang import WangAAR
from algos.family_wu import WuAAR
from algos.gdp_assign import (
    add,
    airline_interval_keys,
    equity_lp,
    group_fill,
    interval_keys,
    leftover_rank,
    mean_pack,
    origin_qp,
    pack,
    rbs,
    remainder_after_zero_cost,
    s0_score,
    score_bundle,
    vs,
)
from algos.instance import bin_panel, day_bins, load_hub
from algos.method_pils import PILS, leftover_prior
from algos.hold_ops import leftover_np, caps as hold_caps
from run_frontier5 import aar_map

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "accept"
CACHE = ROOT / "results" / "cache"
AIRPORTS = ("ATL", "DFW", "EWR")
SEASONS = ("summer", "winter")
RATES = ("Predictive", "Declared", "Conservative")


def cache_path(airport, season):
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{airport}_{season}.pkl"


def load_cached(airport, season):
    p = cache_path(airport, season)
    if p.exists() and p.stat().st_size > 1000:
        with p.open("rb") as f:
            return pickle.load(f)
    print("load", airport, season, flush=True)
    blob = load_hub(airport, season)
    with p.open("wb") as f:
        pickle.dump(blob, f, protocol=4)
    return blob


def fit_rates(train, wx):
    bins_tr = bin_panel(train, wx)
    return {
        "Predictive": RibeiroAAR().fit(bins_tr),
        "Declared": WangAAR().fit(bins_tr),
        "Conservative": WuAAR().fit(bins_tr),
    }


def budgets_for_day(day, wx, models):
    a = pack(day)
    gbin = day_bins(day, wx)
    out = {}
    for name, model in models.items():
        aar = model.aar(gbin)
        out[name] = fluid_budget(a["sched"], a["hod"], aar_map(gbin, aar))
    return a, out


def interval_liquid(pils, day, a, pred, budget):
    """Zero-cost fill per interval, then liquid score on the remainder."""
    keys = interval_keys(a["sched"])
    h0 = leftover_rank(
        pred, a["s1"], a["s2"], a["h1"], a["h2"], budget, keys, s0_score(a["s2"], a["h1"], a["h2"])
    )
    # rebuild remainder per interval from room after zero-cost
    from algos.gdp_assign import slack0, caps, match_sum

    cap = caps(a["h1"])
    sl = slack0(pred, a["s1"], a["h1"], cap)
    h_zero = np.minimum(h0, sl)
    rem_vec = budget.copy()
    # assign zero-cost first per interval then leftover by liquid
    h = np.zeros(len(budget))
    for t in np.unique(a["sched"]):
        idx = np.flatnonzero(a["sched"] == t)
        B = float(budget[idx].sum())
        hz = np.zeros(len(idx))
        rem = B
        sli = sl[idx]
        for j in np.argsort(-sli):
            take = min(cap[idx][j] - hz[j], sli[j], rem)
            hz[j] += take
            rem -= take
            if rem <= 1e-9:
                break
        if rem <= 1e-9:
            h[idx] = hz
            continue
        room = cap[idx] - hz
        try:
            with np.errstate(all="ignore"):
                score = pils._score(day, pred, np.zeros(len(a["sched"])), cap - h_zero)
            sc = np.asarray(score.detach().cpu().numpy() if hasattr(score, "detach") else score, float)
            extra = leftover_np(sc[idx], room, rem)
        except Exception:
            extra = leftover_np(s0_score(a["s2"], a["h1"], a["h2"])[idx], room, rem)
        h[idx] = match_sum(hz + extra, B, cap[idx])
    return h


def assign_all(a, pred, bud, pils, day, use_qp=True):
    keys_i = interval_keys(a["sched"])
    keys_ai = airline_interval_keys(a["sched"], a["carrier"])
    h_rbs = rbs(bud)
    rec = {
        "RBS": h_rbs,
        "Airline": group_fill(pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, keys_ai),
        "Compression": group_fill(pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, keys_i),
        "Remainder": leftover_rank(
            pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, keys_i, s0_score(a["s2"], a["h1"], a["h2"])
        ),
    }
    rec["Exact"] = rec["Compression"]
    eq, info_eq = equity_lp(pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, a["sched"], a["carrier"])
    rec["Equity"] = eq if eq is not None else rec["Airline"]
    rec["equity_info"] = info_eq
    orig_cap = float(h_rbs[~a["h1"]].sum())
    eqo, info_o = equity_lp(
        pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, a["sched"], a["carrier"], origin_cap=orig_cap
    )
    rec["OriginCap"] = eqo if eqo is not None else rec["Equity"]
    rec["origincap_info"] = info_o
    if use_qp:
        qp, info_q = origin_qp(
            pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, a["sched"], a["origin"], a["hod"]
        )
        rec["OriginQP"] = qp if qp is not None else rec["Exact"]
        rec["qp_info"] = info_q
    if pils is not None:
        rec["Liquid"] = interval_liquid(pils, day, a, pred, bud)
    else:
        rec["Liquid"] = rec["Remainder"]
    rem, absorb, tot = remainder_after_zero_cost(pred, a["s1"], a["h1"], bud, keys_i)
    rec["remainder"] = rem
    rec["absorb"] = absorb
    rec["budget_total"] = tot
    rem_a, _, _ = remainder_after_zero_cost(pred, a["s1"], a["h1"], bud, keys_ai)
    rec["remainder_airline"] = rem_a
    return rec


def run_scene(airport, season, smoke=False, use_qp=True):
    train, val, test, wx, te, table = load_cached(airport, season)
    if smoke:
        test = test[:1]
        val = val[:1]
    rates = fit_rates(train, wx)
    pred_model = LateAircraftHGB().fit(train, wx, te)
    pils = None
    if os.environ.get("TRC_FIT_LIQUID", "0") == "1" and not smoke:
        try:
            pils = PILS().fit(train, val, wx, te, table, epochs=4)
        except Exception as e:
            print("pils fail", airport, season, e, flush=True)
            pils = None
    acc = {}
    rem_acc = {r: {"rem": 0.0, "abs": 0.0, "tot": 0.0, "rem_al": 0.0} for r in RATES}
    eq_ok = 0
    n_days = 0
    for day in test:
        a, buds = budgets_for_day(day, wx, rates)
        pred = pred_model.pred(day)
        n_days += 1
        for rname, bud in buds.items():
            holds = assign_all(a, pred, bud, pils, day, use_qp=use_qp and not smoke)
            rem_acc[rname]["rem"] += holds["remainder"]
            rem_acc[rname]["abs"] += holds["absorb"]
            rem_acc[rname]["tot"] += holds["budget_total"]
            rem_acc[rname]["rem_al"] += holds["remainder_airline"]
            if holds.get("equity_info", {}).get("obj") is not None:
                eq_ok += 1
            for mname in ("RBS", "Airline", "Compression", "Remainder", "Exact", "Equity", "OriginCap", "Liquid"):
                rec = score_bundle(holds[mname], pred, a, bud)
                if "OriginQP" in holds:
                    pass
                acc.setdefault(f"{rname}_{mname}", {})
                acc[f"{rname}_{mname}"] = add(acc[f"{rname}_{mname}"], rec)
            if "OriginQP" in holds:
                acc.setdefault(f"{rname}_OriginQP", {})
                acc[f"{rname}_OriginQP"] = add(acc[f"{rname}_OriginQP"], score_bundle(holds["OriginQP"], pred, a, bud))
    packed = {k: mean_pack(v) for k, v in acc.items()}
    table_out = {
        "airport": airport,
        "season": season,
        "n_test_days": n_days,
        "equity_solved": eq_ok,
        "remainder": rem_acc,
    }
    methods = ["RBS", "Airline", "Compression", "Remainder", "Exact", "Equity", "OriginCap", "Liquid"]
    if any(k.endswith("_OriginQP") for k in packed):
        methods.append("OriginQP")
    for rname in RATES:
        block = {}
        for m in methods:
            key = f"{rname}_{m}"
            if key in packed:
                block[m] = packed[key]
        if "RBS" in block:
            for m in methods:
                if m in block:
                    block[f"{m}_vs_RBS"] = vs(block["RBS"], block[m])
            if "Exact" in block and "Remainder" in block:
                block["Exact_vs_Remainder"] = vs(block["Remainder"], block["Exact"])
            if "Liquid" in block and "Remainder" in block:
                block["Liquid_vs_Remainder"] = vs(block["Remainder"], block["Liquid"])
            if "Equity" in block and "Exact" in block:
                block["Equity_vs_Exact"] = vs(block["Exact"], block["Equity"])
        table_out[rname] = block
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if smoke else "full"
    out = OUTDIR / f"{airport}_{season}_{tag}.json"
    out.write_text(json.dumps({"test": packed, "table": table_out}, indent=2), encoding="utf-8")
    print(json.dumps(table_out, indent=2)[:4000], flush=True)
    print("wrote", out, flush=True)
    return table_out


def _job(args):
    airport, season = args
    return run_scene(airport, season, smoke=False, use_qp=True)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] == "smoke":
        airport = argv[1] if len(argv) > 1 else "DFW"
        season = argv[2] if len(argv) > 2 else "summer"
        run_scene(airport, season, smoke=True, use_qp=False)
        return
    if argv[0] == "all":
        jobs = [(a, s) for a in AIRPORTS for s in SEASONS]
        workers = min(3, max(1, (os.cpu_count() or 4) // 8))
        print("all jobs", jobs, "workers", workers, flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_job, j) for j in jobs]
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                    print("done", rec.get("airport"), rec.get("season"), flush=True)
                except Exception as e:
                    print("job fail", e, flush=True)
        return
    run_scene(argv[0], argv[1] if len(argv) > 1 else "summer", smoke=False, use_qp=True)


if __name__ == "__main__":
    main()
