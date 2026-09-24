"""Train liquid to fit interval-conserving assignment; deploy the network only.

Usage:
  python run_liquid_replace.py smoke
  python run_liquid_replace.py DFW summer
  python run_liquid_replace.py all
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from algos.delay_map import LateAircraftHGB
from algos.gdp_assign import (
    add,
    airline_interval_keys,
    closed_cascade,
    equity_lp,
    group_fill,
    interval_keys,
    mean_pack,
    pack,
    rbs,
    score_bundle,
    vs,
)
from algos.instance import day_bins
from algos.liquid_replace import LiquidEDCT
from algos.assign_ours import fluid_budget
from run_frontier5 import aar_map
from run_trc_accept import AIRPORTS, CACHE, RATES, SEASONS, fit_rates, load_cached

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "liquid_replace"


def budget_from(day, a, wx, model):
    gbin = day_bins(day, wx)
    aar = model.aar(gbin)
    return fluid_budget(a["sched"], a["hod"], aar_map(gbin, aar))


def run_scene(airport, season, smoke=False, epochs=12):
    train, val, test, wx, te, table = load_cached(airport, season)
    if smoke:
        train, val, test = train[:4], val[:2], test[:2]
        epochs = 2
    rates = fit_rates(train, wx)
    pred_model = LateAircraftHGB().fit(train, wx, te)
    rib = rates["Predictive"]

    def pred_fn(day):
        return pred_model.pred(day)

    def budget_fn(day):
        a = pack(day)
        return budget_from(day, a, wx, rib)

    print("fit liquid", airport, season, flush=True)
    liquid = LiquidEDCT().fit(train, val, wx, pred_fn, budget_fn, epochs=epochs)
    acc = {}
    n_days = 0
    fit_abs = fit_den = 0.0
    for day in test:
        a = pack(day)
        pred = pred_model.pred(day)
        n_days += 1
        for rname, model in rates.items():
            bud = budget_from(day, a, wx, model)
            h_rbs = rbs(bud)
            h_air = group_fill(pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, airline_interval_keys(a["sched"], a["carrier"]))
            h_pred = group_fill(pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, interval_keys(a["sched"]))
            h_liq = liquid.assign(day, pred, bud)
            real = np.clip(a["delay"], -30.0, 180.0)
            h_hind = group_fill(real, a["s1"], a["s2"], a["h1"], a["h2"], bud, interval_keys(a["sched"]))
            eq, _ = equity_lp(pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, a["sched"], a["carrier"])
            h_eq = eq if eq is not None else h_air
            if "NASDelay" in day.columns:
                import pandas as pd

                nas = pd.to_numeric(day["NASDelay"], errors="coerce").fillna(0.0).to_numpy(float)
                nas_d = np.clip(a["delay"] - nas, -30.0, 180.0)
            else:
                nas_d = real
            if rname == "Predictive":
                fit_abs += float(np.abs(h_liq - h_pred).sum())
                fit_den += float(np.abs(h_pred).sum()) + 1.0
            for name, h in (
                ("RBS", h_rbs),
                ("Airline", h_air),
                ("PredFill", h_pred),
                ("Liquid", h_liq),
                ("Hindsight", h_hind),
                ("Equity", h_eq),
            ):
                rec = score_bundle(h, pred, a, bud)
                rec["cascade_real"] = closed_cascade(h, real, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                rec["cascade_nas"] = closed_cascade(h, nas_d, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                key = f"{rname}_{name}"
                acc[key] = add(acc.get(key, {}), rec)
    packed = {k: mean_pack(v) for k, v in acc.items()}
    table_out = {
        "airport": airport,
        "season": season,
        "n_test_days": n_days,
        "val_mae": float(getattr(liquid, "val_mae", 0.0)),
        "test_hold_mae": float(fit_abs / max(fit_den, 1.0)),
        "smoke": bool(smoke),
    }
    methods = ("RBS", "Airline", "PredFill", "Liquid", "Hindsight", "Equity")
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
                    if "cascade_real" in block["RBS"] and "cascade_real" in block[m]:
                        block[f"{m}_vs_RBS_real"] = vs(block["RBS"], block[m], key="cascade_real")
                    if "cascade_nas" in block["RBS"] and "cascade_nas" in block[m]:
                        block[f"{m}_vs_RBS_nas"] = vs(block["RBS"], block[m], key="cascade_nas")
            if "Liquid" in block and "PredFill" in block:
                block["Liquid_vs_PredFill"] = vs(block["PredFill"], block["Liquid"])
                block["Liquid_vs_PredFill_real"] = vs(block["PredFill"], block["Liquid"], key="cascade_real")
                block["Liquid_vs_PredFill_nas"] = vs(block["PredFill"], block["Liquid"], key="cascade_nas")
            if "Hindsight" in block and "PredFill" in block:
                block["Hindsight_vs_PredFill_real"] = vs(block["PredFill"], block["Hindsight"], key="cascade_real")
            if "Hindsight" in block and "Liquid" in block:
                block["Liquid_vs_Hindsight_real"] = vs(block["Hindsight"], block["Liquid"], key="cascade_real")
        table_out[rname] = block
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if smoke else "full"
    out = OUTDIR / f"{airport}_{season}_{tag}.json"
    out.write_text(json.dumps({"test": packed, "table": table_out}, indent=2), encoding="utf-8")
    print(json.dumps(table_out, indent=2)[:5000], flush=True)
    print("wrote", out, flush=True)
    return table_out


def _job(args):
    return run_scene(*args, smoke=False, epochs=12)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] == "smoke":
        airport = argv[1] if len(argv) > 1 else "DFW"
        season = argv[2] if len(argv) > 2 else "summer"
        run_scene(airport, season, smoke=True, epochs=2)
        return
    if argv[0] == "all":
        jobs = [(a, s) for a in AIRPORTS for s in SEASONS]
        workers = min(2, max(1, (os.cpu_count() or 4) // 8))
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
    run_scene(argv[0], argv[1] if len(argv) > 1 else "summer", smoke=False, epochs=12)


if __name__ == "__main__":
    main()
