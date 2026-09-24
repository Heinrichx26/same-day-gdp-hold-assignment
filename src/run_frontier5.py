"""Five 2024--2026 journal methods vs PILS on DFW July.

1 Delay-predictive hybrid rate, TR-C 171 (2025) 104947.
2 DTW k-medoids declared rate, TR-C 171 (2025) 105012.
3 Distributionally robust ground holding, arXiv 2509.18492 (2025).
4 Incremental space-time DCB search, TR-C 181 (2025) 105382 analogue.
5 Hybrid SA delay allocation, TR-C 180 (2025) 105306 analogue.

Each rate paper is run as published: its AAR plus the delay allocation
that paper uses (equal extra hold inside the 15-min bin). Equal hold is
not a 2025 method; it is the second stage inside those papers.
PILS uses the same extra-hold total. Ablation: frozen delay-map transport.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from algos.assign_ours import fluid_budget, rbs
from algos.family_ribeiro import RibeiroAAR
from algos.family_wang import WangAAR
from algos.family_wu import WuAAR
from algos.instance import arrays, bin_panel, day_bins, load_dfw
from algos.method_pils import PILS
from algos.metrics import add, mean_pack, score_hold
from algos.transport_hold import allocate

OUT = Path(__file__).resolve().parents[1] / "results" / "winter" / "frontier5_dfw.json"


def aar_map(gbin, aar):
    return {int(b): float(u) for b, u in zip(gbin["bin"].to_numpy(int), aar)}


def vs(base, ours):
    b, o = base["cascade"], ours["cascade"]
    return float((b - o) / b) if b else None


def _sa_job(args):
    hold0, pred, s1, s2, h1, h2, seed = args
    from algos.family_sa import anneal_budget

    return anneal_budget(hold0, pred, s1, s2, h1, h2, seed=seed)


def _search_job(args):
    sched, table, s1, s2, h1, h2, pred = args
    from algos.assign_window import incremental_search

    return incremental_search(sched, table, s1, s2, h1, h2, pred)


def main():
    print("load", flush=True)
    may, june, july, wx, te, table = load_dfw()
    bins_may = bin_panel(may, wx)
    rates = {
        "DelayHybrid": RibeiroAAR().fit(bins_may),
        "DeclaredDTW": WangAAR().fit(bins_may),
        "DistRobust": WuAAR().fit(bins_may),
    }
    print("fit PILS", flush=True)
    method = PILS().fit(may, june, wx, te, table, epochs=6)
    acc = {}

    def bump(name, rec):
        acc[name] = add(acc.get(name, {}), rec)

    print("july serial PILS", flush=True)
    cache = []
    sa_jobs, sa_meta, search_jobs = [], [], []
    for di, day in enumerate(july):
        a = arrays(day)
        gbin = day_bins(day, wx)
        pred_hgb = method.base.pred(day)
        rec = {"a": a, "pred_hgb": pred_hgb, "buds": {}, "published": {}, "pils": {}, "map_t": {}}
        for fname, model in rates.items():
            aar = model.aar(gbin)
            bud = fluid_budget(a["sched"], a["hod"], aar_map(gbin, aar))
            published = rbs(bud)
            rec["buds"][fname] = bud
            rec["published"][fname] = published
            rec["map_t"][fname] = allocate(pred_hgb, a["s1"], a["s2"], a["h1"], a["h2"], bud)
            rec["pils"][fname] = method.assign(day, bud)
            bump(f"{fname}_published", score_hold(published, a))
            bump(f"{fname}_map_transport", score_hold(rec["map_t"][fname], a))
            bump(f"{fname}_PILS", score_hold(rec["pils"][fname], a))
            sa_jobs.append((published, pred_hgb, a["s1"], a["s2"], a["h1"], a["h2"], 1000 * di + abs(hash(fname)) % 997))
            sa_meta.append((di, fname))
        hour_bud = fluid_budget(
            a["sched"], a["hod"], {int(b): float(table.get((int(b) * 15) // 60, 4.0)) for b in a["sched"]}
        )
        rec["pils_hour"] = method.assign(day, hour_bud)
        bump("PILS_hourcap", score_hold(rec["pils_hour"], a))
        search_jobs.append((a["sched"], table, a["s1"], a["s2"], a["h1"], a["h2"], pred_hgb))
        cache.append(rec)
    workers = max(4, min(20, (os.cpu_count() or 8) - 4))
    print("july parallel SA/search workers", workers, "jobs", len(sa_jobs) + len(search_jobs), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        sa_holds = list(pool.map(_sa_job, sa_jobs, chunksize=1))
        search_holds = list(pool.map(_search_job, search_jobs, chunksize=1))
    for (di, fname), h in zip(sa_meta, sa_holds):
        bump(f"{fname}_SA", score_hold(h, cache[di]["a"]))
    for di, h in enumerate(search_holds):
        bump("IncrSearch", score_hold(h, cache[di]["a"]))
    packed = {k: mean_pack(v) for k, v in acc.items()}
    inc = {
        "alpha_med": getattr(method, "alpha_med", None),
        "alpha_tight": getattr(method, "alpha_tight", None),
        "papers": {
            "DelayHybrid": "TR-C 171 (2025) 104947 delay-predictive hybrid",
            "DeclaredDTW": "TR-C 171 (2025) 105012 DTW k-medoids declared rate",
            "DistRobust": "arXiv 2509.18492 (2025) distributionally robust GHP",
            "IncrSearch": "TR-C 181 (2025) 105382 incremental DCB search analogue",
            "SA": "TR-C 180 (2025) 105306 hybrid SA delay allocation analogue",
        },
    }
    for fam in ("DelayHybrid", "DeclaredDTW", "DistRobust"):
        inc[fam] = {
            "PILS_vs_published": vs(packed[f"{fam}_published"], packed[f"{fam}_PILS"]),
            "PILS_vs_SA": vs(packed[f"{fam}_SA"], packed[f"{fam}_PILS"]),
            "PILS_vs_map_transport": vs(packed[f"{fam}_map_transport"], packed[f"{fam}_PILS"]),
            "SA_vs_published": vs(packed[f"{fam}_published"], packed[f"{fam}_SA"]),
            "published": packed[f"{fam}_published"]["cascade"],
            "SA": packed[f"{fam}_SA"]["cascade"],
            "map_transport": packed[f"{fam}_map_transport"]["cascade"],
            "PILS": packed[f"{fam}_PILS"]["cascade"],
        }
    inc["IncrSearch"] = {
        "PILS_vs_search": vs(packed["IncrSearch"], packed["PILS_hourcap"]),
        "search": packed["IncrSearch"]["cascade"],
        "PILS_hourcap": packed["PILS_hourcap"]["cascade"],
    }
    OUT.write_text(json.dumps({"july": packed, "increment": inc}, indent=2), encoding="utf-8")
    print(json.dumps(inc, indent=2))


if __name__ == "__main__":
    main()
