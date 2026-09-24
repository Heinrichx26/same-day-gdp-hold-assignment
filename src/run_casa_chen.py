"""Score CASA and Chen incremental search on the same Predictive AAR as Liquid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from algos.delay_map import LateAircraftHGB
from algos.gdp_assign import caps, closed_cascade, interval_keys, match_sum, pack
from algos.instance import day_bins
from run_frontier5 import aar_map
from run_liquid_replace import budget_from
from run_trc_accept import AIRPORTS, SEASONS, fit_rates, load_cached

OUT = Path(__file__).resolve().parents[1] / "results" / "casa_chen.json"


def cap_of(b, aar_by_bin):
    return max(int(np.floor(aar_by_bin.get(int(b), 4.0))), 1)


def casa_aar(sched, aar_by_bin):
    n = len(sched)
    wait = np.zeros(n)
    used = {}
    for i in np.argsort(sched, kind="mergesort"):
        b = int(sched[i])
        while used.get(b, 0) >= cap_of(b, aar_by_bin):
            b += 1
        used[b] = used.get(b, 0) + 1
        wait[i] = 15.0 * max(0, b - int(sched[i]))
    return wait


def search_aar(sched, aar_by_bin, s1, s2, h1, h2, pred, passes=3):
    n = len(sched)
    wait = casa_aar(sched, aar_by_bin)
    release = sched + (wait / 15.0).astype(int)
    used = {}
    for b in release:
        used[int(b)] = used.get(int(b), 0) + 1
    slack = s1 - np.clip(pred, -30, 180)

    def cost_of(w):
        leg2 = np.where(h1, np.maximum(0.0, w - slack), 0.0)
        leg3 = np.where(h2, np.maximum(0.0, leg2 - np.clip(s2, 0.0, None)), 0.0)
        return leg2 + leg3

    conn = np.where(h1)[0]
    term = np.where(~h1)[0]
    for _ in range(passes):
        improved = False
        cur = cost_of(wait)
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
                if used.get(b, 0) >= cap_of(b, aar_by_bin) and b != b0:
                    cand = [j for j in term if int(release[j]) == b and int(sched[j]) <= b0]
                    if not cand:
                        continue
                    partner = int(cand[0])
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


def main():
    liq_root = Path(__file__).resolve().parents[1] / "results" / "liquid_replace"
    rows = []
    for airport in AIRPORTS:
        for season in SEASONS:
            train, val, test, wx, te, _ = load_cached(airport, season)
            rates = fit_rates(train, wx)
            pred_model = LateAircraftHGB().fit(train, wx, te)
            rib = rates["Predictive"]
            tot = {
                k: 0.0
                for k in (
                    "casa_p",
                    "casa_r",
                    "search_p",
                    "search_r",
                    "fair_p",
                    "fair_r",
                    "fs_p",
                    "fs_r",
                    "casa_w",
                    "search_w",
                    "liq_w",
                    "rbs_w",
                )
            }
            n = 0
            for day in test:
                a = pack(day)
                gbin = day_bins(day, wx)
                aar = aar_map(gbin, rib.aar(gbin))
                pred = pred_model.pred(day)
                real = np.clip(a["delay"], -30.0, 180.0)
                h_c = casa_aar(a["sched"], aar)
                h_s = search_aar(a["sched"], aar, a["s1"], a["s2"], a["h1"], a["h2"], pred)
                bud = budget_from(day, a, wx, rib)
                tot["casa_p"] += closed_cascade(h_c, pred, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                tot["casa_r"] += closed_cascade(h_c, real, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                tot["search_p"] += closed_cascade(h_s, pred, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                tot["search_r"] += closed_cascade(h_s, real, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                tot["casa_w"] += float(h_c.sum())
                tot["search_w"] += float(h_s.sum())
                tot["rbs_w"] += float(np.asarray(bud).sum())
                keys = interval_keys(a["sched"])
                cap = caps(a["h1"])
                h_cf = np.zeros(len(h_c))
                h_sf = np.zeros(len(h_s))
                for k in np.unique(keys):
                    idx = np.flatnonzero(keys == k)
                    B = float(np.asarray(bud)[idx].sum())
                    raw_c = h_c[idx]
                    raw_s = h_s[idx]
                    if raw_c.sum() < 1e-9:
                        raw_c = np.ones(len(idx))
                    if raw_s.sum() < 1e-9:
                        raw_s = np.ones(len(idx))
                    h_cf[idx] = match_sum(raw_c, B, cap[idx])
                    h_sf[idx] = match_sum(raw_s, B, cap[idx])
                tot["fair_p"] += closed_cascade(h_cf, pred, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                tot["fair_r"] += closed_cascade(h_cf, real, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                tot["fs_p"] += closed_cascade(h_sf, pred, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                tot["fs_r"] += closed_cascade(h_sf, real, a["s1"], a["s2"], a["h1"], a["h2"])[0]
                n += 1
            liq = json.loads((liq_root / f"{airport}_{season}_full.json").read_text(encoding="utf-8"))
            s = liq["table"]["Predictive"]
            rec = {
                "airport": airport,
                "season": season,
                "days": n,
                "CASA_pred": tot["casa_p"],
                "CASA_real": tot["casa_r"],
                "Search_pred": tot["search_p"],
                "Search_real": tot["search_r"],
                "Liquid_pred": s["Liquid"]["cascade"],
                "Liquid_real": s["Liquid"]["cascade_real"],
                "RBS_pred": s["RBS"]["cascade"],
                "RBS_real": s["RBS"]["cascade_real"],
                "CASA_wait": tot["casa_w"],
                "Search_wait": tot["search_w"],
                "RBS_wait": tot["rbs_w"],
                "Liquid_wait": s["Liquid"]["wait"],
            }
            rec["Liquid_vs_CASA_pred"] = (rec["CASA_pred"] - rec["Liquid_pred"]) / max(rec["CASA_pred"], 1)
            rec["Liquid_vs_CASA_real"] = (rec["CASA_real"] - rec["Liquid_real"]) / max(rec["CASA_real"], 1)
            rec["Liquid_vs_Search_pred"] = (rec["Search_pred"] - rec["Liquid_pred"]) / max(rec["Search_pred"], 1)
            rec["Liquid_vs_Search_real"] = (rec["Search_real"] - rec["Liquid_real"]) / max(rec["Search_real"], 1)
            rec["Search_vs_CASA_pred"] = (rec["CASA_pred"] - rec["Search_pred"]) / max(rec["CASA_pred"], 1)
            rec["FairCASA_pred"] = tot["fair_p"]
            rec["FairCASA_real"] = tot["fair_r"]
            rec["FairSearch_pred"] = tot["fs_p"]
            rec["FairSearch_real"] = tot["fs_r"]
            rec["Liquid_vs_FairCASA_pred"] = (tot["fair_p"] - rec["Liquid_pred"]) / max(tot["fair_p"], 1)
            rec["Liquid_vs_FairCASA_real"] = (tot["fair_r"] - rec["Liquid_real"]) / max(tot["fair_r"], 1)
            rec["Liquid_vs_FairSearch_pred"] = (tot["fs_p"] - rec["Liquid_pred"]) / max(tot["fs_p"], 1)
            rec["Liquid_vs_FairSearch_real"] = (tot["fs_r"] - rec["Liquid_real"]) / max(tot["fs_r"], 1)
            rows.append(rec)
            print(
                airport,
                season,
                "fairCASA pred %.1f real %.1f fairSearch pred %.1f real %.1f"
                % (
                    rec["Liquid_vs_FairCASA_pred"] * 100,
                    rec["Liquid_vs_FairCASA_real"] * 100,
                    rec["Liquid_vs_FairSearch_pred"] * 100,
                    rec["Liquid_vs_FairSearch_real"] * 100,
                ),
                flush=True,
            )
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
