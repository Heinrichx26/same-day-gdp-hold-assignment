"""Build TTL-SPO first, then pick a scene by the hindsight–PTO gap.

The method is fixed: two-timescale liquid costs, SPO+ through min-cost
flow, noon test-time training on landed flights. A scene is used only
when hindsight cascade is at least 15% below two-stage GBM-PTO on July.
Turn = 45, May lower-third rate.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor

import run_expand_full as ex
import run_trc_gate as g
import ttl_spo as m

OUT = Path(__file__).resolve().parents[1] / "results" / "winter" / "ttl_spo.json"
AIRPORTS = ("DFW",)
DEVICE = g.DEVICE
NOON = m.NOON


def hourly_weather(airport, days):
    paths = ex.metar_for(airport)
    tz = ex.TZ[airport]
    table = {d: np.zeros((17, 6), dtype=np.float32) for d in days}
    for d in days:
        hod = np.arange(6, 23, dtype=np.float32)
        ang = 2 * np.pi * (hod - 6) / 16.0
        table[d][:, 2] = np.sin(ang)
        table[d][:, 3] = np.cos(ang)
        table[d][:, 4] = hod / 24.0
        table[d][:, 0] = 1.0
    if not paths:
        return table
    try:
        met = ex.load_metar(paths)
    except Exception as e:
        print("metar fail", airport, e, flush=True)
        return table
    met = met.copy()
    met["local"] = met["valid"].dt.tz_convert(tz)
    met["day"] = met["local"].dt.strftime("%Y-%m-%d")
    met["hod"] = met["local"].dt.hour
    grp = met.groupby(["day", "hod"]).agg(vis=("vsby", "mean"), sknt=("sknt", "mean"), wx=("wx", "first"))
    for (day, hod), r in grp.iterrows():
        if day not in table or hod < 6 or hod > 22:
            continue
        i = int(hod) - 6
        vis = 10.0 if pd.isna(r.vis) else float(r.vis)
        sk = 8.0 if pd.isna(r.sknt) else float(r.sknt)
        ifr = 1.0 if str(r.wx) in ("IMC", "IFR", "LIFR") else 0.0
        table[day][i, 0] = vis / 10.0
        table[day][i, 1] = sk / 20.0
        table[day][i, 5] = ifr
    return table


def tensors(day, wx):
    key = day["day"].iloc[0]
    X, y = g.feat_rows(day)
    hod = day["hod"].to_numpy(int)
    hod_idx = np.clip(hod - 6, 0, 16)
    slow = torch.tensor(wx[key], device=DEVICE)
    fast = torch.tensor(X, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)
    return slow, fast, hod_idx, y_t


def split_days(arr):
    def grab(pref):
        return [x.sort_values("bin") for _, x in arr.loc[arr["day"].str.startswith(pref)].groupby("day") if len(x) >= 30]

    return grab("2024-05"), grab("2024-06"), grab("2024-07")


def gbr_fit(may, wx):
    Xtr, ytr = [], []
    for day in may:
        X, y = g.feat_rows(day)
        key = day["day"].iloc[0]
        hod_idx = np.clip(day["hod"].to_numpy(int) - 6, 0, 16)
        extra = wx[key][hod_idx]
        Xtr.append(np.concatenate([X, extra], 1))
        ytr.append(y)
    return GradientBoostingRegressor(max_depth=3, n_estimators=80, random_state=0).fit(
        np.vstack(Xtr), np.concatenate(ytr)
    )


def gbr_pred(gbr, day, wx):
    X, _ = g.feat_rows(day)
    key = day["day"].iloc[0]
    hod_idx = np.clip(day["hod"].to_numpy(int) - 6, 0, 16)
    extra = wx[key][hod_idx]
    return np.clip(gbr.predict(np.concatenate([X, extra], 1)), -30, 180)


def pack(acc):
    return {"cascade": float(acc[0]), "leg2": float(acc[1]), "leg3": float(acc[2]), "wait": float(acc[3])}


def probe_airport(arr, wx, table, may, july):
    gbr = gbr_fit(may, wx)
    acc = {k: np.zeros(4) for k in ("hour", "pto", "hindsight")}
    for day in july:
        sched, s1, s2, h1, h2, _, _ = m.day_arrays(day)
        wait0 = g.mcf(sched, s1, s2, h1, h2, table)
        acc["hour"] += g.score_wait(wait0, day)
        pred = gbr_pred(gbr, day, wx)
        acc["pto"] += m.assign_with_delay(day, table, pred)[0]
        acc["hindsight"] += m.hindsight_assign(day, table)[0]
    pto_c = float(acc["pto"][0])
    hin_c = float(acc["hindsight"][0])
    hour_c = float(acc["hour"][0])
    gap = float((pto_c - hin_c) / pto_c) if pto_c else 0.0
    rec = {
        "hour": pack(acc["hour"]),
        "gbr_pto": pack(acc["pto"]),
        "hindsight": pack(acc["hindsight"]),
        "hindsight_vs_pto": gap,
        "hindsight_vs_hour": float((hour_c - hin_c) / hour_c) if hour_c else None,
        "pto_vs_hour": float((hour_c - pto_c) / hour_c) if hour_c else None,
        "july_days": len(july),
    }
    return rec, gbr


def total_pred(model, day, wx, gbr):
    slow, fast, hod_idx, _ = tensors(day, wx)
    residual = model.forward_day(slow, fast, hod_idx).detach().cpu().numpy()
    return np.clip(gbr_pred(gbr, day, wx) + residual, -30, 180)


def train_ttl(may, june, july, wx, table, gbr):
    print("cache hindsight", flush=True)
    cache = {}
    gbr_np = {}
    for split in (may, june, july):
        for day in split:
            key = day["day"].iloc[0]
            cache[key] = m.cache_day(day, table)
            gbr_np[key] = gbr_pred(gbr, day, wx)
    model = m.TwoTimescaleLiquid().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best, best_val = None, 1e18
    history = []
    for epoch in range(1, 11):
        model.train()
        ep = 0.0
        agree = spo_ag = abs_p = abs_r = 0.0
        for day in may:
            key = day["day"].iloc[0]
            slow, fast, hod_idx, _ = tensors(day, wx)
            loss, st = m.spo_plus_loss(model, table, slow, fast, hod_idx, cache[key], gbr_np[key])
            opt.zero_grad()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ep += float(loss.detach().cpu())
            agree += st["agree0"]
            spo_ag += st["agree_spo"]
            abs_p += st["mean_abs_pred"]
            abs_r += st["mean_abs_res"]
        n_may = max(len(may), 1)
        model.eval()
        val = 0.0
        june_agree = 0.0
        with torch.no_grad():
            for day in june:
                key = day["day"].iloc[0]
                pred = total_pred(model, day, wx, gbr)
                val += m.assign_with_delay(day, table, pred)[0][0]
                c_pred = m._pred_costs(cache[key], pred)
                y0, _ = m.mcf_bins(cache[key]["sched"], c_pred, table, cache[key]["bins"])
                june_agree += float((y0 == cache[key]["y_star"]).mean())
        rec = {
            "epoch": epoch,
            "spo": ep / n_may,
            "june": float(val),
            "may_agree0": agree / n_may,
            "may_agree_spo": spo_ag / n_may,
            "june_agree0": june_agree / max(len(june), 1),
            "abs_pred": abs_p / n_may,
            "abs_res": abs_r / n_may,
        }
        history.append(rec)
        print(
            f"epoch {epoch} spo {rec['spo']:.4f} june {val:.0f} "
            f"agree0 {rec['may_agree0']:.3f}/{rec['june_agree0']:.3f} "
            f"spo_ag {rec['may_agree_spo']:.3f} |pred| {rec['abs_pred']:.2f} |res| {rec['abs_res']:.2f}",
            flush=True,
        )
        if val < best_val:
            best_val = val
            best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best)
    model.to(DEVICE)
    acc = {k: np.zeros(4) for k in ("spo", "ttl")}
    july_agree = 0.0
    model.eval()
    with torch.no_grad():
        for day in july:
            key = day["day"].iloc[0]
            pred = total_pred(model, day, wx, gbr)
            acc["spo"] += m.assign_with_delay(day, table, pred)[0]
            c_pred = m._pred_costs(cache[key], pred)
            y0, _ = m.mcf_bins(cache[key]["sched"], c_pred, table, cache[key]["bins"])
            july_agree += float((y0 == cache[key]["y_star"]).mean())
    for day in july:
        key = day["day"].iloc[0]
        slow, fast, hod_idx, y_t = tensors(day, wx)
        clone = copy.deepcopy(model)
        inner = torch.optim.Adam(clone.parameters(), lr=1e-3)
        morning = day["bin"].to_numpy(int) < NOON
        target = torch.tensor(y_t.cpu().numpy() - gbr_np[key], dtype=torch.float32, device=DEVICE)
        m.ttt_adapt(clone, inner, slow, fast, hod_idx, target, torch.tensor(morning, device=DEVICE))
        clone.eval()
        with torch.no_grad():
            residual = clone.forward_day(slow, fast, hod_idx).cpu().numpy()
        pred = np.clip(gbr_np[key] + residual, -30, 180)
        wait0 = g.mcf(
            day["bin"].to_numpy(int),
            np.nan_to_num(day["s1"].to_numpy(float), nan=180.0),
            np.nan_to_num(day["s2"].to_numpy(float), nan=180.0),
            day["has1"].to_numpy(bool),
            day["has2"].to_numpy(bool),
            table,
        )
        wait_p = m.assign_with_delay(day, table, pred)[1]
        wait = wait0.copy()
        sub = day["bin"].to_numpy(int) >= NOON
        wait[sub] = wait_p[sub]
        tot, l2, l3, wt = g.score_wait(wait, day)
        acc["ttl"] += (tot, l2, l3, wt)
    extra = {"history": history, "july_agree0": july_agree / max(len(july), 1)}
    return pack(acc["spo"]), pack(acc["ttl"]), best_val, extra


def main():
    print("device", DEVICE, "method TTL-SPO", flush=True)
    df = g.read_months((5, 6, 7))
    gaps = {}
    chosen = None
    for ap in AIRPORTS:
        print("probe", ap, flush=True)
        arr = g.build(df, ap, ex.TZ[ap])
        may, june, july = split_days(arr)
        days = [d["day"].iloc[0] for d in may + june + july]
        wx = hourly_weather(ap, days)
        table = g.hour_rate(arr, set(arr.loc[arr["day"].str.startswith("2024-05"), "day"]))
        rec, gbr = probe_airport(arr, wx, table, may, july)
        rec["may_days"] = len(may)
        rec["june_days"] = len(june)
        gaps[ap] = rec
        print(ap, json.dumps(rec, indent=2), flush=True)
        if rec["hindsight_vs_pto"] >= 0.15 and (chosen is None or rec["hindsight_vs_pto"] > gaps[chosen]["hindsight_vs_pto"]):
            chosen = ap
            chosen_pack = (arr, wx, table, may, june, july, gbr)
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    summary = {"method": "TTL-SPO", "probe": prev.get("probe", gaps), "chosen": chosen}
    summary["probe"].update(gaps)
    if chosen is None:
        summary["train"] = None
        summary["note"] = "no scene with hindsight-PTO gap >= 15%"
        OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    print("train on", chosen, "gap", gaps[chosen]["hindsight_vs_pto"], flush=True)
    arr, wx, table, may, june, july, gbr = chosen_pack
    spo, ttl, june_best, extra = train_ttl(may, june, july, wx, table, gbr)
    pto_c = gaps[chosen]["gbr_pto"]["cascade"]
    hour_c = gaps[chosen]["hour"]["cascade"]
    summary["train"] = {
        "airport": chosen,
        "june_best": june_best,
        "july_spo": spo,
        "july_ttl": ttl,
        "july_agree0": extra["july_agree0"],
        "history": extra["history"],
        "ttl_vs_pto": float((pto_c - ttl["cascade"]) / pto_c) if pto_c else None,
        "ttl_vs_hour": float((hour_c - ttl["cascade"]) / hour_c) if hour_c else None,
        "spo_vs_pto": float((pto_c - spo["cascade"]) / pto_c) if pto_c else None,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
