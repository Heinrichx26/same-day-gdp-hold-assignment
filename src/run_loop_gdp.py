"""Closed-loop delay-triggered GDP vs Ribeiro / Wang / Wu.

Chronological split (fog/IMC in both sides). cap = realized landed.
Trigger uses lagged realized ArrDelay only (tactical feedback).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from run_bakeoff_2025 import (
    add_lags,
    eval_day,
    hod_series,
    kmedoids_dtw,
    load_month,
    nv_quantile,
    queue_feature,
    wavg,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "winter"

JOBS = [
    dict(name="BOS_JF", dest="BOS", pairs=[
        (ROOT / "data" / "bts" / "On_Time_2024_1.zip", ROOT / "data" / "metar" / "bos_metar_202401.csv"),
        (ROOT / "data" / "bts" / "On_Time_2024_2.zip", ROOT / "data" / "metar" / "bos_metar_202402.csv"),
    ]),
    dict(name="EWR_Jan", dest="EWR", pairs=[
        (ROOT / "data" / "bts" / "On_Time_2024_1.zip", ROOT / "data" / "metar" / "ewr_metar_202401.csv"),
    ]),
    dict(name="EWR_Jun", dest="EWR", pairs=[
        (ROOT / "data" / "bts" / "On_Time_2024_6.zip", ROOT / "data" / "metar" / "ewr_metar_202406.csv"),
    ]),
    dict(name="EWR_Jul", dest="EWR", pairs=[
        (ROOT / "data" / "bts" / "On_Time_2024_7.zip", ROOT / "data" / "metar" / "ewr_metar_202407.csv"),
    ]),
    dict(name="EWR_Summer", dest="EWR", pairs=[
        (ROOT / "data" / "bts" / "On_Time_2024_6.zip", ROOT / "data" / "metar" / "ewr_metar_202406.csv"),
        (ROOT / "data" / "bts" / "On_Time_2024_7.zip", ROOT / "data" / "metar" / "ewr_metar_202407.csv"),
    ]),
]


def lookup(s, hods, default=4.0):
    return np.array([max(float(s.get(int(h), default) or default), 0.5) for h in hods], float)


def rib_X(g, nom_hod):
    dem = g["demand"].to_numpy(float)
    nom = lookup(nom_hod, g["hod"])
    qf = queue_feature(dem, nom)
    lag1 = g["lag1"].to_numpy(float) if "lag1" in g.columns else dem.copy()
    lag1 = np.where(np.isnan(lag1), dem, lag1)
    lag2 = g["lag2"].to_numpy(float) if "lag2" in g.columns else lag1.copy()
    lag2 = np.where(np.isnan(lag2), lag1, lag2)
    vlag = g["vlag"].to_numpy(float) if "vlag" in g.columns else g["vsby"].to_numpy(float)
    vlag = np.where(np.isnan(vlag), g["vsby"].to_numpy(float), vlag)
    ceil = np.log1p(g["skyl1"].to_numpy(float).clip(0, 20000) / 100.0)
    return np.column_stack(
        [
            g["vsby"], g["sknt"], g["hod"], dem, g["imc"], g["low"],
            qf, lag1, lag2, vlag, ceil,
            np.sin(2 * np.pi * g["hod"] / 24.0),
            np.cos(2 * np.pi * g["hod"] / 24.0),
        ]
    )


def ewma(x, span=4):
    s = 0.0
    out = np.zeros(len(x))
    a = 2.0 / (span + 1.0)
    for i, v in enumerate(x):
        s = a * v + (1.0 - a) * s if i else v
        out[i] = s
    return out


def rate_aar(demand, delay, alpha):
    return np.clip(demand * 15.0 / (15.0 + np.clip(alpha * delay, 0, 180)), 0.5, 30.0)


def trigger_aar(demand, roll, landed_lag, tau, alpha, mode="rate"):
    on = roll > tau
    if mode == "rate":
        alt = rate_aar(demand, roll, alpha)
    elif mode == "ema":
        alt = np.clip(np.minimum(demand, np.where(np.isnan(landed_lag), demand, landed_lag)), 0.5, 30)
    else:  # mix
        r = rate_aar(demand, roll, alpha)
        e = np.clip(np.where(np.isnan(landed_lag), demand, landed_lag), 0.5, 30)
        alt = 0.5 * r + 0.5 * e
    return np.where(on, alt, np.clip(demand, 0.5, 30.0))


def split_days(days):
    days = sorted(days)
    k = max(8, int(round(len(days) * 0.65)))
    k = min(k, len(days) - 4) if len(days) > 12 else max(1, len(days) - max(3, len(days) // 3))
    return days[:k], days[k:]


def run_job(job: dict) -> dict:
    print("JOB", job["name"], flush=True)
    parts = []
    for zp, met in job["pairs"]:
        if zp.exists() and met.exists():
            parts.append(load_month(zp, met, dest=job["dest"]))
    if not parts:
        return {"name": job["name"], "skip": True}
    bins = pd.concat(parts, ignore_index=True)
    days = sorted(bins["day"].unique())
    train_days, test_days = split_days(days)
    train = bins.loc[bins["day"].isin(train_days)].copy()
    test = bins.loc[bins["day"].isin(test_days)].copy()
    print(" ", job["name"], "train", train_days[0], train_days[-1], len(train_days),
          "test", test_days[0], test_days[-1], len(test_days), flush=True)

    hod_mean = train.groupby("hod")["landed"].mean()
    hod_p80 = train.groupby("hod")["landed"].quantile(0.80)
    hod_p33 = train.groupby("hod")["landed"].quantile(1.0 / 3.0)

    train_f = pd.concat([add_lags(g) for _, g in train.groupby("day")], ignore_index=True)
    Xtr = rib_X(train_f, hod_mean)
    gbm_d = GradientBoostingRegressor(max_depth=3, n_estimators=120, random_state=0)
    gbm_c = GradientBoostingRegressor(max_depth=3, n_estimators=120, random_state=1)
    gbm_d.fit(Xtr, train_f["arr"].to_numpy(float))
    gbm_c.fit(Xtr, train_f["landed"].to_numpy(float))

    series = [hod_series(gb) for _, gb in train.groupby("day")]
    pi, medoids, _ = kmedoids_dtw(series, k=min(4, len(series)))
    wang_q33 = {}
    for hi, hod in enumerate(range(6, 23)):
        samples = np.array([medoids[k][hi] for k in range(len(medoids))])
        wang_q33[hod] = max(nv_quantile(samples, pi, 1.0 / 3.0), 0.5)

    def pred_delay(g):
        return np.clip(gbm_d.predict(rib_X(g, hod_mean)), 0, 120)

    def pred_land(g):
        return np.clip(gbm_c.predict(rib_X(g, hod_mean)), 0.5, 20)

    def aar_rib(g, alpha=1.0):
        return np.clip(0.5 * rate_aar(g["demand"].to_numpy(float), pred_delay(g), alpha) + 0.5 * pred_land(g), 0.5, 20)

    # DFL alpha for open-loop Ribeiro (stronger family)
    rib_scores = {}
    for a in (0.15, 0.25, 0.4, 0.6, 1.0):
        cs, ns = [], []
        for _, g0 in train.groupby("day"):
            g = add_lags(g0.sort_values("bin15"))
            r = eval_day(g["demand"].to_numpy(float), aar_rib(g, a), g["landed"].to_numpy(float))
            cs.append(r["cost"]); ns.append(float(g["demand"].sum()))
        rib_scores[a] = float(np.average(cs, weights=ns))
    rib_alpha = min(rib_scores, key=rib_scores.get)
    print("  RibeiroDFL alpha", rib_alpha, rib_scores, flush=True)

    # DFL trigger on train
    tau_grid = [8, 12, 16, 20, 28, 40]
    alpha_grid = [0.2, 0.35, 0.5]
    best = None
    for tau in tau_grid:
        for alpha in alpha_grid:
            for mode in ("rate", "ema", "mix"):
                cs, ns = [], []
                for _, g0 in train.groupby("day"):
                    g = add_lags(g0.sort_values("bin15"))
                    dem = g["demand"].to_numpy(float)
                    arr_lag = g["arr"].shift(1).to_numpy(float)
                    arr_lag = np.where(np.isnan(arr_lag), 0.0, arr_lag)
                    roll = ewma(arr_lag, 4)
                    lagl = g["lag1"].to_numpy(float) if "lag1" in g.columns else dem.copy()
                    aar = trigger_aar(dem, roll, lagl, tau, alpha, mode)
                    r = eval_day(dem, aar, g["landed"].to_numpy(float))
                    cs.append(r["cost"])
                    ns.append(float(dem.sum()))
                c = float(np.average(cs, weights=ns))
                if best is None or c < best[0]:
                    best = (c, tau, alpha, mode)
    dfl_cost, dfl_tau, dfl_alpha, dfl_mode = best
    print("  DFL", best, flush=True)

    policies = {
        "NoGDP": lambda g: np.clip(g["demand"].to_numpy(float), 0.5, 30),
        "Oracle": lambda g: np.clip(g["landed"].to_numpy(float), 0.5, 30),
        "HourP80": lambda g: lookup(hod_p80, g["hod"]),
        "Wang": lambda g: lookup(wang_q33, g["hod"]),
        "Wu": lambda g: np.clip(lookup(hod_p33, g["hod"]) - 0.5, 0.5, 20),
        "Ribeiro": lambda g: aar_rib(g, 1.0),
        "RibeiroDFL": lambda g: aar_rib(g, rib_alpha),
        "Loop": lambda g: trigger_aar(
            g["demand"].to_numpy(float),
            ewma(np.where(np.isnan(g["arr"].shift(1).to_numpy(float)), 0.0, g["arr"].shift(1).to_numpy(float)), 4),
            g["lag1"].to_numpy(float) if "lag1" in g.columns else g["demand"].to_numpy(float),
            dfl_tau, dfl_alpha, dfl_mode,
        ),
    }

    rows = []
    for day, g0 in test.groupby("day"):
        g = add_lags(g0.sort_values("bin15"))
        dem = g["demand"].to_numpy(float)
        cap = g["landed"].to_numpy(float)
        rec = {
            "day": day, "n": float(dem.sum()), "arr": float(g["arr"].mean()),
            "vis": float(g["vsby"].mean()), "low": float(g["low"].mean()),
        }
        for name, fn in policies.items():
            r = eval_day(dem, fn(g), cap)
            rec[f"{name}_cost"] = r["cost"]
            rec[f"{name}_gnd"] = r["gnd"]
            rec[f"{name}_air"] = r["air"]
        rows.append(rec)

    p = pd.DataFrame(rows)
    costs = {name: {"cost": wavg(p, f"{name}_cost"), "gnd": wavg(p, f"{name}_gnd"), "air": wavg(p, f"{name}_air")}
             for name in policies}
    # high-delay subset
    hi = p.loc[p["arr"] >= 15]
    costs_hi = None
    if len(hi) >= 2:
        costs_hi = {name: {"cost": wavg(hi, f"{name}_cost"), "gnd": wavg(hi, f"{name}_gnd"), "air": wavg(hi, f"{name}_air")}
                    for name in policies}

    def vs_block(cdict):
        out = {}
        for on in cdict:
            out[on] = {lab: float((cdict[lab]["cost"] - cdict[on]["cost"]) / cdict[lab]["cost"]) if cdict[lab]["cost"] else 0.0
                       for lab in ("Ribeiro", "RibeiroDFL", "Wang", "Wu", "NoGDP", "HourP80")}
            out[on]["gate_2025"] = bool(
                out[on]["Ribeiro"] >= 0.15 and out[on]["Wang"] >= 0.15 and out[on]["Wu"] >= 0.15
            )
            out[on]["gate_strong"] = bool(
                out[on]["RibeiroDFL"] >= 0.15 and out[on]["Wang"] >= 0.15 and out[on]["Wu"] >= 0.15
            )
            out[on]["gate_value"] = bool(out[on]["NoGDP"] >= 0.05)
        return out

    vs = vs_block(costs)
    keepers = [k for k, v in vs.items() if v["gate_2025"] and v["gate_value"] and k not in ("Oracle",)]
    out = {
        "name": job["name"],
        "dest": job["dest"],
        "n_train": len(train_days),
        "n_test": len(test_days),
        "test_days": test_days,
        "dfl": {"cost": dfl_cost, "tau": dfl_tau, "alpha": dfl_alpha, "mode": dfl_mode},
        "costs": costs,
        "costs_hi": costs_hi,
        "n_hi_days": int(len(hi)),
        "vs": vs,
        "vs_hi": vs_block(costs_hi) if costs_hi else None,
        "keepers": keepers,
        "mean_test_arr": float(p["arr"].mean()),
    }
    print(" ", job["name"], "Loop", costs["Loop"]["cost"], "Rib", costs["Ribeiro"]["cost"],
          "Wang", costs["Wang"]["cost"], "Wu", costs["Wu"]["cost"], "NoGDP", costs["NoGDP"]["cost"],
          "keepers", keepers, flush=True)
    if costs_hi:
        print("  HI Loop", costs_hi["Loop"]["cost"], "Rib", costs_hi["Ribeiro"]["cost"],
              "Wang", costs_hi["Wang"]["cost"], "NoGDP", costs_hi["NoGDP"]["cost"], flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for job in JOBS:
        try:
            results.append(run_job(job))
        except Exception as e:
            print("FAIL", job["name"], e, flush=True)
            results.append({"name": job["name"], "error": str(e)})
    summary = {"jobs": results}
    # pool Loop vs families across jobs with flights
    keepers = []
    for r in results:
        if r.get("keepers"):
            keepers.append(r["name"])
    summary["job_keepers"] = keepers
    (OUT / "loop_gdp.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({r.get("name"): {"keepers": r.get("keepers"), "costs": r.get("costs"), "dfl": r.get("dfl"), "n_hi": r.get("n_hi_days")} for r in results}, indent=2, default=float))
    if keepers:
        (OUT / "VS2025_KEEP.md").write_text(
            "# KEEP vs Ribeiro/Wang/Wu (closed-loop GDP)\n\n" + json.dumps(summary, indent=2, default=float),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
