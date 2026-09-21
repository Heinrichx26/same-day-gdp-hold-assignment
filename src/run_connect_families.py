"""EWR summer: same 2025 AAR budgets, connection QP vs family RBS.

First stage = Ribeiro / Wang / Wu AAR (chronological train).
Budget_i = 15 * (d-AAR)+ / d in the flight's 15-min bin.
Second stage = RBS vs time-expanded QP on tail slack.
cap=landed for 1g+2a (unchanged by allocation). Spill is the lever.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from run_bakeoff_2025 import eval_day, hod_series, kmedoids_dtw, nv_quantile, queue_feature, wavg
from run_connect_sweep import hhmm
from run_gdp_gate import wx_class
from run_bakeoff_2025 import add_lags
from run_loop_gdp import lookup, rate_aar, rib_X, split_days

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "winter"
TZ = "America/New_York"
PAIRS = [
    (ROOT / "data" / "bts" / "On_Time_2024_6.zip", ROOT / "data" / "metar" / "ewr_metar_202406.csv"),
    (ROOT / "data" / "bts" / "On_Time_2024_7.zip", ROOT / "data" / "metar" / "ewr_metar_202407.csv"),
]
COLS = [
    "FlightDate", "Tail_Number", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "ArrTime",
    "ArrDelay", "Cancelled", "Diverted",
]


def daily_qp(arr, slack, has, budget):
    n = len(budget)
    h = cp.Variable(n, nonneg=True)
    ns = int(has.sum())
    if ns == 0:
        return np.asarray(budget, float)
    spill = cp.Variable(ns, nonneg=True)
    cons = [cp.sum(h) == float(np.asarray(budget, float).sum()), h <= 90]
    for k, i in enumerate(np.where(has)[0]):
        cons += [spill[k] >= arr[i] + h[i] - slack[i]]
    cp.Problem(cp.Minimize(cp.sum(spill) + 0.01 * cp.sum_squares(h)), cons).solve(solver=cp.OSQP, verbose=False)
    if h.value is None:
        return np.asarray(budget, float)
    return np.array(h.value, float).ravel()


def daily_fill(arr, slack, has, budget, hmax=90.0, hmax_term=240.0):
    """Time-expanded slack fill. Terminating flights (no rotation) absorb hold first."""
    n = len(budget)
    rem = float(np.asarray(budget, float).sum())
    h = np.zeros(n)
    cap = np.where(has, hmax, hmax_term).astype(float)
    residual = np.where(has, np.clip(np.asarray(slack, float) - np.asarray(arr, float), 0.0, None), hmax_term)
    order = np.argsort(-residual)
    for j in order:
        take = min(cap[j], rem, residual[j])
        h[j] = take
        rem -= take
        if rem <= 1e-9:
            break
    if rem > 1e-9:
        room = cap - h
        for j in np.argsort(-np.asarray(slack, float)):
            take = min(max(room[j], 0.0), rem)
            h[j] += take
            rem -= take
            if rem <= 1e-9:
                break
        if rem > 1e-9:
            h += rem / n
    return h


def greedy15(g, budget):
    n = len(g)
    h = np.zeros(n)
    pos = 0
    g = g.copy()
    g["budget"] = budget
    for _, gb in g.groupby("bin15", sort=True):
        nn = len(gb)
        need = float(gb["budget"].sum())
        sl = gb["slack"].to_numpy()
        rem = need
        hg = np.zeros(nn)
        for j in np.argsort(-sl):
            take = min(max(sl[j], 0.0), rem, 90.0)
            hg[j] = take
            rem -= take
            if rem <= 1e-9:
                break
        if rem > 1e-9:
            hg += rem / nn
        h[pos:pos + nn] = hg
        pos += nn
    return h


def load_flights(zp, metar, dest="EWR"):
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=COLS, dtype=str, low_memory=False)
    for c in ["ArrDelay", "Cancelled", "Diverted"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[(df["Cancelled"].fillna(0) < 0.5) & (df["Diverted"].fillna(0) < 0.5)]
    inn = df.loc[df["Dest"] == dest].copy()
    outb = df.loc[df["Origin"] == dest].copy()
    mins = inn["CRSArrTime"].map(hhmm)
    d0 = pd.to_datetime(inn["FlightDate"])
    dmin = inn["CRSDepTime"].map(hhmm)
    add = ((mins < dmin) & dmin.notna() & mins.notna()).astype(int)
    inn["sched"] = (d0 + pd.to_timedelta(add, unit="D") + pd.to_timedelta(mins.fillna(0), unit="m")).dt.tz_localize(
        TZ, nonexistent="shift_forward", ambiguous="NaT"
    )
    amins = inn["ArrTime"].map(hhmm)
    inn["actual"] = (d0 + pd.to_timedelta(add, unit="D") + pd.to_timedelta(amins.fillna(mins), unit="m")).dt.tz_localize(
        TZ, nonexistent="shift_forward", ambiguous="NaT"
    )
    inn = inn.loc[inn["sched"].notna()]
    inn["hod"] = inn["sched"].dt.hour
    inn = inn.loc[inn["hod"].between(6, 22)]
    inn["day"] = inn["sched"].dt.strftime("%Y-%m-%d")
    inn["bin15"] = inn["sched"].dt.floor("15min")
    inn["sched_utc"] = inn["sched"].dt.tz_convert("UTC")
    inn["crs_arr"] = inn["CRSArrTime"].map(hhmm)
    inn["tail"] = inn["Tail_Number"].astype(str).str.strip()
    inn["date"] = pd.to_datetime(inn["FlightDate"])
    outb["tail"] = outb["Tail_Number"].astype(str).str.strip()
    outb["crs_dep"] = outb["CRSDepTime"].map(hhmm)
    outb["date"] = pd.to_datetime(outb["FlightDate"])
    rot = inn.merge(outb, on=["tail", "date"], suffixes=("_in", "_out"))
    rot["gnd"] = rot["crs_dep"] - rot["crs_arr"]
    rot = rot.loc[(rot["gnd"] > 20) & (rot["gnd"] < 300)].sort_values("gnd").drop_duplicates(["tail", "date", "crs_arr"])
    slack_map = rot.set_index(rot["tail"] + "|" + rot["day"] + "|" + rot["crs_arr"].astype(str))["gnd"] - 45
    inn["key"] = inn["tail"] + "|" + inn["day"] + "|" + inn["crs_arr"].astype(str)
    inn["slack"] = inn["key"].map(slack_map).fillna(150.0)
    inn["has"] = inn["key"].isin(slack_map.index)
    inn["arr"] = inn["ArrDelay"].fillna(0)
    met = pd.read_csv(metar)
    met["valid"] = pd.to_datetime(met["valid"], utc=True)
    met["vsby"] = pd.to_numeric(met["vsby"], errors="coerce")
    met["sknt"] = pd.to_numeric(met["sknt"], errors="coerce")
    met["skyl1"] = pd.to_numeric(met.get("skyl1"), errors="coerce")
    met["wxcodes"] = met["wxcodes"].replace({"null": np.nan})
    met["wx"] = [wx_class(s, v, w) for s, v, w in zip(met["sknt"], met["vsby"], met["wxcodes"])]
    inn = pd.merge_asof(
        inn.sort_values("sched_utc"),
        met.sort_values("valid").rename(columns={"valid": "sched_utc"})[["sched_utc", "vsby", "sknt", "skyl1", "wx"]],
        on="sched_utc", direction="backward", tolerance=pd.Timedelta("90min"),
    )
    inn["vsby"] = inn["vsby"].fillna(10)
    inn["sknt"] = inn["sknt"].fillna(8)
    inn["skyl1"] = inn["skyl1"].fillna(5000)
    inn["wx"] = inn["wx"].fillna("VMC")
    return inn


def bins_from(inn: pd.DataFrame) -> pd.DataFrame:
    act = inn.loc[inn["actual"].notna()].groupby(inn.loc[inn["actual"].notna(), "actual"].dt.floor("15min")).size().rename("landed")
    b = inn.groupby(["day", "bin15", "hod"], sort=True).agg(
        demand=("day", "size"), vsby=("vsby", "mean"), sknt=("sknt", "mean"), skyl1=("skyl1", "mean"),
        arr=("arr", "mean"), wx=("wx", "first"),
    ).reset_index()
    b = b.merge(act, left_on="bin15", right_index=True, how="left")
    b["landed"] = b["landed"].fillna(0.0)
    b["imc"] = (b["wx"] == "IMC").astype(float)
    b["low"] = (b["vsby"] <= 3).astype(float)
    return b.sort_values(["day", "bin15"])


def binkey(s):
    t = pd.to_datetime(s, utc=True, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(TZ)
    return t.dt.strftime("%Y-%m-%d %H:%M")


def run_job(job_name: str, dest: str, pairs) -> dict:
    print("JOB", job_name, dest, flush=True)
    inn = pd.concat([load_flights(z, m, dest=dest) for z, m in pairs if z.exists() and m.exists()], ignore_index=True)
    bins = bins_from(inn)
    days = sorted(bins["day"].unique())
    train_days, test_days = split_days(days)
    train = bins.loc[bins["day"].isin(train_days)].copy()
    print("train", len(train_days), "test", len(test_days), "rot", int(inn["has"].sum()), flush=True)

    hod_mean = train.groupby("hod")["landed"].mean()
    hod_p33 = train.groupby("hod")["landed"].quantile(1 / 3)
    tr = pd.concat([add_lags(g) for _, g in train.groupby("day")], ignore_index=True)
    Xtr = rib_X(tr, hod_mean)
    gbm_d = GradientBoostingRegressor(max_depth=3, n_estimators=140, random_state=0).fit(Xtr, tr["arr"].to_numpy(float))
    gbm_c = GradientBoostingRegressor(max_depth=3, n_estimators=140, random_state=1).fit(Xtr, tr["landed"].to_numpy(float))
    series = [hod_series(gb) for _, gb in train.groupby("day")]
    pi, medoids, _ = kmedoids_dtw(series, k=min(4, len(series)))
    wang = {hod: max(nv_quantile(np.array([medoids[k][hi] for k in range(len(medoids))]), pi, 1 / 3), 0.5)
            for hi, hod in enumerate(range(6, 23))}

    def rib_aar(g):
        X = rib_X(g, hod_mean)
        delay = np.clip(gbm_d.predict(X), 0, 90)
        land = np.clip(gbm_c.predict(X), 0.5, 20)
        dem = g["demand"].to_numpy(float)
        return np.clip(0.5 * rate_aar(dem, delay, 1.0) + 0.5 * land, 0.5, 20)

    def wang_aar(g):
        return lookup(wang, g["hod"])

    def wu_aar(g):
        return np.clip(lookup(hod_p33, g["hod"]) - 0.5, 0.5, 20)

    families = {"Ribeiro": rib_aar, "Wang": wang_aar, "Wu": wu_aar}

    def queue_wait(dem, aar):
        """Per-bin average ground wait (min) from the fluid queue."""
        q_g = 0.0
        w = np.zeros(len(dem))
        for t in range(len(dem)):
            d = float(dem[t]) + q_g
            send = min(d, float(aar[t]))
            q_g = d - send
            w[t] = 15.0 * q_g / max(float(dem[t]), 1.0)
        return np.clip(w, 0.0, 90.0)

    def to_budget(gfl, gbin, aar):
        wait = queue_wait(gbin["demand"].to_numpy(float), aar)
        m = dict(zip(gbin["bin15"], wait))
        return np.array([float(m.get(b, 0.0)) for b in gfl["bin15"]], float)

    rows = []
    for day in test_days:
        gfl = inn.loc[inn["day"] == day].sort_values("bin15").reset_index(drop=True)
        gbin = bins.loc[bins["day"] == day].sort_values("bin15").copy()
        if len(gfl) < 20 or len(gbin) < 4:
            continue
        gfl = gfl.copy()
        gfl["bin15"] = binkey(gfl["bin15"])
        gbin["bin15"] = binkey(gbin["bin15"])
        gbin = add_lags(gbin)
        arr = gfl["arr"].to_numpy()
        slack = gfl["slack"].to_numpy()
        has = gfl["has"].to_numpy()
        dem = gbin["demand"].to_numpy(float)
        capn = gbin["landed"].to_numpy(float)
        rec = {"day": day, "n": len(gfl), "n_rot": int(has.sum())}
        for name, fn in families.items():
            aar = fn(gbin)
            bud = to_budget(gfl, gbin, aar)
            h_rbs = bud.copy()
            h_gr = greedy15(gfl, bud)
            h_qp = daily_qp(arr, slack, has, bud)
            h_df = daily_fill(arr, slack, has, bud)
            sim = eval_day(dem, aar, capn)
            def sp(hh):
                return float(np.clip(arr[has] + hh[has] - slack[has], 0, None).mean()) if has.any() else 0.0
            rec[f"{name}_hold"] = float(bud.mean())
            rec[f"{name}_gnd"] = sim["gnd"]
            rec[f"{name}_air"] = sim["air"]
            rec[f"{name}_cost"] = sim["cost"]
            rec[f"{name}_spill_rbs"] = sp(h_rbs)
            rec[f"{name}_spill_gr"] = sp(h_gr)
            rec[f"{name}_spill_qp"] = sp(h_qp)
            rec[f"{name}_spill_df"] = sp(h_df)
        rows.append(rec)
        print("day", day, {n: round(rec[f"{n}_hold"], 2) for n in families},
              "qp_vs_rbs", {n: round((rec[f"{n}_spill_rbs"] - rec[f"{n}_spill_qp"]) / rec[f"{n}_spill_rbs"], 3) if rec[f"{n}_spill_rbs"] else 0 for n in families},
              flush=True)

    p = pd.DataFrame(rows)
    w = p["n_rot"].clip(1).to_numpy(float)
    summary = {"n_days": int(len(p)), "n_rot": float(p["n_rot"].sum()), "test_days": test_days}
    for name in families:
        rbs, qp, gr, df = (float(np.average(p[f"{name}_spill_{k}"], weights=w)) for k in ("rbs", "qp", "gr", "df"))
        hold = float(np.average(p[f"{name}_hold"], weights=p["n"].to_numpy(float)))
        cost = float(np.average(p[f"{name}_cost"], weights=p["n"].to_numpy(float)))
        tot_r = cost + rbs
        tot_d = cost + df
        summary[name] = {
            "hold": hold, "cost_1g2a": cost, "spill_rbs": rbs, "spill_gr": gr, "spill_qp": qp, "spill_df": df,
            "qp_vs_rbs_spill": float((rbs - qp) / rbs) if rbs else 0,
            "df_vs_rbs_spill": float((rbs - df) / rbs) if rbs else 0,
            "df_vs_gr_spill": float((gr - df) / gr) if gr else 0,
            "total_rbs": tot_r, "total_df": tot_d,
            "df_vs_rbs_total": float((tot_r - tot_d) / tot_r) if tot_r else 0,
        }
    summary["gate_spill"] = bool(
        summary["Ribeiro"]["df_vs_rbs_spill"] >= 0.15
        and summary["Wang"]["df_vs_rbs_spill"] >= 0.15
        and summary["Wu"]["df_vs_rbs_spill"] >= 0.15
    )
    summary["gate_total"] = bool(
        summary["Ribeiro"]["df_vs_rbs_total"] >= 0.15
        and summary["Wang"]["df_vs_rbs_total"] >= 0.15
        and summary["Wu"]["df_vs_rbs_total"] >= 0.15
    )
    summary["gate_vs_greedy"] = bool(
        summary["Ribeiro"]["df_vs_gr_spill"] >= 0.15
        and summary["Wang"]["df_vs_gr_spill"] >= 0.15
        and summary["Wu"]["df_vs_gr_spill"] >= 0.15
    )
    summary["job"] = job_name
    summary["dest"] = dest
    print(job_name, "spill vs RBS", {k: round(summary[k]["df_vs_rbs_spill"], 3) for k in families},
          "vs greedy", {k: round(summary[k]["df_vs_gr_spill"], 3) for k in families},
          "gate", summary["gate_spill"], summary["gate_vs_greedy"], flush=True)
    return summary


def main():
    jobs = [
        ("EWR_Summer", "EWR", PAIRS),
        ("BOS_JF", "BOS", [
            (ROOT / "data" / "bts" / "On_Time_2024_1.zip", ROOT / "data" / "metar" / "bos_metar_202401.csv"),
            (ROOT / "data" / "bts" / "On_Time_2024_2.zip", ROOT / "data" / "metar" / "bos_metar_202402.csv"),
        ]),
    ]
    out = {}
    for job_name, dest, pairs in jobs:
        out[job_name] = run_job(job_name, dest, pairs)
    (OUT / "connect_families.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: {fk: v[fk] for fk in ("Ribeiro", "Wang", "Wu", "gate_spill", "gate_vs_greedy", "n_rot")} for k, v in out.items()}, indent=2))
    if all(v.get("gate_spill") for v in out.values()):
        (OUT / "VS2025_KEEP.md").write_text(
            "# KEEP vs Ribeiro/Wang/Wu\n\n"
            "Time-expanded slack fill of GDP queue-wait; terminating flights absorb hold.\n"
            "First stage = 2025 AAR. Second stage vs that family's RBS and 15-min greedy.\n\n"
            + json.dumps(out, indent=2, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
