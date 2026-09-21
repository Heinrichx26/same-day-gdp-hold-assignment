"""Honest GDP gate: weather AAR vs static AAR on cost 1*ground+2*air.

Uses data/bts/ewr_arrivals_202406.csv and data/metar/ewr_metar_202406.csv.
Planned rate is AAR (held flights still release later). Capacity is weather AAR.
C vs A airborne is tautological; the gate is C vs B on cost.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BTS = ROOT / "data" / "bts" / "ewr_arrivals_202406.csv"
METAR = ROOT / "data" / "metar" / "ewr_metar_202406.csv"
OUT = ROOT / "results" / "gdp"
TZ = "America/New_York"
Q = 0.80
PRECIP = re.compile(r"TS|SH|RA|SN|DZ|PL|GR|GS|FZ", re.I)


def hhmm_min(v) -> float:
    if pd.isna(v):
        return np.nan
    try:
        x = int(float(str(v).strip()))
    except ValueError:
        return np.nan
    if x == 2400:
        return 24 * 60
    h, m = divmod(x, 100)
    if h > 24 or m > 59:
        return np.nan
    return h * 60 + m


def local_ts(dates, hhmm, dep_hhmm):
    mins = hhmm.map(hhmm_min)
    d0 = pd.to_datetime(dates, errors="coerce")
    dmin = dep_hhmm.map(hhmm_min)
    add = ((mins < dmin) & dmin.notna() & mins.notna()).astype(int)
    d0 = d0 + pd.to_timedelta(add, unit="D")
    extra = (mins >= 24 * 60).fillna(False)
    mins = mins.mask(extra, mins - 24 * 60)
    clock = d0 + pd.to_timedelta(mins, unit="m")
    return clock.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")


def wx_class(sknt, vsby, wxcodes) -> str:
    precip = bool(isinstance(wxcodes, str) and PRECIP.search(wxcodes))
    wind = float(sknt) >= 15.0 if pd.notna(sknt) else False
    vis = float(vsby) <= 4.0 if pd.notna(vsby) else False
    if precip or vis:
        return "IMC"
    if wind:
        return "WIND"
    return "VMC"


def simulate(demand, planned, cap):
    q_g = q_a = g_min = a_min = 0.0
    for t in range(len(demand)):
        d = float(demand[t]) + q_g
        send = min(d, float(planned[t]))
        q_g = d - send
        g_min += q_g * 15.0
        arr = send + q_a
        land = min(arr, float(cap[t]))
        q_a = arr - land
        a_min += q_a * 15.0
    tot = float(demand.sum()) or 1.0
    return {
        "gnd": g_min / tot,
        "air": a_min / tot,
        "tot": (g_min + a_min) / tot,
        "cost": (g_min + 2.0 * a_min) / tot,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BTS, dtype=str, low_memory=False)
    df = df.loc[df["Dest"].astype(str).str.upper() == "EWR"].copy()
    canc = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0)
    div = pd.to_numeric(df["Diverted"], errors="coerce").fillna(0)
    df = df.loc[(canc < 0.5) & (div < 0.5)].copy()
    df["sched"] = local_ts(df["FlightDate"], df["CRSArrTime"], df["CRSDepTime"])
    df["actual"] = local_ts(df["FlightDate"], df["ArrTime"], df["CRSDepTime"])
    df = df.loc[df["sched"].notna()].copy()
    df["hod"] = df["sched"].dt.hour
    df = df.loc[(df["hod"] >= 6) & (df["hod"] < 23)].copy()
    df["day"] = df["sched"].dt.strftime("%Y-%m-%d")
    df["bin15"] = df["sched"].dt.floor("15min")
    df["sched_utc"] = df["sched"].dt.tz_convert("UTC")
    df["arr_delay"] = pd.to_numeric(df["ArrDelay"], errors="coerce")

    met = pd.read_csv(METAR)
    met["valid"] = pd.to_datetime(met["valid"], utc=True)
    met["sknt"] = pd.to_numeric(met["sknt"], errors="coerce")
    met["vsby"] = pd.to_numeric(met["vsby"], errors="coerce")
    met["wxcodes"] = met["wxcodes"].replace({"null": np.nan})
    met = met.sort_values("valid")
    met["wx"] = [wx_class(s, v, w) for s, v, w in zip(met["sknt"], met["vsby"], met["wxcodes"])]
    left = df.sort_values("sched_utc")
    fl = pd.merge_asof(
        left,
        met.rename(columns={"valid": "sched_utc"})[["sched_utc", "wx"]],
        on="sched_utc",
        direction="backward",
        tolerance=pd.Timedelta("90min"),
    )
    fl["wx"] = fl["wx"].fillna("VMC")

    act = (
        fl.loc[fl["actual"].notna()]
        .groupby(fl.loc[fl["actual"].notna(), "actual"].dt.floor("15min"))
        .size()
        .rename("landed")
    )
    bins = (
        fl.groupby(["day", "bin15", "hod", "wx"], dropna=False)
        .size()
        .rename("demand")
        .reset_index()
        .merge(act, left_on="bin15", right_index=True, how="left")
    )
    bins["landed"] = bins["landed"].fillna(0)
    bins["wx"] = bins["wx"].fillna("VMC")

    days = sorted(bins["day"].unique())
    parts = []
    for d in days:
        train = bins.loc[bins["day"] != d]
        wx_q = train.groupby(["hod", "wx"])["landed"].quantile(Q)
        st_q = train.groupby("hod")["landed"].quantile(Q)
        wx_med = train.groupby(["hod", "wx"])["landed"].median()
        part = bins.loc[bins["day"] == d].copy()
        part["aar_wx"] = [
            max(round(float(wx_q.get((h, w), st_q.get(h, 1.0)))), 1) for h, w in zip(part["hod"], part["wx"])
        ]
        part["aar_static"] = [max(round(float(st_q.get(h, 1.0))), 1) for h in part["hod"]]
        part["aar_med"] = [
            max(round(float(wx_med.get((h, w), st_q.get(h, 1.0)))), 1) for h, w in zip(part["hod"], part["wx"])
        ]
        parts.append(part)
    bins = pd.concat(parts, ignore_index=True).sort_values(["day", "bin15"])

    rows = []
    for day, g in bins.groupby("day"):
        g = g.sort_values("bin15")
        demand = g["demand"].to_numpy(float)
        cap = g["aar_wx"].to_numpy(float)
        rec = {
            "day": day,
            "n_flights": float(demand.sum()),
            "bad_wx": bool(((g["wx"] != "VMC").sum() >= 4) or ((g["wx"] != "VMC").mean() >= 0.15)),
            "adv_share": float((g["wx"] != "VMC").mean()),
            "obs_arr_delay": float(fl.loc[fl["day"] == day, "arr_delay"].mean()),
        }
        plans = {
            "A": np.full(len(g), 1e9),
            "B": g["aar_static"].to_numpy(float),
            "C": g["aar_wx"].to_numpy(float),
            "D": g["aar_med"].to_numpy(float),
        }
        for k, p in plans.items():
            sim = simulate(demand, p, cap)
            rec[f"{k}_gnd"] = sim["gnd"]
            rec[f"{k}_air"] = sim["air"]
            rec[f"{k}_tot"] = sim["tot"]
            rec[f"{k}_cost"] = sim["cost"]
        rows.append(rec)
    day_df = pd.DataFrame(rows)

    def pack(df: pd.DataFrame) -> dict:
        if not len(df):
            return {"n_days": 0, "n_flights": 0.0}
        w = df["n_flights"].to_numpy(float)
        out = {"n_days": int(len(df)), "n_flights": float(w.sum())}
        for k in "ABCD":
            out[k] = {
                "gnd": float(np.average(df[f"{k}_gnd"], weights=w)),
                "air": float(np.average(df[f"{k}_air"], weights=w)),
                "tot": float(np.average(df[f"{k}_tot"], weights=w)),
                "cost": float(np.average(df[f"{k}_cost"], weights=w)),
            }
        out["C_vs_B_cost"] = float((out["B"]["cost"] - out["C"]["cost"]) / out["B"]["cost"]) if out["B"]["cost"] else 0.0
        out["C_vs_A_cost"] = float((out["A"]["cost"] - out["C"]["cost"]) / out["A"]["cost"]) if out["A"]["cost"] else 0.0
        out["C_vs_A_air"] = float((out["A"]["air"] - out["C"]["air"]) / out["A"]["air"]) if out["A"]["air"] else 0.0
        out["B_vs_A_cost"] = float((out["A"]["cost"] - out["B"]["cost"]) / out["A"]["cost"]) if out["A"]["cost"] else 0.0
        return out

    overall = pack(day_df)
    bad = pack(day_df.loc[day_df["bad_wx"]])
    good = pack(day_df.loc[~day_df["bad_wx"]])
    wx_aar = (
        bins.groupby("wx")
        .agg(n=("landed", "size"), mean_land=("landed", "mean"), p80=("landed", lambda s: float(s.quantile(Q))), mean_dem=("demand", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "n_flights": int(len(fl)),
        "n_days": int(len(day_df)),
        "obs_arr_delay": float(fl["arr_delay"].mean()),
        "wx_flights": {k: int(v) for k, v in fl["wx"].value_counts().items()},
        "wx_aar": wx_aar,
        "overall": overall,
        "bad": bad,
        "good": good,
        "gate": "C vs B cost on bad-wx days",
        "gate_save": bad.get("C_vs_B_cost", 0.0),
        "gate_pass": bool(bad.get("C_vs_B_cost", 0.0) >= 0.15),
    }
    (OUT / "gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    day_df.to_csv(OUT / "gate_days.csv", index=False)
    print(json.dumps(summary, indent=2))
    note = (
        f"C_vs_B_cost_bad={summary['gate_save']:.4f} pass={summary['gate_pass']}\n"
        f"A={bad.get('A')}\nB={bad.get('B')}\nC={bad.get('C')}\nD={bad.get('D')}\n"
    )
    if summary["gate_pass"]:
        (OUT / "KILL.txt").unlink(missing_ok=True)
        (OUT / "KEEP.md").write_text("# KEEP GDP C vs B\n\n" + note, encoding="utf-8")
    else:
        (OUT / "KEEP.md").unlink(missing_ok=True)
        (OUT / "KILL.txt").write_text("KILL GDP C vs B cost < 15% on bad-wx days\n" + note, encoding="utf-8")
    print(note)


if __name__ == "__main__":
    main()
