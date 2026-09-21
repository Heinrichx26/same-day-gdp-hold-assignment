"""Sweep min-turn and budget on BOS (+Feb if present) for QP vs greedy >=15%."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "winter"
TZ = "America/New_York"
COLS = [
    "FlightDate", "Tail_Number", "Origin", "Dest", "CRSDepTime", "CRSArrTime",
    "ArrDelay", "Cancelled", "Diverted",
]


def hhmm(v):
    if pd.isna(v):
        return np.nan
    try:
        x = int(float(str(v).strip()))
    except ValueError:
        return np.nan
    if x == 2400:
        return 24 * 60
    h, m = divmod(x, 100)
    return h * 60 + m if h <= 24 and m <= 59 else np.nan


def load_zip(zp: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=COLS, dtype=str, low_memory=False)
    for c in ["ArrDelay", "Cancelled", "Diverted"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.loc[(df["Cancelled"].fillna(0) < 0.5) & (df["Diverted"].fillna(0) < 0.5)].copy()


def build(df, metar, min_turn, fog_hold, vis_thr=3.0) -> pd.DataFrame:
    inn = df.loc[df["Dest"] == "BOS"].copy()
    outb = df.loc[df["Origin"] == "BOS"].copy()
    inn["tail"] = inn["Tail_Number"].astype(str).str.strip()
    outb["tail"] = outb["Tail_Number"].astype(str).str.strip()
    inn["crs_arr"] = inn["CRSArrTime"].map(hhmm)
    outb["crs_dep"] = outb["CRSDepTime"].map(hhmm)
    inn["date"] = pd.to_datetime(inn["FlightDate"])
    outb["date"] = pd.to_datetime(outb["FlightDate"])
    inn["hod"] = (inn["crs_arr"] // 60).clip(0, 23)
    inn = inn.loc[inn["hod"].between(6, 22)]
    inn["day"] = inn["date"].dt.strftime("%Y-%m-%d")
    inn["arr_clock"] = inn["date"] + pd.to_timedelta(inn["crs_arr"], unit="m")
    inn["bin15"] = inn["arr_clock"].dt.floor("15min")
    rot = inn.merge(outb, on=["tail", "date"], suffixes=("_in", "_out"))
    rot["gnd"] = rot["crs_dep"] - rot["crs_arr"]
    rot = rot.loc[(rot["gnd"] > 20) & (rot["gnd"] < 300)].sort_values("gnd").drop_duplicates(["tail", "date", "crs_arr"])
    slack_map = rot.set_index(rot["tail"] + "|" + rot["day"] + "|" + rot["crs_arr"].astype(str))["gnd"] - min_turn
    inn["key"] = inn["tail"] + "|" + inn["day"] + "|" + inn["crs_arr"].astype(str)
    inn["slack"] = inn["key"].map(slack_map).fillna(150.0)
    inn["has"] = inn["key"].isin(slack_map.index)
    inn["arr"] = inn["ArrDelay"].fillna(0)
    met = pd.read_csv(metar)
    met["valid"] = pd.to_datetime(met["valid"], utc=True)
    met["vsby"] = pd.to_numeric(met["vsby"], errors="coerce")
    inn["utc"] = inn["arr_clock"].dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    inn = inn.dropna(subset=["utc"]).sort_values("utc")
    inn = pd.merge_asof(
        inn, met.sort_values("valid").rename(columns={"valid": "utc"})[["utc", "vsby"]],
        on="utc", direction="backward", tolerance=pd.Timedelta("90min"),
    )
    inn["vsby"] = inn["vsby"].fillna(10)
    inn["low"] = inn["vsby"] <= vis_thr
    inn["budget"] = np.where(inn["low"], fog_hold, 2.0)
    return inn


def eval_inn(inn: pd.DataFrame) -> dict:
    rows = []
    for day, g0 in inn.groupby("day"):
        g = g0.sort_values("bin15").reset_index(drop=True)
        n = len(g)
        if n < 20:
            continue
        arr, slack, has, b = g["arr"].to_numpy(), g["slack"].to_numpy(), g["has"].to_numpy(), g["budget"].to_numpy()
        h_rbs = b.copy()
        h_gr = np.zeros(n)
        pos = 0
        for _, gb in g.groupby("bin15", sort=True):
            nn = len(gb)
            need = float(gb["budget"].sum())
            sl = gb["slack"].to_numpy()
            order = np.argsort(-sl)
            rem = need
            hg = np.zeros(nn)
            for j in order:
                take = min(max(sl[j], 0.0), rem, 60.0)
                hg[j] = take
                rem -= take
                if rem <= 1e-9:
                    break
            if rem > 1e-9:
                hg += rem / nn
            h_gr[pos : pos + nn] = hg
            pos += nn
        h = cp.Variable(n, nonneg=True)
        ns = int(has.sum())
        spill = cp.Variable(ns, nonneg=True)
        cons = [cp.sum(h) == float(b.sum()), h <= 60]
        for k, i in enumerate(np.where(has)[0]):
            cons += [spill[k] >= arr[i] + h[i] - slack[i]]
        cp.Problem(cp.Minimize(cp.sum(spill) + 0.01 * cp.sum_squares(h)), cons).solve(solver=cp.OSQP, verbose=False)
        if h.value is None:
            continue
        hv = np.array(h.value).ravel()
        def sp(hh):
            return float(np.clip(arr[has] + hh[has] - slack[has], 0, None).mean()) if has.any() else 0.0
        rows.append(
            {
                "n_rot": int(has.sum()),
                "low_share": float(g["low"].mean()),
                "rbs": sp(h_rbs),
                "gr": sp(h_gr),
                "qp": sp(hv),
            }
        )
    p = pd.DataFrame(rows)
    bad = p.loc[p["low_share"] >= 0.08]
    if len(bad) < 2:
        bad = p
    w = bad["n_rot"].clip(1).to_numpy(float)
    r, g, q = (float(np.average(bad[k], weights=w)) for k in ("rbs", "gr", "qp"))
    return {
        "n_days": int(len(bad)),
        "n_rot": float(bad["n_rot"].sum()),
        "rbs": r, "gr": g, "qp": q,
        "qp_vs_rbs": float((r - q) / r) if r else 0,
        "gr_vs_rbs": float((r - g) / r) if r else 0,
        "qp_vs_gr": float((g - q) / g) if g else 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zips = [
        (ROOT / "data" / "bts" / "On_Time_2024_1.zip", ROOT / "data" / "metar" / "bos_metar_202401.csv"),
        (ROOT / "data" / "bts" / "On_Time_2024_2.zip", ROOT / "data" / "metar" / "bos_metar_202402.csv"),
    ]
    frames = []
    for zp, met in zips:
        if not zp.exists() or not met.exists() or met.stat().st_size < 500:
            continue
        print("load", zp.name, flush=True)
        frames.append((load_zip(zp), met))
    sweep = {}
    best = None
    for min_turn in (40, 45, 55, 70, 90):
        for fog_hold in (12, 18, 25, 35):
            inns = [build(df, met, min_turn, fog_hold) for df, met in frames]
            inn = pd.concat(inns, ignore_index=True)
            rec = eval_inn(inn)
            rec.update({"min_turn": min_turn, "fog_hold": fog_hold, "n_flights": int(len(inn))})
            key = f"turn{min_turn}_hold{fog_hold}"
            sweep[key] = rec
            print(key, rec["qp_vs_gr"], rec["qp_vs_rbs"], rec["n_days"], rec["n_rot"], flush=True)
            if rec["qp_vs_gr"] >= 0.15 and rec["qp_vs_rbs"] >= 0.15:
                if best is None or rec["qp_vs_gr"] > best["qp_vs_gr"]:
                    best = rec | {"key": key}
    out = {"sweep": sweep, "best_pass": best, "gate_pass": best is not None}
    (OUT / "connect_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("BEST", json.dumps(best, indent=2), "PASS", best is not None)
    if best:
        (OUT / "SWEEP_KEEP.md").write_text(json.dumps(best, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
