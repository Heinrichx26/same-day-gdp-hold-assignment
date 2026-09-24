"""Capture the DFW hindsight gap on July.

Late-aircraft delay of the previous same-tail leg is known before this
flight's departure. It enters the delay map; min-cost flow is unchanged.
Turn=45, May lower-third rate. May+June fit, July scored once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

import run_expand_full as ex
import run_trc_gate as g
import run_ttl_spo as ttl
import ttl_spo as m

OUT = Path(__file__).resolve().parents[1] / "results" / "winter" / "close_july_dfw.json"
DEVICE = g.DEVICE


def build_with_prev(df, dest, tz):
    legs = df.sort_values(["date", "tail", "dep_m"]).copy()
    grp = legs.groupby(["date", "tail"], sort=False)
    legs["prev_arr_delay"] = pd.to_numeric(grp["ArrDelay"].shift(1), errors="coerce")
    arr = g.build(df, dest, tz)
    key = ["date", "tail", "dep_m"]
    extra = legs[key + ["prev_arr_delay", "Origin"]].drop_duplicates(key)
    arr = arr.merge(extra, on=key, how="left", suffixes=("", "_y"))
    if "Origin_y" in arr.columns:
        arr["Origin"] = arr["Origin"].fillna(arr["Origin_y"])
        arr = arr.drop(columns=["Origin_y"])
    arr["prev_arr_delay"] = arr["prev_arr_delay"].fillna(0.0)
    arr["has_prev"] = (arr["prev_arr_delay"].abs() > 1e-6).astype(np.float32)
    arr["dow"] = pd.to_datetime(arr["day"]).dt.dayofweek.astype(np.float32)
    return arr


def rich_xy(day, wx, origin_te):
    X, y = g.feat_rows(day)
    key = day["day"].iloc[0]
    hod_idx = np.clip(day["hod"].to_numpy(int) - 6, 0, 16)
    extra_wx = wx[key][hod_idx]
    prev = day["prev_arr_delay"].to_numpy(float) / 60.0
    hasp = day["has_prev"].to_numpy(float) if "has_prev" in day.columns else (np.abs(prev) > 0).astype(float)
    orig = day["Origin"].map(origin_te).fillna(0.0).to_numpy(float) / 60.0
    dow = day["dow"].to_numpy(float) / 6.0 if "dow" in day.columns else np.zeros(len(day))
    extra = np.column_stack([extra_wx, prev, hasp, orig, dow])
    return np.concatenate([X, extra], 1).astype(np.float32), y


def origin_te(may):
    cat = pd.concat(may, ignore_index=True)
    return cat.groupby("Origin")["arr_delay"].mean().to_dict()


def fit_models(train_days, wx, te):
    Xtr, ytr = [], []
    for day in train_days:
        X, y = rich_xy(day, wx, te)
        Xtr.append(X)
        ytr.append(y)
    Xtr = np.vstack(Xtr)
    ytr = np.concatenate(ytr)
    gbr = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=0).fit(Xtr, ytr)
    hgb = HistGradientBoostingRegressor(max_depth=6, max_iter=200, learning_rate=0.08, random_state=0).fit(Xtr, ytr)
    return gbr, hgb


def pred_model(model, day, wx, te):
    X, y = rich_xy(day, wx, te)
    return np.clip(model.predict(X), -30, 180), y


def score_model(model, days, wx, te, table):
    acc = np.zeros(4)
    mae = 0.0
    n = 0
    for day in days:
        pred, y = pred_model(model, day, wx, te)
        acc += m.assign_with_delay(day, table, pred)[0]
        mae += np.abs(pred - y).sum()
        n += len(y)
    rec = ttl.pack(acc)
    rec["mae"] = float(mae / max(n, 1))
    return rec


def main():
    print("device", DEVICE, flush=True)
    df = g.read_months((5, 6, 7))
    arr = build_with_prev(df, "DFW", ex.TZ["DFW"])
    may, june, july = ttl.split_days(arr)
    days = [d["day"].iloc[0] for d in may + june + july]
    wx = ttl.hourly_weather("DFW", days)
    table = g.hour_rate(arr, set(arr.loc[arr["day"].str.startswith("2024-05"), "day"]))
    te = origin_te(may)
    print("fit may", len(may), "june", len(june), flush=True)
    gbr_may, hgb_may = fit_models(may, wx, te)
    gbr_mj, hgb_mj = fit_models(may + june, wx, te)
    hin = np.zeros(4)
    for day in july:
        hin += m.hindsight_assign(day, table)[0]
    hour = np.zeros(4)
    for day in july:
        sched, s1, s2, h1, h2, _, _ = m.day_arrays(day)
        hour += g.score_wait(g.mcf(sched, s1, s2, h1, h2, table), day)
    out = {
        "hour": ttl.pack(hour),
        "hindsight": ttl.pack(hin),
        "gbr_may": score_model(gbr_may, july, wx, te, table),
        "hgb_may": score_model(hgb_may, july, wx, te, table),
        "gbr_mayjune": score_model(gbr_mj, july, wx, te, table),
        "hgb_mayjune": score_model(hgb_mj, july, wx, te, table),
    }
    pto0 = 592614.0
    hin_c = out["hindsight"]["cascade"]
    for k, rec in out.items():
        rec["vs_old_pto"] = float((pto0 - rec["cascade"]) / pto0)
        rec["vs_hindsight"] = float((rec["cascade"] - hin_c) / hin_c) if hin_c else None
        rec["gap_captured"] = float((pto0 - rec["cascade"]) / (pto0 - hin_c)) if pto0 != hin_c else None
        print(k, rec["cascade"], "cap", rec.get("gap_captured"), "mae", rec.get("mae"), flush=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
