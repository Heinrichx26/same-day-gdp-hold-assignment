"""Full operators on an expanded airport set. No tree-count shortcut.

Train months are before the test month. The pinball level is chosen on a
validation slice that is not the test month. A liquid cell may then add a
correction of at most 1.5 landings, checkpointed on that same validation slice.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor

import run_connect_families as rcf
import run_liquid_decision as ld
import run_liquid_wx_queue as lq
from run_bakeoff_2025 import hhmm
from run_gdp_gate import wx_class

ROOT = Path(__file__).resolve().parents[1]
OUT = rcf.OUT / "expand_full.json"
COLS = ["FlightDate", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "ArrTime", "ArrDelay", "Cancelled", "Diverted"]
TZ = {
    "EWR": "America/New_York", "BOS": "America/New_York", "JFK": "America/New_York",
    "ATL": "America/New_York", "CLT": "America/New_York", "MIA": "America/New_York",
    "DTW": "America/Detroit", "ORD": "America/Chicago", "MSP": "America/Chicago",
    "DEN": "America/Denver", "PHX": "America/Phoenix", "IAH": "America/Chicago",
    "DFW": "America/Chicago", "SEA": "America/Los_Angeles", "SFO": "America/Los_Angeles",
    "LAX": "America/Los_Angeles", "LAS": "America/Los_Angeles",
}
TREES = 200
ALPHAS = (0.40, 0.50, 0.60, 0.70)


def metar_for(airport):
    folder = ROOT / "data" / "metar"
    hits = [p for p in folder.glob(f"{airport.lower()}_metar_*.csv") if p.stat().st_size > 5000]
    return sorted(hits)


def read_flights(dests):
    frames = []
    for month in (5, 6, 7, 8):
        zp = ROOT / "data" / "bts" / f"On_Time_2024_{month}.zip"
        if not zp.exists():
            print("missing", zp.name, flush=True)
            continue
        with zipfile.ZipFile(zp) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(inner) as f:
                df = pd.read_csv(f, usecols=COLS, dtype=str, low_memory=False)
        df = df.loc[df["Dest"].isin(dests)]
        frames.append(df)
        print("zip", month, len(df), flush=True)
    return pd.concat(frames, ignore_index=True)


def load_metar(paths):
    met = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    met["valid"] = pd.to_datetime(met["valid"], utc=True)
    met["vsby"] = pd.to_numeric(met["vsby"], errors="coerce")
    met["sknt"] = pd.to_numeric(met["sknt"], errors="coerce")
    if "skyl1" not in met.columns:
        met["skyl1"] = np.nan
    else:
        met["skyl1"] = pd.to_numeric(met["skyl1"], errors="coerce")
    met["wxcodes"] = met["wxcodes"].replace({"null": np.nan, "M": np.nan})
    met["wx"] = [wx_class(s, v, w) for s, v, w in zip(met["sknt"], met["vsby"], met["wxcodes"])]
    return met.drop_duplicates("valid").sort_values("valid")


def to_bins(raw, dest, tz, met):
    df = raw.loc[raw["Dest"] == dest].copy()
    for c in ["ArrDelay", "Cancelled", "Diverted"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[(df["Cancelled"].fillna(0) < 0.5) & (df["Diverted"].fillna(0) < 0.5)]
    mins = df["CRSArrTime"].map(hhmm)
    d0 = pd.to_datetime(df["FlightDate"])
    dmin = df["CRSDepTime"].map(hhmm)
    add = ((mins < dmin) & dmin.notna() & mins.notna()).astype(int)
    df["sched"] = (d0 + pd.to_timedelta(add, unit="D") + pd.to_timedelta(mins.fillna(0), unit="m")).dt.tz_localize(
        tz, nonexistent="shift_forward", ambiguous="NaT"
    )
    amins = df["ArrTime"].map(hhmm)
    df["actual"] = (d0 + pd.to_timedelta(add, unit="D") + pd.to_timedelta(amins.fillna(mins), unit="m")).dt.tz_localize(
        tz, nonexistent="shift_forward", ambiguous="NaT"
    )
    df = df.loc[df["sched"].notna()]
    df["hod"] = df["sched"].dt.hour
    df = df.loc[df["hod"].between(6, 22)]
    df["day"] = df["sched"].dt.strftime("%Y-%m-%d")
    df["bin15"] = df["sched"].dt.floor("15min")
    df["sched_utc"] = df["sched"].dt.tz_convert("UTC")
    df["arr"] = df["ArrDelay"].fillna(0.0)
    fl = pd.merge_asof(
        df.sort_values("sched_utc"),
        met.rename(columns={"valid": "sched_utc"})[["sched_utc", "vsby", "sknt", "skyl1", "wx"]],
        on="sched_utc", direction="backward", tolerance=pd.Timedelta("90min"),
    )
    for col, fill in (("vsby", 10.0), ("sknt", 8.0), ("skyl1", 5000.0)):
        fl[col] = fl[col].fillna(fill)
    fl["wx"] = fl["wx"].fillna("VMC")
    act = fl.loc[fl["actual"].notna()].groupby(fl.loc[fl["actual"].notna(), "actual"].dt.floor("15min")).size().rename("landed")
    bins = (
        fl.groupby(["day", "bin15", "hod"], sort=True)
        .agg(demand=("day", "size"), vsby=("vsby", "mean"), sknt=("sknt", "mean"), skyl1=("skyl1", "mean"), arr=("arr", "mean"), wx=("wx", "first"))
        .reset_index()
    )
    bins = bins.merge(act, left_on="bin15", right_index=True, how="left")
    bins["landed"] = bins["landed"].fillna(0.0)
    bins["imc"] = (bins["wx"] == "IMC").astype(float)
    bins["low"] = (bins["vsby"] <= 3.0).astype(float)
    return bins.sort_values(["day", "bin15"]).reset_index(drop=True)


class Corrector(lq.LiquidQueue):
    def __init__(self):
        super().__init__(xdim=7, hidden=32)
        self.read.weight.data.mul_(0.01)
        self.read.bias.data.zero_()

    def forward_day(self, x, dem, base):
        h = torch.zeros(self.hidden, device=x.device)
        q = x.new_zeros(())
        out = []
        for t in range(x.shape[0]):
            z = torch.cat([x[t], h], 0)
            cand = torch.tanh(self.ff(z))
            tau = torch.nn.functional.softplus(self.tau(z)) + 0.05
            decay = torch.exp(-1.0 / tau)
            h = decay * h + (1.0 - decay) * cand
            delta = 1.5 * torch.tanh(self.read(torch.cat([h, q.reshape(1)], 0))).squeeze()
            aar = (base[t] + delta).clamp(0.5, 25.0)
            avail = q + dem[t]
            q = avail - torch.minimum(avail, aar)
            out.append(aar)
        return torch.stack(out)


def liquid_correct(days, base_map, fit_days, val_days, test_days):
    def prep(group):
        ready = []
        for d in group:
            extra = np.column_stack([
                d["x"].detach().cpu().numpy()[:, :4],
                base_map[d["day"]] / 10.0,
            ]).astype(np.float32)
            # keep hour sin/cos from original x columns 5 and 6 if present
            x0 = d["x"].detach().cpu().numpy()
            hour = x0[:, 5:7] if x0.shape[1] >= 7 else np.zeros((len(x0), 2), np.float32)
            x = np.concatenate([extra, hour], 1).astype(np.float32)
            ready.append({
                "day": d["day"],
                "x": torch.tensor(x, device=lq.DEVICE),
                "dem": d["dem"],
                "land": d["land"],
                "base": torch.tensor(base_map[d["day"]], dtype=torch.float32, device=lq.DEVICE),
                "hod": d["hod"],
            })
        return ready

    fit, val, test = prep(fit_days), prep(val_days), prep(test_days)
    model = Corrector().to(lq.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    best_state, best_val = None, 1e9
    for epoch in range(1, 81):
        model.train()
        opt.zero_grad()
        parts, count = [], 0
        for d in fit:
            aar = model.forward_day(d["x"], d["dem"], d["base"])
            count += aar.numel()
            parts.append(ld.gdp_cost(d["dem"], aar, d["land"]) * aar.numel())
        loss = torch.stack(parts).sum() / count
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                rates = [model.forward_day(d["x"], d["dem"], d["base"]).cpu().numpy() for d in val]
            cost = ld.score_rates(val, rates)["cost"]
            print("  liquid", epoch, round(cost, 2), flush=True)
            if cost < best_val:
                best_val = cost
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(lq.DEVICE)
    model.eval()
    with torch.no_grad():
        rates = [model.forward_day(d["x"], d["dem"], d["base"]).cpu().numpy() for d in test]
    return ld.score_rates(test, rates), best_val


def one(bins):
    labeled = lq.days_from(bins)
    months = sorted({d["day"][:7] for d in labeled})
    print("  months", months, "days", len(labeled), flush=True)
    if "2024-08" in months and "2024-05" in months:
        fit = [d for d in labeled if d["day"] < "2024-07-01"]
        val = [d for d in labeled if d["day"].startswith("2024-07")]
        test = [d for d in labeled if d["day"].startswith("2024-08")]
        split = "train May-Jun, val July, test August"
    else:
        june = [d for d in labeled if d["day"].startswith("2024-06")]
        fit, val = june[:-6], june[-6:]
        test = [d for d in labeled if d["day"].startswith("2024-07")]
        split = "train June except last 6 days, val those 6, test July"
    if len(fit) < 20 or len(val) < 4 or len(test) < 10:
        return {"error": "short", "split": split, "n_fit": len(fit), "n_val": len(val), "n_test": len(test)}
    fit_ids = [d["day"] for d in fit]
    train = bins.loc[bins["day"].isin(fit_ids)]
    hod_mean = train.groupby("hod")["landed"].mean()
    feat = {}
    for day, g in bins.groupby("day", sort=True):
        g = rcf.add_lags(g.sort_values("bin15"))
        feat[str(day)] = (rcf.rib_X(g, hod_mean), g["arr"].to_numpy(float), g["landed"].to_numpy(float), g["demand"].to_numpy(float))
    Xfit = np.vstack([feat[d][0] for d in fit_ids])
    y_delay = np.concatenate([feat[d][1] for d in fit_ids])
    y_land = np.concatenate([feat[d][2] for d in fit_ids])
    print("  trees", TREES, "rows", len(y_land), flush=True)
    gbm_d = GradientBoostingRegressor(max_depth=3, n_estimators=TREES, random_state=0).fit(Xfit, y_delay)
    gbm_c = GradientBoostingRegressor(max_depth=3, n_estimators=TREES, random_state=1).fit(Xfit, y_land)
    hybrid = {}
    for day, (X, _, _, dem) in feat.items():
        delay = np.clip(gbm_d.predict(X), 0, 90)
        land = np.clip(gbm_c.predict(X), 0.5, 20)
        hybrid[day] = np.clip(0.5 * rcf.rate_aar(dem, delay, 1.0) + 0.5 * land, 0.5, 20)
    best = None
    for alpha in ALPHAS:
        model = GradientBoostingRegressor(loss="quantile", alpha=alpha, max_depth=3, n_estimators=TREES, random_state=2).fit(Xfit, y_land)
        pred = {day: np.clip(model.predict(feat[day][0]), 0.5, 20) for day in feat}
        cost = ld.score_rates(val, [pred[d["day"]] for d in val])["cost"]
        print("  pinball", alpha, round(cost, 2), flush=True)
        if best is None or cost < best[0]:
            best = (cost, alpha, pred)
    hour = {}
    means = train.groupby("hod")["landed"].mean()
    for d in test:
        hour[d["day"]] = np.array([float(means.get(int(h), 8.0)) for h in d["hod"]])
    h = ld.score_rates(test, [hour[d["day"]] for d in test])
    hy = ld.score_rates(test, [hybrid[d["day"]] for d in test])
    q = ld.score_rates(test, [best[2][d["day"]] for d in test])
    out = {
        "split": split,
        "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
        "pinball_alpha": best[1],
        "hour": h, "hybrid": hy, "pinball": q,
        "pinball_vs_hour": float((h["cost"] - q["cost"]) / h["cost"]),
        "pinball_vs_hybrid": float((hy["cost"] - q["cost"]) / hy["cost"]),
    }
    print("  liquid on pinball", flush=True)
    try:
        liq, val_liq = liquid_correct(labeled, best[2], fit, val, test)
        out["liquid_on_pinball"] = liq
        out["val_liquid_cost"] = val_liq
        out["liquid_vs_pinball"] = float((q["cost"] - liq["cost"]) / q["cost"])
        out["liquid_vs_hour"] = float((h["cost"] - liq["cost"]) / h["cost"])
    except Exception as e:
        out["liquid_error"] = str(e)
        print("  liquid error", e, flush=True)
    return out


def main():
    airports = [a for a in TZ if metar_for(a)]
    print("airports", airports, flush=True)
    raw = read_flights(airports)
    summary = {}
    for airport in airports:
        print("AIRPORT", airport, flush=True)
        try:
            bins = to_bins(raw, airport, TZ[airport], load_metar(metar_for(airport)))
            summary[airport] = one(bins)
        except Exception as e:
            summary[airport] = {"error": str(e)}
            print("  ERROR", e, flush=True)
        OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
