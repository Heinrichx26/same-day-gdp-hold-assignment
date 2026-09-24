"""July gate: liquid delay model plus min-cost release.

Landing rate = lower third of May hourly arrival counts, divided by four.
Turn = 45 minutes. Cost = delay pushed onto the next two same-tail departures.
May fits models. June picks the liquid checkpoint. July is scored once.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor

import run_expand_full as ex

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "winter" / "trc_gate_july.json"
TURN = 45
AIRPORTS = ("EWR", "DFW", "SFO")
COLS = [
    "FlightDate", "Tail_Number", "Origin", "Dest",
    "CRSDepTime", "CRSArrTime", "ArrTime", "ArrDelay", "Cancelled", "Diverted",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_months(months):
    frames = []
    for month in months:
        zp = ROOT / "data" / "bts" / f"On_Time_2024_{month}.zip"
        with zipfile.ZipFile(zp) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(inner) as f:
                df = pd.read_csv(f, usecols=COLS, dtype=str, low_memory=False)
        frames.append(df)
        print("zip", month, len(df), flush=True)
    df = pd.concat(frames, ignore_index=True)
    for c in ("Cancelled", "Diverted", "ArrDelay"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[(df["Cancelled"].fillna(0) < 0.5) & (df["Diverted"].fillna(0) < 0.5)].copy()
    df["tail"] = df["Tail_Number"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["FlightDate"])
    from run_bakeoff_2025 import hhmm
    df["dep_m"] = df["CRSDepTime"].map(hhmm)
    df["arr_m"] = df["CRSArrTime"].map(hhmm)
    return df


def build(df, dest, tz):
    from run_bakeoff_2025 import hhmm  # noqa: F401
    legs = df.sort_values(["date", "tail", "dep_m"]).copy()
    g = legs.groupby(["date", "tail"], sort=False)
    legs["n1_dep"] = g["dep_m"].shift(-1)
    legs["n1_arr"] = g["arr_m"].shift(-1)
    legs["n1_origin"] = g["Origin"].shift(-1)
    legs["n2_dep"] = g["dep_m"].shift(-2)
    arr = legs.loc[legs["Dest"] == dest].copy()
    add = ((arr["arr_m"] < arr["dep_m"]) & arr["dep_m"].notna() & arr["arr_m"].notna()).astype(int)
    sched = (
        arr["date"] + pd.to_timedelta(add, unit="D") + pd.to_timedelta(arr["arr_m"].fillna(0), unit="m")
    ).dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    arr = arr.loc[sched.notna()].copy()
    arr["sched"] = sched.loc[arr.index]
    arr = arr.loc[arr["sched"].dt.hour.between(6, 22)].copy()
    arr["day"] = arr["sched"].dt.strftime("%Y-%m-%d")
    arr["hod"] = arr["sched"].dt.hour.astype(int)
    arr["bin"] = ((arr["sched"].dt.hour * 60 + arr["sched"].dt.minute) // 15).astype(int)
    a = arr["arr_m"].to_numpy(float)
    d = arr["dep_m"].to_numpy(float)
    a = np.where(a < d - 30, a + 1440, a)
    n1 = arr["n1_dep"].to_numpy(float)
    n1a = arr["n1_arr"].to_numpy(float)
    n2 = arr["n2_dep"].to_numpy(float)
    n1 = np.where(np.isfinite(n1) & (n1 < a - 30), n1 + 1440, n1)
    n1a = np.where(np.isfinite(n1a) & (n1a < n1 - 30), n1a + 1440, n1a)
    n2 = np.where(np.isfinite(n2) & (n2 < n1a - 30), n2 + 1440, n2)
    has1 = arr["n1_origin"].notna().to_numpy() & np.isfinite(n1)
    has2 = has1 & np.isfinite(n2)
    s1 = np.where(has1, n1 - a - TURN, np.nan)
    s2 = np.where(has2, n2 - n1a - TURN, np.nan)
    arr["s1"] = np.where(has1 & (s1 > -30) & (s1 < 600), s1, np.nan)
    arr["s2"] = np.where(has2 & (s2 > -30) & (s2 < 600) & np.isfinite(arr["s1"]), s2, np.nan)
    arr["has1"] = np.isfinite(arr["s1"])
    arr["has2"] = np.isfinite(arr["s2"])
    arr["arr_delay"] = arr["ArrDelay"].fillna(0.0).to_numpy(float)
    return arr.reset_index(drop=True)


def hour_rate(frame, train_days):
    sub = frame.loc[frame["day"].isin(train_days)]
    counts = sub.groupby(["day", "hod"]).size()
    levels = counts.index.get_level_values("hod")
    table = {}
    for hod in range(6, 23):
        vals = counts.xs(hod, level="hod").to_numpy(float) if hod in levels else np.array([4.0])
        table[hod] = max(float(np.quantile(vals, 1.0 / 3.0)) / 4.0, 1.0)
    return table


def cascade(wait, s1, s2, has1, has2):
    leg2 = np.zeros(len(wait))
    leg3 = np.zeros(len(wait))
    if has1.any():
        leg2[has1] = np.maximum(0.0, wait[has1] - s1[has1])
    if has2.any():
        leg3[has2] = np.maximum(0.0, leg2[has2] - s2[has2])
    return leg2 + leg3, leg2, leg3


def realized_slack(g):
    s1 = g["s1"].to_numpy(float).copy()
    s2 = g["s2"].to_numpy(float).copy()
    has1 = g["has1"].to_numpy(bool)
    has2 = g["has2"].to_numpy(bool)
    delay = g["arr_delay"].to_numpy(float)
    s1[has1] = s1[has1] - delay[has1]
    return s1, s2, has1, has2


def mcf(sched, s1, s2, has1, has2, table):
    n = len(sched)
    first = int(min(sched.min(), 6 * 4))
    last = int(max(sched.max(), 23 * 4)) + 1
    bins = list(range(first, last + 1))
    cap = {}
    for b in bins:
        hod = (b * 15) // 60
        cap[b] = max(int(np.floor(table.get(hod, 4.0))), 0) if hod <= 22 else n
    cap[last] = n
    graph = nx.DiGraph()
    graph.add_node("src", demand=-n)
    graph.add_node("sink", demand=n)
    for b, c in cap.items():
        if c > 0:
            graph.add_edge(f"b{b}", "sink", capacity=int(c), weight=0)
    for i in range(n):
        graph.add_edge("src", f"f{i}", capacity=1, weight=0)
        connected = False
        for b in bins:
            if b < int(sched[i]) or cap.get(b, 0) <= 0:
                continue
            wait = 15.0 * (b - int(sched[i]))
            cost = int(
                cascade(
                    np.array([wait]),
                    np.array([s1[i] if has1[i] else 0.0]),
                    np.array([s2[i] if has2[i] else 0.0]),
                    np.array([has1[i]]),
                    np.array([has2[i]]),
                )[0][0]
            )
            graph.add_edge(f"f{i}", f"b{b}", capacity=1, weight=cost)
            connected = True
        if not connected:
            graph.add_edge(f"f{i}", f"b{last}", capacity=1, weight=10_000)
    flow = nx.min_cost_flow(graph)
    wait = np.zeros(n)
    for i in range(n):
        for dst, qty in flow[f"f{i}"].items():
            if qty and dst.startswith("b"):
                b = int(dst[1:])
                wait[i] = 15.0 * max(0, b - int(sched[i]))
    return wait


def feat_rows(g):
    s1 = np.nan_to_num(g["s1"].to_numpy(float), nan=180.0)
    s2 = np.nan_to_num(g["s2"].to_numpy(float), nan=180.0)
    hod = g["hod"].to_numpy(float)
    ang = 2 * np.pi * (hod - 6) / 16.0
    X = np.column_stack(
        [
            s1 / 60.0,
            s2 / 60.0,
            g["has1"].to_numpy(float),
            g["has2"].to_numpy(float),
            np.sin(ang),
            np.cos(ang),
            hod / 24.0,
        ]
    )
    y = g["arr_delay"].to_numpy(float)
    return X.astype(np.float32), y.astype(np.float32)


class LiquidDelay(torch.nn.Module):
    def __init__(self, xdim=7, hidden=24):
        super().__init__()
        self.hidden = hidden
        self.ff = torch.nn.Linear(xdim + hidden, hidden)
        self.tau = torch.nn.Linear(xdim + hidden, hidden)
        self.read = torch.nn.Linear(hidden, 1)

    def forward_day(self, x):
        h = torch.zeros(self.hidden, device=x.device)
        outs = []
        for t in range(x.shape[0]):
            z = torch.cat([x[t], h], 0)
            cand = torch.tanh(self.ff(z))
            tau = torch.nn.functional.softplus(self.tau(z)) + 0.05
            decay = torch.exp(-1.0 / tau)
            h = decay * h + (1.0 - decay) * cand
            outs.append(self.read(h).squeeze())
        return torch.stack(outs)


class FeedDelay(torch.nn.Module):
    def __init__(self, xdim=7, hidden=24):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(xdim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )

    def forward_day(self, x):
        return self.net(x).squeeze(-1)


def train_torch(model, days_xy, val_xy, epochs=40):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    best, best_v = None, 1e9
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = 0.0
        n = 0
        for X, y in days_xy:
            pred = model.forward_day(X)
            loss = loss + torch.nn.functional.smooth_l1_loss(pred, y, reduction="sum")
            n += y.numel()
        loss = loss / max(n, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                err = n = 0.0
                for X, y in val_xy:
                    pred = model.forward_day(X)
                    err += torch.abs(pred - y).sum().item()
                    n += y.numel()
            v = err / max(n, 1)
            if v < best_v:
                best_v = v
                best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best)
    model.to(DEVICE)
    return best_v


def predict_torch(model, days_g):
    model.eval()
    out = []
    with torch.no_grad():
        for g in days_g:
            X, _ = feat_rows(g)
            xt = torch.tensor(X, device=DEVICE)
            pred = model.forward_day(xt).cpu().numpy()
            out.append(np.clip(pred, -30, 180))
    return out


def score_wait(wait, g):
    s1, s2, has1, has2 = realized_slack(g)
    tot, leg2, leg3 = cascade(wait, s1, s2, has1, has2)
    return float(tot.sum()), float(leg2.sum()), float(leg3.sum()), float(wait.sum())


def adjust_slack(g, pred_delay):
    s1 = g["s1"].to_numpy(float).copy()
    s2 = g["s2"].to_numpy(float).copy()
    has1 = g["has1"].to_numpy(bool)
    has2 = g["has2"].to_numpy(bool)
    s1[has1] = s1[has1] - pred_delay[has1]
    return s1, s2, has1, has2


def run_airport(df, dest):
    print(dest, flush=True)
    arr = build(df, dest, ex.TZ[dest])
    may = arr.loc[arr["day"].str.startswith("2024-05")]
    june = arr.loc[arr["day"].str.startswith("2024-06")]
    july = arr.loc[arr["day"].str.startswith("2024-07")]
    table = hour_rate(arr, set(may["day"]))
    may_days = [g.sort_values("bin") for _, g in may.groupby("day") if len(g) >= 30]
    june_days = [g.sort_values("bin") for _, g in june.groupby("day") if len(g) >= 30]
    july_days = [g.sort_values("bin") for _, g in july.groupby("day") if len(g) >= 30]
    Xtr, ytr = zip(*[feat_rows(g) for g in may_days])
    Xva, yva = zip(*[feat_rows(g) for g in june_days])
    gbr = GradientBoostingRegressor(max_depth=3, n_estimators=80, random_state=0)
    gbr.fit(np.vstack(Xtr), np.concatenate(ytr))
    may_t = [(torch.tensor(X, device=DEVICE), torch.tensor(y, device=DEVICE)) for X, y in zip(Xtr, ytr)]
    june_t = [(torch.tensor(X, device=DEVICE), torch.tensor(y, device=DEVICE)) for X, y in zip(Xva, yva)]
    liquid = LiquidDelay().to(DEVICE)
    feed = FeedDelay().to(DEVICE)
    print("  train liquid", flush=True)
    lv = train_torch(liquid, may_t, june_t, 30)
    print("  train feed", lv, flush=True)
    fv = train_torch(feed, may_t, june_t, 30)
    print("  val mae liquid", lv, "feed", fv, flush=True)

    acc = {k: np.zeros(4) for k in ("mcf", "pto", "liquid", "feed")}
    n_flights = 0
    for g in july_days:
        n_flights += len(g)
        sched = g["bin"].to_numpy(int)
        s1s, s2s, h1, h2 = g["s1"].to_numpy(float), g["s2"].to_numpy(float), g["has1"].to_numpy(bool), g["has2"].to_numpy(bool)
        s1s = np.nan_to_num(s1s, nan=180.0)
        s2s = np.nan_to_num(s2s, nan=180.0)
        w_mcf = mcf(sched, s1s, s2s, h1, h2, table)
        X, _ = feat_rows(g)
        pred_gb = np.clip(gbr.predict(X), -30, 180)
        s1p, s2p, h1p, h2p = adjust_slack(g, pred_gb)
        s1p = np.nan_to_num(s1p, nan=180.0)
        s2p = np.nan_to_num(s2p, nan=180.0)
        w_pto = mcf(sched, s1p, s2p, h1p, h2p, table)
        pred_l = predict_torch(liquid, [g])[0]
        s1l, s2l, h1l, h2l = adjust_slack(g, pred_l)
        s1l = np.nan_to_num(s1l, nan=180.0)
        s2l = np.nan_to_num(s2l, nan=180.0)
        w_liq = mcf(sched, s1l, s2l, h1l, h2l, table)
        pred_f = predict_torch(feed, [g])[0]
        s1f, s2f, h1f, h2f = adjust_slack(g, pred_f)
        s1f = np.nan_to_num(s1f, nan=180.0)
        s2f = np.nan_to_num(s2f, nan=180.0)
        w_ff = mcf(sched, s1f, s2f, h1f, h2f, table)
        for name, w in (("mcf", w_mcf), ("pto", w_pto), ("liquid", w_liq), ("feed", w_ff)):
            tot, leg2, leg3, wait = score_wait(w, g)
            acc[name] += (tot, leg2, leg3, wait)
        print(" ", g["day"].iloc[0], "mcf", int(acc["mcf"][0]), "pto", int(acc["pto"][0]), "liq", int(acc["liquid"][0]), flush=True)
    out = {"n_july_days": len(july_days), "flights": n_flights}
    for name in acc:
        out[name] = {
            "cascade": float(acc[name][0]),
            "leg2": float(acc[name][1]),
            "leg3": float(acc[name][2]),
            "wait": float(acc[name][3]),
        }
    mcf_c, pto_c, liq_c, ff_c = (out[k]["cascade"] for k in ("mcf", "pto", "liquid", "feed"))
    out["liquid_vs_mcf"] = float((mcf_c - liq_c) / mcf_c) if mcf_c else None
    out["liquid_vs_pto"] = float((pto_c - liq_c) / pto_c) if pto_c else None
    out["liquid_vs_feed"] = float((ff_c - liq_c) / ff_c) if ff_c else None
    waits = [out[k]["wait"] for k in acc]
    out["wait_spread"] = float((max(waits) - min(waits)) / max(max(waits), 1.0))
    return out


def main():
    print("device", DEVICE, flush=True)
    df = read_months((5, 6, 7))
    summary = {}
    for dest in AIRPORTS:
        try:
            summary[dest] = run_airport(df, dest)
        except Exception as e:
            summary[dest] = {"error": str(e)}
            print("ERR", dest, e, flush=True)
        OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(dest, json.dumps(summary[dest]), flush=True)
    print("done")


if __name__ == "__main__":
    main()
