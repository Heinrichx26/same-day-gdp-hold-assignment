"""Bakeoff vs Ribeiro / Wang / Wu on BOS Jan+Feb, cap=landed.

Shift: train vis-ok days, test vis-bad days (Wu-style).
Score: 1*gnd + 2*air from fluid queue vs realized landings.

Families (2025):
  Ribeiro: hybrid queue feature + GBM delay, AAR from delay+landed mix (TR-C 171 104947)
  Wang: DTW k-medoids daily profiles + two-stage newsvendor SP (TR-C 171 105012)
  Wu: Wasserstein-1 shift of newsvendor quantile (dr-SAGHP / DR-MAGHP)

Ours (sweep; KEEP only if >=15% vs each of the three on cost):
  CQR-RH: quantile GBM on weather+queue+lags, conformal residual, receding
  DFL-q: pick quantile by train 1g+2a
  CondSP: Wang medoids reweighted by vis-profile DTW (weather-conditional SP)
  IsoVis: isotonic landed vs vis per hod, extrapolate
  OnlineEMA: receding AAR from lagged landed
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from run_gdp_gate import simulate, wx_class

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "winter"
TZ = "America/New_York"
COLS = [
    "FlightDate", "Dest", "CRSDepTime", "CRSArrTime", "ArrTime", "ArrDelay",
    "Cancelled", "Diverted",
]
PAIRS = [
    (ROOT / "data" / "bts" / "On_Time_2024_1.zip", ROOT / "data" / "metar" / "bos_metar_202401.csv"),
    (ROOT / "data" / "bts" / "On_Time_2024_2.zip", ROOT / "data" / "metar" / "bos_metar_202402.csv"),
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


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            D[i, j] = abs(ai - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def kmedoids_dtw(series: list[np.ndarray], k: int = 4, n_iter: int = 10):
    rng = np.random.default_rng(0)
    n = len(series)
    k = min(k, n)
    med = rng.choice(n, size=k, replace=False)
    assign = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist = np.zeros((n, k))
        for i in range(n):
            for j, m in enumerate(med):
                dist[i, j] = dtw(series[i], series[m])
        assign = dist.argmin(axis=1)
        new = []
        for j in range(k):
            idx = np.where(assign == j)[0]
            if len(idx) == 0:
                new.append(int(rng.integers(0, n)))
                continue
            best, best_d = int(idx[0]), np.inf
            for u in idx:
                s = 0.0
                for v in idx:
                    s += dtw(series[u], series[v])
                if s < best_d:
                    best, best_d = int(u), s
            new.append(best)
        if np.array_equal(new, med):
            break
        med = np.array(new)
    pi = np.array([(assign == j).mean() for j in range(k)])
    medoids = [series[int(m)] for m in med]
    return pi, medoids, assign


def load_month(zp: Path, metar: Path, dest: str = "BOS") -> pd.DataFrame:
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=COLS, dtype=str, low_memory=False)
    df = df.loc[df["Dest"] == dest].copy()
    for c in ["ArrDelay", "Cancelled", "Diverted"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[(df["Cancelled"].fillna(0) < 0.5) & (df["Diverted"].fillna(0) < 0.5)]
    mins = df["CRSArrTime"].map(hhmm)
    d0 = pd.to_datetime(df["FlightDate"])
    dmin = df["CRSDepTime"].map(hhmm)
    add = ((mins < dmin) & dmin.notna() & mins.notna()).astype(int)
    df["sched"] = (
        d0 + pd.to_timedelta(add, unit="D") + pd.to_timedelta(mins.fillna(0), unit="m")
    ).dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    amins = df["ArrTime"].map(hhmm)
    df["actual"] = (
        d0 + pd.to_timedelta(add, unit="D") + pd.to_timedelta(amins.fillna(mins), unit="m")
    ).dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    df = df.loc[df["sched"].notna()]
    df["hod"] = df["sched"].dt.hour
    df = df.loc[df["hod"].between(6, 22)]
    df["day"] = df["sched"].dt.strftime("%Y-%m-%d")
    df["bin15"] = df["sched"].dt.floor("15min")
    df["sched_utc"] = df["sched"].dt.tz_convert("UTC")
    met = pd.read_csv(metar)
    met["valid"] = pd.to_datetime(met["valid"], utc=True)
    met["vsby"] = pd.to_numeric(met["vsby"], errors="coerce")
    met["sknt"] = pd.to_numeric(met["sknt"], errors="coerce")
    met["skyl1"] = pd.to_numeric(met.get("skyl1"), errors="coerce")
    met["wxcodes"] = met["wxcodes"].replace({"null": np.nan})
    met["wx"] = [wx_class(s, v, w) for s, v, w in zip(met["sknt"], met["vsby"], met["wxcodes"])]
    fl = pd.merge_asof(
        df.sort_values("sched_utc"),
        met.sort_values("valid").rename(columns={"valid": "sched_utc"})[
            ["sched_utc", "vsby", "sknt", "skyl1", "wx"]
        ],
        on="sched_utc",
        direction="backward",
        tolerance=pd.Timedelta("90min"),
    )
    fl["vsby"] = fl["vsby"].fillna(10.0)
    fl["sknt"] = fl["sknt"].fillna(8.0)
    fl["skyl1"] = fl["skyl1"].fillna(5000.0)
    fl["wx"] = fl["wx"].fillna("VMC")
    fl["arr"] = pd.to_numeric(fl["ArrDelay"], errors="coerce").fillna(0.0)
    act = (
        fl.loc[fl["actual"].notna()]
        .groupby(fl.loc[fl["actual"].notna(), "actual"].dt.floor("15min"))
        .size()
        .rename("landed")
    )
    bins = (
        fl.groupby(["day", "bin15", "hod"], sort=True)
        .agg(
            demand=("day", "size"),
            vsby=("vsby", "mean"),
            sknt=("sknt", "mean"),
            skyl1=("skyl1", "mean"),
            arr=("arr", "mean"),
            wx=("wx", "first"),
        )
        .reset_index()
    )
    bins = bins.merge(act, left_on="bin15", right_index=True, how="left")
    bins["landed"] = bins["landed"].fillna(0.0)
    bins["imc"] = (bins["wx"] == "IMC").astype(float)
    bins["low"] = (bins["vsby"] <= 3.0).astype(float)
    return bins.sort_values(["day", "bin15"]).reset_index(drop=True)


def hod_series(g: pd.DataFrame) -> np.ndarray:
    s = g.groupby("hod")["landed"].mean()
    return np.array([float(s.get(h, 0.0)) for h in range(6, 23)], dtype=float)


def vis_series(g: pd.DataFrame) -> np.ndarray:
    s = g.groupby("hod")["vsby"].mean()
    return np.array([float(s.get(h, 10.0)) for h in range(6, 23)], dtype=float)


def add_lags(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("bin15").copy()
    g["lag1"] = g["landed"].shift(1)
    g["lag2"] = g["landed"].shift(2)
    g["lag4"] = g["landed"].shift(4)
    g["vlag"] = g["vsby"].shift(2)
    return g


def queue_feature(demand, nom):
    q = 0.0
    out = np.zeros(len(demand))
    for i, (d, n) in enumerate(zip(demand, nom)):
        q = max(0.0, q + float(d) - float(n))
        out[i] = q
    return out


def nv_quantile(samples: np.ndarray, pi: np.ndarray, p: float) -> float:
    order = np.argsort(samples)
    cdf = np.cumsum(pi[order])
    k = int(np.searchsorted(cdf, p, side="left"))
    k = min(max(k, 0), len(samples) - 1)
    return float(samples[order][k])


def eval_day(dem, aar, cap) -> dict:
    aar = np.clip(np.asarray(aar, float), 0.5, 30.0)
    return simulate(dem, aar, cap)


def wavg(df: pd.DataFrame, col: str, wcol="n") -> float:
    w = df[wcol].to_numpy(float)
    return float(np.average(df[col].to_numpy(float), weights=w)) if w.sum() else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("load", flush=True)
    parts = []
    for zp, met in PAIRS:
        if zp.exists() and met.exists():
            print(" ", zp.name, flush=True)
            parts.append(load_month(zp, met))
    bins = pd.concat(parts, ignore_index=True)
    bins = bins.loc[bins["day"].str.startswith("2024-0")].copy()

    day_low = bins.groupby("day")["low"].mean()
    vis_bad = set(day_low[day_low >= 0.08].index)
    days = sorted(bins["day"].unique())
    train_days = [d for d in days if d not in vis_bad]
    test_days = [d for d in days if d in vis_bad]
    print("train", len(train_days), "test", test_days, flush=True)

    train = bins.loc[bins["day"].isin(train_days)].copy()
    test = bins.loc[bins["day"].isin(test_days)].copy()

    # diagnostics
    diag = {
        "n_train_days": len(train_days),
        "n_test_days": len(test_days),
        "train_landed_mean": float(train["landed"].mean()),
        "test_landed_mean": float(test["landed"].mean()),
        "train_demand_mean": float(train["demand"].mean()),
        "test_demand_mean": float(test["demand"].mean()),
        "train_vis_mean": float(train["vsby"].mean()),
        "test_vis_mean": float(test["vsby"].mean()),
        "corr_landed_vis_train": float(np.corrcoef(train["landed"], train["vsby"])[0, 1]),
        "corr_landed_vis_test": float(np.corrcoef(test["landed"], test["vsby"])[0, 1]) if len(test) > 3 else 0.0,
        "corr_lag1_landed_train": float(
            np.corrcoef(train.groupby("day", group_keys=False).apply(add_lags, include_groups=True)["lag1"].fillna(0),
                        train["landed"])[0, 1]
        ) if False else None,
    }
    tr_lag = pd.concat([add_lags(g) for _, g in train.groupby("day")], ignore_index=True)
    m = tr_lag["lag1"].notna()
    diag["corr_lag1_landed_train"] = float(np.corrcoef(tr_lag.loc[m, "lag1"], tr_lag.loc[m, "landed"])[0, 1])
    te_lag = pd.concat([add_lags(g) for _, g in test.groupby("day")], ignore_index=True)
    m2 = te_lag["lag1"].notna()
    diag["corr_lag1_landed_test"] = float(np.corrcoef(te_lag.loc[m2, "lag1"], te_lag.loc[m2, "landed"])[0, 1])
    print("diag", json.dumps(diag, indent=2), flush=True)

    hod_mean = train.groupby("hod")["landed"].mean()
    hod_p80 = train.groupby("hod")["landed"].quantile(0.80)
    hod_p50 = train.groupby("hod")["landed"].quantile(0.50)
    hod_p33 = train.groupby("hod")["landed"].quantile(1.0 / 3.0)
    hod_p67 = train.groupby("hod")["landed"].quantile(2.0 / 3.0)
    hod_std = train.groupby("hod")["landed"].std().fillna(1.0)

    def lookup(s, hods, default=4.0):
        return np.array([max(float(s.get(int(h), default) or default), 0.5) for h in hods], float)

    # --- Wang DTW k-medoids + SP ---
    series = [hod_series(gb) for _, gb in train.groupby("day")]
    vis_tr = [vis_series(gb) for _, gb in train.groupby("day")]
    pi, medoids, assign = kmedoids_dtw(series, k=min(4, len(series)))
    print("wang pi", np.round(pi, 3), "assign", np.bincount(assign, minlength=len(pi)), flush=True)
    wang_q33, wang_q50, wang_q67 = {}, {}, {}
    for hi, hod in enumerate(range(6, 23)):
        samples = np.array([medoids[k][hi] for k in range(len(medoids))])
        wang_q33[hod] = max(nv_quantile(samples, pi, 1.0 / 3.0), 0.5)
        wang_q50[hod] = max(nv_quantile(samples, pi, 0.50), 0.5)
        wang_q67[hod] = max(nv_quantile(samples, pi, 2.0 / 3.0), 0.5)

    # --- Wu: W1 shift of newsvendor quantile (1/3) ---
    # 1D W1 ball: Q_p^worst ≈ Q_p - ε. ε in aircraft/15min.
    wu_eps = [0.0, 0.25, 0.5, 1.0, 1.5]

    # --- Ribeiro features ---
    def rib_X(g: pd.DataFrame, nom_hod) -> np.ndarray:
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

    train_f = pd.concat([add_lags(g) for _, g in train.groupby("day")], ignore_index=True)
    Xtr = rib_X(train_f, hod_mean)
    y_delay = train_f["arr"].to_numpy(float)
    y_land = train_f["landed"].to_numpy(float)
    gbm_d = GradientBoostingRegressor(max_depth=3, n_estimators=150, random_state=0)
    gbm_c = GradientBoostingRegressor(max_depth=3, n_estimators=150, random_state=1)
    gbm_d.fit(Xtr, y_delay)
    gbm_c.fit(Xtr, y_land)
    # quantile GBMs for CQR / DFL
    q_models = {}
    for q in (0.20, 0.33, 0.40, 0.50, 0.60, 0.67, 0.80):
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q, max_depth=3, n_estimators=150, random_state=2
        )
        m.fit(Xtr, y_land)
        q_models[q] = m
        print("fitted q", q, flush=True)

    pred_tr = gbm_c.predict(Xtr)
    resid = y_land - pred_tr
    conf_q = float(np.quantile(resid, 0.20))  # lower residual for conservative AAR

    # isotonic landed vs vis per hod (min 12 obs)
    iso = {}
    for hod, gh in train.groupby("hod"):
        if len(gh) < 12:
            continue
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ir.fit(gh["vsby"].to_numpy(float), gh["landed"].to_numpy(float))
        iso[int(hod)] = ir

    # DFL: pick q minimizing train cost (vis-ok, still has vis variation)
    dfl_scores = {}
    for q, m in q_models.items():
        costs = []
        ns = []
        for d, g in train.groupby("day"):
            g = add_lags(g)
            aar = np.clip(m.predict(rib_X(g, hod_mean)), 0.5, 30)
            r = eval_day(g["demand"].to_numpy(float), aar, g["landed"].to_numpy(float))
            costs.append(r["cost"])
            ns.append(float(g["demand"].sum()))
        dfl_scores[q] = float(np.average(costs, weights=ns))
    dfl_q = min(dfl_scores, key=dfl_scores.get)
    print("dfl_q", dfl_q, dfl_scores, flush=True)

    # also DFL mix: hour p80 vs cond
    def aar_ribeiro(g):
        X = rib_X(g, hod_mean)
        delay = np.clip(gbm_d.predict(X), 0, 90)
        land = np.clip(gbm_c.predict(X), 0.5, 20)
        dem = g["demand"].to_numpy(float)
        aar = np.clip(dem * 15.0 / (15.0 + delay), 0.5, None)
        return np.clip(0.5 * aar + 0.5 * land, 0.5, 20.0)

    def aar_ribeiro_nolig(g):
        # strategic Ribeiro: zero out lag landed (set to hod mean)
        g2 = g.copy()
        hm = lookup(hod_mean, g2["hod"])
        g2["lag1"] = hm
        g2["lag2"] = hm
        g2["lag4"] = hm
        return aar_ribeiro(g2)

    def aar_wang(g, table):
        return lookup(table, g["hod"])

    def aar_wu(g, eps):
        # Q_1/3(hod) - eps
        return np.clip(lookup(hod_p33, g["hod"]) - eps, 0.5, 20.0)

    def aar_wu_cvar20(g):
        p20 = train.groupby("hod")["landed"].quantile(0.20)
        return lookup(p20, g["hod"])

    def aar_condsp(g):
        # reweight Wang medoids by vis-profile DTW to today's vis (same-day METAR = perfect wx forecast)
        v = vis_series(g)
        w = np.array([np.exp(-dtw(v, vis_tr[i]) / 8.0) for i in range(len(vis_tr))])
        w = w / (w.sum() + 1e-9)
        # expected capacity by hod, then newsvendor 1/3 of weighted daily profiles
        # use weighted quantile across train days
        out = []
        for hi, hod in enumerate(range(6, 23)):
            samples = np.array([series[i][hi] for i in range(len(series))])
            out.append(max(nv_quantile(samples, w, 1.0 / 3.0), 0.5))
        table = {hod: out[i] for i, hod in enumerate(range(6, 23))}
        return lookup(table, g["hod"])

    def aar_iso(g):
        out = []
        for r in g.itertuples(index=False):
            ir = iso.get(int(r.hod))
            if ir is None:
                out.append(float(hod_p50.get(int(r.hod), 4.0)))
            else:
                out.append(float(ir.predict([float(r.vsby)])[0]))
        return np.clip(np.array(out, float), 0.5, 20.0)

    def aar_cqr(g, q, conf=0.0):
        X = rib_X(g, hod_mean)
        return np.clip(q_models[q].predict(X) + conf, 0.5, 20.0)

    def aar_online_ema(g, alpha=0.6):
        h80 = lookup(hod_p80, g["hod"])
        lag = g["lag1"].to_numpy(float)
        lag = np.where(np.isnan(lag), h80, lag)
        return np.clip(alpha * lag + (1.0 - alpha) * h80, 0.5, 20.0)

    def aar_hour(g, s):
        return lookup(s, g["hod"])

    def aar_oracle(g):
        return np.clip(g["landed"].to_numpy(float), 0.5, 30.0)

    def aar_nogdp(g):
        return np.clip(g["demand"].to_numpy(float), 0.5, 30.0)

    # vis-p80 with min count
    vis_p80 = train.copy()
    vis_p80["vbin"] = pd.cut(vis_p80["vsby"], [-0.1, 1, 3, 6, 9, 20], labels=list("abcde"))
    cnt = vis_p80.groupby(["hod", "vbin"], observed=False)["landed"].size()
    vp = vis_p80.groupby(["hod", "vbin"], observed=False)["landed"].quantile(0.80)

    def aar_visp80(g):
        vb = pd.cut(g["vsby"], [-0.1, 1, 3, 6, 9, 20], labels=list("abcde"))
        out = []
        for hod, v in zip(g["hod"], vb):
            nobs = int(cnt.get((hod, v), 0))
            x = vp.get((hod, v), np.nan) if nobs >= 8 else np.nan
            if x != x:
                x = hod_p80.get(hod, 4.0)
            out.append(max(float(x if x == x else 4.0), 0.5))
        return np.array(out, float)

    policies = {
        "NoGDP": aar_nogdp,
        "Oracle": aar_oracle,
        "HourP80": lambda g: aar_hour(g, hod_p80),
        "HourP50": lambda g: aar_hour(g, hod_p50),
        "HourP33": lambda g: aar_hour(g, hod_p33),
        "HourP67": lambda g: aar_hour(g, hod_p67),
        "WangQ33": lambda g: aar_wang(g, wang_q33),
        "WangQ50": lambda g: aar_wang(g, wang_q50),
        "WangQ67": lambda g: aar_wang(g, wang_q67),
        "WuCVaR20": aar_wu_cvar20,
        "WuW0": lambda g: aar_wu(g, 0.0),
        "WuW025": lambda g: aar_wu(g, 0.25),
        "WuW05": lambda g: aar_wu(g, 0.5),
        "WuW10": lambda g: aar_wu(g, 1.0),
        "Ribeiro": aar_ribeiro,
        "RibeiroNoLag": aar_ribeiro_nolig,
        "VisP80": aar_visp80,
        "CondSP": aar_condsp,
        "IsoVis": aar_iso,
        "CQR33": lambda g: aar_cqr(g, 0.33),
        "CQR50": lambda g: aar_cqr(g, 0.50),
        "CQR67": lambda g: aar_cqr(g, 0.67),
        "CQR80": lambda g: aar_cqr(g, 0.80),
        "CQR_DFL": lambda g: aar_cqr(g, dfl_q),
        "CQR_conf": lambda g: aar_cqr(g, 0.50, conf_q),
        "OnlineEMA": lambda g: aar_online_ema(g, 0.6),
        "OnlineEMA80": lambda g: aar_online_ema(g, 0.8),
    }

    rows = []
    for day, g0 in test.groupby("day"):
        g = add_lags(g0.sort_values("bin15"))
        if len(g) < 8:
            continue
        dem = g["demand"].to_numpy(float)
        cap = g["landed"].to_numpy(float)
        rec = {"day": day, "n": float(dem.sum()), "vis": float(g["vsby"].mean()), "low": float(g["low"].mean())}
        for name, fn in policies.items():
            r = eval_day(dem, fn(g), cap)
            rec[f"{name}_cost"] = r["cost"]
            rec[f"{name}_gnd"] = r["gnd"]
            rec[f"{name}_air"] = r["air"]
            rec[f"{name}_aar"] = float(np.mean(fn(g)))
        rows.append(rec)
        print(
            "day", day,
            "Wang33", round(rec["WangQ33_cost"], 2),
            "Rib", round(rec["Ribeiro_cost"], 2),
            "Wu05", round(rec["WuW05_cost"], 2),
            "CQR_DFL", round(rec["CQR_DFL_cost"], 2),
            "EMA", round(rec["OnlineEMA_cost"], 2),
            "Hour80", round(rec["HourP80_cost"], 2),
            "Oracle", round(rec["Oracle_cost"], 2),
            flush=True,
        )

    p = pd.DataFrame(rows)
    summary = {"diag": diag, "dfl_q": dfl_q, "dfl_scores": {str(k): v for k, v in dfl_scores.items()},
               "wang_pi": pi.tolist(), "n_days": int(len(p)), "n_flights": float(p["n"].sum()),
               "test_days": test_days, "train_days": train_days}
    costs = {}
    for name in policies:
        costs[name] = {
            "cost": wavg(p, f"{name}_cost"),
            "gnd": wavg(p, f"{name}_gnd"),
            "air": wavg(p, f"{name}_air"),
            "aar": wavg(p, f"{name}_aar"),
        }
    summary["costs"] = costs

    # named 2025 comparators (pre-registered, not swept after looking)
    # Ribeiro = hybrid queue+GBM with lags (paper M6/M7 near-real-time)
    # Wang = DTW k-medoids newsvendor 1/3 (air=2,gnd=1)
    # Wu = W1 ε=0.5 around Q_1/3  (mild DRO; CVaR20 is the old over-hold analogue)
    named = {"Ribeiro": "Ribeiro", "Wang": "WangQ33", "Wu": "WuW05"}
    ours_names = ["CQR_DFL", "CQR50", "CQR67", "CondSP", "IsoVis", "OnlineEMA", "OnlineEMA80", "CQR_conf", "CQR33"]
    vs = {}
    for on in ours_names:
        vs[on] = {}
        for lab, fam in named.items():
            c0, c1 = costs[fam]["cost"], costs[on]["cost"]
            vs[on][lab] = float((c0 - c1) / c0) if c0 else 0.0
        vs[on]["vs_HourP80"] = float((costs["HourP80"]["cost"] - costs[on]["cost"]) / costs["HourP80"]["cost"]) if costs["HourP80"]["cost"] else 0
        vs[on]["vs_NoGDP"] = float((costs["NoGDP"]["cost"] - costs[on]["cost"]) / costs["NoGDP"]["cost"]) if costs["NoGDP"]["cost"] else 0
        vs[on]["gate_2025"] = bool(vs[on]["Ribeiro"] >= 0.15 and vs[on]["Wang"] >= 0.15 and vs[on]["Wu"] >= 0.15)
    summary["vs"] = vs
    summary["named_costs"] = {k: costs[v] for k, v in named.items()}
    keepers = [k for k, v in vs.items() if v["gate_2025"]]
    summary["keepers"] = keepers
    (OUT / "bakeoff_2025.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: costs[k] for k in list(policies)}, indent=2))
    print("VS", json.dumps(vs, indent=2))
    print("KEEPERS", keepers, flush=True)
    if keepers:
        (OUT / "VS2025_KEEP.md").write_text(
            "# KEEP vs Ribeiro/Wang/Wu\n\n" + json.dumps(summary, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
