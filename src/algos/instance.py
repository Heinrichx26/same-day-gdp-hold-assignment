"""DFW May/June/July instance: rotations, late-aircraft, 15-min bins, METAR."""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_close_july as cj
import run_expand_full as ex
import run_trc_gate as g
import run_ttl_spo as ttl


def _days(arr, prefix):
    return [
        x.sort_values("bin")
        for _, x in arr.loc[arr["day"].str.startswith(prefix)].groupby("day")
        if len(x) >= 30
    ]


def load_dfw():
    return load_hub("DFW", "summer")


def load_hub(airport, season):
    tz = ex.TZ[airport]
    if season == "summer":
        df = g.read_months((5, 6, 7))
        arr = cj.build_with_prev(df, airport, tz)
        train, val, test = _days(arr, "2024-05"), _days(arr, "2024-06"), _days(arr, "2024-07")
        train_keys = set(arr.loc[arr["day"].str.startswith("2024-05"), "day"])
    elif season == "winter":
        df = g.read_months((1, 2))
        arr = cj.build_with_prev(df, airport, tz)
        jan = sorted(_days(arr, "2024-01"), key=lambda d: d["day"].iloc[0])
        feb = _days(arr, "2024-02")
        if len(jan) < 12:
            raise RuntimeError(f"{airport} winter jan days {len(jan)}")
        train, val, test = jan[:-7], jan[-7:], feb
        train_keys = {d["day"].iloc[0] for d in train}
    else:
        raise ValueError(season)
    days = [d["day"].iloc[0] for d in train + val + test]
    wx = ttl.hourly_weather(airport, days)
    te = cj.origin_te(train)
    table = g.hour_rate(arr, train_keys)
    print(airport, season, "train", len(train), "val", len(val), "test", len(test), flush=True)
    return train, val, test, wx, te, table


def day_bins(day, wx):
    key = day["day"].iloc[0]
    sched = day["bin"].to_numpy(int)
    delay = day["arr_delay"].to_numpy(float)
    hod = day["hod"].to_numpy(int)
    act = sched + np.round(delay / 15.0).astype(int)
    rows = []
    for b in range(int(sched.min()), int(max(sched.max(), act.max())) + 1):
        h = (b * 15) // 60
        if h < 6 or h > 22:
            continue
        dem = int((sched == b).sum())
        land = int((act == b).sum())
        i = int(np.clip(h - 6, 0, 16))
        vis, sk = 10.0, 8.0
        if key in wx:
            vis = float(wx[key][i, 0] * 10.0)
            sk = float(wx[key][i, 1] * 20.0)
        imc = float(wx[key][i, 5]) if key in wx else 0.0
        rows.append(
            dict(day=key, bin=b, bin15=b, hod=h, demand=dem, landed=land, vsby=vis, sknt=sk, skyl1=5000.0, arr=float(delay[sched == b].mean()) if dem else 0.0, imc=imc, low=float(vis <= 3), wx="IMC" if imc > 0.5 else "VMC")
        )
    return pd.DataFrame(rows).sort_values("bin")


def bin_panel(days, wx):
    return pd.concat([day_bins(d, wx) for d in days], ignore_index=True)


def arrays(day):
    dep = day["dep_m"].to_numpy(float) if "dep_m" in day.columns else np.zeros(len(day))
    arr_m = day["arr_m"].to_numpy(float) if "arr_m" in day.columns else dep + 90.0
    block = arr_m - dep
    block = np.where(block < 20.0, block + 1440.0, block)
    return dict(
        sched=day["bin"].to_numpy(int),
        hod=day["hod"].to_numpy(int),
        s1=np.nan_to_num(day["s1"].to_numpy(float), nan=180.0),
        s2=np.nan_to_num(day["s2"].to_numpy(float), nan=180.0),
        h1=day["has1"].to_numpy(bool),
        h2=day["has2"].to_numpy(bool),
        delay=day["arr_delay"].to_numpy(float),
        prev=day["prev_arr_delay"].to_numpy(float) if "prev_arr_delay" in day.columns else np.zeros(len(day)),
        dist=np.clip(block, 30.0, 600.0),
    )
