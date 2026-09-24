"""Liquid assignment: cell corrects inbound delay, then fills unused buffer.

Remaining unused turn buffer is computed from corrected inbound delay.
Training minimises cascade under recorded inbound delay and NAS-substituted
delay, with a squared error on the corrected inbound delay.
"""
from __future__ import annotations

import numpy as np
import torch

import run_trc_gate as g
from algos.gdp_assign import caps, closed_cascade, group_fill, interval_keys, match_sum, slack0
from algos.instance import arrays
from algos.method_pils import leftover_np, leftover_soft, SitCapacityLiquid

FAST_DIM = 12
TEMP = 6.0


def _wx_slow(wx, key, device):
    arr = wx.get(key)
    if arr is None:
        arr = np.zeros((17, 6), dtype=np.float32)
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = np.repeat(x.reshape(1, -1), 17, axis=0)
    if x.shape[0] < 17:
        pad = np.repeat(x[-1:], 17 - x.shape[0], axis=0)
        x = np.concatenate([x, pad], 0)
    if x.shape[1] < 6:
        x = np.pad(x, ((0, 0), (0, 6 - x.shape[1])))
    return torch.tensor(x[:17, :6], device=device)


def fast_features(pred, a, budget):
    cap = caps(a["h1"])
    sl = slack0(pred, a["s1"], a["h1"], cap)
    conn = np.asarray(a["h1"], bool)
    room1 = np.zeros(len(sl))
    room1[conn] = np.minimum(np.clip(a["s2"][conn], 0.0, None), np.maximum(cap[conn] - sl[conn], 0.0))
    keys = interval_keys(a["sched"])
    bud = np.asarray(budget, float)
    b_int = np.zeros(len(sl))
    n_int = np.ones(len(sl))
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        b_int[idx] = float(bud[idx].sum())
        n_int[idx] = max(len(idx), 1)
    term = 1.0 - conn.astype(np.float64)
    hod = np.clip(np.asarray(a["hod"], float), 6.0, 22.0)
    prev = np.asarray(a.get("prev", np.zeros(len(sl))), float)
    return np.column_stack(
        [
            sl / 60.0,
            room1 / 60.0,
            cap / 240.0,
            np.clip(pred, -30.0, 180.0) / 60.0,
            np.asarray(a["s1"], float) / 60.0,
            np.clip(a["s2"], 0.0, None) / 60.0,
            term,
            conn.astype(np.float64),
            np.asarray(a["h2"], float),
            b_int / np.maximum(n_int, 1.0) / 60.0,
            (hod - 6.0) / 16.0,
            np.clip(prev, -30.0, 180.0) / 60.0,
        ]
    ).astype(np.float32)


def _rooms(pred, a):
    cap = caps(a["h1"])
    sl = slack0(pred, a["s1"], a["h1"], cap)
    conn = np.asarray(a["h1"], bool)
    room1 = np.zeros(len(sl))
    room1[conn] = np.minimum(
        np.clip(a["s2"][conn], 0.0, None), np.maximum(cap[conn] - sl[conn], 0.0)
    )
    return cap, sl, room1


def apply_room_residual(sl, room1, cap, d_sl, d_r1):
    sl2 = np.clip(np.asarray(sl, float) + np.asarray(d_sl, float), 0.0, cap)
    room2 = np.clip(np.asarray(room1, float) + np.asarray(d_r1, float), 0.0, np.maximum(cap - sl2, 0.0))
    return sl2, room2


def structured_spend(score, sl, room1, cap, sched, budget):
    h = np.zeros(len(score), dtype=np.float64)
    bud = np.asarray(budget, float)
    sc = np.asarray(score, float)
    keys = interval_keys(sched)
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        B = float(bud[idx].sum())
        h0 = leftover_np(sc[idx], sl[idx], B)
        rem = max(0.0, B - float(h0.sum()))
        h1 = leftover_np(sc[idx], np.minimum(room1[idx], cap[idx] - h0), rem)
        rem = max(0.0, rem - float(h1.sum()))
        h2 = leftover_np(sc[idx], np.maximum(cap[idx] - h0 - h1, 0.0), rem)
        h[idx] = match_sum(h0 + h1 + h2, B, cap[idx])
    return h


def structured_spend_soft(score, sl, room1, cap, sched, budget, temp=TEMP):
    h = torch.zeros_like(score)
    keys = sched.detach().cpu().numpy() if torch.is_tensor(sched) else np.asarray(sched)
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        ii = torch.as_tensor(idx, device=score.device, dtype=torch.long)
        B = budget[ii].sum()
        h0 = leftover_soft(score[ii], sl[ii], B, temp=temp)
        rem = (B - h0.sum()).clamp(min=0.0)
        room_1 = torch.minimum(room1[ii], (cap[ii] - h0).clamp(min=0.0))
        h1 = leftover_soft(score[ii], room_1, rem, temp=temp)
        rem = (rem - h1.sum()).clamp(min=0.0)
        room_2 = (cap[ii] - h0 - h1).clamp(min=0.0)
        h2 = leftover_soft(score[ii], room_2, rem, temp=temp)
        h[ii] = h0 + h1 + h2
    return h


def rooms_from_delay(delay, a):
    cap = caps(a["h1"])
    sl = slack0(delay, a["s1"], a["h1"], cap)
    conn = np.asarray(a["h1"], bool)
    room1 = np.zeros(len(sl))
    room1[conn] = np.minimum(
        np.clip(a["s2"][conn], 0.0, None), np.maximum(cap[conn] - sl[conn], 0.0)
    )
    return cap, sl, room1


class LiquidEDCT:
    """Two-timescale liquid policy. Inbound-delay residual sets unused buffer."""

    def __init__(self, hidden=32):
        self.net = None
        self.wx = None
        self.device = g.DEVICE
        self.hidden = hidden

    def _heads_t(self, day, pred, a, budget):
        X = fast_features(pred, a, budget)
        hod_idx = np.clip(np.asarray(a["hod"], int) - 6, 0, 16)
        key = day["day"].iloc[0]
        slow = _wx_slow(self.wx, key, self.device)
        fast = torch.tensor(X, device=self.device)
        score_delta, delay_res, d_r1 = self.net.heads(slow, fast, hod_idx)
        sl = slack0(pred, a["s1"], a["h1"], caps(a["h1"]))
        room1 = np.where(a["h1"], np.clip(a["s2"], 0.0, None), 0.0)
        prior = torch.tensor(
            4.0 * (1.0 - a["h1"].astype(float)) + sl / 240.0 + room1 / 180.0,
            device=self.device,
            dtype=score_delta.dtype,
        )
        return prior + score_delta, delay_res, d_r1

    def assign(self, day, pred, budget):
        a = arrays(day)
        self.net.eval()
        with torch.no_grad():
            score, delay_res, d_r1 = self._heads_t(day, pred, a, budget)
            d_iss = np.clip(np.asarray(pred, float) + delay_res.detach().cpu().numpy(), -30.0, 180.0)
            cap, sl, room1 = rooms_from_delay(d_iss, a)
            room1 = np.clip(room1 + d_r1.detach().cpu().numpy(), 0.0, np.maximum(cap - sl, 0.0))
            sc = score.detach().cpu().numpy()
        return structured_spend(sc, sl, room1, cap, a["sched"], budget)

    def fit(self, train, val, wx, pred_fn, budget_fn, epochs=8):
        from algos.gdp_assign import pack
        from algos.method_pils import torch_cascade
        import pandas as pd

        self.wx = wx
        self.device = g.DEVICE
        self.net = SitCapacityLiquid(slow_dim=6, fast_dim=FAST_DIM, hidden=self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)

        def pred_fill(day, pred, bud):
            a = arrays(day)
            return group_fill(pred, a["s1"], a["s2"], a["h1"], a["h2"], bud, interval_keys(a["sched"]))

        def truth_delays(day, a):
            real = np.clip(a["delay"], -30.0, 180.0)
            if "NASDelay" in day.columns:
                nas = pd.to_numeric(day["NASDelay"], errors="coerce").fillna(0.0).to_numpy(float)
                nas_d = np.clip(a["delay"] - nas, -30.0, 180.0)
            else:
                nas_d = real
            return real, nas_d

        def cas_np(h, delay, a):
            return closed_cascade(h, delay, a["s1"], a["s2"], a["h1"], a["h2"])[0]

        def val_real():
            self.net.eval()
            real_l = real_p = pred_l = pred_p = 0.0
            with torch.no_grad():
                for day in val:
                    pred = pred_fn(day)
                    bud = budget_fn(day)
                    a = pack(day)
                    h_p = pred_fill(day, pred, bud)
                    h = self.assign(day, pred, bud)
                    real, nas_d = truth_delays(day, a)
                    real_l += 0.5 * cas_np(h, real, a) + 0.5 * cas_np(h, nas_d, a)
                    real_p += 0.5 * cas_np(h_p, real, a) + 0.5 * cas_np(h_p, nas_d, a)
                    pred_l += cas_np(h, pred, a)
                    pred_p += cas_np(h_p, pred, a)
            return real_l, real_p, pred_l, pred_p

        best = 1e18
        best_state = None
        for epoch in range(1, epochs + 1):
            self.net.train()
            ep = 0.0
            order = np.random.default_rng(epoch).permutation(len(train))
            for di in order:
                day = train[int(di)]
                pred = pred_fn(day)
                bud = budget_fn(day)
                a = pack(day)
                score, delay_res, d_r1 = self._heads_t(day, pred, a, bud)
                pred_t = torch.tensor(np.clip(pred, -30.0, 180.0), device=self.device, dtype=score.dtype)
                d_iss = (pred_t + delay_res).clamp(-30.0, 180.0)
                cap_np = caps(a["h1"])
                cap_t = torch.tensor(cap_np, device=self.device, dtype=score.dtype)
                s1_t = torch.tensor(a["s1"], device=self.device, dtype=score.dtype)
                h1_b = torch.tensor(a["h1"], device=self.device)
                sl_t = torch.where(h1_b, (s1_t - d_iss).clamp(min=0.0), cap_t)
                sl_t = torch.minimum(sl_t, cap_t)
                s2_t = torch.tensor(np.clip(a["s2"], 0.0, None), device=self.device, dtype=score.dtype)
                room1_t = torch.where(h1_b, torch.minimum(s2_t, (cap_t - sl_t).clamp(min=0.0)), torch.zeros_like(sl_t))
                room1_t = (room1_t + d_r1).clamp(min=0.0)
                room1_t = torch.minimum(room1_t, (cap_t - sl_t).clamp(min=0.0))
                bud_t = torch.tensor(np.asarray(bud, float), device=self.device, dtype=score.dtype)
                h = structured_spend_soft(score, sl_t, room1_t, cap_t, a["sched"], bud_t)
                h2_t = torch.tensor(a["h2"], device=self.device)
                real, nas_d = truth_delays(day, a)
                real_t = torch.tensor(real, device=self.device, dtype=score.dtype)
                nas_t = torch.tensor(nas_d, device=self.device, dtype=score.dtype)
                true_sl = torch.tensor(
                    slack0(real, a["s1"], a["h1"], cap_np), device=self.device, dtype=score.dtype
                )
                conn = h1_b.float()
                slack_loss = ((sl_t - true_sl) ** 2 * conn).sum() / (conn.sum().clamp(min=1.0) * 3600.0)
                h_or = pred_fill(day, real, bud)
                h_or_t = torch.tensor(h_or, device=self.device, dtype=score.dtype)
                distill = ((h - h_or_t) ** 2).mean() / 3600.0
                cas = (
                    0.30 * torch_cascade(h, real_t, s1_t, s2_t, h1_b, h2_t)
                    + 0.30 * torch_cascade(h, nas_t, s1_t, s2_t, h1_b, h2_t)
                    + 0.05 * torch_cascade(h, pred_t, s1_t, s2_t, h1_b, h2_t)
                )
                loss = cas / max(len(a["sched"]), 1) + 0.25 * slack_loss + 0.15 * distill
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep += float(loss.detach().cpu())
            real_l, real_p, pred_l, pred_p = val_real()
            print(
                f"liquid-delay epoch {epoch} loss {ep:.3f} "
                f"val_real liquid {real_l:.0f} fill {real_p:.0f} "
                f"val_pred liquid {pred_l:.0f} fill {pred_p:.0f}",
                flush=True,
            )
            if real_l < best:
                best = real_l
                best_state = {k: t.detach().cpu().clone() for k, t in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.to(self.device)
        self.net.eval()
        self.val_real = best
        print("liquid-delay deploy val_real", round(best, 1), flush=True)
        return self
