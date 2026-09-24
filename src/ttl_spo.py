"""Test-time liquid predict-then-optimize (TTL-SPO).

Min-cost flow stays the assignment operator. A frozen delay map (late-
aircraft plus histogram gradient boosting) supplies the base cost. A
two-timescale closed-form liquid cell outputs a bounded residual. SPO+
trains the residual; a scalar mix selected on June can drop the residual
and return the base map. Fast features include previous-leg delay so the
late-aircraft signal sits inside the cell, not only in the booster.

Paper name: TTL-SPO.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

import run_trc_gate as g

NOON = 12 * 4
BOUND = 45.0
COST_SCALE = 100.0
SPO_LAMBDA = 15.0


def cfc_step(cell, x, h):
    z = torch.cat([x, h], 0)
    cand = torch.tanh(cell.ff(z))
    tau = F.softplus(cell.tau(z)) + 0.05
    decay = torch.exp(-1.0 / tau)
    return decay * h + (1.0 - decay) * cand


class CfcBlock(torch.nn.Module):
    def __init__(self, xdim, hidden):
        super().__init__()
        self.hidden = hidden
        self.ff = torch.nn.Linear(xdim + hidden, hidden)
        self.tau = torch.nn.Linear(xdim + hidden, hidden)


class TwoTimescaleLiquid(torch.nn.Module):
    """Slow cell over the day's weather hours; fast cell over flights."""

    def __init__(self, slow_dim=6, fast_dim=7, hidden=24):
        super().__init__()
        self.hidden = hidden
        self.slow = CfcBlock(slow_dim, hidden)
        self.fast = CfcBlock(fast_dim + hidden, hidden)
        self.read = torch.nn.Linear(hidden, 1)
        self.read.weight.data.mul_(0.01)
        self.read.bias.data.zero_()

    def slow_states(self, slow_x):
        h = torch.zeros(self.hidden, device=slow_x.device)
        out = []
        for t in range(slow_x.shape[0]):
            h = cfc_step(self.slow, slow_x[t], h)
            out.append(h)
        return torch.stack(out)

    def hidden_day(self, slow_x, fast_x, hod_idx):
        slow_h = self.slow_states(slow_x)
        h = torch.zeros(self.hidden, device=fast_x.device)
        hid = []
        for i in range(fast_x.shape[0]):
            hs = slow_h[int(hod_idx[i])]
            z = torch.cat([fast_x[i], hs], 0)
            h = cfc_step(self.fast, z, h)
            hid.append(h)
        return torch.stack(hid)

    def forward_day(self, slow_x, fast_x, hod_idx):
        hid = self.hidden_day(slow_x, fast_x, hod_idx)
        return BOUND * torch.tanh(self.read(hid).squeeze(-1))


def flight_cost_matrix(sched, s1, s2, has1, has2, bins):
    b = np.asarray(bins, dtype=float)
    wait = np.maximum(0.0, 15.0 * (b[None, :] - sched[:, None]))
    leg2 = np.where(has1[:, None], np.maximum(0.0, wait - s1[:, None]), 0.0)
    leg3 = np.where(has2[:, None], np.maximum(0.0, leg2 - s2[:, None]), 0.0)
    cost = leg2 + leg3
    cost = np.where(b[None, :] < sched[:, None], 10_000.0, cost)
    return cost, wait


def _int_costs(cost):
    c = np.asarray(cost, dtype=float)
    infeas = c >= 5_000.0
    finite = c[~infeas]
    if finite.size:
        c = c - finite.min()
    scaled = np.where(infeas, 10_000_000, np.round(c * COST_SCALE))
    return scaled.astype(np.int64)


def mcf_bins(sched, cost, table, bins):
    n = len(sched)
    last = int(bins[-1])
    cap = {}
    for b in bins:
        hod = (int(b) * 15) // 60
        cap[int(b)] = max(int(np.floor(table.get(hod, 4.0))), 0) if hod <= 22 else n
    cap[last] = n
    weights = _int_costs(cost)
    graph = nx.DiGraph()
    graph.add_node("src", demand=-n)
    graph.add_node("sink", demand=n)
    for b, c in cap.items():
        if c > 0:
            graph.add_edge(f"b{b}", "sink", capacity=int(c), weight=0)
    for i in range(n):
        graph.add_edge("src", f"f{i}", capacity=1, weight=0)
        connected = False
        for j, b in enumerate(bins):
            if b < int(sched[i]) or cap.get(int(b), 0) <= 0:
                continue
            graph.add_edge(f"f{i}", f"b{int(b)}", capacity=1, weight=int(weights[i, j]))
            connected = True
        if not connected:
            graph.add_edge(f"f{i}", f"b{last}", capacity=1, weight=10_000_000)
    flow = nx.min_cost_flow(graph)
    chosen = np.zeros(n, dtype=int)
    wait = np.zeros(n)
    for i in range(n):
        for dst, qty in flow[f"f{i}"].items():
            if qty and dst.startswith("b"):
                b = int(dst[1:])
                chosen[i] = b
                wait[i] = 15.0 * max(0, b - int(sched[i]))
    return chosen, wait


def day_arrays(day):
    sched = day["bin"].to_numpy(int)
    s1 = np.nan_to_num(day["s1"].to_numpy(float), nan=180.0)
    s2 = np.nan_to_num(day["s2"].to_numpy(float), nan=180.0)
    h1, h2 = day["has1"].to_numpy(bool), day["has2"].to_numpy(bool)
    delay = day["arr_delay"].to_numpy(float)
    hod = day["hod"].to_numpy(int)
    return sched, s1, s2, h1, h2, delay, hod


def bins_for(sched):
    first = int(min(sched.min(), 6 * 4))
    last = int(max(sched.max(), 23 * 4)) + 1
    return list(range(first, last + 1))


def assign_with_delay(day, table, pred):
    sched, s1, s2, h1, h2, _, _ = day_arrays(day)
    s1p = s1.copy()
    s1p[h1] = s1p[h1] - pred[h1]
    wait = g.mcf(sched, s1p, s2, h1, h2, table)
    return g.score_wait(wait, day), wait


def hindsight_assign(day, table):
    sched, s1, s2, h1, h2, delay, _ = day_arrays(day)
    s1r = s1.copy()
    s1r[h1] = s1r[h1] - delay[h1]
    wait = g.mcf(sched, s1r, s2, h1, h2, table)
    return g.score_wait(wait, day), wait


def torch_cascade(wait, pred, s1, s2, h1, h2):
    slack1 = s1 - pred
    leg2 = torch.where(h1, torch.clamp(wait - slack1, min=0.0), torch.zeros_like(wait))
    leg3 = torch.where(h2, torch.clamp(leg2 - s2, min=0.0), torch.zeros_like(wait))
    return leg2 + leg3


def cache_day(day, table):
    sched, s1, s2, h1, h2, delay, _ = day_arrays(day)
    bins = bins_for(sched)
    s1r = s1.copy()
    s1r[h1] = s1r[h1] - delay[h1]
    c_true, _ = flight_cost_matrix(sched, s1r, s2, h1, h2, bins)
    y_star, w_star = mcf_bins(sched, c_true, table, bins)
    return {
        "sched": sched,
        "s1": s1,
        "s2": s2,
        "h1": h1,
        "h2": h2,
        "delay": delay,
        "bins": bins,
        "c_true": c_true,
        "y_star": y_star,
        "w_star": w_star,
    }


def _pred_costs(cache, pred_np):
    s1p = cache["s1"].copy()
    h1 = cache["h1"]
    s1p[h1] = s1p[h1] - np.clip(pred_np[h1], -30, 180)
    c_pred, _ = flight_cost_matrix(cache["sched"], s1p, cache["s2"], h1, cache["h2"], cache["bins"])
    return c_pred


def spo_plus_loss(model, table, slow_x, fast_x, hod_idx, cache, gbr_np):
    """SPO+ on 2c-c*, plus Fenchel-Young on the unperturbed MCF.

    Costs start from the GBM delay so 2c-c* is not the anti-hindsight
    problem. Integer weights are hundredths of a minute. A lambda-shift
    of the unperturbed costs supplies a black-box gradient when the two
    assignments differ.
    """
    residual = model.forward_day(slow_x, fast_x, hod_idx)
    gbr_t = torch.tensor(gbr_np, dtype=residual.dtype, device=residual.device)
    pred = (gbr_t + residual).clamp(-30.0, 180.0)
    pred_np = pred.detach().cpu().numpy()
    sched = cache["sched"]
    bins = cache["bins"]
    c_pred = _pred_costs(cache, pred_np)
    y_star = cache["y_star"]
    y_spo, w_spo = mcf_bins(sched, 2.0 * c_pred - cache["c_true"], table, bins)
    y0, w0 = mcf_bins(sched, c_pred, table, bins)
    if np.array_equal(y0, y_star):
        w_bb = w0
    else:
        c_bb = c_pred.copy()
        index = {int(b): j for j, b in enumerate(bins)}
        for i in range(len(sched)):
            j0 = index.get(int(y0[i]))
            js = index.get(int(y_star[i]))
            if j0 is None or js is None or j0 == js:
                continue
            c_bb[i, j0] += SPO_LAMBDA
            c_bb[i, js] -= SPO_LAMBDA
        _, w_bb = mcf_bins(sched, c_bb, table, bins)
    device = pred.device
    s1t = torch.tensor(cache["s1"], dtype=pred.dtype, device=device)
    s2t = torch.tensor(cache["s2"], dtype=pred.dtype, device=device)
    h1t = torch.tensor(cache["h1"], device=device)
    h2t = torch.tensor(cache["h2"], device=device)
    w_spo_t = torch.tensor(w_spo, dtype=pred.dtype, device=device)
    w0_t = torch.tensor(w0, dtype=pred.dtype, device=device)
    w_star_t = torch.tensor(cache["w_star"], dtype=pred.dtype, device=device)
    w_bb_t = torch.tensor(w_bb, dtype=pred.dtype, device=device)
    cc = lambda w: torch_cascade(w, pred, s1t, s2t, h1t, h2t)
    loss = (cc(w_spo_t) - cc(w_star_t)).mean() + (cc(w0_t) - cc(w_star_t)).mean()
    loss = loss + (cc(w0_t) - cc(w_bb_t)).mean()
    stats = {
        "agree0": float((y0 == y_star).mean()),
        "agree_spo": float((y_spo == y_star).mean()),
        "mean_abs_pred": float(np.abs(pred_np).mean()),
        "mean_abs_res": float(np.abs(residual.detach().cpu().numpy()).mean()),
        "n_diff0": int((y0 != y_star).sum()),
    }
    return loss, stats


def ttt_adapt(model, opt, slow_x, fast_x, hod_idx, resid_target, morning, steps=6):
    """Same-day inner loop on flights already landed. Target is realized minus GBM."""
    if int(morning.sum()) < 8:
        return
    was = model.training
    model.train()
    for _ in range(steps):
        residual = model.forward_day(slow_x, fast_x, hod_idx)
        loss = F.smooth_l1_loss(residual[morning], resid_target[morning])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.train(was)
