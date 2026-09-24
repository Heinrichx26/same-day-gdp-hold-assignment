"""TLNEHA ablations: drop first ranking, weather timescale, or reverse scan."""
from __future__ import annotations

import numpy as np
import torch

import ttl_spo as core
from algos.hold_ops import leftover_np, leftover_prior
from algos.instance import arrays
from algos.method_pils import DELTA_BOUND, _caps, _fast_rows, keep_pass1


def assign_first(method, day, budget):
    from algos.family_priority import hierarchical_priority

    a = arrays(day)
    hgb = method.base.pred(day)
    return hierarchical_priority(hgb, a["s1"], a["h1"], a["s2"], a["h2"], budget)


def assign_s0(method, day, budget):
    a = arrays(day)
    hgb = method.base.pred(day)
    h0, rem, cap = keep_pass1(hgb, a["s1"], a["h1"], budget)
    if rem <= 1e-9:
        return h0
    return h0 + leftover_np(leftover_prior(a), cap - h0, rem)


def assign_fwd(method, day, budget):
    return _liquid(method, day, budget, backward=False, weather=True, first=True)


def assign_nowx(method, day, budget):
    return _liquid(method, day, budget, backward=True, weather=False, first=True)


def assign_nofirst(method, day, budget):
    return _liquid(method, day, budget, backward=True, weather=True, first=False)


def _liquid(method, day, budget, backward, weather, first):
    a = arrays(day)
    hgb = method.base.pred(day)
    cap = _caps(a["h1"])
    if first:
        h0, rem, _ = keep_pass1(hgb, a["s1"], a["h1"], budget)
        if rem <= 1e-9:
            return h0
        room = cap - h0
    else:
        h0 = np.zeros(len(a["s1"]))
        rem = float(np.asarray(budget, float).sum())
        room = cap
    key = day["day"].iloc[0]
    X = _fast_rows(day, method.te, hgb, h0, room)
    hod_idx = np.clip(day["hod"].to_numpy(int) - 6, 0, 16)
    slow = torch.tensor(method.wx[key], device=method.device)
    if not weather:
        slow = torch.zeros_like(slow)
    fast = torch.tensor(X, device=method.device)
    prior = torch.tensor(leftover_prior(a), device=method.device)
    net = method.net
    with torch.no_grad():
        slow_h = net.slow_states(slow)
        n = fast.shape[0]
        h = torch.zeros(net.hidden, device=fast.device)
        fwd = []
        for i in range(n):
            z = torch.cat([fast[i], slow_h[int(hod_idx[i])]], 0)
            h = core.cfc_step(net.fwd, z, h)
            fwd.append(h)
        if backward:
            h = torch.zeros(net.hidden, device=fast.device)
            bwd = [None] * n
            for i in range(n - 1, -1, -1):
                z = torch.cat([fast[i], slow_h[int(hod_idx[i])]], 0)
                h = core.cfc_step(net.bwd, z, h)
                bwd[i] = h
            hid = torch.cat([torch.stack(fwd), torch.stack(bwd)], dim=-1)
        else:
            hid = torch.cat([torch.stack(fwd), torch.zeros_like(torch.stack(fwd))], dim=-1)
        delta = DELTA_BOUND * torch.tanh(net.read(hid).squeeze(-1))
        score = (prior + delta).detach().cpu().numpy()
    return h0 + leftover_np(score, room, rem)
