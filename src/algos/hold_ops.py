"""Shared extra-hold operators: caps, predicted cascade, leftover ranking."""
from __future__ import annotations

import numpy as np
import torch

CAP_CONN = 90.0
CAP_TERM = 240.0
QUANT = 15.0
TEMP = 8.0


def caps(has1):
    return np.where(has1, CAP_CONN, CAP_TERM).astype(np.float64)


def pred_cascade(hold, pred, s1, s2, h1, h2):
    slack = s1 - np.clip(np.asarray(pred, float), -30.0, 180.0)
    h = np.asarray(hold, float)
    leg2 = np.where(h1, np.maximum(0.0, h - slack), 0.0)
    leg3 = np.where(h2, np.maximum(0.0, leg2 - s2), 0.0)
    return float((leg2 + leg3).sum())


def leftover_np(score, room, rem):
    extra = np.zeros(len(score), dtype=np.float64)
    rem = float(rem)
    if rem <= 1e-9:
        return extra
    for j in np.argsort(-np.asarray(score, float)):
        take = min(max(float(room[j]), 0.0), rem)
        extra[j] = take
        rem -= take
        if rem <= 1e-9:
            return extra
    return extra


def leftover_torch(score, room, rem):
    extra = torch.zeros_like(score)
    rem = torch.as_tensor(rem, dtype=score.dtype, device=score.device)
    if float(rem) <= 1e-9:
        return extra
    for i in torch.argsort(score, descending=True):
        take = torch.minimum(room[i].clamp(min=0.0), rem.clamp(min=0.0))
        extra[i] = take
        rem = rem - take
    return extra


def leftover_soft(score, room, rem, temp=TEMP, steps=25):
    rem = torch.as_tensor(rem, dtype=score.dtype, device=score.device).reshape(())
    room = room.clamp_min(0.0)
    rem = torch.clamp(rem, min=0.0, max=room.sum())
    if float(rem) <= 1e-9:
        return torch.zeros_like(score)
    lo = score.amin() - 4.0 * temp
    hi = score.amax() + 4.0 * temp
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        h = room * torch.sigmoid((score - mid) / temp)
        too = h.sum() > rem
        lo = torch.where(too, mid, lo)
        hi = torch.where(too, hi, mid)
    tau = 0.5 * (lo + hi)
    h = room * torch.sigmoid((score - tau) / temp)
    return h * (rem / h.sum().clamp_min(1e-6))


def leftover_ste(score, room, rem):
    hard = leftover_torch(score.detach(), room, rem)
    soft = leftover_soft(score, room, rem)
    return hard + soft - soft.detach()


def torch_cascade(h, delay, s1, s2, h1, h2):
    slack = s1 - delay
    leg2 = torch.where(h1, (h - slack).clamp(min=0.0), torch.zeros_like(h))
    leg3 = torch.where(h2, (leg2 - s2).clamp(min=0.0), torch.zeros_like(h))
    return (leg2 + leg3).sum()


def leftover_prior(a):
    s2 = np.where(a["h2"], a["s2"], 200.0)
    return (1.0 - a["h1"].astype(np.float32)) * 400.0 + s2.astype(np.float32)
