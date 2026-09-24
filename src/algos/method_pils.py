"""KEEP sit-capacity liquid that spends leftover mass on downstream buffer.

KEEP residual fill is the free-slack pass and stays frozen: connecting
flights take only predicted remaining slack, terminating flights are
sinks. Extra hold that remains after that pass is cascade-causing mass.
Original KEEP dumps that mass by scheduled slack. The liquid state is
the downstream (second-leg) sit-capacity of that leftover mass, scanned
bidirectionally with the day's weather, so evening buffers pull leftover
off tighter turns.
"""
from __future__ import annotations

import numpy as np
import torch

import run_trc_gate as g
import ttl_spo as core
from algos.assign_ours import fluid_budget, pred_fill, rbs
from algos.delay_map import LateAircraftHGB
from algos.instance import arrays

FAST_DIM = 16
CAP_CONN = 90.0
CAP_TERM = 240.0
DELTA_BOUND = 15.0
ROOM_BOUND = 45.0
DELAY_BOUND = 90.0
TEMP = 8.0
JUNE_TOL = 3e-3


def _caps(has1):
    return np.where(has1, CAP_CONN, CAP_TERM).astype(np.float32)


def keep_pass1(pred, s1, has1, budget):
    rem = float(np.asarray(budget, float).sum())
    h = np.zeros(len(s1), dtype=np.float64)
    cap = _caps(has1)
    resid = np.where(has1, np.clip(s1 - np.clip(pred, -30.0, 180.0), 0.0, None), CAP_TERM)
    for j in np.argsort(-resid):
        take = min(float(cap[j]), rem, float(resid[j]))
        h[j] = take
        rem -= take
        if rem <= 1e-9:
            return h, 0.0, cap
    return h, rem, cap


def leftover_np(score, room, rem):
    extra = np.zeros(len(score), dtype=np.float64)
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
    rem = rem.clamp(min=0.0)
    rem = torch.minimum(rem, room.sum())
    if float(rem.detach()) <= 1e-9:
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


def leftover_prior(a):
    s2 = np.where(a["h2"], a["s2"], 200.0)
    return (1.0 - a["h1"].astype(np.float32)) * 400.0 + s2.astype(np.float32)


def _fast_rows(day, te, hgb, h_pass, room):
    X, _ = g.feat_rows(day)
    prev = day["prev_arr_delay"].to_numpy(float) / 60.0
    hasp = day["has_prev"].to_numpy(float)
    orig = day["Origin"].map(te).fillna(0.0).to_numpy(float) / 60.0
    dow = day["dow"].to_numpy(float) / 6.0
    extra = np.column_stack(
        [
            prev,
            hasp,
            orig,
            dow,
            np.clip(hgb, -30.0, 180.0) / 60.0,
            np.asarray(h_pass, float) / 60.0,
            np.asarray(room, float) / 60.0,
            1.0 - day["has1"].to_numpy(float),
            np.nan_to_num(day["s2"].to_numpy(float), nan=180.0) / 60.0,
        ]
    )
    return np.concatenate([X, extra], 1).astype(np.float32)


def torch_cascade(h, delay, s1, s2, h1, h2):
    slack = s1 - delay
    leg2 = torch.where(h1, (h - slack).clamp(min=0.0), torch.zeros_like(h))
    leg3 = torch.where(h2, (leg2 - s2).clamp(min=0.0), torch.zeros_like(h))
    return (leg2 + leg3).sum()


class SitCapacityLiquid(torch.nn.Module):
    def __init__(self, slow_dim=6, fast_dim=FAST_DIM, hidden=24):
        super().__init__()
        self.hidden = hidden
        self.slow = core.CfcBlock(slow_dim, hidden)
        self.fwd = core.CfcBlock(fast_dim + hidden, hidden)
        self.bwd = core.CfcBlock(fast_dim + hidden, hidden)
        self.read = torch.nn.Linear(2 * hidden, 1)
        self.read.weight.data.zero_()
        self.read.bias.data.zero_()
        self.room = torch.nn.Linear(2 * hidden, 2)
        self.room.weight.data.zero_()
        self.room.bias.data.zero_()
        self.delay = torch.nn.Linear(2 * hidden, 1)
        self.delay.weight.data.zero_()
        self.delay.bias.data.zero_()

    def slow_states(self, slow_x):
        h = torch.zeros(self.hidden, device=slow_x.device)
        out = []
        for t in range(slow_x.shape[0]):
            h = core.cfc_step(self.slow, slow_x[t], h)
            out.append(h)
        return torch.stack(out)

    def hidden_states(self, slow_x, fast_x, hod_idx):
        slow_h = self.slow_states(slow_x)
        n = fast_x.shape[0]
        h = torch.zeros(self.hidden, device=fast_x.device)
        fwd = []
        for i in range(n):
            z = torch.cat([fast_x[i], slow_h[int(hod_idx[i])]], 0)
            h = core.cfc_step(self.fwd, z, h)
            fwd.append(h)
        h = torch.zeros(self.hidden, device=fast_x.device)
        bwd = [None] * n
        for i in range(n - 1, -1, -1):
            z = torch.cat([fast_x[i], slow_h[int(hod_idx[i])]], 0)
            h = core.cfc_step(self.bwd, z, h)
            bwd[i] = h
        return torch.cat([torch.stack(fwd), torch.stack(bwd)], dim=-1)

    def delta(self, slow_x, fast_x, hod_idx):
        hid = self.hidden_states(slow_x, fast_x, hod_idx)
        return DELTA_BOUND * torch.tanh(self.read(hid).squeeze(-1))

    def heads(self, slow_x, fast_x, hod_idx):
        hid = self.hidden_states(slow_x, fast_x, hod_idx)
        score = DELTA_BOUND * torch.tanh(self.read(hid).squeeze(-1))
        delay_res = DELAY_BOUND * torch.tanh(self.delay(hid).squeeze(-1))
        rm = ROOM_BOUND * torch.tanh(self.room(hid))
        return score, delay_res, rm[:, 1]


class PILS:
    def __init__(self):
        self.alpha = 1.0
        self.alpha_med = 1.0
        self.alpha_tight = 1.0
        self.te = None
        self.wx = None
        self.table = None
        self.base = None
        self.net = None
        self.device = g.DEVICE

    def _score(self, day, hgb, h_pass, room):
        key = day["day"].iloc[0]
        X = _fast_rows(day, self.te, hgb, h_pass, room)
        hod_idx = np.clip(day["hod"].to_numpy(int) - 6, 0, 16)
        slow = torch.tensor(self.wx[key], device=self.device)
        fast = torch.tensor(X, device=self.device)
        prior = torch.tensor(leftover_prior(arrays(day)), device=self.device)
        return prior + self.net.delta(slow, fast, hod_idx)

    def assign(self, day, budget):
        a = arrays(day)
        hgb = self.base.pred(day)
        h0, rem, cap = keep_pass1(hgb, a["s1"], a["h1"], budget)
        if rem <= 1e-9:
            return h0
        room = cap - h0
        with torch.no_grad():
            score = self._score(day, hgb, h0, room).detach().cpu().numpy()
        return h0 + leftover_np(score, room, rem)

    def fit(self, may, june, wx, te, table, epochs=6):
        from algos.family_ribeiro import RibeiroAAR
        from algos.family_wang import WangAAR
        from algos.family_wu import WuAAR
        from algos.instance import bin_panel, day_bins
        from algos.metrics import score_hold

        self.wx, self.te, self.table = wx, te, table
        self.device = g.DEVICE
        self.base = LateAircraftHGB().fit(may, wx, te)
        self.net = SitCapacityLiquid().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=4e-3)
        bins_may = bin_panel(may, wx)
        aar_models = {
            "DelayHybrid": RibeiroAAR().fit(bins_may),
            "DeclaredDTW": WangAAR().fit(bins_may),
            "DistRobust": WuAAR().fit(bins_may),
        }
        families = ("DelayHybrid", "DeclaredDTW", "DistRobust")

        def packs(day):
            a = arrays(day)
            gbin = day_bins(day, wx)
            out = {}
            for nm, mdl in aar_models.items():
                aar = mdl.aar(gbin)
                amap = {int(b): float(u) for b, u in zip(gbin["bin"].to_numpy(int), aar)}
                out[nm] = fluid_budget(a["sched"], a["hod"], amap)
            return out

        bases = {d["day"].iloc[0]: self.base.pred(d) for d in may + june}
        bud, keep_h = {}, {}
        for d in may + june:
            key = d["day"].iloc[0]
            a = arrays(d)
            bud[key] = packs(d)
            keep_h[key] = {
                nm: pred_fill(bases[key], a["s1"], a["h1"], b) for nm, b in bud[key].items()
            }

        def june_table():
            tot = {nm: 0.0 for nm in families}
            keep_tot = {nm: 0.0 for nm in families}
            self.net.eval()
            with torch.no_grad():
                for day in june:
                    key = day["day"].iloc[0]
                    a = arrays(day)
                    for nm in families:
                        tot[nm] += score_hold(self.assign(day, bud[key][nm]), a)["cascade"]
                        keep_tot[nm] += score_hold(keep_h[key][nm], a)["cascade"]
            return tot, keep_tot

        cas0, keep0 = june_table()
        rel0 = {nm: (keep0[nm] - cas0[nm]) / max(keep0[nm], 1.0) for nm in families}
        feasible0 = all(cas0[nm] <= keep0[nm] * (1.0 + JUNE_TOL) for nm in families)
        best_val = min(rel0.values()) if feasible0 else -1.0
        best_sum = sum(rel0.values()) if feasible0 else -1e9
        best_state = {k: t.detach().cpu().clone() for k, t in self.net.state_dict().items()}
        print(
            "leftover-liquid epoch 0 feasible",
            int(feasible0),
            {nm: round(cas0[nm]) for nm in families},
            "keep",
            {nm: round(keep0[nm]) for nm in families},
            "vs_keep",
            {nm: round(rel0[nm], 4) for nm in families},
            flush=True,
        )

        for epoch in range(1, epochs + 1):
            self.net.train()
            ep = 0.0
            order = np.random.default_rng(epoch).permutation(len(may))
            for di in order:
                day = may[int(di)]
                a = arrays(day)
                key = day["day"].iloc[0]
                hgb = bases[key]
                delay = torch.tensor(a["delay"], device=self.device, dtype=torch.float32)
                s1 = torch.tensor(a["s1"], device=self.device, dtype=torch.float32)
                s2 = torch.tensor(a["s2"], device=self.device, dtype=torch.float32)
                h1t = torch.tensor(a["h1"], device=self.device)
                h2t = torch.tensor(a["h2"], device=self.device)
                rels, hinges = [], []
                n_fam = 0
                for nm in families:
                    B = float(np.asarray(bud[key][nm], float).sum())
                    h0, rem, cap = keep_pass1(hgb, a["s1"], a["h1"], B)
                    if rem <= 1.0:
                        continue
                    room_np = cap - h0
                    score = self._score(day, hgb, h0, room_np)
                    room = torch.tensor(room_np, device=self.device, dtype=score.dtype)
                    extra = leftover_ste(score, room, rem)
                    h = torch.tensor(h0, device=self.device, dtype=score.dtype) + extra
                    cas = torch_cascade(h, delay, s1, s2, h1t, h2t)
                    keep_day = float(score_hold(keep_h[key][nm], a)["cascade"])
                    rels.append(cas / max(keep_day, 1.0))
                    hinges.append((cas - keep_day).clamp(min=0.0) / max(keep_day, 1.0))
                    n_fam += 1
                if n_fam == 0:
                    continue
                rel_t = torch.stack(rels)
                hinge_t = torch.stack(hinges)
                loss = rel_t.mean() + rel_t.max() + 8.0 * hinge_t.sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep += float(loss.detach().cpu())
            cas, keep_cas = june_table()
            rel = {nm: (keep_cas[nm] - cas[nm]) / max(keep_cas[nm], 1.0) for nm in families}
            feasible = all(cas[nm] <= keep_cas[nm] * (1.0 + JUNE_TOL) for nm in families)
            val = min(rel.values()) if feasible else -1.0
            sval = sum(rel.values()) if feasible else -1e9
            print(
                f"leftover-liquid epoch {epoch} loss {ep:.3f} feasible {int(feasible)} "
                f"june { {nm: round(cas[nm]) for nm in families} } "
                f"vs_keep { {nm: round(rel[nm], 4) for nm in families} } min {val:.4f}",
                flush=True,
            )
            if feasible and (val > best_val + 1e-6 or (abs(val - best_val) <= 1e-6 and sval > best_sum)):
                best_val, best_sum = val, sval
                best_state = {k: t.detach().cpu().clone() for k, t in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        self.net.to(self.device)
        self.net.eval()
        self.june_min_rel = best_val
        print("leftover-liquid deploy vs_keep_min", round(best_val, 4), flush=True)
        return self
