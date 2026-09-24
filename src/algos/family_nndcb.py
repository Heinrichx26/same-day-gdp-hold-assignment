"""Learning family: Chen, Zhao, Fei and Yang, Aerospace 11 (2024) 966.

NN-DCB uses neural diving (a complete delay vector) plus neural branching
(binary local adjustments). Architecture from that paper: three fully
connected layers, ReLU, dropout 0.3, Xavier initialisation. The diving
network scores inbound flights and spends the extra-hold total under the
connecting and terminating caps. The branching network then accepts or
rejects 15-minute transfers. Both networks are pointwise: no liquid
time-constant and no bidirectional flight order.
"""
from __future__ import annotations

import numpy as np
import torch

import run_trc_gate as g
from algos.assign_ours import fluid_budget
from algos.delay_map import LateAircraftHGB
from algos.hold_ops import QUANT, caps, leftover_np, leftover_ste, torch_cascade
from algos.instance import arrays, bin_panel, day_bins
from algos.metrics import score_hold

HIDDEN = 64
DROPOUT = 0.3
EPOCHS = 8
LR = 3e-3


def _xavier(mod):
    for m in mod.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)


class DivingNet(torch.nn.Module):
    def __init__(self, din, hidden=HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(din, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden, 1),
        )
        _xavier(self)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BranchNet(torch.nn.Module):
    def __init__(self, din, hidden=HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * din, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden, 1),
        )
        _xavier(self)

    def forward(self, xi, xj):
        return self.net(torch.cat([xi, xj], dim=-1)).squeeze(-1)


def _rows(day, te, hgb, wx_day):
    X, _ = g.feat_rows(day)
    prev = day["prev_arr_delay"].to_numpy(float) / 60.0
    hasp = day["has_prev"].to_numpy(float)
    orig = day["Origin"].map(te).fillna(0.0).to_numpy(float) / 60.0
    dow = day["dow"].to_numpy(float) / 6.0
    has1 = day["has1"].to_numpy(float)
    s1 = np.nan_to_num(day["s1"].to_numpy(float), nan=180.0)
    s2 = np.nan_to_num(day["s2"].to_numpy(float), nan=180.0)
    pred = np.clip(np.asarray(hgb, float), -30.0, 180.0)
    resid = np.where(has1 > 0.5, np.clip(s1 - pred, 0.0, None), 240.0)
    hod_idx = np.clip(day["hod"].to_numpy(int) - 6, 0, 16)
    wx = np.asarray(wx_day, float)
    if wx.ndim == 1:
        wx = np.repeat(wx.reshape(1, -1), 17, axis=0)
    extra = np.column_stack(
        [prev, hasp, orig, dow, pred / 60.0, 1.0 - has1, s2 / 60.0, resid / 60.0, s1 / 60.0]
    )
    return np.concatenate([X, extra, wx[hod_idx]], 1).astype(np.float32)


class NeuralDCB:
    def __init__(self):
        self.base = None
        self.dive = None
        self.branch = None
        self.te = None
        self.wx = None
        self.device = g.DEVICE
        self.june_cascade = None

    def _feat(self, day, hgb):
        key = day["day"].iloc[0]
        return torch.tensor(_rows(day, self.te, hgb, self.wx[key]), device=self.device)

    def _dive_hold(self, day, budget, hgb):
        a = arrays(day)
        B = float(np.asarray(budget, float).sum())
        cap = caps(a["h1"])
        with torch.no_grad():
            score = self.dive(self._feat(day, hgb)).detach().cpu().numpy()
        return leftover_np(score, cap, B)

    def _branch_refine(self, day, h, hgb, n_try=40):
        a = arrays(day)
        cap = caps(a["h1"])
        slack = a["s1"] - np.clip(hgb, -30.0, 180.0)
        x = self._feat(day, hgb)
        donors = np.flatnonzero(h >= QUANT)
        takers = np.flatnonzero(h + QUANT <= cap)
        if len(donors) == 0 or len(takers) == 0:
            return h
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for _ in range(n_try):
                i = int(rng.choice(donors))
                j = int(rng.choice(takers))
                if i == j:
                    continue
                logit = float(self.branch(x[i], x[j]).detach().cpu())
                if logit <= 0:
                    continue
                # accept only if predicted cascade does not rise
                def cas(hold, idx):
                    if not a["h1"][idx]:
                        return 0.0
                    leg2 = max(0.0, hold[idx] - slack[idx])
                    return leg2 + (max(0.0, leg2 - a["s2"][idx]) if a["h2"][idx] else 0.0)

                before = cas(h, i) + cas(h, j)
                h[i] -= QUANT
                h[j] += QUANT
                after = cas(h, i) + cas(h, j)
                if after > before + 1e-9:
                    h[i] += QUANT
                    h[j] -= QUANT
        return h

    def assign(self, day, budget):
        hgb = self.base.pred(day)
        h = self._dive_hold(day, budget, hgb)
        return self._branch_refine(day, h, hgb)

    def fit(self, may, june, wx, te, table, epochs=EPOCHS):
        from algos.family_ribeiro import RibeiroAAR
        from algos.family_wang import WangAAR
        from algos.family_wu import WuAAR

        self.wx, self.te = wx, te
        self.device = g.DEVICE
        self.base = LateAircraftHGB().fit(may, wx, te)
        bins_may = bin_panel(may, wx)
        rates = {
            "DelayHybrid": RibeiroAAR().fit(bins_may),
            "DeclaredDTW": WangAAR().fit(bins_may),
            "DistRobust": WuAAR().fit(bins_may),
        }
        families = tuple(rates)

        def packs(day):
            a = arrays(day)
            gbin = day_bins(day, wx)
            out = {}
            for nm, mdl in rates.items():
                aar = mdl.aar(gbin)
                amap = {int(b): float(u) for b, u in zip(gbin["bin"].to_numpy(int), aar)}
                out[nm] = fluid_budget(a["sched"], a["hod"], amap)
            return out

        bases = {d["day"].iloc[0]: self.base.pred(d) for d in may + june}
        bud = {d["day"].iloc[0]: packs(d) for d in may + june}
        probe = may[0]
        din = _rows(probe, te, bases[probe["day"].iloc[0]], wx[probe["day"].iloc[0]]).shape[1]
        self.dive = DivingNet(din).to(self.device)
        self.branch = BranchNet(din).to(self.device)
        opt = torch.optim.Adam(list(self.dive.parameters()) + list(self.branch.parameters()), lr=LR)
        bce = torch.nn.BCEWithLogitsLoss()

        def june_sum():
            tot = 0.0
            self.dive.eval()
            self.branch.eval()
            with torch.no_grad():
                for day in june:
                    key = day["day"].iloc[0]
                    a = arrays(day)
                    for nm in families:
                        tot += score_hold(self.assign(day, bud[key][nm]), a)["cascade"]
            return tot

        best_val = june_sum()
        best_dive = {k: t.detach().cpu().clone() for k, t in self.dive.state_dict().items()}
        best_br = {k: t.detach().cpu().clone() for k, t in self.branch.state_dict().items()}
        print("nndcb epoch 0 june", round(best_val), flush=True)

        for epoch in range(1, epochs + 1):
            self.dive.train()
            self.branch.train()
            ep = 0.0
            order = np.random.default_rng(epoch).permutation(len(may))
            for di in order:
                day = may[int(di)]
                a = arrays(day)
                key = day["day"].iloc[0]
                hgb = bases[key]
                feat = self._feat(day, hgb)
                delay = torch.tensor(a["delay"], device=self.device, dtype=torch.float32)
                s1 = torch.tensor(a["s1"], device=self.device, dtype=torch.float32)
                s2 = torch.tensor(a["s2"], device=self.device, dtype=torch.float32)
                h1t = torch.tensor(a["h1"], device=self.device)
                h2t = torch.tensor(a["h2"], device=self.device)
                cap = torch.tensor(caps(a["h1"]), device=self.device, dtype=torch.float32)
                score = self.dive(feat)
                losses = []
                for nm in families:
                    B = float(np.asarray(bud[key][nm], float).sum())
                    if B <= 1.0:
                        continue
                    h = leftover_ste(score, cap, B)
                    losses.append(torch_cascade(h, delay, s1, s2, h1t, h2t))
                if not losses:
                    continue
                # branching labels: 15-minute transfer that cuts predicted cascade
                rng = np.random.default_rng(epoch * 1000 + int(di))
                n = len(a["s1"])
                br_loss = torch.zeros((), device=self.device)
                n_br = 0
                slack = a["s1"] - np.clip(hgb, -30.0, 180.0)
                for _ in range(8):
                    i, j = int(rng.integers(n)), int(rng.integers(n))
                    if i == j:
                        continue
                    def cas(idx, hold):
                        if not a["h1"][idx]:
                            return 0.0
                        leg2 = max(0.0, hold - slack[idx])
                        return leg2 + (max(0.0, leg2 - a["s2"][idx]) if a["h2"][idx] else 0.0)

                    y = 1.0 if cas(i, QUANT) + cas(j, 2 * QUANT) < cas(i, 2 * QUANT) + cas(j, QUANT) else 0.0
                    br_loss = br_loss + bce(self.branch(feat[i], feat[j]), torch.tensor(y, device=self.device))
                    n_br += 1
                loss = torch.stack(losses).mean() + 0.2 * (br_loss / max(n_br, 1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.dive.parameters()) + list(self.branch.parameters()), 1.0)
                opt.step()
                ep += float(loss.detach().cpu())
            val = june_sum()
            print(f"nndcb epoch {epoch} loss {ep:.1f} june {round(val)}", flush=True)
            if val < best_val - 1.0:
                best_val = val
                best_dive = {k: t.detach().cpu().clone() for k, t in self.dive.state_dict().items()}
                best_br = {k: t.detach().cpu().clone() for k, t in self.branch.state_dict().items()}
        self.dive.load_state_dict(best_dive)
        self.branch.load_state_dict(best_br)
        self.dive.to(self.device).eval()
        self.branch.to(self.device).eval()
        self.june_cascade = best_val
        print("nndcb deploy june", round(best_val), flush=True)
        return self
