"""TTL-SPO with two listwise heads switched by hold-budget load.

Medium budgets (delay-hybrid, declared-capacity) train ListNet on
hindsight hold shares. Tight DRO budgets train ListMLE on remaining
turn slack s-d. At test, load = total hold / (n*90) picks the head.
June chooses each head's mix so that head's family cascade falls.
"""
from __future__ import annotations

import numpy as np
import torch

import run_trc_gate as g
import ttl_spo as core
from algos.assign_ours import fluid_budget
from algos.delay_map import LateAircraftHGB
from algos.instance import arrays
from algos.transport_hold import allocate

FAST_DIM = 12


def _fast_rows(day, te, load=0.0):
    X, y = g.feat_rows(day)
    prev = day["prev_arr_delay"].to_numpy(float) / 60.0
    hasp = day["has_prev"].to_numpy(float)
    orig = day["Origin"].map(te).fillna(0.0).to_numpy(float) / 60.0
    dow = day["dow"].to_numpy(float) / 6.0
    extra = np.column_stack([prev, hasp, orig, dow, np.full(len(day), float(load))])
    return np.concatenate([X, extra], 1).astype(np.float32), y


def _load(budget, n):
    return float(np.asarray(budget, float).sum()) / (max(n, 1) * 90.0)


def hour_budget(day, table):
    a = arrays(day)
    amap = {int(b): float(table.get((int(b) * 15) // 60, 4.0)) for b in np.unique(a["sched"])}
    return fluid_budget(a["sched"], a["hod"], amap)


def listmle(pred_score, true_score):
    order = torch.argsort(true_score, descending=True)
    s = pred_score[order]
    rev = torch.flip(s, [0])
    lse = torch.flip(torch.logcumsumexp(rev, 0), [0])
    return -(s - lse).mean()


class TTLSPO:
    def __init__(self, bound=20.0):
        self.bound = bound
        self.alpha_med = 0.0
        self.alpha_tight = 0.0
        self.te = None
        self.wx = None
        self.table = None
        self.base = None
        self.model_med = None
        self.model_tight = None
        self.device = g.DEVICE

    def _tensors(self, day, load=0.0):
        key = day["day"].iloc[0]
        X, y = _fast_rows(day, self.te, load)
        hod_idx = np.clip(day["hod"].to_numpy(int) - 6, 0, 16)
        slow = torch.tensor(self.wx[key], device=self.device)
        fast = torch.tensor(X, device=self.device)
        return slow, fast, hod_idx, torch.tensor(y, device=self.device)

    def _resid(self, net, day, load):
        if net is None:
            return np.zeros(len(day))
        slow, fast, hod_idx, _ = self._tensors(day, load)
        core.BOUND = self.bound
        return net.forward_day(slow, fast, hod_idx).detach().cpu().numpy()

    def pred(self, day, budget=None):
        base = self.base.pred(day)
        if budget is None:
            return base
        load = _load(budget, len(day))
        if load >= 0.78:
            return np.clip(base + self.alpha_tight * self._resid(self.model_tight, day, load), -30, 180)
        return np.clip(base + self.alpha_med * self._resid(self.model_med, day, load), -30, 180)

    def assign(self, day, budget):
        a = arrays(day)
        return allocate(self.pred(day, budget), a["s1"], a["s2"], a["h1"], a["h2"], budget)

    def fit(self, may, june, wx, te, table, epochs=6):
        from algos.family_ribeiro import RibeiroAAR
        from algos.family_wang import WangAAR
        from algos.family_wu import WuAAR
        from algos.instance import bin_panel, day_bins
        from algos.metrics import score_hold

        self.wx, self.te, self.table = wx, te, table
        self.device = g.DEVICE
        self.base = LateAircraftHGB().fit(may, wx, te)
        core.BOUND = self.bound
        self.model_med = core.TwoTimescaleLiquid(fast_dim=FAST_DIM).to(self.device)
        self.model_tight = core.TwoTimescaleLiquid(fast_dim=FAST_DIM).to(self.device)
        self.model_med.read.weight.data.mul_(4.0)
        opt_m = torch.optim.Adam(self.model_med.parameters(), lr=8e-3)
        opt_t = torch.optim.Adam(self.model_tight.parameters(), lr=5e-3)
        bins_may = bin_panel(may, wx)
        aar_models = {
            "DelayHybrid": RibeiroAAR().fit(bins_may),
            "DeclaredDTW": WangAAR().fit(bins_may),
            "DistRobust": WuAAR().fit(bins_may),
        }

        def packs(day):
            a = arrays(day)
            gbin = day_bins(day, wx)
            out = {"hour": hour_budget(day, table)}
            for nm, mdl in aar_models.items():
                aar = mdl.aar(gbin)
                amap = {int(b): float(u) for b, u in zip(gbin["bin"].to_numpy(int), aar)}
                out[nm] = fluid_budget(a["sched"], a["hod"], amap)
            return out

        bases = {d["day"].iloc[0]: self.base.pred(d) for d in may + june}
        bud, star = {}, {}
        for d in may + june:
            key = d["day"].iloc[0]
            a = arrays(d)
            bud[key] = packs(d)
            star[key] = {
                nm: allocate(a["delay"], a["s1"], a["s2"], a["h1"], a["h2"], b)
                for nm, b in bud[key].items()
            }

        best_m, best_t = None, None
        best_am, best_at, best_val = 0.0, 0.0, -1e18
        for epoch in range(1, epochs + 1):
            self.model_med.train()
            self.model_tight.train()
            ep = 0.0
            for day in may:
                a = arrays(day)
                key = day["day"].iloc[0]
                s1t = torch.tensor(a["s1"], device=self.device, dtype=torch.float32)
                dly = torch.tensor(a["delay"], device=self.device, dtype=torch.float32)
                msk = torch.tensor(a["h1"], device=self.device)
                prev = torch.tensor(a["prev"], device=self.device, dtype=torch.float32)
                hasp = torch.tensor(day["has_prev"].to_numpy(float), device=self.device, dtype=torch.float32)
                phy = torch.relu(s1t - prev)
                hgb_t = torch.tensor(bases[key], device=self.device, dtype=torch.float32)
                # medium head: ListNet on delay-hybrid hold shares
                load_m = _load(bud[key]["DelayHybrid"], len(day))
                slow, fast, hod_idx, _ = self._tensors(day, load_m)
                sm = self.model_med.forward_day(slow, fast, hod_idx)
                h_star = torch.tensor(star[key]["DelayHybrid"], device=sm.device, dtype=sm.dtype)
                logp = torch.nn.functional.log_softmax(sm, dim=0)
                share = h_star / h_star.sum().clamp_min(1.0)
                loss_m = -(share * logp).sum()
                opt_m.zero_grad()
                loss_m.backward()
                torch.nn.utils.clip_grad_norm_(self.model_med.parameters(), 1.0)
                opt_m.step()
                # tight head: ListMLE on remaining slack
                load_t = _load(bud[key]["DistRobust"], len(day))
                slow, fast, hod_idx, _ = self._tensors(day, load_t)
                st = self.model_tight.forward_day(slow, fast, hod_idx)
                pred = (hgb_t + st).clamp(-30.0, 180.0)
                sh_t = torch.relu(s1t - pred)
                loss_t = listmle((s1t - pred)[msk], (s1t - dly)[msk])
                loss_t = loss_t + 0.15 * ((sh_t - phy).pow(2) * hasp).sum() / hasp.sum().clamp_min(1.0)
                opt_t.zero_grad()
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(self.model_tight.parameters(), 1.0)
                opt_t.step()
                ep += float(loss_m.detach().cpu() + loss_t.detach().cpu())
            self.model_med.eval()
            self.model_tight.eval()
            alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
            cas_m = {a: {"DelayHybrid": 0.0, "DeclaredDTW": 0.0} for a in alphas}
            cas_t = {a: 0.0 for a in alphas}
            with torch.no_grad():
                for a_mix in alphas:
                    for day in june:
                        key = day["day"].iloc[0]
                        a = arrays(day)
                        rm = self._resid(self.model_med, day, _load(bud[key]["DelayHybrid"], len(day)))
                        rt = self._resid(self.model_tight, day, _load(bud[key]["DistRobust"], len(day)))
                        pm = np.clip(bases[key] + a_mix * rm, -30, 180)
                        pt = np.clip(bases[key] + a_mix * rt, -30, 180)
                        for nm in ("DelayHybrid", "DeclaredDTW"):
                            h0 = allocate(pm, a["s1"], a["s2"], a["h1"], a["h2"], bud[key][nm])
                            cas_m[a_mix][nm] += score_hold(h0, a)["cascade"]
                        h1 = allocate(pt, a["s1"], a["s2"], a["h1"], a["h2"], bud[key]["DistRobust"])
                        cas_t[a_mix] += score_hold(h1, a)["cascade"]
            hy0 = cas_m[0.0]["DelayHybrid"]
            dtw0 = cas_m[0.0]["DeclaredDTW"]
            dro0 = cas_t[0.0]
            feas_m = [
                a for a in alphas
                if cas_m[a]["DelayHybrid"] <= hy0 and cas_m[a]["DeclaredDTW"] <= dtw0
            ] or [0.0]
            a_med = min(feas_m, key=lambda a: cas_m[a]["DelayHybrid"] + cas_m[a]["DeclaredDTW"])
            feas_t = [a for a in alphas if cas_t[a] <= dro0] or [0.0]
            a_tight = min(feas_t, key=lambda a: cas_t[a])
            print(
                f"ttlspo epoch {epoch} med {a_med} tight {a_tight} "
                f"hyb { {a: round(cas_m[a]['DelayHybrid']) for a in alphas} } "
                f"dtw { {a: round(cas_m[a]['DeclaredDTW']) for a in alphas} } "
                f"dro { {a: round(cas_t[a]) for a in alphas} }",
                flush=True,
            )
            val = (hy0 - cas_m[a_med]["DelayHybrid"]) / hy0 + (dtw0 - cas_m[a_med]["DeclaredDTW"]) / dtw0 + (
                dro0 - cas_t[a_tight]
            ) / dro0
            if val > best_val + 1e-6:
                best_val = val
                best_am, best_at = a_med, a_tight
                best_m = {k: t.detach().cpu().clone() for k, t in self.model_med.state_dict().items()}
                best_t = {k: t.detach().cpu().clone() for k, t in self.model_tight.state_dict().items()}
        if best_m is not None:
            self.model_med.load_state_dict(best_m)
            self.model_tight.load_state_dict(best_t)
            self.model_med.to(self.device)
            self.model_tight.to(self.device)
        self.alpha_med, self.alpha_tight = best_am, best_at
        self.alpha = self.alpha_med
        self.model_med.eval()
        self.model_tight.eval()
        print("ttlspo alpha_med", self.alpha_med, "alpha_tight", self.alpha_tight, flush=True)
        return self
