"""Differentiable finite-volume solver (PyTorch) and physics-embedded neural network.

The same discretisation as ``fd_solver.FDSolver`` (geometric mesh, symmetry plane,
Dirichlet drying face) is integrated with a linearly-implicit Euler scheme on a
logarithmic time grid. Every operation is a torch operation, so gradients of the
predicted average saturation with respect to the transport parameters — and to the
weights of a network that predicts them — are obtained by automatic differentiation.

``PhysicsEmbeddedNet``: theta = g(composition) (small MLP), S_avg(t) = Solver(theta).
The governing equation is satisfied by construction; only the data misfit is minimised.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .fd_solver import make_mesh
from .physics import PhysicsConfig, SECONDS_PER_DAY, diffusivity, param_bounds, theta_to_physical
from .pinn import DTYPE, MLP, composition_features


class TorchFVSolver:
    def __init__(self, cfg: PhysicsConfig, n_cells: int = 60, ratio: float = 1.08,
                 t_min: float = 1e-4, t_max: float = 100.0, n_steps: int = 240):
        self.cfg = cfg
        m = make_mesh(cfg.L, n_cells, ratio)
        self.w = torch.tensor(m.widths, dtype=DTYPE)
        self.dxc = torch.tensor(np.diff(m.centers), dtype=DTYPE)
        self.dx0 = float(m.centers[0])
        self.centers = m.centers
        self.t = torch.tensor(np.concatenate([[0.0], np.geomspace(t_min, t_max, n_steps)]), dtype=DTYPE)

    def D(self, S, th):
        """S: (B, N); th: dict of (B,) tensors in log10 / physical form."""
        p = theta_to_physical({k: v[:, None] for k, v in th.items()}, self.cfg)
        return diffusivity(S, p["Dv"], p["alpha0"], p["n"], p["K"], self.cfg) * SECONDS_PER_DAY

    def run(self, th: dict, return_fields: bool = False):
        """Integrate for a batch of parameter sets. Returns S_avg (B, T) [and S (B, T, N)]."""
        cfg = self.cfg
        B = next(iter(th.values())).shape[0]
        N = self.w.shape[0]
        S = torch.full((B, N), cfg.S_init, dtype=DTYPE)
        avg = [S @ self.w / cfg.L]
        fields = [S] if return_fields else None
        eye = torch.eye(N, dtype=DTYPE)
        for k in range(1, len(self.t)):
            dt = self.t[k] - self.t[k - 1]
            D = self.D(S, th)                                     # (B, N) at the old state
            Df = 0.5 * (D[:, :-1] + D[:, 1:]) / self.dxc          # face conductances (B, N-1)
            Sb = cfg.S_e
            Db = self.D(0.5 * (S[:, :1] + Sb), th)[:, 0] / self.dx0   # drying-face conductance (B,)
            # assemble (I + dt * A) S_new = S_old + dt * b   (A: diffusion operator / widths)
            diag = torch.zeros(B, N, dtype=DTYPE)
            diag[:, :-1] += Df
            diag[:, 1:] += Df
            diag[:, 0] += Db
            M = eye + dt * (torch.diag_embed(diag / self.w)
                            - torch.diag_embed(Df / self.w[:-1], 1) - torch.diag_embed(Df / self.w[1:], -1))
            rhs = S.clone()
            rhs[:, 0] = rhs[:, 0] + dt * Db * Sb / self.w[0]
            S = torch.linalg.solve(M, rhs)
            avg.append(S @ self.w / cfg.L)
            if return_fields:
                fields.append(S)
        avg = torch.stack(avg, 1)
        return (avg, torch.stack(fields, 1)) if return_fields else avg

    def interp(self, avg, t_query):
        """Linear interpolation of S_avg (B, T) in log-time at t_query (B, Q) [day]."""
        lt = torch.log(self.t[1:])
        lq = torch.log(t_query)
        idx = torch.clamp(torch.searchsorted(lt, lq.contiguous()), 1, len(lt) - 1)
        x0, x1 = lt[idx - 1], lt[idx]
        y = avg[:, 1:]
        y0 = torch.gather(y, 1, idx - 1)
        y1 = torch.gather(y, 1, idx)
        return y0 + (y1 - y0) * (lq - x0) / (x1 - x0)


def make_torch_solver(cfg: PhysicsConfig, t_max: float):
    if cfg.geometry == "slab":
        return TorchFVSolver(cfg, t_max=t_max)
    if cfg.table_path:
        from .table_solver import TableSolver
        return TableSolver(cfg.table_path, cfg)
    from .axisym import AxisymTorch
    return AxisymTorch(cfg, t_max=t_max)


class PhysicsEmbeddedNet(nn.Module):
    """theta(c) = lo + (hi - lo) * sigmoid(MLP(c)); S_avg = TorchFVSolver(theta)."""

    def __init__(self, cfg: PhysicsConfig, hidden: int = 16, layers: int = 2, per_mix: int = 0):
        super().__init__()
        self.cfg = cfg
        self.names = list(cfg.identified)
        lo, hi = np.array(param_bounds(cfg)).T
        self.register_buffer("lo", torch.tensor(lo, dtype=DTYPE))
        self.register_buffer("hi", torch.tensor(hi, dtype=DTYPE))
        self.mlp = MLP(3, len(lo), hidden, layers).to(DTYPE)
        z0 = np.array([cfg.defaults[k] for k in self.names])
        p0 = (z0 - lo) / (hi - lo)
        with torch.no_grad():
            self.mlp.net[-1].bias.copy_(torch.tensor(np.log(p0 / (1 - p0)), dtype=DTYPE))
            self.mlp.net[-1].weight.mul_(0.1)
        # optional free offsets per mix (calibration mode, no generalisation)
        self.offsets = nn.Parameter(torch.zeros(per_mix, len(lo), dtype=DTYPE)) if per_mix else None

    def theta(self, c, mix_idx=None):
        z = self.mlp(c)
        if self.offsets is not None and mix_idx is not None:
            z = z + self.offsets[mix_idx]
        return self.lo + (self.hi - self.lo) * torch.sigmoid(z)


def _group(df):
    mixes = list(dict.fromkeys(df["Mix"]))
    first = df.drop_duplicates("Mix").set_index("Mix").loc[mixes].reset_index()
    c = torch.tensor(composition_features(first), dtype=DTYPE)
    Q = max((df.Mix == m).sum() for m in mixes)
    tq = np.ones((len(mixes), Q))
    sq = np.zeros((len(mixes), Q))
    mask = np.zeros((len(mixes), Q))
    for i, m in enumerate(mixes):
        g = df[df.Mix == m]
        tq[i, :len(g)] = g.Time_d.values
        sq[i, :len(g)] = g.S_frac.values
        mask[i, :len(g)] = 1
    return mixes, c, torch.tensor(tq, dtype=DTYPE), torch.tensor(sq, dtype=DTYPE), torch.tensor(mask, dtype=DTYPE)


class PENTrainer:
    def __init__(self, train_df, cfg: PhysicsConfig, hidden=16, layers=2, lr=1e-2, iters=600,
                 weight_decay=1e-4, seed=0, per_mix=False):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.mixes, self.c, self.tq, self.sq, self.mask = _group(train_df)
        self.model = PhysicsEmbeddedNet(cfg, hidden, layers, per_mix=len(self.mixes) if per_mix else 0)
        self.solver = make_torch_solver(cfg, t_max=float(train_df.Time_d.max()) * 1.05 + 1)
        self.lr, self.iters, self.wd = lr, iters, weight_decay
        self.history = []

    def predict_train(self):
        th = self.model.theta(self.c, torch.arange(len(self.mixes)))
        avg = self.solver.run({k: th[:, i] for i, k in enumerate(self.model.names)})
        return self.solver.interp(avg, self.tq)

    def fit(self, verbose=True, log_every=50):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
        t0 = time.time()
        for it in range(self.iters):
            opt.zero_grad()
            pred = self.predict_train()
            loss = ((pred - self.sq) ** 2 * self.mask).sum() / self.mask.sum()
            loss.backward()
            opt.step()
            sched.step()
            if it % log_every == 0 or it == self.iters - 1:
                self.history.append({"it": it, "mse": loss.item(), "time_s": time.time() - t0})
                if verbose:
                    print(f"[PEN {it:4d}] rmse={loss.item() ** 0.5:.4f} ({time.time() - t0:.0f}s)", flush=True)
        self.train_time = time.time() - t0
        return self

    @torch.no_grad()
    def predict(self, df):
        mixes, c, tq, _, mask = _group(df)
        th = self.model.theta(c)
        solver = make_torch_solver(self.cfg, t_max=float(df.Time_d.max()) * 1.05 + 1)
        s = solver.interp(solver.run({k: th[:, i] for i, k in enumerate(self.model.names)}), tq).numpy()
        out = np.empty(len(df))
        for i, m in enumerate(mixes):
            idx = np.where(df.Mix.values == m)[0]
            out[idx] = s[i, :len(idx)]
        return out

    @torch.no_grad()
    def predict_theta(self, df):
        mixes = df.drop_duplicates("Mix")
        th = self.model.theta(torch.tensor(composition_features(mixes), dtype=DTYPE)).numpy()
        out = mixes[["Mix"]].reset_index(drop=True).copy()
        for i, k in enumerate(self.model.names):
            out[k] = th[:, i]
        return out
