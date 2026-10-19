"""Tabulated solution manifold of the disc model, with differentiable interpolation.

Only two parameters are identified, so the average-saturation response of the
axisymmetric disc model can be computed once on a dense grid of
(log10 D_v, log10 K_int) with the reference BDF solver, and interpolated in
between with bicubic convolution (Keys, 1981) in the parameters and linear
interpolation in log-time. Every table entry is an exact (BDF) solution of the
governing equation; the interpolation error is reported by ``check``.

The class mimics the interface of the differentiable solvers (``t``, ``run``,
``interp``) so that ``PENTrainer`` can use it unchanged.
"""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from .physics import PhysicsConfig

DTYPE = torch.float64


def _solve_row(args):
    cfg, lDv, lKs, t = args
    from .fd_solver import make_solver
    solver = make_solver(cfg)
    return np.array([solver.solve({"log10_Dv": lDv, "log10_K": lK}, t)[0] for lK in lKs])


def build_table(cfg: PhysicsConfig, path: Path, n_dv=49, n_k=49, n_t=120, t_min=0.05, t_max=110.0, workers=2):
    lo, hi = np.array([cfg.log10_Dv_range, cfg.log10_K_range]).T
    g_dv = np.linspace(lo[0], hi[0], n_dv)
    g_k = np.linspace(lo[1], hi[1], n_k)
    t = np.geomspace(t_min, t_max, n_t)
    t0 = time.time()
    with Pool(workers) as pool:
        rows = pool.map(_solve_row, [(cfg, d, g_k, t) for d in g_dv])
    S = np.stack(rows)                                   # (n_dv, n_k, n_t)
    np.savez_compressed(path, S=S, g_dv=g_dv, g_k=g_k, t=t, n=cfg.defaults["n"], alpha0=cfg.defaults["alpha0"],
                        geometry=cfg.geometry, build_time_s=time.time() - t0)
    return path


def _keys_weights(u):
    """Bicubic convolution weights (a = -0.5) for fractional offsets u in [0, 1)."""
    a = -0.5
    x = torch.stack([1 + u, u, 1 - u, 2 - u], -1)
    w_near = (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
    w_far = a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
    return torch.where(x <= 1, w_near, w_far)


class TableSolver:
    def __init__(self, path: Path, cfg: PhysicsConfig):
        z = np.load(path)
        if float(z["n"]) != cfg.defaults["n"] or str(z["geometry"]) != cfg.geometry:
            raise ValueError("table built for different constants or geometry")
        S = torch.tensor(z["S"], dtype=DTYPE)
        # pad by one cell on each side (linear extrapolation) for the 4x4 stencil
        S = torch.cat([2 * S[:1] - S[1:2], S, 2 * S[-1:] - S[-2:-1]], 0)
        S = torch.cat([2 * S[:, :1] - S[:, 1:2], S, 2 * S[:, -1:] - S[:, -2:-1]], 1)
        self.S = S
        self.g_dv, self.g_k = z["g_dv"], z["g_k"]
        self.h = (self.g_dv[1] - self.g_dv[0], self.g_k[1] - self.g_k[0])
        self.t = torch.tensor(np.concatenate([[0.0], z["t"]]), dtype=DTYPE)
        self.names = ["log10_Dv", "log10_K"]

    def run(self, th: dict):
        x = (th["log10_Dv"] - float(self.g_dv[0])) / self.h[0]
        y = (th["log10_K"] - float(self.g_k[0])) / self.h[1]
        n1, n2 = len(self.g_dv), len(self.g_k)
        x = x.clamp(0, n1 - 1 - 1e-9)
        y = y.clamp(0, n2 - 1 - 1e-9)
        ix, iy = torch.floor(x).long(), torch.floor(y).long()
        wx, wy = _keys_weights(x - ix), _keys_weights(y - iy)          # (B, 4)
        # padded indices: original index i -> i + 1; stencil i-1..i+2 -> i..i+3
        rx = ix[:, None] + torch.arange(4)
        ry = iy[:, None] + torch.arange(4)
        block = self.S[rx[:, :, None], ry[:, None, :]]                 # (B, 4, 4, T)
        avg = torch.einsum("bi,bj,bijt->bt", wx, wy, block)
        ones = torch.ones(avg.shape[0], 1, dtype=DTYPE)
        return torch.cat([ones, avg], 1)

    def interp(self, avg, t_query):
        lt = torch.log(self.t[1:])
        lq = torch.log(t_query.clamp(min=float(self.t[1])))
        idx = torch.clamp(torch.searchsorted(lt, lq.contiguous()), 1, len(lt) - 1)
        x0, x1 = lt[idx - 1], lt[idx]
        y = avg[:, 1:]
        y0, y1 = torch.gather(y, 1, idx - 1), torch.gather(y, 1, idx)
        return y0 + (y1 - y0) * (lq - x0) / (x1 - x0)


def check(path: Path, cfg: PhysicsConfig, n_test=20, seed=0, t_eval=None):
    """Interpolation error at random parameter sets, against direct BDF solutions."""
    from .fd_solver import make_solver
    rng = np.random.default_rng(seed)
    ts = TableSolver(path, cfg)
    solver = make_solver(cfg)
    t_eval = np.array([0.5, 1, 2, 4, 7, 14, 28, 56, 90]) if t_eval is None else t_eval
    lo, hi = np.array([cfg.log10_Dv_range, cfg.log10_K_range]).T
    errs = []
    for _ in range(n_test):
        z = lo + 0.1 * (hi - lo) + 0.8 * (hi - lo) * rng.random(2)
        ref, _ = solver.solve({"log10_Dv": z[0], "log10_K": z[1]}, t_eval)
        a = ts.run({"log10_Dv": torch.tensor([z[0]], dtype=DTYPE), "log10_K": torch.tensor([z[1]], dtype=DTYPE)})
        q = ts.interp(a, torch.tensor(t_eval[None], dtype=DTYPE)).numpy()[0]
        errs.append(np.max(np.abs(q - ref)))
    return np.array(errs)
