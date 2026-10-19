"""Axisymmetric (r, z) model of a disc drying through all its faces.

Geometry: disc of radius R and thickness 2H; by symmetry the domain is
r in [0, R], z in [0, H] with the axis (r = 0) and the mid-plane (z = 0) as
symmetry lines, and the lateral surface (r = R) and the flat face (z = H) as
drying faces, S = S_e.

Two solvers share the same finite-volume discretisation (cells graded towards
the drying faces):
* ``AxisymBDF``   — method of lines + BDF (scipy), the reference solver;
* ``AxisymTorch`` — differentiable, linearly implicit, locally one-dimensional
  (LOD) splitting: a backward-Euler sweep in r followed by one in z, with the
  diffusivity lagged at the previous step. Every step solves small tridiagonal
  systems batched over mixes and lines, so gradients with respect to the
  parameters are obtained by automatic differentiation.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from .physics import PhysicsConfig, SECONDS_PER_DAY, diffusivity, surface_saturation, theta_to_physical

DTYPE = torch.float64


def graded(length, n, ratio):
    """Cell widths shrinking towards the end of the interval (the drying face)."""
    w = ratio ** np.arange(n)[::-1]
    return w / w.sum() * length


class DiscMesh:
    def __init__(self, R, H, nr, nz, ratio):
        self.wr, self.wz = graded(R, nr, ratio), graded(H, nz, ratio)
        rf = np.concatenate([[0.0], np.cumsum(self.wr)])
        zf = np.concatenate([[0.0], np.cumsum(self.wz)])
        self.rf, self.zf = rf, zf
        self.rc, self.zc = 0.5 * (rf[1:] + rf[:-1]), 0.5 * (zf[1:] + zf[:-1])
        self.nr, self.nz = nr, nz
        ring = 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2)              # annulus area per radian
        self.vol = np.outer(ring, self.wz)                    # (nr, nz)
        self.Tr = np.outer(rf[1:-1], self.wz) / np.diff(self.rc)[:, None]      # internal r-faces (nr-1, nz)
        self.TrR = rf[-1] * self.wz / (rf[-1] - self.rc[-1])                   # lateral face (nz,)
        self.Tz = ring[:, None] / np.diff(self.zc)[None, :]                    # internal z-faces (nr, nz-1)
        self.TzH = ring / (zf[-1] - self.zc[-1])                               # flat face (nr,)


class AxisymBDF:
    def __init__(self, cfg: PhysicsConfig, nr=36, nz=24, ratio=1.10):
        self.cfg = cfg
        self.m = DiscMesh(cfg.R_disc, cfg.H_disc, nr, nz, ratio)
        n = nr * nz
        J = lil_matrix((n, n), dtype=bool)
        idx = np.arange(n).reshape(nr, nz)
        for i in range(nr):
            for j in range(nz):
                J[idx[i, j], idx[i, j]] = True
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if 0 <= i + di < nr and 0 <= j + dj < nz:
                        J[idx[i, j], idx[i + di, j + dj]] = True
        self.jac = J.tocsr()

    def rhs(self, t, y, p):
        m, cfg = self.m, self.cfg
        S = y.reshape(m.nr, m.nz)
        Dfun = lambda s: diffusivity(s, p["Dv"], p["alpha0"], p["n"], p["K"], cfg) * SECONDS_PER_DAY  # noqa: E731
        D = Dfun(S)
        Sb = surface_saturation(t, cfg)
        out = np.zeros_like(S)                                  # net outflow of each cell
        F = 0.5 * (D[1:] + D[:-1]) * m.Tr * (S[1:] - S[:-1])    # flow from i+1 to i
        out[1:] += F
        out[:-1] -= F
        out[-1] += Dfun(0.5 * (S[-1] + Sb)) * m.TrR * (S[-1] - Sb)
        F = 0.5 * (D[:, 1:] + D[:, :-1]) * m.Tz * (S[:, 1:] - S[:, :-1])
        out[:, 1:] += F
        out[:, :-1] -= F
        out[:, -1] += Dfun(0.5 * (S[:, -1] + Sb)) * m.TzH * (S[:, -1] - Sb)
        return (-out / m.vol).ravel()

    def solve(self, theta, t_eval, rtol=1e-6, atol=1e-9, return_fields=False):
        p = theta_to_physical(theta, self.cfg)
        t_eval = np.asarray(t_eval, float)
        y0 = np.full(self.m.nr * self.m.nz, self.cfg.S_init)
        sol = solve_ivp(self.rhs, (0.0, float(t_eval.max())), y0, method="BDF", t_eval=t_eval, args=(p,),
                        rtol=rtol, atol=atol, jac_sparsity=self.jac)
        if not sol.success:
            raise RuntimeError(sol.message)
        S = sol.y.reshape(self.m.nr, self.m.nz, -1)
        avg = (S * self.m.vol[:, :, None]).sum((0, 1)) / self.m.vol.sum()
        return (avg, S) if return_fields else (avg, None)


def _tridiag_solve(a, b, c, d):
    """Batched tridiagonal solve: a (sub), b (diag), c (super), d (rhs), all (..., n)."""
    M = torch.diag_embed(b) + torch.diag_embed(c[..., :-1], 1) + torch.diag_embed(a[..., 1:], -1)
    return torch.linalg.solve(M, d.unsqueeze(-1)).squeeze(-1)


class AxisymTorch:
    def __init__(self, cfg: PhysicsConfig, nr=30, nz=20, ratio=1.10, t_min=1e-4, t_max=100.0, n_steps=240):
        self.cfg = cfg
        self.m = DiscMesh(cfg.R_disc, cfg.H_disc, nr, nz, ratio)
        T = lambda a: torch.tensor(a, dtype=DTYPE)  # noqa: E731
        self.vol, self.Tr, self.TrR, self.Tz, self.TzH = map(T, (self.m.vol, self.m.Tr, self.m.TrR,
                                                                 self.m.Tz, self.m.TzH))
        self.t = torch.tensor(np.concatenate([[0.0], np.geomspace(t_min, t_max, n_steps)]), dtype=DTYPE)

    def D(self, S, th):
        p = theta_to_physical({k: v.view(-1, *([1] * (S.dim() - 1))) for k, v in th.items()}, self.cfg)
        return diffusivity(S, p["Dv"], p["alpha0"], p["n"], p["K"], self.cfg) * SECONDS_PER_DAY

    def run(self, th: dict, return_fields: bool = False):
        cfg, m = self.cfg, self.m
        B = next(iter(th.values())).shape[0]
        S = torch.full((B, m.nr, m.nz), cfg.S_init, dtype=DTYPE)
        Sb = cfg.S_e
        vtot = self.vol.sum()
        avg = [(S * self.vol).sum((1, 2)) / vtot]
        fields = [S] if return_fields else None
        for k in range(1, len(self.t)):
            dt = self.t[k] - self.t[k - 1]
            D = self.D(S, th)
            # --- sweep in r (lines of constant z): unknowns along dim 1 ---------------------
            Cr = 0.5 * (D[:, 1:] + D[:, :-1]) * self.Tr                       # (B, nr-1, nz)
            CR = self.D(0.5 * (S[:, -1] + Sb), th) * self.TrR                  # (B, nz)
            V = (self.vol / dt).expand(B, -1, -1)
            diag = V.clone()
            diag[:, 1:] += Cr
            diag[:, :-1] += Cr
            diag[:, -1] += CR
            off = torch.zeros_like(V)
            off[:, :-1] = -Cr                                                 # coupling i <-> i+1
            rhs = V * S
            rhs[:, -1] += CR * Sb
            tr = lambda x: x.transpose(1, 2)  # noqa: E731  (B, nz, nr)
            sub = torch.zeros_like(V)
            sub[:, 1:] = -Cr
            S = tr(_tridiag_solve(tr(sub), tr(diag), tr(off), tr(rhs)))
            # --- sweep in z (lines of constant r): unknowns along dim 2 ---------------------
            D = self.D(S, th)
            Cz = 0.5 * (D[:, :, 1:] + D[:, :, :-1]) * self.Tz                  # (B, nr, nz-1)
            CH = self.D(0.5 * (S[:, :, -1] + Sb), th) * self.TzH               # (B, nr)
            diag = V.clone()
            diag[:, :, 1:] += Cz
            diag[:, :, :-1] += Cz
            diag[:, :, -1] += CH
            off = torch.zeros_like(V)
            off[:, :, :-1] = -Cz
            sub = torch.zeros_like(V)
            sub[:, :, 1:] = -Cz
            rhs = V * S
            rhs[:, :, -1] += CH * Sb
            S = _tridiag_solve(sub, diag, off, rhs)
            avg.append((S * self.vol).sum((1, 2)) / vtot)
            if return_fields:
                fields.append(S)
        avg = torch.stack(avg, 1)
        return (avg, torch.stack(fields, 1)) if return_fields else avg

    def interp(self, avg, t_query):
        lt = torch.log(self.t[1:])
        lq = torch.log(t_query)
        idx = torch.clamp(torch.searchsorted(lt, lq.contiguous()), 1, len(lt) - 1)
        x0, x1 = lt[idx - 1], lt[idx]
        y = avg[:, 1:]
        y0, y1 = torch.gather(y, 1, idx - 1), torch.gather(y, 1, idx)
        return y0 + (y1 - y0) * (lq - x0) / (x1 - x0)
