"""Finite-volume / method-of-lines solver for the 1D drying problem.

Used (i) as the classical forward model for least-squares identification,
(ii) to verify the parameters identified by the PINN, and (iii) to produce
reference saturation profiles.

The mesh is refined geometrically towards the drying face (x = 0), time
integration uses an implicit BDF scheme with a tridiagonal Jacobian pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

from .physics import PhysicsConfig, SECONDS_PER_DAY, diffusivity, surface_saturation, theta_to_physical


@dataclass
class Mesh:
    faces: np.ndarray    # (N+1,) face positions [m]
    centers: np.ndarray  # (N,)
    widths: np.ndarray   # (N,)


def make_mesh(L: float, n_cells: int = 80, ratio: float = 1.06) -> Mesh:
    w = ratio ** np.arange(n_cells)
    w = w / w.sum() * L
    faces = np.concatenate([[0.0], np.cumsum(w)])
    return Mesh(faces, 0.5 * (faces[:-1] + faces[1:]), w)


class FDSolver:
    def __init__(self, cfg: PhysicsConfig, n_cells: int = 80, ratio: float = 1.06, D_fn=None):
        self.cfg = cfg
        # D_fn(S, p) -> D [m^2/s]; defaults to the dual-regime model of physics.py
        self.D_fn = D_fn or (lambda S, p: diffusivity(S, p["Dv"], p["alpha0"], p["n"], p["K"], cfg))
        self.mesh = make_mesh(cfg.L, n_cells, ratio)
        m = self.mesh
        self.dxc = np.diff(m.centers)                 # distance between centres
        self.dx0 = m.centers[0]                       # centre of first cell to the face
        n = len(m.centers)
        self.jac_pattern = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(n, n)).toarray() > 0

    def rhs(self, t, S, p):
        cfg, m = self.cfg, self.mesh
        D = self.D_fn(S, p) * SECONDS_PER_DAY        # m^2/day
        Sb = surface_saturation(t, cfg)
        Db = self.D_fn(np.array([0.5 * (Sb + S[0])]), p)[0] * SECONDS_PER_DAY
        flux = np.empty(len(S) + 1)                   # flux = -D dS/dx at faces (positive towards +x)
        flux[0] = -Db * (S[0] - Sb) / self.dx0
        Dface = 0.5 * (D[:-1] + D[1:])
        flux[1:-1] = -Dface * np.diff(S) / self.dxc
        flux[-1] = 0.0                                # symmetry plane
        return -(flux[1:] - flux[:-1]) / m.widths

    def solve(self, theta: dict, t_eval_day, rtol=1e-6, atol=1e-8):
        """Return (S_avg(t_eval), S(x, t_eval) [N x T]) for parameter dict ``theta``."""
        p = theta_to_physical(theta, self.cfg)
        t_eval = np.asarray(t_eval_day, dtype=float)
        S0 = np.full(len(self.mesh.centers), self.cfg.S_init)
        sol = solve_ivp(self.rhs, (0.0, float(t_eval.max())), S0, method="BDF", t_eval=t_eval,
                        args=(p,), rtol=rtol, atol=atol, jac_sparsity=self.jac_pattern)
        if not sol.success:
            raise RuntimeError(sol.message)
        S = sol.y
        S_avg = (S * self.mesh.widths[:, None]).sum(0) / self.cfg.L
        return S_avg, S


def make_solver(cfg: PhysicsConfig, **kw):
    """Reference (BDF) forward solver for the geometry selected in ``cfg``.

    Both solvers expose ``solve(theta, t_eval) -> (S_avg, fields)``.
    """
    if cfg.geometry == "slab":
        return FDSolver(cfg, **kw)
    if cfg.geometry == "disc":
        from .axisym import AxisymBDF
        kw.pop("n_cells", None)
        return AxisymBDF(cfg, **{"nr": 24, "nz": 16, "ratio": 1.12, **kw})
    raise ValueError(cfg.geometry)
