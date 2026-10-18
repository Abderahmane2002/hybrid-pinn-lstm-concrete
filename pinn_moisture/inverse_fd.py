"""Classical inverse identification: FD forward model + nonlinear least squares.

For each mix, the identified parameters minimise sum_j [S_avg,FD(t_j; theta) - S_avg,exp(t_j)]^2
with a trust-region reflective algorithm (scipy.optimize.least_squares).
Several starting points are used to avoid local minima. The Jacobian at the
optimum gives approximate standard errors and parameter correlations, used to
assess identifiability.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from .fd_solver import FDSolver
from .physics import PhysicsConfig, param_bounds


def fit_mix(t, s_obs, cfg: PhysicsConfig, n_starts: int = 4, seed: int = 0, n_cells: int = 60):
    solver = FDSolver(cfg, n_cells=n_cells)
    names = list(cfg.identified)
    lo, hi = np.array(param_bounds(cfg)).T
    rng = np.random.default_rng(seed)
    n_eval = [0]

    def resid(z):
        n_eval[0] += 1
        s, _ = solver.solve(dict(zip(names, z)), t)
        return s - s_obs

    starts = [np.array([cfg.defaults[k] for k in names])]
    starts += [lo + (hi - lo) * rng.uniform(0.2, 0.8, size=len(names)) for _ in range(n_starts - 1)]
    best, t0 = None, time.time()
    for z0 in starts:
        z0 = np.clip(z0, lo + 1e-6, hi - 1e-6)
        try:
            r = least_squares(resid, z0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3, max_nfev=200)
        except RuntimeError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    dof = max(len(t) - len(names), 1)
    s2 = 2 * best.cost / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        se = np.sqrt(np.diag(cov))
        corr = cov / np.outer(se, se)
        cond = float(np.linalg.cond(J))
    except np.linalg.LinAlgError:
        se, corr, cond = np.full(len(names), np.nan), np.full((len(names),) * 2, np.nan), np.inf
    pred = s_obs + best.fun
    ss_res = np.sum(best.fun ** 2)
    ss_tot = np.sum((s_obs - s_obs.mean()) ** 2)
    return {
        "theta": dict(zip(names, best.x.tolist())),
        "se": dict(zip(names, se.tolist())),
        "corr": corr.tolist(),
        "cond": cond,
        "at_bound": [bool(np.isclose(v, l, atol=1e-3 * (h - l)) or np.isclose(v, h, atol=1e-3 * (h - l)))
                     for v, l, h in zip(best.x, lo, hi)],
        "r2": float(1 - ss_res / ss_tot),
        "rmse": float(np.sqrt(np.mean(best.fun ** 2))),
        "pred": pred.tolist(),
        "n_eval": n_eval[0],
        "time_s": time.time() - t0,
    }
