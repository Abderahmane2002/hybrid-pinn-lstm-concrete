"""Experiment B (+H): sensitivity of the identified parameters to the modelling assumptions.

For 8 representative mixes the classical least-squares identification (direct BDF disc
solver, no tabulation) is repeated with one assumption changed at a time:
    mesh_36x24       finer mesh (discretisation effect)
    sealed_lateral   lateral surface sealed: drying through the two flat faces only (1D, L = 25 mm)
    alpha0, n, S_c   vapour branch constants
    S_cap, k         regime switch
    beta             capillary-pressure exponent (P_c0 and eta only scale K_int, see note)
    S_init           incomplete vacuum saturation
    tau_b            finite surface exchange (surface reaches S_e with time constant tau_b)
Output: results/q1/physics_sensitivity/{fits.csv, summary.csv}
Note: D_cap is proportional to P_c0 K_int / eta_l, so a change of P_c0 or eta_l by a factor c
shifts log10 K_int by exactly -log10(c) and leaves the fit unchanged.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from common import ROOT, load_data, out_dir
from pinn_moisture.axisym import AxisymBDF
from pinn_moisture.fd_solver import FDSolver
from pinn_moisture.physics import config_for, param_bounds

MIXES = ["C-0.3", "C-0.5", "F20-0.4", "F40-0.3", "S20-0.3", "S40-0.5", "F10S10-0.4", "F20S20-0.5"]
VARIANTS = {
    "base": {}, "mesh_36x24": {"mesh": (36, 24, 1.08)}, "sealed_lateral": {"sealed": True},
    "alpha0_0.02": {"alpha0": 0.02}, "alpha0_0.10": {"alpha0": 0.10},
    "n_4": {"n": 4.0}, "n_16": {"n": 16.0},
    "S_c_0.75": {"S_c": 0.75},
    "S_cap_0.70": {"S_cap": 0.70}, "S_cap_0.80": {"S_cap": 0.80},
    "k_25": {"k_switch": 25.0},
    "beta_2.0": {"beta": 2.0}, "beta_3.0": {"beta": 3.0},
    "S_init_0.97": {"S_init": 0.97}, "tau_b_0.1d": {"tau_b": 0.1},
}


def make(variant):
    v = dict(VARIANTS[variant])
    if v.pop("sealed", False):
        cfg = config_for("slab", L=0.025)
        cfg.defaults = dict(cfg.defaults, n=8.0)
        cfg.log10_Dv_range, cfg.log10_K_range = (-10.0, -7.5), (-21.0, -18.5)
        return cfg, FDSolver(cfg, n_cells=60)
    mesh = v.pop("mesh", (24, 16, 1.12))
    defaults = {k: v.pop(k) for k in ("alpha0", "n") if k in v}
    cfg = config_for("disc", **v)
    cfg.defaults = dict(cfg.defaults, **defaults)
    return cfg, AxisymBDF(cfg, nr=mesh[0], nz=mesh[1], ratio=mesh[2])


def job(args):
    variant, mix, x0 = args
    d = load_data()
    g = d[d.Mix == mix]
    t, s = g.Time_d.to_numpy(), g.S_frac.to_numpy()
    cfg, solver = make(variant)
    lo, hi = np.array(param_bounds(cfg)).T
    t0 = time.time()
    best = None
    for z0 in (np.array(x0), np.array([-9.0, -19.77])):
        try:
            r = least_squares(lambda z: solver.solve({"log10_Dv": z[0], "log10_K": z[1]}, t)[0] - s,
                              np.clip(z0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale="jac", diff_step=1e-3,
                              max_nfev=100)
        except RuntimeError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    cov = np.linalg.inv(J.T @ J) * (2 * best.cost / (len(t) - 2))
    se = np.sqrt(np.diag(cov))
    return {"variant": variant, "Mix": mix, "log10_Dv": best.x[0], "log10_K": best.x[1], "se_Dv": se[0], "se_K": se[1],
            "corr": cov[0, 1] / se[0] / se[1], "rmse": float(np.sqrt(np.mean(best.fun ** 2))),
            "at_bound": bool(np.any(np.isclose(best.x, lo, atol=1e-3) | np.isclose(best.x, hi, atol=1e-3))),
            "time_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    out = out_dir("physics_sensitivity")
    cl = pd.read_csv(ROOT / "results/disc/classical_fits.csv").set_index("Mix")
    done = pd.read_csv(out / "fits.csv") if (out / "fits.csv").exists() else pd.DataFrame(columns=["variant", "Mix"])
    jobs = [(v, m, (cl.loc[m, "log10_Dv"], cl.loc[m, "log10_K"])) for v in a.variants for m in MIXES
            if not ((done.variant == v) & (done.Mix == m)).any()]
    rows = done.to_dict("records")
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows.append(r)
            pd.DataFrame(rows).to_csv(out / "fits.csv", index=False)
            print(f"{r['variant']:15s} {r['Mix']:11s} Dv={r['log10_Dv']:.3f} K={r['log10_K']:.3f} "
                  f"rmse={r['rmse']:.4f} ({r['time_s']:.0f} s)", flush=True)
    F = pd.DataFrame(rows)
    base = F[F.variant == "base"].set_index("Mix")
    S = []
    for v, g in F.groupby("variant"):
        g = g.set_index("Mix")
        g = g.loc[base.index.intersection(g.index)]
        S.append({"variant": v, "dDv_mean": (g.log10_Dv - base.log10_Dv).mean(),
                  "dDv_maxabs": (g.log10_Dv - base.log10_Dv).abs().max(),
                  "dK_mean": (g.log10_K - base.log10_K).mean(), "dK_maxabs": (g.log10_K - base.log10_K).abs().max(),
                  "rmse_mean": g.rmse.mean(), "rmse_ratio": (g.rmse / base.rmse).mean(),
                  "corr_max": g["corr"].abs().max(), "n_at_bound": int(g.at_bound.sum()),
                  # does the ranking of the mixes survive? (Spearman of the identified parameters)
                  "rank_Dv": g.log10_Dv.rank().corr(base.log10_Dv.rank()), "rank_K": g.log10_K.rank().corr(base.log10_K.rank())})
    S = pd.DataFrame(S)
    S.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
