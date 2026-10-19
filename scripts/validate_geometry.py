"""Bias introduced by the 1D equivalent-slab idealisation of the disc.

For the disc-model parameters of four mixes (results/disc/classical_fits.csv):
  (a) difference between the disc and the 1D slab (L = V/A = 12.5 mm, same constants)
      average-saturation curves computed with the same parameters;
  (b) parameters obtained when the slab model is fitted to the disc curve.
Output: results/geometry_check.md, results/geometry_check.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.fd_solver import FDSolver, make_solver  # noqa: E402
from pinn_moisture.physics import config_for, param_bounds  # noqa: E402


def main():
    cfg_d = config_for("disc")
    cfg_s = config_for("disc")
    cfg_s.geometry = "slab"                       # same constants (n = 8), 1D slab of half-thickness V/A
    cl = pd.read_csv(ROOT / "results/disc/classical_fits.csv").set_index("Mix")
    data = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
    disc, slab = make_solver(cfg_d), FDSolver(cfg_s, n_cells=60)
    names = list(cfg_d.identified)
    lo, hi = np.array(param_bounds(cfg_s)).T
    rows = []
    for mix in ["C-0.5", "C-0.4", "F20-0.4", "F40-0.3"]:
        th = {k: float(cl.loc[mix, k]) for k in names}
        t = data[data.Mix == mix].Time_d.to_numpy()
        s2, _ = disc.solve(th, t)
        s1, _ = slab.solve(th, t)
        r = least_squares(lambda z: slab.solve(dict(zip(names, z)), t)[0] - s2,
                          np.array([th[k] for k in names]), bounds=(lo - 1, hi + 1), diff_step=1e-3, x_scale="jac")
        d = r.x - np.array([th[k] for k in names])
        rows.append({"Mix": mix, "max_diff": float(np.max(np.abs(s2 - s1))),
                     "rmse": float(np.sqrt(np.mean((s2 - s1) ** 2))), "dlog10_Dv": float(d[0]),
                     "dlog10_K": float(d[1]), "refit_rmse": float(np.sqrt(np.mean(r.fun ** 2)))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/geometry_check.csv", index=False)
    md = ["# Equivalent-slab idealisation vs disc model (same constants, n = 8)", "",
          "| Mix | max \\|ΔS_avg\\| same parameters | RMSE | slab refit: Δlog10 D_v | Δlog10 K_int | refit RMSE |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r.Mix} | {r.max_diff:.3f} | {r.rmse:.3f} | {r.dlog10_Dv:+.2f} | {r.dlog10_K:+.2f} | {r.refit_rmse:.4f} |"
           for r in df.itertuples()]
    (ROOT / "results/geometry_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
