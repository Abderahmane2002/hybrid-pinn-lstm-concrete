"""Numerical verification of the solvers used in the paper.

1. Spatial convergence of the BDF finite-volume solver (S_avg error vs a 640-cell reference).
2. Temporal convergence of the differentiable (linearly implicit Euler) solver.
3. Global mass balance: water lost through the drying face vs change of stored water.

Parameter sets: the classical fits of the fastest (C-0.5), a median (C-0.4) and the
slowest-drying mix (F40-0.3). Output: results/numerical_verification.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.diff_solver import TorchFVSolver  # noqa: E402
from pinn_moisture.fd_solver import FDSolver  # noqa: E402
from pinn_moisture.physics import SECONDS_PER_DAY, config_for, surface_saturation, theta_to_physical  # noqa: E402
from pinn_moisture.axisym import AxisymBDF  # noqa: E402
from pinn_moisture.pinn import DTYPE  # noqa: E402

T_EVAL = np.array([0.25, 0.5, 1, 2, 4, 7, 14, 28, 56, 90])


def main():
    cfg = config_for("slab")
    cl = pd.read_csv(ROOT / "results/slab/classical_fits.csv").set_index("Mix")
    mixes = ["C-0.5", "C-0.4", "F40-0.3"]
    lines = ["# Numerical verification", ""]

    # 1. spatial convergence of the BDF solver
    lines += ["## 1. Spatial convergence (BDF solver, rtol 1e-9)", "",
              "max |S_avg - S_avg,ref| over t = 0.25-90 d; reference: 640 cells", "",
              "| Mix | 20 cells | 40 cells | 60 cells | 80 cells | 160 cells | observed order (40->80) |",
              "|---|---|---|---|---|---|---|"]
    conv = {}
    for mix in mixes:
        th = {k: float(cl.loc[mix, k]) for k in cfg.identified}
        ref, _ = FDSolver(cfg, n_cells=640, ratio=1.006).solve(th, T_EVAL, rtol=1e-9, atol=1e-12)
        errs = []
        for n in (20, 40, 60, 80, 160):
            ratio = 1.06 ** (80 / n)          # same geometric stretching for every mesh
            s, _ = FDSolver(cfg, n_cells=n, ratio=ratio).solve(th, T_EVAL, rtol=1e-9, atol=1e-12)
            errs.append(np.max(np.abs(s - ref)))
        order = np.log2(errs[1] / errs[3])
        conv[mix] = errs
        lines.append(f"| {mix} | " + " | ".join(f"{e:.1e}" for e in errs) + f" | {order:.2f} |")

    # 2. temporal convergence of the differentiable solver (60 cells, as in the paper)
    lines += ["", "## 2. Differentiable solver (linearly implicit Euler, 60 cells, log time grid)", "",
              "max |S_avg - S_avg,BDF(160 cells)|", "",
              "| Mix | 60 steps | 120 steps | 240 steps (paper) | 480 steps | 960 steps |", "|---|---|---|---|---|---|"]
    for mix in mixes:
        th = {k: float(cl.loc[mix, k]) for k in cfg.identified}
        ref, _ = FDSolver(cfg, n_cells=160).solve(th, T_EVAL, rtol=1e-9, atol=1e-12)
        errs = []
        for ns in (60, 120, 240, 480, 960):
            sv = TorchFVSolver(cfg, n_steps=ns)
            a = sv.run({k: torch.tensor([v], dtype=DTYPE) for k, v in th.items()})
            q = sv.interp(a, torch.tensor(T_EVAL[None, :], dtype=DTYPE)).numpy()[0]
            errs.append(np.max(np.abs(q - ref)))
        lines.append(f"| {mix} | " + " | ".join(f"{e:.1e}" for e in errs) + " |")

    # 3. global mass balance of the BDF solver
    lines += ["", "## 3. Global mass balance (BDF solver, 80 cells)", "",
              "Water lost through the drying face, integrated in time, vs decrease of stored water.", "",
              "| Mix | t [d] | lost through face | decrease of S_avg L | relative error |", "|---|---|---|---|---|"]
    for mix in mixes:
        th = {k: float(cl.loc[mix, k]) for k in cfg.identified}
        solver = FDSolver(cfg, n_cells=80)
        p = theta_to_physical(th, cfg)
        t_end = 28.0

        def rhs_aug(t, y):
            S = y[:-1]
            dS = solver.rhs(t, S, p)
            Sb = surface_saturation(t, cfg)
            Db = solver.D_fn(np.array([0.5 * (Sb + S[0])]), p)[0] * SECONDS_PER_DAY
            out_flux = Db * (S[0] - Sb) / solver.dx0           # outward flux at x = 0 [m/day]
            return np.concatenate([dS, [out_flux]])

        y0 = np.concatenate([np.full(len(solver.mesh.centers), cfg.S_init), [0.0]])
        sol = solve_ivp(rhs_aug, (0, t_end), y0, method="BDF", rtol=1e-9, atol=1e-12)
        S_end = sol.y[:-1, -1]
        lost = sol.y[-1, -1]
        stored = (cfg.S_init * cfg.L) - (S_end * solver.mesh.widths).sum()
        lines.append(f"| {mix} | {t_end:.0f} | {lost * 1e3:.5f} mm | {stored * 1e3:.5f} mm | "
                     f"{abs(lost - stored) / stored:.1e} |")

    # 4. disc model: mesh convergence of the BDF solver used for identification (24 x 16 cells)
    cfg_d = config_for("disc")
    cl_d = pd.read_csv(ROOT / "results/disc/classical_fits.csv").set_index("Mix")
    lines += ["", "## 4. Disc model: mesh convergence (BDF, reference 96 x 64 cells)", "",
              "| Mix | 12 x 8 | 24 x 16 (paper) | 48 x 32 |", "|---|---|---|---|"]
    for mix in mixes:
        th = {k: float(cl_d.loc[mix, k]) for k in cfg_d.identified}
        ref, _ = AxisymBDF(cfg_d, nr=96, nz=64, ratio=1.03).solve(th, T_EVAL, rtol=1e-8, atol=1e-11)
        errs = []
        for nr, nz, ra in ((12, 8, 1.25), (24, 16, 1.12), (48, 32, 1.055)):
            a, _ = AxisymBDF(cfg_d, nr=nr, nz=nz, ratio=ra).solve(th, T_EVAL, rtol=1e-8, atol=1e-11)
            errs.append(np.max(np.abs(a - ref)))
        lines.append(f"| {mix} | " + " | ".join(f"{e:.1e}" for e in errs) + " |")
    tc = (ROOT / "results/disc/table_check.md")
    if tc.exists():
        lines += ["", "## 5. Tabulated disc solutions", "", tc.read_text().split("\n\n", 1)[1].strip()]

    text = "\n".join(lines) + "\n"
    (ROOT / "results/numerical_verification.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
