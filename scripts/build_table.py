"""Tabulate the disc-model solutions on a (log10 D_v, log10 K_int) grid and check the interpolation.

python scripts/build_table.py  ->  results/disc/solution_table.npz, results/disc/table_check.md
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.physics import config_for  # noqa: E402
from pinn_moisture.table_solver import build_table, check  # noqa: E402

if __name__ == "__main__":
    cfg = config_for("disc")
    path = ROOT / "results/disc/solution_table.npz"
    if not path.exists():
        build_table(cfg, path, n_dv=49, n_k=49)
    errs = check(path, cfg, n_test=30)
    z = np.load(path)
    txt = (f"# Tabulated disc solutions\n\nGrid: {len(z['g_dv'])} x {len(z['g_k'])} parameter values, "
           f"log10 D_v in [{z['g_dv'][0]}, {z['g_dv'][-1]}], log10 K_int in [{z['g_k'][0]}, {z['g_k'][-1]}], "
           f"{len(z['t'])} times from {z['t'][0]:.2f} to {z['t'][-1]:.0f} d; build time {float(z['build_time_s']):.0f} s.\n\n"
           f"Interpolation error against direct BDF solutions at 30 random parameter sets "
           f"(max over t = 0.5-90 d): median {np.median(errs):.1e}, max {errs.max():.1e}.\n")
    (ROOT / "results/disc/table_check.md").write_text(txt)
    print(txt)
