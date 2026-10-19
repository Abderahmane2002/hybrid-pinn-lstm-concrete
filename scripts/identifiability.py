"""Identifiability study: fit 2, 3 and 4 parameters per mix and report fit quality,
largest parameter correlation and condition number of the Jacobian.

python scripts/identifiability.py --geometry disc  ->  results/<geometry>/identifiability.md
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.inverse_fd import fit_mix  # noqa: E402
from pinn_moisture.physics import config_for  # noqa: E402

DATA = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
MIXES = ["C-0.4", "F20-0.3", "S40-0.5", "F20S20-0.5"]
SETS = [("log10_Dv", "alpha0", "log10_K"), ("log10_Dv", "alpha0", "n", "log10_K"), ("log10_Dv", "log10_K")]
SYMBOL = {"log10_Dv": "D_v", "alpha0": "α0", "n": "n", "log10_K": "K_int"}


def run(job):
    geometry, mix, ident = job
    g = DATA[DATA.Mix == mix]
    cfg = config_for(geometry)
    cfg.identified = ident
    cfg.log10_Dv_range, cfg.log10_K_range = (-12.0, -6.0), (-23.0, -18.0)   # wide: alpha0 trades with D_v
    r = fit_mix(g.Time_d.to_numpy(), g.S_frac.to_numpy(), cfg, n_starts=3)
    corr = np.abs(np.array(r["corr"]))
    np.fill_diagonal(corr, 0)
    i, j = np.unravel_index(np.nanargmax(corr), corr.shape)
    return {"Mix": mix, "identified": ", ".join(SYMBOL[k] for k in ident), "n_par": len(ident),
            "r2": r["r2"], "rmse": r["rmse"], "max_corr": float(corr[i, j]),
            "pair": f"{SYMBOL[ident[i]]}–{SYMBOL[ident[j]]}", "cond": r["cond"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="disc")
    a = ap.parse_args()
    jobs = [(a.geometry, m, s) for m in MIXES for s in SETS]
    with Pool(2) as pool:
        rows = list(pool.imap(run, jobs))
    df = pd.DataFrame(rows)
    out = ROOT / "results" / a.geometry
    df.to_csv(out / "identifiability.csv", index=False)
    md = [f"# Identifiability study ({a.geometry}; n = {config_for(a.geometry).defaults['n']:g} when not identified)",
          "", "| Mix | identified | R² | RMSE | max \\|ρ\\| (pair) | cond(J) |", "|---|---|---|---|---|---|"]
    md += [f"| {r.Mix} | {r.identified} | {r.r2:.4f} | {r.rmse:.4f} | {r.max_corr:.2f} ({r.pair}) | {r.cond:.1e} |"
           for r in df.itertuples()]
    (out / "identifiability.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
