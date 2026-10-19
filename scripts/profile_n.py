"""Profile the shared shape exponent n: fit (log10 D_v, log10 K_int) per mix for several fixed n.

python scripts/profile_n.py --geometry disc   ->  results/<geometry>/profile_n.md
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.inverse_fd import fit_mix  # noqa: E402
from pinn_moisture.physics import PhysicsConfig  # noqa: E402

DATA = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
MIXES = ["C-0.3", "C-0.5", "F40-0.4", "S20-0.4", "F10S10-0.5", "F20S20-0.3"]


def run(job):
    geometry, mix, n = job
    g = DATA[DATA.Mix == mix]
    cfg = PhysicsConfig(geometry=geometry)
    cfg.defaults = dict(cfg.defaults, n=n)
    r = fit_mix(g.Time_d.to_numpy(), g.S_frac.to_numpy(), cfg, n_starts=2)
    return {"mix": mix, "n": n, "rmse": r["rmse"], "r2": r["r2"], "N": len(g)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="disc")
    ap.add_argument("--n", nargs="+", type=float, default=[1.5, 2.0, 3.0, 4.0])
    a = ap.parse_args()
    with Pool(2) as pool:
        rows = list(pool.imap_unordered(run, [(a.geometry, m, n) for n in a.n for m in MIXES]))
    df = pd.DataFrame(rows)
    df["sse"] = df.rmse ** 2 * df.N
    tab = df.groupby("n").agg(sse=("sse", "sum"), r2_mean=("r2", "mean"), r2_min=("r2", "min")).reset_index()
    out = ROOT / "results" / a.geometry
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "profile_n_runs.csv", index=False)
    md = ["# Profile of the shared exponent n (" + a.geometry + ")", "", "| n | total SSE | mean R² | min R² |",
          "|---|---|---|---|"] + [f"| {r.n:g} | {r.sse:.5f} | {r.r2_mean:.5f} | {r.r2_min:.5f} |" for r in tab.itertuples()]
    (out / "profile_n.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
