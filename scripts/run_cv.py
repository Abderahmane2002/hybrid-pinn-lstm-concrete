"""Extended cross-validation of the composition-to-parameter models.

Schemes
-------
lomo : leave-one-mix-out, 20 folds (each mix predicted from the other 19)
lofo : leave-one-binder-family-out, 7 folds (e.g. all F40 mixes predicted from the
       other six families) -- extrapolation in SCM type/content.

Methods: PENN (physics-embedded network), FD + regression, data-driven MLP.
Writes results/cv/pred_<scheme>.csv (one row per data point, method, fold).

python scripts/run_cv.py --geometry disc --schemes lomo lofo --workers 2
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.baselines import DataDrivenMLP, fd_regression  # noqa: E402
from pinn_moisture.diff_solver import PENTrainer  # noqa: E402
from pinn_moisture.physics import config_for  # noqa: E402

DATA = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
DATA = DATA[DATA.reliable].reset_index(drop=True)
GEOMETRY = "disc"


def physics():
    cfg = config_for(GEOMETRY)
    table = ROOT / "results/disc/solution_table.npz"
    if GEOMETRY == "disc" and table.exists():
        cfg.table_path = str(table)
    return cfg


def folds(scheme):
    mixes = sorted(DATA.Mix.unique())
    if scheme == "lomo":
        return [(m, [m]) for m in mixes]
    fams = sorted({m.split("-")[0] for m in mixes})
    return [(f, [m for m in mixes if m.split("-")[0] == f]) for f in fams]


def run(job):
    scheme, name, test_mixes, method, geometry = job
    global GEOMETRY
    GEOMETRY = geometry
    torch.set_num_threads(1)
    cfg = physics()
    OUT = ROOT / "results" / geometry / "cv"
    CLASSICAL = pd.read_csv(ROOT / "results" / geometry / "classical_fits.csv")
    train, test = DATA[~DATA.Mix.isin(test_mixes)], DATA[DATA.Mix.isin(test_mixes)]
    t0 = time.time()
    if method == "pen":
        tr = PENTrainer(train, cfg, seed=0, iters=400, lr=0.02).fit(verbose=False)
        pred = tr.predict(test)
        th = tr.predict_theta(test)
    elif method == "mlp":
        pred = DataDrivenMLP(seed=0).fit(train).predict(test)
        th = None
    elif method == "fdreg":
        train_mixes = sorted(train.Mix.unique())
        p, th, _ = fd_regression(CLASSICAL, DATA, train_mixes, test_mixes, cfg)
        pred = p.set_index(["Mix", "Time_d"]).loc[list(zip(test.Mix, test.Time_d)), "S_pred"].to_numpy()
    df = pd.DataFrame({"scheme": scheme, "fold": name, "method": method, "Mix": test.Mix.values,
                       "Time_d": test.Time_d.values, "S_obs": test.S_frac.values, "S_pred": pred})
    df.to_csv(OUT / f"{scheme}_{name}_{method}.csv", index=False)
    if th is not None:
        th.assign(scheme=scheme, fold=name, method=method).to_csv(OUT / f"theta_{scheme}_{name}_{method}.csv",
                                                                  index=False)
    return f"{scheme} {name:8s} {method:6s} {time.time() - t0:6.0f} s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["lomo", "lofo"])
    ap.add_argument("--methods", nargs="+", default=["fdreg", "mlp", "pen"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--geometry", default="disc")
    a = ap.parse_args()
    OUT = ROOT / "results" / a.geometry / "cv"
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, n, t, m, a.geometry) for m in a.methods for s in a.schemes for n, t in folds(s)
            if not (OUT / f"{s}_{n}_{m}.csv").exists()]
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)
