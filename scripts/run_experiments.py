"""Train the physics-embedded network (pen), the PINN, its ablation and the data-driven
baseline on the training mixes,
evaluate on the six held-out mixes, for several random seeds.

Each run writes results/runs/<method>_seed<k>/{pred.csv, theta.csv, history.csv, meta.json, model.pt}.
Run `python scripts/summarize.py` afterwards to build the tables used in the paper.

Examples
--------
python scripts/run_experiments.py --methods pen mlp --seeds 0 1 2 3 4
python scripts/run_experiments.py --methods pinn pinn_nopde --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.baselines import DataDrivenMLP  # noqa: E402
from pinn_moisture.diff_solver import PENTrainer  # noqa: E402
from pinn_moisture.physics import config_for  # noqa: E402
from pinn_moisture.pinn import Trainer, TrainConfig  # noqa: E402

TEST_MIXES = ["C-0.4", "C-0.5", "F10S10-0.3", "F20-0.3", "F20S20-0.4", "S40-0.4"]
DATA = ROOT / "data/processed/drying_clean.csv"


def physics(geometry):
    cfg = config_for(geometry)
    table = ROOT / "results/disc/solution_table.npz"
    if geometry == "disc" and table.exists():
        cfg.table_path = str(table)
    return cfg


def run(job):
    method, seed, overrides, geometry, noise = job
    torch.set_num_threads(1)
    d = pd.read_csv(DATA)
    d = d[d.reliable].reset_index(drop=True)          # w/b = 0.6 curves excluded until re-digitised
    train, test = d[~d.Mix.isin(TEST_MIXES)].copy(), d[d.Mix.isin(TEST_MIXES)]
    if method in ("pinn", "pinn_nopde"):
        geometry = "slab"                              # the PINN is formulated on the 1D slab only
    tag = f"{method}_seed{seed}" + (f"_noise{noise:g}" if noise else "")
    if noise:                                          # robustness study: perturb the training data
        rng = np.random.default_rng(1000 + seed)
        train["S_frac"] = np.clip(train["S_frac"] + rng.normal(0.0, noise, len(train)), 0.0, 1.0)
    out = ROOT / "results" / geometry / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    pcfg = physics(geometry)
    t0 = time.time()
    meta = {"method": method, "seed": seed, "noise": noise, "geometry": geometry, "test_mixes": TEST_MIXES,
            "physics": pcfg.to_dict()}
    if method in ("pinn", "pinn_nopde"):
        tcfg = TrainConfig(seed=seed, use_pde=(method == "pinn"), **overrides)
        tr = Trainer(train, pcfg, tcfg).fit(verbose=False)
        pred_fn = tr.predict_avg
        tr.predict_theta(d).to_csv(out / "theta.csv", index=False)
        pd.DataFrame(tr.history).to_csv(out / "history.csv", index=False)
        torch.save(tr.model.state_dict(), out / "model.pt")
        meta.update(train=tcfg.to_dict(), pde_rms=tr.pde_residual_stats(), train_time_s=tr.train_time)
    elif method == "pen":
        tr = PENTrainer(train, pcfg, seed=seed, **{"iters": 400, "lr": 0.02, **overrides}).fit(verbose=False)
        pred_fn = tr.predict
        tr.predict_theta(d).to_csv(out / "theta.csv", index=False)
        pd.DataFrame(tr.history).to_csv(out / "history.csv", index=False)
        torch.save(tr.model.state_dict(), out / "model.pt")
        meta.update(train_time_s=tr.train_time)
    elif method == "mlp":
        m = DataDrivenMLP(seed=seed, **overrides).fit(train)
        pred_fn = m.predict
        meta.update(train_time_s=m.train_time)
    else:
        raise ValueError(method)
    pred = pd.concat([
        pd.DataFrame({"Mix": df.Mix.values, "Time_d": df.Time_d.values, "S_obs": df.S_frac.values,
                      "S_pred": pred_fn(df), "split": split})
        for split, df in (("train", train), ("test", test))])
    pred.to_csv(out / "pred.csv", index=False)
    meta["wall_time_s"] = time.time() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    test_pred = pred[pred.split == "test"]
    rmse = float(np.sqrt(np.mean((test_pred.S_pred - test_pred.S_obs) ** 2)))
    return f"{geometry} {tag}: test RMSE = {rmse:.4f} ({meta['wall_time_s']:.0f} s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["pen", "pinn", "pinn_nopde", "mlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--geometry", default="disc")
    ap.add_argument("--noise", type=float, default=0.0, help="std of Gaussian noise added to training data")
    ap.add_argument("--overrides", type=json.loads, default={},
                    help='JSON dict of TrainConfig overrides, e.g. \'{"adam_iters": 2000}\'')
    a = ap.parse_args()
    jobs = [(m, s, a.overrides if m in ("pinn", "pinn_nopde") else {}, a.geometry, a.noise)
            for m in a.methods for s in a.seeds]
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)
