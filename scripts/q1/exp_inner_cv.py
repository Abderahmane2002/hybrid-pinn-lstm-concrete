"""Experiment A: were the hyper-parameters chosen without the withheld mixes?

The settings used in the paper (PENN: 2 x 16 tanh, 400 iterations, lr 0.02; data-driven
network: 4 x 48, 4000 iterations, weight decay 1e-5) are compared with alternatives by
an INNER leave-one-family-out cross-validation that uses the 14 training mixes only.
If the inner CV selects (or nearly selects) the paper settings, the hold-out results are
not the product of test-set tuning. The selected configuration is then evaluated once on
the six withheld mixes.
Output: results/q1/inner_cv/{inner.csv, selected.csv}
"""
from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd
import torch

from common import TEST_MIXES, disc_cfg, family, load_data, metrics, mix_table, predict_from_theta, out_dir

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from pinn_moisture.baselines import DataDrivenMLP  # noqa: E402
from pinn_moisture.diff_solver import PENTrainer  # noqa: E402

PEN_GRID = [dict(hidden=h, layers=2, iters=it) for h, it in itertools.product((8, 16, 32), (200, 400, 800))]
MLP_GRID = [dict(hidden=h, layers=l, weight_decay=wd) for (h, l), wd in itertools.product(((16, 2), (48, 4)), (1e-5, 1e-3))]


def fit_predict(kind, cfg, train, test, seed=0):
    if kind == "pen":
        tr = PENTrainer(train, disc_cfg(), lr=0.02, seed=seed, **cfg).fit(verbose=False)
        th = tr.predict_theta(test)
        return predict_from_theta(test, {r.Mix: (r.log10_Dv, r.log10_K) for r in th.itertuples()})
    return DataDrivenMLP(seed=seed, **cfg).fit(train).predict(test)


def main():
    out = out_dir("inner_cv")
    d = load_data()
    train_all = d[~d.Mix.isin(TEST_MIXES)]
    test = d[d.Mix.isin(TEST_MIXES)]
    fams = sorted(set(family(m) for m in train_all.Mix))
    rows, t0 = [], time.time()
    for kind, grid in (("pen", PEN_GRID), ("mlp", MLP_GRID)):
        for i, cfg in enumerate(grid):
            pred = []
            for f in fams:
                tr = train_all[train_all.Mix.map(family) != f]
                va = train_all[train_all.Mix.map(family) == f]
                pred.append(va.assign(S_pred=fit_predict(kind, cfg, tr, va)))
            p = pd.concat(pred)
            rows.append({"kind": kind, "cfg": i, **cfg, **{f"inner_{k}": v for k, v in metrics(p.S_frac, p.S_pred).items()}})
            print(kind, cfg, f"inner rmse {rows[-1]['inner_rmse']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    R = pd.DataFrame(rows)
    sel = []
    for kind, g in R.groupby("kind"):
        best = g.loc[g.inner_rmse.idxmin()]
        cfg = {k: best[k] for k in (["hidden", "layers", "iters"] if kind == "pen" else ["hidden", "layers", "weight_decay"])}
        cfg = {k: (int(v) if k != "weight_decay" else float(v)) for k, v in cfg.items()}
        sc = [metrics(test.S_frac, fit_predict(kind, cfg, train_all, test, seed=s))["rmse"] for s in range(5)]
        paper = (g.hidden == (16 if kind == "pen" else 48)) & ((g.iters == 400) if kind == "pen" else (g.weight_decay == 1e-5))
        sel.append({"kind": kind, **{f"sel_{k}": v for k, v in cfg.items()}, "sel_inner_rmse": best.inner_rmse,
                    "paper_inner_rmse": float(g[paper].inner_rmse.iloc[0]), "inner_rank_of_paper": int((g.inner_rmse < g[paper].inner_rmse.iloc[0]).sum()) + 1,
                    "n_configs": len(g), "test_rmse_selected": float(np.mean(sc)), "test_rmse_selected_sd": float(np.std(sc, ddof=1))})
    R.to_csv(out / "inner.csv", index=False)
    S = pd.DataFrame(sel)
    S.to_csv(out / "selected.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
