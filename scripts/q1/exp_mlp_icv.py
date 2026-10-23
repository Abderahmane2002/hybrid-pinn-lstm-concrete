"""Data-driven baseline with the configuration selected by the inner cross-validation
(4 x 48 tanh, weight decay 1e-3; results/q1/inner_cv/selected.csv), evaluated under exactly the
same folds as every other model. Its predictions are added to results/q1/baselines/pred_<scheme>.csv
as method "mlp_icv" and the summary tables are rebuilt.
"""
from __future__ import annotations

import time

import pandas as pd
import torch

from common import ROOT, folds, load_data, out_dir
from exp_baselines import summarize

torch.set_num_threads(1)
from pinn_moisture.baselines import DataDrivenMLP  # noqa: E402


def main():
    sel = pd.read_csv(ROOT / "results/q1/inner_cv/selected.csv").set_index("kind").loc["mlp"]
    cfg = dict(hidden=int(sel.sel_hidden), layers=int(sel.sel_layers), weight_decay=float(sel.sel_weight_decay))
    print("inner-CV configuration:", cfg, flush=True)
    out = out_dir("baselines")
    d = load_data()
    mixes = list(dict.fromkeys(d.Mix))
    t0 = time.time()
    for scheme in ("holdout", "lomo", "lofo"):
        P = pd.read_csv(out / f"pred_{scheme}.csv")
        P = P[P.method != "mlp_icv"]
        new = []
        for fold, test_m in folds(scheme, mixes):
            train, test = d[~d.Mix.isin(test_m)], d[d.Mix.isin(test_m)]
            for seed in (range(5) if scheme == "holdout" else [0]):
                p = DataDrivenMLP(seed=seed, **cfg).fit(train).predict(test)
                new.append(pd.DataFrame({"method": "mlp_icv", "seed": seed, "fold": fold, "Mix": test.Mix.values,
                                         "Time_d": test.Time_d.values, "S_obs": test.S_frac.values, "S_pred": p}))
            print(scheme, fold, f"({time.time() - t0:.0f} s)", flush=True)
        pd.concat([P] + new).to_csv(out / f"pred_{scheme}.csv", index=False)
    summarize(out, ["holdout", "lomo", "lofo"])


if __name__ == "__main__":
    main()
