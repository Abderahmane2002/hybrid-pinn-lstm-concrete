"""Why does the PENN shrink K_int towards the mean? (SYNTHETIC data only)

Experiment C2 showed that the PENN recovers the composition dependence of log10 K_int
less well than classical identification + OLS. Candidate causes: weight decay, the
number of iterations, the flexibility of the composition map, and the weak sensitivity
of S_avg to K_int compared with D_v. Each PENN variant is trained on the same synthetic
curves (truth_lin and truth_mix of exp_synthetic, iid noise 0.01, replicate 0) and scored
on parameter recovery for all 20 mixes. Selection is done on synthetic data only, never on
the withheld experimental mixes.
Output: results/q1/pen_shrinkage/{theta.csv, summary.csv}
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

from common import NAMES, ROOT, TEST_MIXES, comp, disc_cfg, load_data, mix_table, out_dir
from exp_robustness import perturb
from exp_synthetic import gen
from pinn_moisture.diff_solver import PENTrainer

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

VARIANTS = {
    "paper (2x16, 400 it, wd 1e-4)": dict(hidden=16, layers=2, iters=400, weight_decay=1e-4, lr=0.02),
    "2x16, 2000 it, wd 1e-4": dict(hidden=16, layers=2, iters=2000, weight_decay=1e-4, lr=0.02),
    "2x16, 2000 it, wd 0": dict(hidden=16, layers=2, iters=2000, weight_decay=0.0, lr=0.02),
    "linear, 400 it, wd 1e-4": dict(hidden=1, layers=0, iters=400, weight_decay=1e-4, lr=0.02),
    "linear, 2000 it, wd 0": dict(hidden=1, layers=0, iters=2000, weight_decay=0.0, lr=0.02),
    "linear, 4000 it, wd 0, lr 0.05": dict(hidden=1, layers=0, iters=4000, weight_decay=0.0, lr=0.05),
    "linear, L-BFGS, wd 0 (revised)": dict(hidden=1, layers=0, iters=500, weight_decay=0.0, optimizer="lbfgs"),
}
import sys
if "--only-new" in sys.argv:
    VARIANTS = {k: v for k, v in VARIANTS.items() if "L-BFGS" in k}


def main():
    out = out_dir("pen_shrinkage")
    d = load_data()
    mixes = list(dict.fromkeys(d.Mix))
    mt = mix_table(d).set_index("Mix")
    X = comp(mt.loc[mixes].reset_index())
    cl = pd.read_csv(ROOT / "results/disc/classical_fits.csv").set_index("Mix").loc[mixes, NAMES].to_numpy()
    A = np.column_stack([np.ones(len(X)), X])
    truths = {"truth_mix": cl, "truth_lin": A @ np.linalg.lstsq(A, cl, rcond=None)[0]}
    tr_m = [m for m in mixes if m not in TEST_MIXES]
    rows = []
    for tname, T in truths.items():
        truth = {m: T[i] for i, m in enumerate(mixes)}
        clean = pd.concat([d[d.Mix == m].assign(S_frac=gen(truth[m], d[d.Mix == m].Time_d.to_numpy())) for m in mixes])
        noisy = perturb(clean, "iid_0.01", np.random.default_rng(700))
        train = noisy[noisy.Mix.isin(tr_m)]
        for vname, kw in VARIANTS.items():
            for s in range(3):
                kw2 = dict(kw)
                opt = kw2.pop("optimizer", "adam")
                tr = PENTrainer(train, disc_cfg(), seed=s, **kw2).fit(verbose=False, optimizer=opt)
                th = tr.predict_theta(mt.reset_index()).set_index("Mix")
                for m in mixes:
                    rows.append({"truth": tname, "variant": vname, "seed": s, "Mix": m,
                                 "split": "test" if m in TEST_MIXES else "train",
                                 "Dv_true": truth[m][0], "K_true": truth[m][1],
                                 "Dv_est": th.loc[m, "log10_Dv"], "K_est": th.loc[m, "log10_K"],
                                 "final_mse": tr.history[-1]["mse"]})
            print(tname, vname, flush=True)
    R = pd.DataFrame(rows)
    if "--only-new" in sys.argv and (out / "theta.csv").exists():
        R = pd.concat([pd.read_csv(out / "theta.csv"), R])
        R = R.drop_duplicates(["truth", "variant", "seed", "Mix"], keep="last")
    R.to_csv(out / "theta.csv", index=False)
    g = R.groupby(["truth", "variant", "split"])
    S = g.apply(lambda x: pd.Series({
        "mae_K": (x.K_est - x.K_true).abs().mean(), "r_K": np.corrcoef(x.K_true, x.K_est)[0, 1],
        "slope_K": np.polyfit(x.K_true, x.K_est, 1)[0], "mae_Dv": (x.Dv_est - x.Dv_true).abs().mean(),
        "train_rmse": np.sqrt(x.final_mse.mean())})).reset_index()
    S.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
