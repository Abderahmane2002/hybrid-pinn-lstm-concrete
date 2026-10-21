"""Experiment D: accuracy against the number of training mixes.

For k in {5, 8, 11, 14} training mixes, 10 random subsets of the 20 mixes are drawn
(same subsets for every method); all remaining mixes are predicted. Methods:
two-stage ols / ridge / gp, end-to-end pen / pen_lin, and the data-driven mlp.
Output: results/q1/data_efficiency/{permix.csv, summary.csv}
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import torch

from common import NAMES, comp, disc_cfg, fit_all_table, load_data, metrics, mix_table, out_dir, predict_from_theta
from exp_baselines import run_pen, two_stage

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from pinn_moisture.baselines import DataDrivenMLP  # noqa: E402


def main(sizes=(5, 8, 11, 14), n_rep=10):
    out = out_dir("data_efficiency")
    d = load_data()
    mixes = list(dict.fromkeys(d.Mix))
    mt = mix_table(d).set_index("Mix")
    cls = fit_all_table(d).set_index("Mix")
    rng = np.random.default_rng(2026)
    rows, t0 = [], time.time()
    for k in sizes:
        for rep in range(n_rep):
            tr_m = list(rng.choice(mixes, k, replace=False))
            te_m = [m for m in mixes if m not in tr_m]
            train, test = d[d.Mix.isin(tr_m)], d[d.Mix.isin(te_m)]
            Xtr, Xte = comp(mt.loc[tr_m].reset_index()), comp(mt.loc[te_m].reset_index())
            preds = {}
            for kind in ("ols", "ridge", "gp"):
                Y, _ = two_stage(kind, Xtr, cls.loc[tr_m, NAMES].to_numpy(), Xte)
                preds[kind] = predict_from_theta(test, {m: Y[i] for i, m in enumerate(te_m)})
            for kind in ("pen", "pen_lin"):
                th, _ = run_pen(kind, train, mt.loc[te_m].reset_index(), seed=rep)
                preds[kind] = predict_from_theta(test, th)
            preds["mlp"] = DataDrivenMLP(seed=rep).fit(train).predict(test)
            for meth, p in preds.items():
                g = test.assign(S_pred=p)
                for m, gm in g.groupby("Mix"):
                    rows.append({"k": k, "rep": rep, "method": meth, "Mix": m, **metrics(gm.S_frac, gm.S_pred)})
                rows.append({"k": k, "rep": rep, "method": meth, "Mix": "__pooled__", **metrics(g.S_frac, g.S_pred)})
            print(f"k={k} rep={rep} ({time.time() - t0:.0f} s)", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "permix.csv", index=False)
    pooled = R[R.Mix == "__pooled__"]
    worst = R[R.Mix != "__pooled__"].groupby(["k", "rep", "method"]).r2.min().rename("r2_worst").reset_index()
    S = pooled.merge(worst, on=["k", "rep", "method"]).groupby(["k", "method"]).agg(
        rmse_med=("rmse", "median"), rmse_q25=("rmse", lambda x: x.quantile(.25)),
        rmse_q75=("rmse", lambda x: x.quantile(.75)), r2_med=("r2", "median"),
        r2_worst_med=("r2_worst", "median"), r2_worst_min=("r2_worst", "min")).reset_index()
    S.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(4).to_string(index=False))


def truncation(cutoffs=(3.0, 4.0, 7.0, 14.0, None)):
    """Training curves truncated at t <= cutoff (hold-out split); withheld curves complete."""
    from common import TEST_MIXES
    out = out_dir("data_efficiency")
    d = load_data()
    mt = mix_table(d).set_index("Mix")
    tr_m = [m for m in dict.fromkeys(d.Mix) if m not in TEST_MIXES]
    test = d[d.Mix.isin(TEST_MIXES)]
    Xtr, Xte = comp(mt.loc[tr_m].reset_index()), comp(mt.loc[TEST_MIXES].reset_index())
    cl_full = fit_all_table(d).set_index("Mix")
    rows = []
    for c in cutoffs:
        train = d[d.Mix.isin(tr_m) & (d.Time_d <= (c or 1e9))]
        npts = train.groupby("Mix").size()
        ok = [m for m in tr_m if npts.get(m, 0) >= 3]          # a 2-parameter fit needs >= 3 points
        cls = fit_all_table(train[train.Mix.isin(ok)]).set_index("Mix")
        Y, _ = two_stage("ols", comp(mt.loc[ok].reset_index()), cls.loc[ok, NAMES].to_numpy(), Xte)
        res = {"ols": [predict_from_theta(test, {m: Y[i] for i, m in enumerate(TEST_MIXES)})]}
        dth = {"ols": [(cls.loc[ok, NAMES] - cl_full.loc[ok, NAMES]).abs().mean()]}
        for kind in ("pen", "pen_lin"):
            res[kind], dth[kind] = [], []
            for s in range(5):
                th, _ = run_pen(kind, train, mt.reset_index(), seed=s)
                res[kind].append(predict_from_theta(test, th))
                dth[kind].append((pd.DataFrame(th, index=NAMES).T.loc[ok] - cl_full.loc[ok, NAMES]).abs().mean())
        res["mlp"] = [DataDrivenMLP(seed=s).fit(train).predict(test) for s in range(5)]
        for meth, ps in res.items():
            sc = [metrics(test.S_frac, p) for p in ps]
            row = {"cutoff_d": c or "full", "n_train_pts": len(train), "n_mix_fit": len(ok), "method": meth,
                   "rmse": np.mean([x["rmse"] for x in sc]), "rmse_sd": np.std([x["rmse"] for x in sc]),
                   "r2": np.mean([x["r2"] for x in sc])}
            if meth in dth:
                row["train_dDv_mae"] = float(np.mean([x["log10_Dv"] for x in dth[meth]]))
                row["train_dK_mae"] = float(np.mean([x["log10_K"] for x in dth[meth]]))
            rows.append(row)
        print("cutoff", c, "done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(out / "truncation.csv", index=False)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    import sys
    truncation() if "truncation" in sys.argv else main()
