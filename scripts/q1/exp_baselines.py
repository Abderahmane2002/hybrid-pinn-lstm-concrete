"""Experiment F (+A): extended model comparison under identical folds.

Two-stage methods (classical per-mix fit on the training mixes -> regression of
theta on composition -> forward solution):
    ols      linear least squares (the paper's 'classical + regression')
    ridge    linear ridge, penalty chosen by leave-one-out CV on the training mixes
    quad     quadratic features + ridge (inner LOO)
    gp       Gaussian-process regression (RBF ARD + white noise, ML-II)
End-to-end methods (composition -> theta -> solver, trained on the curves):
    pen      PENN as in the paper (2 x 16 tanh)
    pen_s    reduced PENN (1 x 8 tanh)
    pen_lin  linear PENN (theta logits linear in composition)
Data-driven reference: mlp (paper settings; predictions re-used from results/disc).

Schemes: holdout (6 mixes), lomo (20 folds), lofo (7 folds). Per-mix classical
fits use only the data of that mix, so fitting them once on all mixes does not
leak information across folds; regressions are always fitted on training mixes only.
Outputs: results/q1/baselines/{pred_<scheme>.csv, theta_<scheme>.csv, permix.csv, pooled.csv, timing.csv}
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures

from common import (NAMES, ROOT, TEST_MIXES, comp, disc_cfg, fit_all_table, folds, load_data, metrics,
                    mix_table, out_dir, per_mix_metrics, predict_from_theta)

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from pinn_moisture.diff_solver import PENTrainer  # noqa: E402

ALPHAS = np.logspace(-4, 1, 11)


def ridge_loo(X, y):
    best = None
    for a in ALPHAS:
        p = cross_val_predict(Ridge(alpha=a), X, y, cv=LeaveOneOut())
        e = np.mean((p - y) ** 2)
        if best is None or e < best[0]:
            best = (e, a)
    return Ridge(alpha=best[1]).fit(X, y), best[1]


def two_stage(kind, Xtr, Ytr, Xte):
    if kind == "ols":
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        coef = np.linalg.lstsq(A, Ytr, rcond=None)[0]
        return np.column_stack([np.ones(len(Xte)), Xte]) @ coef, {}
    if kind in ("ridge", "quad"):
        if kind == "quad":
            pf = PolynomialFeatures(2, include_bias=False).fit(Xtr)
            Xtr, Xte = pf.transform(Xtr), pf.transform(Xte)
        out, info = [], {}
        for j in range(Ytr.shape[1]):
            m, a = ridge_loo(Xtr, Ytr[:, j])
            out.append(m.predict(Xte))
            info[f"alpha_{j}"] = a
        return np.column_stack(out), info
    if kind == "gp":
        out = []
        for j in range(Ytr.shape[1]):
            k = ConstantKernel(1.0, (1e-3, 1e2)) * RBF([1.0] * Xtr.shape[1], (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-6, 1e0))
            gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(Xtr, Ytr[:, j])
            out.append(gp.predict(Xte))
        return np.column_stack(out), {}
    raise ValueError(kind)


PEN_ARCH = {"pen": dict(hidden=16, layers=2), "pen_s": dict(hidden=8, layers=1), "pen_lin": dict(hidden=1, layers=0)}


def run_pen(kind, train, test_mixes_df, seed):
    tr = PENTrainer(train, disc_cfg(), lr=0.02, iters=400, seed=seed, **PEN_ARCH[kind]).fit(verbose=False)
    th = tr.predict_theta(test_mixes_df)
    return {r.Mix: (r.log10_Dv, r.log10_K) for r in th.itertuples()}, tr.train_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["holdout", "lomo", "lofo"])
    ap.add_argument("--holdout-seeds", type=int, default=5)
    a = ap.parse_args()
    out = out_dir("baselines")
    d = load_data()
    mixes = list(dict.fromkeys(d.Mix))
    mt = mix_table(d).set_index("Mix")
    t0 = time.time()
    cls = fit_all_table(d).set_index("Mix")                       # per-mix classical fits
    t_cls = (time.time() - t0) / len(cls)
    cls.to_csv(out / "classical_table_fits.csv")
    timing = []
    for scheme in a.schemes:
        preds, thetas = [], []
        for fold, test_m in folds(scheme, mixes):
            train = d[~d.Mix.isin(test_m)]
            test = d[d.Mix.isin(test_m)]
            tr_m = [m for m in mixes if m not in test_m]
            Xtr = comp(mt.loc[tr_m].reset_index())
            Xte = comp(mt.loc[test_m].reset_index())
            Ytr = cls.loc[tr_m, NAMES].to_numpy()
            for kind in ("ols", "ridge", "quad", "gp"):
                t1 = time.time()
                Y, _ = two_stage(kind, Xtr, Ytr, Xte)
                dt = time.time() - t1 + t_cls * len(tr_m)
                th = {m: Y[i] for i, m in enumerate(test_m)}
                preds.append(test.assign(method=kind, seed=0, fold=fold, S_pred=predict_from_theta(test, th)))
                thetas += [{"method": kind, "seed": 0, "fold": fold, "Mix": m, "log10_Dv": th[m][0],
                            "log10_K": th[m][1]} for m in test_m]
                timing.append({"scheme": scheme, "method": kind, "time_s": dt})
            seeds = range(a.holdout_seeds) if scheme == "holdout" else [0]
            for kind in PEN_ARCH:
                for seed in seeds:
                    th, dt = run_pen(kind, train, mt.loc[test_m].reset_index(), seed)
                    preds.append(test.assign(method=kind, seed=seed, fold=fold, S_pred=predict_from_theta(test, th)))
                    thetas += [{"method": kind, "seed": seed, "fold": fold, "Mix": m, "log10_Dv": th[m][0],
                                "log10_K": th[m][1]} for m in test_m]
                    timing.append({"scheme": scheme, "method": kind, "time_s": dt})
            print(f"{scheme} {fold} done ({time.time() - t0:.0f} s)", flush=True)
        P = pd.concat(preds).rename(columns={"S_frac": "S_obs"})[["method", "seed", "fold", "Mix", "Time_d", "S_obs", "S_pred"]]
        # data-driven reference, same folds (re-used from the main runs)
        if scheme == "holdout":
            for s in range(5):
                q = pd.read_csv(ROOT / f"results/disc/runs/mlp_seed{s}/pred.csv")
                q = q[q.split == "test"]
                P = pd.concat([P, q.assign(method="mlp", seed=s, fold="holdout")[P.columns]])
        else:
            for f in (ROOT / "results/disc/cv").glob(f"{scheme}_*_mlp.csv"):
                q = pd.read_csv(f)
                P = pd.concat([P, q.assign(method="mlp", seed=0, fold=q.fold)[P.columns]])
        P.to_csv(out / f"pred_{scheme}.csv", index=False)
        pd.DataFrame(thetas).to_csv(out / f"theta_{scheme}.csv", index=False)
    pd.DataFrame(timing).groupby(["scheme", "method"]).time_s.mean().reset_index().to_csv(out / "timing.csv", index=False)
    summarize(out, a.schemes)


def summarize(out, schemes):
    rows, permix = [], []
    for scheme in schemes:
        P = pd.read_csv(out / f"pred_{scheme}.csv")
        for (meth, seed), g in P.groupby(["method", "seed"]):
            pm = per_mix_metrics(g).assign(scheme=scheme, method=meth, seed=seed)
            permix.append(pm)
            rows.append({"scheme": scheme, "method": meth, "seed": seed, **metrics(g.S_obs, g.S_pred),
                         "r2_mix_min": pm.r2.min(), "r2_mix_median": pm.r2.median(),
                         "rmse_mix_mean": pm.rmse.mean(), "n_mix_below_095": int((pm.r2 < 0.95).sum())})
    R = pd.DataFrame(rows)
    R.to_csv(out / "pooled_by_seed.csv", index=False)
    agg = R.groupby(["scheme", "method"]).agg(r2=("r2", "mean"), r2_sd=("r2", "std"), rmse=("rmse", "mean"),
                                               rmse_sd=("rmse", "std"), rmse_mix_mean=("rmse_mix_mean", "mean"),
                                               r2_mix_min=("r2_mix_min", "mean"),
                                               n_below=("n_mix_below_095", "mean")).reset_index()
    agg.to_csv(out / "pooled.csv", index=False)
    pd.concat(permix).to_csv(out / "permix.csv", index=False)
    pd.set_option("display.width", 200)
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
