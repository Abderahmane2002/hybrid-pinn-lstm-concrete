"""Experiment C: recovery of known parameters from SYNTHETIC drying curves.

All curves in this experiment are generated with the disc solver on a finer mesh
(36 x 24 cells) than the one used for identification (tabulated 24 x 16 solutions),
so that the inversion does not use exactly the generating model. They are labelled
synthetic everywhere and are never mixed with the experimental data.

C1  single-curve inversion: 6 known parameter sets spanning the experimental range,
    sampled at the measurement times of C-0.4, under
      exact | iid noise 0.01 | correlated (digitisation-like) noise 0.01 |
      time-origin error 0.25 d | data limited to t <= 14 d (exact)
    3 replicates for the noisy conditions.
C2  composition-to-parameter learning: synthetic curves for the 20 mixes (measurement
    times of each mix) generated from
      truth_lin  : parameters linear in composition (OLS fit of the classical parameters)
      truth_mix  : the per-mix classical parameters (includes scatter not explained by composition)
    with iid noise 0.01. PENN (5 seeds) and classical identification + OLS are trained on the
    14 training mixes; the recovered parameters are compared with the truth.
Output: results/q1/synthetic/{c1.csv, c2_theta.csv, c2_scores.csv, summary_c1.csv, summary_c2.csv}
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

from common import (NAMES, ROOT, TEST_MIXES, comp, fit_table, load_data, metrics, mix_table, out_dir,
                    predict_from_theta)
from exp_baselines import run_pen, two_stage
from exp_robustness import perturb
from pinn_moisture.axisym import AxisymBDF
from pinn_moisture.physics import config_for

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
GEN = AxisymBDF(config_for("disc"), nr=36, nz=24, ratio=1.08)


def gen(theta, t):
    return GEN.solve({"log10_Dv": theta[0], "log10_K": theta[1]}, t, rtol=1e-8, atol=1e-11)[0]


def c1(out):
    d = load_data()
    t = d[d.Mix == "C-0.4"].Time_d.to_numpy()
    sets = [(dv, k) for dv in (-9.2, -8.8, -8.4) for k in (-20.0, -19.4)]
    rows = []
    for i, th in enumerate(sets):
        s_true = gen(th, t)
        conds = [("exact", 0)] + [(c, r) for c in ("iid_0.01", "corr_0.01", "t0_0.25") for r in range(3)] + [("short_14d", 0)]
        for cond, r in conds:
            rng = np.random.default_rng(100 * i + r)
            df = pd.DataFrame({"Mix": "syn", "Time_d": t, "S_frac": s_true})
            if cond in ("iid_0.01", "corr_0.01", "t0_0.25"):
                df = perturb(df, cond, rng)
            if cond == "short_14d":
                df = df[df.Time_d <= 14]
            est, rmse = fit_table(df.Time_d.to_numpy(), df.S_frac.to_numpy())
            rows.append({"set": i, "Dv_true": th[0], "K_true": th[1], "cond": cond, "rep": r,
                         "Dv_est": est[0], "K_est": est[1], "rmse_fit": rmse})
    C = pd.DataFrame(rows)
    C["err_Dv"], C["err_K"] = C.Dv_est - C.Dv_true, C.K_est - C.K_true
    C.to_csv(out / "c1.csv", index=False)
    S = C.groupby("cond").agg(maxabs_Dv=("err_Dv", lambda x: x.abs().max()), mean_Dv=("err_Dv", "mean"),
                              maxabs_K=("err_K", lambda x: x.abs().max()), mean_K=("err_K", "mean"),
                              rmse_fit=("rmse_fit", "mean")).reset_index()
    S.to_csv(out / "summary_c1.csv", index=False)
    print(S.round(4).to_string(index=False))


def c2(out):
    d = load_data()
    mixes = list(dict.fromkeys(d.Mix))
    mt = mix_table(d).set_index("Mix")
    X = comp(mt.loc[mixes].reset_index())
    cl = pd.read_csv(ROOT / "results/disc/classical_fits.csv").set_index("Mix").loc[mixes, NAMES].to_numpy()
    A = np.column_stack([np.ones(len(X)), X])
    truths = {"truth_mix": cl, "truth_lin": A @ np.linalg.lstsq(A, cl, rcond=None)[0]}
    tr_m = [m for m in mixes if m not in TEST_MIXES]
    Xtr, Xte = comp(mt.loc[tr_m].reset_index()), comp(mt.loc[TEST_MIXES].reset_index())
    th_rows, sc_rows = [], []
    for tname, T in truths.items():
        truth = {m: T[i] for i, m in enumerate(mixes)}
        clean = pd.concat([d[d.Mix == m].assign(S_frac=gen(truth[m], d[d.Mix == m].Time_d.to_numpy())) for m in mixes])
        test_true = clean[clean.Mix.isin(TEST_MIXES)]
        for r in range(3):
            noisy = perturb(clean, "iid_0.01", np.random.default_rng(700 + r))
            train = noisy[noisy.Mix.isin(tr_m)]
            # classical + OLS
            fits = {m: fit_table(g.Time_d.to_numpy(), g.S_frac.to_numpy())[0] for m, g in train.groupby("Mix")}
            Y, _ = two_stage("ols", Xtr, np.array([fits[m] for m in tr_m]), Xte)
            est = {**fits, **{m: Y[i] for i, m in enumerate(TEST_MIXES)}}
            th_rows += [{"truth": tname, "rep": r, "seed": 0, "method": "classical+ols", "Mix": m,
                         "Dv_true": truth[m][0], "K_true": truth[m][1], "Dv_est": est[m][0], "K_est": est[m][1]} for m in mixes]
            sc_rows.append({"truth": tname, "rep": r, "seed": 0, "method": "classical+ols",
                            **metrics(test_true.S_frac, predict_from_theta(test_true, est))})
            for s in range(5 if r == 0 else 1):
                thp, _ = run_pen("pen", train, mt.reset_index(), seed=s)
                th_rows += [{"truth": tname, "rep": r, "seed": s, "method": "pen", "Mix": m, "Dv_true": truth[m][0],
                             "K_true": truth[m][1], "Dv_est": thp[m][0], "K_est": thp[m][1]} for m in mixes]
                sc_rows.append({"truth": tname, "rep": r, "seed": s, "method": "pen",
                                **metrics(test_true.S_frac, predict_from_theta(test_true, thp))})
            print(f"{tname} rep {r} done", flush=True)
    Tt = pd.DataFrame(th_rows)
    Tt["split"] = np.where(Tt.Mix.isin(TEST_MIXES), "test", "train")
    Tt["err_Dv"], Tt["err_K"] = Tt.Dv_est - Tt.Dv_true, Tt.K_est - Tt.K_true
    Tt.to_csv(out / "c2_theta.csv", index=False)
    pd.DataFrame(sc_rows).to_csv(out / "c2_scores.csv", index=False)
    S = Tt.groupby(["truth", "method", "split"]).agg(
        mae_Dv=("err_Dv", lambda x: x.abs().mean()), mae_K=("err_K", lambda x: x.abs().mean()),
        bias_K=("err_K", "mean")).reset_index()
    corr = Tt.groupby(["truth", "method", "split"]).apply(
        lambda g: pd.Series({"r_Dv": np.corrcoef(g.Dv_true, g.Dv_est)[0, 1], "r_K": np.corrcoef(g.K_true, g.K_est)[0, 1]})).reset_index()
    S = S.merge(corr, on=["truth", "method", "split"])
    sc = pd.DataFrame(sc_rows).groupby(["truth", "method"]).rmse.mean().rename("test_rmse_vs_truth").reset_index()
    S = S.merge(sc, on=["truth", "method"])
    S.to_csv(out / "summary_c2.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    o = out_dir("synthetic")
    c1(o)
    c2(o)
