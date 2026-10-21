"""Experiment E: robustness to measurement and data-processing uncertainty.

The training data (14 mixes of the hold-out split) are perturbed and every method is
re-trained; predictions are always scored against the *unperturbed* withheld curves.
Variants
    clean            reference
    iid_0.01/0.02    independent Gaussian noise on S
    corr_0.01        digitisation-like error: per-curve offset N(0, 0.005) + AR(1) noise
                     (rho = 0.9 between successive points, marginal s.d. 0.01)
    t0_0.25          time-origin error: each curve shifted by N(0, 0.25 d)
    shift_0.97/1.13  curve reconstruction with the time offset at the bounds of its IQR
    shift_train      offset estimated from the training mixes only (leakage check)
    B_only           only the short series (t <= 14 d)
    A_only           only the long series (time-corrected), no short series
Noise variants use 5 replicates (seed r); deterministic variants one replicate (PENN/MLP 5 seeds).
Output: results/q1/robustness/{scores.csv, theta.csv, summary.csv}
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import torch

from common import (NAMES, ROOT, TEST_MIXES, comp, fit_all_table, load_data, metrics, mix_table, out_dir,
                    predict_from_theta)
from exp_baselines import run_pen, two_stage

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from pinn_moisture.baselines import DataDrivenMLP  # noqa: E402

RAW = ROOT / "data/raw/tidy_dataset_correctedV1.csv"


def rebuild(shift=1.05, mode="merged"):
    """Re-assemble the reliable curves (w/b <= 0.5) from the raw file, as scripts/clean_data.py does."""
    df = pd.read_csv(RAW)
    df["block"] = (df.Mix != df.Mix.shift()).cumsum()
    df["series"] = np.where(df.groupby("block").Time_d.transform("max") <= 20.0, "B", "A")
    ref = load_data().drop_duplicates("Mix").set_index("Mix")
    rows = []
    for mix in ref.index:
        g = df[df.Mix == mix]
        a, b = g[g.series == "A"].copy(), g[g.series == "B"].copy()
        a["Time_d"] -= shift
        if mode == "merged":
            cur = pd.concat([b, a[a.Time_d > b.Time_d.max()]])
        elif mode == "B":
            cur = b
        else:
            cur = a
        cur = cur[cur.Time_d > 0].sort_values("Time_d")
        r = ref.loc[mix]
        rows.append(pd.DataFrame({"Mix": mix, "Time_d": cur.Time_d.values, "S_frac": cur.S_frac.values,
                                  "wb": r.wb, "FA_frac": r.FA_frac, "SL_frac": r.SL_frac}))
    return pd.concat(rows, ignore_index=True)


def train_shift():
    """Median A-B offset over the consistent *training* mixes (cleaning report, section 2)."""
    rep = (ROOT / "data/processed/cleaning_report.md").read_text().split("## 2.")[1].split("##")[0]
    vals = []
    for line in rep.splitlines():
        p = [x.strip() for x in line.split("|")]
        if len(p) > 5 and p[1] and p[1][0] in "CFS" and p[5] == "consistent" and p[1] not in TEST_MIXES:
            vals.append(float(p[2]))
    return float(np.median(vals))


def perturb(train, kind, rng):
    t = train.copy()
    if kind.startswith("iid"):
        t["S_frac"] = np.clip(t.S_frac + rng.normal(0, float(kind.split("_")[1]), len(t)), 0, 1)
    elif kind.startswith("corr"):
        sd = float(kind.split("_")[1])
        for m, idx in t.groupby("Mix").groups.items():
            n = len(idx)
            e = np.zeros(n)
            e[0] = rng.normal(0, sd)
            for i in range(1, n):
                e[i] = 0.9 * e[i - 1] + rng.normal(0, sd * np.sqrt(1 - 0.81))
            t.loc[idx, "S_frac"] = np.clip(t.loc[idx, "S_frac"] + e + rng.normal(0, 0.005), 0, 1)
    elif kind.startswith("t0"):
        sd = float(kind.split("_")[1])
        for m, idx in t.groupby("Mix").groups.items():
            t.loc[idx, "Time_d"] = np.maximum(t.loc[idx, "Time_d"] + rng.normal(0, sd), 0.05)
    return t


def main():
    out = out_dir("robustness")
    clean = load_data()
    test = clean[clean.Mix.isin(TEST_MIXES)]
    mt = mix_table(clean).set_index("Mix")
    tr_m = [m for m in dict.fromkeys(clean.Mix) if m not in TEST_MIXES]
    Xtr, Xte = comp(mt.loc[tr_m].reset_index()), comp(mt.loc[TEST_MIXES].reset_index())
    s_train = train_shift()
    variants = {"clean": ("data", clean), "shift_0.97": ("data", rebuild(0.97)), "shift_1.13": ("data", rebuild(1.13)),
                f"shift_train": ("data", rebuild(s_train)), "B_only": ("data", rebuild(mode="B")),
                "A_only": ("data", rebuild(mode="A")),
                "iid_0.01": ("noise", None), "iid_0.02": ("noise", None), "corr_0.01": ("noise", None),
                "t0_0.25": ("noise", None)}
    scores, thetas, t0 = [], [], time.time()
    for name, (kind, data) in variants.items():
        reps = range(5) if kind == "noise" else [0]
        for r in reps:
            rng = np.random.default_rng(500 + r)
            base = clean if kind == "noise" else data
            train = base[base.Mix.isin(tr_m)]
            if kind == "noise":
                train = perturb(train, name, rng)
            cls = fit_all_table(base if kind == "data" else pd.concat([train, base[base.Mix.isin(TEST_MIXES)]]))
            cls = cls.set_index("Mix")
            thetas += [{"variant": name, "rep": r, "method": "classical_fit", "Mix": m, "log10_Dv": cls.loc[m, NAMES[0]],
                        "log10_K": cls.loc[m, NAMES[1]]} for m in cls.index]
            Y, _ = two_stage("ols", Xtr, cls.loc[tr_m, NAMES].to_numpy(), Xte)
            th = {m: Y[i] for i, m in enumerate(TEST_MIXES)}
            scores.append({"variant": name, "rep": r, "seed": 0, "method": "ols", **metrics(test.S_frac, predict_from_theta(test, th))})
            seeds = [r] if kind == "noise" else range(5)
            for s in seeds:
                for pk in ("pen", "pen_r"):
                    thp, _ = run_pen(pk, train, mt.reset_index(), seed=s)
                    scores.append({"variant": name, "rep": r, "seed": s, "method": pk,
                                   **metrics(test.S_frac, predict_from_theta(test, thp))})
                    thetas += [{"variant": name, "rep": r, "seed": s, "method": pk, "Mix": m, "log10_Dv": v[0],
                                "log10_K": v[1]} for m, v in thp.items()]
                p = DataDrivenMLP(seed=s).fit(train).predict(test)
                scores.append({"variant": name, "rep": r, "seed": s, "method": "mlp", **metrics(test.S_frac, p)})
            print(f"{name} rep {r} ({time.time() - t0:.0f} s)", flush=True)
    S, T = pd.DataFrame(scores), pd.DataFrame(thetas)
    S.to_csv(out / "scores.csv", index=False)
    T.to_csv(out / "theta.csv", index=False)
    # parameter shifts relative to the clean reference
    ref = T[T.variant == "clean"].groupby(["method", "Mix"])[NAMES].mean()
    sh = []
    for (v, meth), g in T.groupby(["variant", "method"]):
        m = g.groupby("Mix")[NAMES].mean()
        dd = (m - ref.loc[meth].loc[m.index]).abs()
        sh.append({"variant": v, "method": meth, "max_dDv": dd.log10_Dv.max(), "mean_dDv": dd.log10_Dv.mean(),
                   "max_dK": dd.log10_K.max(), "mean_dK": dd.log10_K.mean()})
    summ = S.groupby(["variant", "method"]).agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), r2=("r2", "mean")).reset_index()
    summ = summ.merge(pd.DataFrame(sh).replace({"classical_fit": "ols"}), on=["variant", "method"], how="left")
    summ.to_csv(out / "summary.csv", index=False)
    (out / "shift_train.txt").write_text(f"{s_train:.4f}\n")
    pd.set_option("display.width", 200)
    print(f"offset from training mixes only: {s_train:.3f} d")
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
