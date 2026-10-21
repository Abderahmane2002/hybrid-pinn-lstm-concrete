"""Experiment A: audit of shared choices and mix-level statistics.

1. Shape exponent n: the original profile used six mixes, one of which (C-0.5) is a
   withheld mix. The profile is recomputed from the stored per-mix fits without it.
2. Time offset of the long series: recomputed from the training mixes only
   (see exp_robustness.train_shift).
3. Mix-level paired comparison of the methods in leave-one-mix-out and
   leave-one-family-out CV: per-mix RMSE differences, bootstrap over mixes (10 000
   resamples) and Wilcoxon signed-rank test. Observations of one curve are not
   independent, so the mix (not the point) is the statistical unit.
Output: results/q1/audit/{profile_n_train.csv, paired.csv}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import ROOT, TEST_MIXES, out_dir
from exp_robustness import train_shift


def main():
    out = out_dir("audit")
    pr = pd.read_csv(ROOT / "results/disc/profile_n_runs.csv")
    used = sorted(pr.mix.unique())
    rows = []
    for label, sub in (("all six (original)", pr), ("training mixes only", pr[~pr.mix.isin(TEST_MIXES)])):
        p = sub.groupby("n").sse.sum()
        rows += [{"set": label, "mixes": ", ".join(sorted(sub.mix.unique())), "n": n, "sse": v,
                  "sse_rel_to_n8": v / p.loc[8.0]} for n, v in p.items()]
    P = pd.DataFrame(rows)
    P.to_csv(out / "profile_n_train.csv", index=False)
    print("mixes in the original profile:", used)
    print(P.pivot(index="n", columns="set", values="sse_rel_to_n8").round(3).to_string())
    print(f"time offset, training mixes only: {train_shift():.3f} d (paper: 1.05 d)")

    pm = pd.read_csv(ROOT / "results/q1/baselines/permix.csv")
    pm = pm[pm.seed == 0]
    rng = np.random.default_rng(0)
    res = []
    for scheme in ("lomo", "lofo"):
        w = pm[pm.scheme == scheme].pivot(index="Mix", columns="method", values="rmse")
        for a, b in (("pen", "ols"), ("pen", "ridge"), ("pen", "pen_lin"), ("pen", "mlp"), ("pen", "gp"),
                     ("pen", "quad"), ("ols", "mlp")):
            diff = (w[a] - w[b]).to_numpy()
            boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
            res.append({"scheme": scheme, "a": a, "b": b, "mean_rmse_a": w[a].mean(), "mean_rmse_b": w[b].mean(),
                        "mean_diff": diff.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                        "a_better_in": int((diff < 0).sum()), "n_mixes": len(diff),
                        "wilcoxon_p": float(wilcoxon(diff).pvalue)})
    R = pd.DataFrame(res)
    R.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
