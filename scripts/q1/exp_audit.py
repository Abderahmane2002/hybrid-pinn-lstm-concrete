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

from common import ROOT, TEST_MIXES, family, out_dir

MARGIN = 0.002   # practical equivalence margin on the per-mix RMSE (about 1/3 of the median direct-fit RMSE)
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
        fam = np.array([family(m) for m in w.index])
        ufam = np.unique(fam)
        for a, b in (("pen", "ols"), ("pen_r", "ols"), ("pen", "ridge"), ("pen", "pen_lin"), ("pen", "pen_r"),
                     ("pen", "mlp"), ("pen_r", "mlp"), ("pen", "mlp_icv"), ("ols", "mlp_icv"), ("pen", "gp"),
                     ("pen", "quad"), ("ols", "mlp"), ("ols", "gp"), ("ols", "quad")):
            if a not in w or b not in w:
                continue
            diff = (w[a] - w[b]).to_numpy()
            boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
            # cluster bootstrap: resample binder families, keep all mixes of a drawn family
            fb = []
            for _ in range(10000):
                pick = rng.choice(ufam, len(ufam), replace=True)
                fb.append(np.concatenate([diff[fam == f] for f in pick]).mean())
            fb = np.array(fb)
            fmeans = np.array([diff[fam == f].mean() for f in ufam])
            lo90, hi90 = np.percentile(fb, 5), np.percentile(fb, 95)
            res.append({"scheme": scheme, "a": a, "b": b, "mean_rmse_a": w[a].mean(), "mean_rmse_b": w[b].mean(),
                        "mean_diff": diff.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                        "a_better_in": int((diff < 0).sum()), "n_mixes": len(diff),
                        "wilcoxon_p": float(wilcoxon(diff).pvalue),
                        "fam_ci_lo": np.percentile(fb, 2.5), "fam_ci_hi": np.percentile(fb, 97.5),
                        "fam_wilcoxon_p": float(wilcoxon(fmeans).pvalue), "fam_better_in": int((fmeans < 0).sum()),
                        "n_families": len(ufam), "fam_ci90_lo": lo90, "fam_ci90_hi": hi90,
                        "equivalent_within_margin": bool(lo90 > -MARGIN and hi90 < MARGIN)})
    R = pd.DataFrame(res)
    R.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
