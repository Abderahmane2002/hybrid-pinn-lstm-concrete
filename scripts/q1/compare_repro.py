"""Stage 1: compare an independent re-run of the original pipeline (clean clone, results
regenerated from the raw data) with the results reported in the manuscript.

usage: python scripts/q1/compare_repro.py /path/to/clean/clone
Output: results/q1/reproducibility/{comparison.csv, report.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, out_dir


def main(repro: Path):
    out = out_dir("reproducibility")
    rows = []

    def add(item, orig, new, tol, unit=""):
        ok = (abs(orig - new) <= tol) if np.isfinite(orig) and np.isfinite(new) else False
        rows.append({"item": item, "manuscript": orig, "reproduced": new, "abs_diff": abs(orig - new),
                     "tolerance": tol, "status": "reproduced" if ok else "DIFFERENT"})

    O, N = ROOT / "results", repro / "results"
    # data
    for f, key in (("data/processed/drying_clean.csv", None),):
        a, b = pd.read_csv(ROOT / f), pd.read_csv(repro / f)
        add("cleaned data: number of points", len(a), len(b), 0)
        add("cleaned data: max |S difference|", 0.0, float((a.S_frac - b.S_frac).abs().max()), 1e-12)
        add("cleaned data: max |t difference| [d]", 0.0, float((a.Time_d - b.Time_d).abs().max()), 1e-12)
    # classical fits
    for geo in ("disc", "slab"):
        a = pd.read_csv(O / geo / "classical_fits.csv").set_index("Mix")
        b = pd.read_csv(N / geo / "classical_fits.csv").set_index("Mix")
        common = a.index.intersection(b.index)          # the original slab file also holds the w/b = 0.6 fits
        a, b = a.loc[common], b.loc[common]
        add(f"{geo}: classical log10 D_v, max |diff|", 0.0, float((a.log10_Dv - b.log10_Dv).abs().max()), 0.005)
        add(f"{geo}: classical log10 K_int, max |diff|", 0.0, float((a.log10_K - b.log10_K).abs().max()), 0.02)
        add(f"{geo}: classical median R2", float(a.r2.median()), float(b.r2.median()), 1e-4)
    # hold-out metrics
    for geo, meths in (("disc", ("pen", "mlp", "fdreg", "fdfit")), ("slab", ("pen", "mlp", "fdreg", "pinn", "pinn_nopde"))):
        ma = json.loads((O / geo / "metrics.json").read_text())
        mb = json.loads((N / geo / "metrics.json").read_text())
        for m in meths:
            if m not in mb["methods_test"]:
                continue
            for k, tol in (("r2_mean", 2e-3 if "pinn" in m else 2e-4), ("rmse_mean", 2e-3 if "pinn" in m else 2e-4)):
                add(f"{geo}: {m} hold-out {k}", ma["methods_test"][m][k], mb["methods_test"][m][k], tol)
        for k in ("forward_pen_fd_vs_model_all", "forward_pinn_fd_vs_model_all"):
            if k in ma and k in mb:
                add(f"{geo}: solver check {k.split('_')[1]} R2", ma[k]["r2"], mb[k]["r2"], 0.02 if "pinn" in k else 1e-3)
    # cross-validation
    a = pd.read_csv(O / "disc/cv_summary.csv").set_index(["scheme", "method"])
    b = pd.read_csv(N / "disc/cv_summary.csv").set_index(["scheme", "method"])
    for idx in a.index:
        add(f"CV {idx[0]} {idx[1]} pooled R2", a.loc[idx, "r2"], b.loc[idx, "r2"], 5e-4)
        add(f"CV {idx[0]} {idx[1]} worst-mix R2", a.loc[idx, "r2_mix_min"], b.loc[idx, "r2_mix_min"], 0.01)
    # numerics: compare every number in the verification files
    import re
    for f, tol in (("numerical_verification.md", 0.05), ("geometry_check.csv", 0.01)):
        if (N / f).exists():
            xa = np.array([float(x) for x in re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", (O / f).read_text())])
            xb = np.array([float(x) for x in re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", (N / f).read_text())])
            if len(xa) == len(xb):
                rel = float(np.max(np.abs(xa - xb) / np.maximum(np.abs(xa), 1e-12)))
                rows.append({"item": f"{f}: largest relative difference of all numbers", "manuscript": 0.0, "reproduced": rel,
                             "abs_diff": rel, "tolerance": tol, "status": "reproduced" if rel <= tol else "DIFFERENT"})
            else:
                rows.append({"item": f"{f}: number count", "manuscript": len(xa), "reproduced": len(xb),
                             "abs_diff": abs(len(xa) - len(xb)), "tolerance": 0, "status": "DIFFERENT"})
    R = pd.DataFrame(rows)
    R.to_csv(out / "comparison.csv", index=False)
    lines = ["# Reproducibility check", "",
             f"Independent re-run from a clean clone (`{repro}`), all results regenerated from the raw data.", "",
             "| Item | Manuscript | Re-run | |diff| | Status |", "|---|---|---|---|---|"]
    lines += [f"| {r.item} | {r.manuscript:.6g} | {r.reproduced:.6g} | {r.abs_diff:.2g} | {r.status} |" for r in R.itertuples()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
