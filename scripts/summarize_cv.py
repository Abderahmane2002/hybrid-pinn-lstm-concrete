"""Summarise the cross-validation runs (results/<geometry>/cv) into tables.

Outputs: results/<geometry>/cv_summary.csv (pooled metrics per scheme and method)
         results/<geometry>/cv_per_mix.csv (R2 and RMSE of each held-out mix)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def metrics(o, p):
    r = p - o
    return {"r2": float(1 - np.sum(r ** 2) / np.sum((o - o.mean()) ** 2)), "rmse": float(np.sqrt(np.mean(r ** 2))),
            "mae": float(np.mean(np.abs(r)))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="disc")
    a = ap.parse_args()
    res = ROOT / "results" / a.geometry
    files = [f for f in (res / "cv").glob("*.csv") if not f.name.startswith("theta_")]
    d = pd.concat([pd.read_csv(f) for f in files])
    rows, per_mix = [], []
    for (scheme, method), g in d.groupby(["scheme", "method"]):
        m = metrics(g.S_obs.to_numpy(), g.S_pred.to_numpy())
        # median of the per-mix R2 and number of mixes below 0.95
        r2_mix = [metrics(h.S_obs.to_numpy(), h.S_pred.to_numpy())["r2"] for _, h in g.groupby("Mix")]
        rows.append({"scheme": scheme, "method": method, "n_mixes": g.Mix.nunique(), "n_points": len(g), **m,
                     "r2_mix_median": float(np.median(r2_mix)), "r2_mix_min": float(np.min(r2_mix)),
                     "n_mix_r2_below_0.95": int(np.sum(np.array(r2_mix) < 0.95))})
        for mix, h in g.groupby("Mix"):
            per_mix.append({"scheme": scheme, "method": method, "Mix": mix,
                            **metrics(h.S_obs.to_numpy(), h.S_pred.to_numpy())})
    s = pd.DataFrame(rows)
    s.to_csv(res / "cv_summary.csv", index=False)
    pd.DataFrame(per_mix).to_csv(res / "cv_per_mix.csv", index=False)
    print(s.round(4).to_string(index=False))
    (res / "cv_summary.json").write_text(json.dumps(rows, indent=1))
