"""Classical inverse identification (FD + least squares) for every mix.

Writes results/classical_fits.csv (parameters, standard errors, R2, RMSE, time)
and results/classical_pred.csv (fitted curves).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.inverse_fd import fit_mix  # noqa: E402
from pinn_moisture.physics import PhysicsConfig  # noqa: E402

DATA = pd.read_csv(ROOT / "data/processed/drying_clean.csv")


def run(mix):
    g = DATA[DATA.Mix == mix]
    r = fit_mix(g.Time_d.to_numpy(), g.S_frac.to_numpy(), PhysicsConfig(), n_starts=3)
    return mix, r


if __name__ == "__main__":
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows, preds = [], []
    with Pool(2) as pool:
        for mix, r in pool.imap(run, sorted(DATA.Mix.unique())):
            row = {"Mix": mix, "r2": r["r2"], "rmse": r["rmse"], "time_s": r["time_s"], "n_eval": r["n_eval"],
                   "cond": r["cond"], "corr_Dv_K": r["corr"][0][1]}
            row.update({k: v for k, v in r["theta"].items()})
            row.update({f"se_{k}": v for k, v in r["se"].items()})
            rows.append(row)
            g = DATA[DATA.Mix == mix]
            preds.append(pd.DataFrame({"Mix": mix, "Time_d": g.Time_d.values, "S_obs": g.S_frac.values,
                                       "S_fd": r["pred"]}))
            print(f"{mix:12s} R2={r['r2']:.4f} RMSE={r['rmse']:.4f} "
                  + " ".join(f"{k}={v:.3f}±{r['se'][k]:.3f}" for k, v in r["theta"].items()), flush=True)
    pd.DataFrame(rows).to_csv(out / "classical_fits.csv", index=False)
    pd.concat(preds).to_csv(out / "classical_pred.csv", index=False)
