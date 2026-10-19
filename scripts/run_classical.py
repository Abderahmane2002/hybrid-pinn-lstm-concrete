"""Classical inverse identification (BDF solver + least squares) for every mix.

python scripts/run_classical.py --geometry disc
Writes results/<geometry>/classical_fits.csv (parameters, standard errors, R2, RMSE, time)
and results/<geometry>/classical_pred.csv (fitted curves).
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.inverse_fd import fit_mix  # noqa: E402
from pinn_moisture.physics import config_for  # noqa: E402

DATA = pd.read_csv(ROOT / "data/processed/drying_clean.csv")


def run(job):
    geometry, mix = job
    g = DATA[DATA.Mix == mix]
    r = fit_mix(g.Time_d.to_numpy(), g.S_frac.to_numpy(), config_for(geometry), n_starts=3)
    return mix, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="disc")
    ap.add_argument("--all-mixes", action="store_true", help="also fit the w/b = 0.6 curves flagged unreliable")
    a = ap.parse_args()
    out = ROOT / "results" / a.geometry
    out.mkdir(parents=True, exist_ok=True)
    mixes = sorted(DATA.Mix.unique() if a.all_mixes else DATA[DATA.reliable].Mix.unique())
    rows, preds = [], []
    with Pool(2) as pool:
        for mix, r in pool.imap(run, [(a.geometry, m) for m in mixes]):
            row = {"Mix": mix, "r2": r["r2"], "rmse": r["rmse"], "time_s": r["time_s"], "n_eval": r["n_eval"],
                   "cond": r["cond"], "corr_Dv_K": r["corr"][0][1]}
            row.update(r["theta"])
            row.update({f"se_{k}": v for k, v in r["se"].items()})
            rows.append(row)
            g = DATA[DATA.Mix == mix]
            preds.append(pd.DataFrame({"Mix": mix, "Time_d": g.Time_d.values, "S_obs": g.S_frac.values,
                                       "S_fd": r["pred"]}))
            print(f"{mix:12s} R2={r['r2']:.4f} RMSE={r['rmse']:.4f} "
                  + " ".join(f"{k}={v:.3f}±{r['se'][k]:.3f}" for k, v in r["theta"].items())
                  + f" ({r['time_s']:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(out / "classical_fits.csv", index=False)
    pd.concat(preds).to_csv(out / "classical_pred.csv", index=False)
