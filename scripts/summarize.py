"""Collect all runs into the tables and numbers used in the manuscript.

Outputs (results/):
  table_methods.csv      Table 5: methods on unseen mixes, mean +- sd over seeds
  table_test_mixes.csv   Table 4: PINN accuracy per held-out mix
  theta_all.csv          parameters per mix: PINN (mean, sd over seeds) and classical FD
  forward_verification.csv  FD solution with PINN parameters vs PINN vs data
  fdreg_pred.csv         predictions of the FD + composition-regression baseline
  metrics.json           every number quoted in the text
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.baselines import fd_regression  # noqa: E402
from pinn_moisture.fd_solver import FDSolver  # noqa: E402
from pinn_moisture.physics import PhysicsConfig  # noqa: E402

RES = ROOT / "results"
TEST_MIXES = ["C-0.4", "C-0.5", "F10S10-0.3", "F20-0.3", "F20S20-0.4", "S40-0.4"]
LABELS = {"pen": "Physics-embedded network (FV solver + composition MLP)", "pinn": "PINN", "pinn_nopde": "PINN without PDE residual", "mlp": "Data-driven network",
          "fdreg": "Classical FD + composition regression", "fdfit": "Classical FD fitted to each mix (calibration)"}


def metrics(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    res = pred - obs
    return {"r2": float(1 - np.sum(res ** 2) / np.sum((obs - obs.mean()) ** 2)),
            "rmse": float(np.sqrt(np.mean(res ** 2))), "mae": float(np.mean(np.abs(res)))}


def load_runs():
    runs = []
    for d in sorted((RES / "runs").glob("*_seed*")):
        if not (d / "pred.csv").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        pred = pd.read_csv(d / "pred.csv")
        theta = pd.read_csv(d / "theta.csv") if (d / "theta.csv").exists() else None
        runs.append((meta["method"], meta["seed"], meta, pred, theta))
    return runs


def main():
    data = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
    data = data[data.reliable].reset_index(drop=True)
    cfg = PhysicsConfig()
    classical = pd.read_csv(RES / "classical_fits.csv")
    classical = classical[classical.Mix.isin(data.Mix.unique())].reset_index(drop=True)
    runs = load_runs()
    out = {"n_points": int(len(data)), "n_mixes": int(data.Mix.nunique()),
           "n_train_points": int((~data.Mix.isin(TEST_MIXES)).sum()),
           "n_test_points": int(data.Mix.isin(TEST_MIXES).sum())}

    # ---- per-method metrics on unseen mixes -------------------------------------------
    rows, per_mix = [], []
    for method, seed, meta, pred, _ in runs:
        for split in ("train", "test"):
            p = pred[pred.split == split]
            rows.append({"method": method, "seed": seed, "split": split, **metrics(p.S_obs, p.S_pred),
                         "train_time_s": meta.get("train_time_s", np.nan), "pde_rms": meta.get("pde_rms", np.nan)})
        for mix, g in pred[pred.split == "test"].groupby("Mix"):
            per_mix.append({"method": method, "seed": seed, "Mix": mix, **metrics(g.S_obs, g.S_pred)})
    # classical FD + regression (deterministic)
    train_mixes = sorted(set(data.Mix) - set(TEST_MIXES))
    fdreg_pred, fdreg_theta, _ = fd_regression(classical, data, train_mixes, TEST_MIXES, cfg)
    fdreg_pred.to_csv(RES / "fdreg_pred.csv", index=False)
    rows.append({"method": "fdreg", "seed": 0, "split": "test", **metrics(fdreg_pred.S_obs, fdreg_pred.S_pred),
                 "train_time_s": classical.set_index("Mix").loc[train_mixes, "time_s"].sum()})
    for mix, g in fdreg_pred.groupby("Mix"):
        per_mix.append({"method": "fdreg", "seed": 0, "Mix": mix, **metrics(g.S_obs, g.S_pred)})
    # classical FD calibrated on each test mix (uses the test data: reference only)
    cp = pd.read_csv(RES / "classical_pred.csv")
    cpt = cp[cp.Mix.isin(TEST_MIXES)]
    rows.append({"method": "fdfit", "seed": 0, "split": "test", **metrics(cpt.S_obs, cpt.S_fd),
                 "train_time_s": classical.set_index("Mix").loc[TEST_MIXES, "time_s"].mean()})
    m = pd.DataFrame(rows)
    m.to_csv(RES / "metrics_all_runs.csv", index=False)
    pm = pd.DataFrame(per_mix)
    pm.to_csv(RES / "metrics_per_mix_all_runs.csv", index=False)

    tab = (m[m.split == "test"].groupby("method")
           .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                mae_mean=("mae", "mean"), time_s=("train_time_s", "mean"), n_seeds=("seed", "nunique")))
    tab["label"] = [LABELS.get(i, i) for i in tab.index]
    tab.to_csv(RES / "table_methods.csv")
    print(tab.round(4).to_string())
    out["methods_test"] = tab.drop(columns="label").round(5).to_dict(orient="index")
    out["methods_train"] = (m[m.split == "train"].groupby("method")[["r2", "rmse"]].mean().round(5)
                            .to_dict(orient="index"))
    tm = (pm.groupby(["method", "Mix"]).agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                                             rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std")).reset_index())
    tm.to_csv(RES / "table_test_mixes.csv", index=False)
    print(tm[tm.method.isin(["pen", "pinn"])].round(4).to_string(index=False))

    # ---- parameters: networks vs classical FD ------------------------------------------
    theta_tab = classical[["Mix", *cfg.identified, *[f"se_{k}" for k in cfg.identified], "r2"]].rename(
        columns={**{k: f"{k}_fd" for k in cfg.identified}, "r2": "r2_fd"})
    solver = FDSolver(cfg)
    for meth in ("pen", "pinn"):
        mr = [(s_, th) for m_, s_, _, _, th in runs if m_ == meth and th is not None]
        if not mr:
            continue
        th = pd.concat([t.assign(seed=s_) for s_, t in mr])
        agg = th.groupby("Mix").agg(**{f"{k}_{meth}": (k, "mean") for k in cfg.identified},
                                    **{f"{k}_{meth}_sd": (k, "std") for k in cfg.identified}).reset_index()
        theta_tab = theta_tab.merge(agg, on="Mix", how="left")
        for k in cfg.identified:
            a, b = theta_tab[f"{k}_{meth}"], theta_tab[f"{k}_fd"]
            out[f"theta_{meth}_{k}_range"] = [float(a.min()), float(a.max())]
            out[f"theta_fd_{k}_range"] = [float(b.min()), float(b.max())]
            out[f"theta_{meth}_vs_fd_{k}_mae"] = float(np.mean(np.abs(a - b)))
            out[f"theta_{meth}_vs_fd_{k}_corr"] = float(np.corrcoef(a, b)[0, 1])
            out[f"theta_{meth}_{k}_seed_sd_mean"] = float(theta_tab[f"{k}_{meth}_sd"].mean())
        # forward verification: scipy FD solver run with the network parameters, no fitting
        seed0 = min(s_ for s_, _ in mr)
        th0 = dict(mr)[seed0].set_index("Mix")
        pred0 = [p_ for m_, s_, _, p_, _ in runs if m_ == meth and s_ == seed0][0]
        fv = []
        for mix, g in data.groupby("Mix"):
            s_fd, _ = solver.solve({k: float(th0.loc[mix, k]) for k in cfg.identified}, g.Time_d.to_numpy())
            p_ = pred0[pred0.Mix == mix]
            fv.append(pd.DataFrame({"Mix": mix, "Time_d": g.Time_d.values, "S_obs": g.S_frac.values,
                                    "S_model": p_.S_pred.values, "S_fd_model_theta": s_fd,
                                    "test": mix in TEST_MIXES}))
        fv = pd.concat(fv)
        fv.to_csv(RES / f"forward_verification_{meth}.csv", index=False)
        for split, g in (("all", fv), ("test", fv[fv.test])):
            out[f"forward_{meth}_fd_vs_model_{split}"] = metrics(g.S_model, g.S_fd_model_theta)
            out[f"forward_{meth}_fd_vs_data_{split}"] = metrics(g.S_obs, g.S_fd_model_theta)
        print(meth, "forward verification (FD with network parameters vs network):",
              out[f"forward_{meth}_fd_vs_model_all"])
    theta_tab["test"] = theta_tab.Mix.isin(TEST_MIXES)
    theta_tab.to_csv(RES / "theta_all.csv", index=False)

    # ---- classical calibration --------------------------------------------------------
    out["classical_calibration"] = {"r2_min": float(classical.r2.min()), "r2_median": float(classical.r2.median()),
                                    "rmse_median": float(classical.rmse.median()),
                                    "time_per_mix_s": float(classical.time_s.mean()),
                                    "n_K_at_bound": int((classical.log10_K <= cfg.log10_K_range[0] + 1e-3).sum())}
    (RES / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
