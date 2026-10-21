"""Shared helpers for the revision experiments (scripts/q1/*).

All experiments use the cleaned data (data/processed/drying_clean.csv, reliable
mixes only) and the disc model of the paper (n = 8, alpha0 = 0.05). Forward
solutions for the baseline physics come from the tabulated disc solutions
(results/disc/solution_table.npz, interpolation error < 1e-4 in S_avg); any
experiment that changes the physics uses the direct BDF solver instead.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pinn_moisture.physics import config_for, param_bounds  # noqa: E402
from pinn_moisture.pinn import DTYPE, composition_features  # noqa: E402
from pinn_moisture.table_solver import TableSolver  # noqa: E402

TEST_MIXES = ["C-0.4", "C-0.5", "F10S10-0.3", "F20-0.3", "F20S20-0.4", "S40-0.4"]
FAMILIES = ["C", "F20", "F40", "S20", "S40", "F10S10", "F20S20"]
TABLE = ROOT / "results/disc/solution_table.npz"
OUT = ROOT / "results/q1"
NAMES = ["log10_Dv", "log10_K"]


def out_dir(name):
    p = OUT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_data(path=None):
    d = pd.read_csv(path or ROOT / "data/processed/drying_clean.csv")
    return d[d.reliable].reset_index(drop=True)


def family(mix):
    return mix.rsplit("-", 1)[0]


def disc_cfg(**kw):
    cfg = config_for("disc", **kw)
    if TABLE.exists() and not kw:
        cfg.table_path = str(TABLE)
    return cfg


_TS = {}


def table():
    if "ts" not in _TS:
        _TS["ts"] = TableSolver(TABLE, config_for("disc"))
    return _TS["ts"]


def forward_table(theta: np.ndarray, times: list[np.ndarray]) -> list[np.ndarray]:
    """theta (B, 2) [log10 Dv, log10 K] -> S_avg at the given times, one array per row."""
    ts = table()
    th = torch.tensor(np.atleast_2d(theta), dtype=DTYPE)
    avg = ts.run({"log10_Dv": th[:, 0], "log10_K": th[:, 1]})
    Q = max(len(t) for t in times)
    tq = np.ones((len(times), Q))
    for i, t in enumerate(times):
        tq[i, :len(t)] = t
    s = ts.interp(avg, torch.tensor(tq, dtype=DTYPE)).numpy()
    return [s[i, :len(t)] for i, t in enumerate(times)]


def fit_table(t, s_obs, starts=((-9.0, -19.77), (-8.5, -19.5), (-9.5, -20.2)), x0=None):
    """Classical least-squares fit with the tabulated disc solver (same model as the BDF fit)."""
    lo, hi = np.array(param_bounds(config_for("disc"))).T
    best = None
    for z0 in ([x0] if x0 is not None else []) + [np.array(z) for z in starts]:
        r = least_squares(lambda z: forward_table(z[None, :], [t])[0] - s_obs, np.clip(z0, lo + 1e-6, hi - 1e-6),
                          bounds=(lo, hi), x_scale="jac", diff_step=1e-4)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(np.sqrt(np.mean(best.fun ** 2)))


def fit_all_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mix, g in d.groupby("Mix", sort=False):
        th, rmse = fit_table(g.Time_d.to_numpy(), g.S_frac.to_numpy())
        rows.append({"Mix": mix, "log10_Dv": th[0], "log10_K": th[1], "rmse": rmse})
    return pd.DataFrame(rows)


def comp(df_mixes: pd.DataFrame) -> np.ndarray:
    """Composition features (w/b, FA, SL), scaled to [0, 1], one row per mix."""
    return composition_features(df_mixes)


def mix_table(d):
    return d.drop_duplicates("Mix").reset_index(drop=True)


def predict_from_theta(d_test: pd.DataFrame, theta_by_mix: dict) -> np.ndarray:
    mixes = list(dict.fromkeys(d_test.Mix))
    times = [d_test[d_test.Mix == m].Time_d.to_numpy() for m in mixes]
    s = forward_table(np.array([theta_by_mix[m] for m in mixes]), times)
    out = np.empty(len(d_test))
    for m, si in zip(mixes, s):
        out[np.where(d_test.Mix.values == m)[0]] = si
    return out


def metrics(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    res = obs - pred
    return {"r2": float(1 - np.sum(res ** 2) / np.sum((obs - obs.mean()) ** 2)),
            "rmse": float(np.sqrt(np.mean(res ** 2))), "mae": float(np.mean(np.abs(res)))}


def per_mix_metrics(df, pred_col="S_pred", obs_col="S_obs"):
    return pd.DataFrame([{"Mix": m, **metrics(g[obs_col], g[pred_col])} for m, g in df.groupby("Mix")])


def folds(scheme: str, mixes: list[str]):
    if scheme == "holdout":
        return [("holdout", TEST_MIXES)]
    if scheme == "lomo":
        return [(m, [m]) for m in mixes]
    if scheme == "lofo":
        return [(f, [m for m in mixes if family(m) == f]) for f in FAMILIES if any(family(m) == f for m in mixes)]
    raise ValueError(scheme)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0
