"""Reference methods for the comparison table.

* ``DataDrivenMLP``  : purely data-driven network (sqrt(t), composition) -> S_avg.
* ``fd_regression``  : classical route for unseen mixes — parameters identified
  by FD + least squares on the training mixes, regressed linearly on composition,
  then the FD solver is run with the regressed parameters.
The PINN without PDE residual is obtained with ``TrainConfig(use_pde=False)``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .fd_solver import FDSolver
from .physics import PhysicsConfig
from .pinn import DTYPE, MLP, composition_features


class DataDrivenMLP:
    def __init__(self, t_ref=100.0, hidden=48, layers=4, iters=4000, lr=2e-3, weight_decay=1e-5, seed=0):
        torch.manual_seed(seed)
        self.t_ref, self.iters, self.lr, self.wd = t_ref, iters, lr, weight_decay
        self.net = MLP(4, 1, hidden, layers).to(DTYPE)

    def _x(self, df):
        tau = np.sqrt(df["Time_d"].to_numpy() / self.t_ref)
        return torch.tensor(np.column_stack([tau, composition_features(df)]) * 2 - 1, dtype=DTYPE)

    def fit(self, df):
        x, y = self._x(df), torch.tensor(df["S_frac"].to_numpy(), dtype=DTYPE)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9993)
        t0 = time.time()
        for _ in range(self.iters):
            opt.zero_grad()
            loss = torch.mean((torch.sigmoid(self.net(x)[:, 0]) - y) ** 2)
            loss.backward()
            opt.step()
            sched.step()
        self.train_time = time.time() - t0
        return self

    @torch.no_grad()
    def predict(self, df):
        return torch.sigmoid(self.net(self._x(df))[:, 0]).numpy()


def fd_regression(classical: pd.DataFrame, data: pd.DataFrame, train_mixes, test_mixes, cfg: PhysicsConfig):
    """Regress classical parameters on composition (training mixes) and predict unseen mixes with FD."""
    comp = data.drop_duplicates("Mix").set_index("Mix")
    feats = lambda mixes: np.column_stack([np.ones(len(mixes)), composition_features(comp.loc[mixes].reset_index())])
    tr = classical.set_index("Mix").loc[train_mixes]
    # exclude fits that ended on a bound (unidentified K for w/b = 0.6 short series)
    keep = np.ones(len(tr), bool)
    for k in cfg.identified:
        lo, hi = getattr(cfg, {"log10_Dv": "log10_Dv_range", "log10_K": "log10_K_range"}[k])
        keep &= (tr[k] > lo + 1e-3) & (tr[k] < hi - 1e-3)
    X = feats(list(tr.index[keep]))
    coef = {k: np.linalg.lstsq(X, tr.loc[keep, k].to_numpy(), rcond=None)[0] for k in cfg.identified}
    solver = FDSolver(cfg)
    preds, thetas = [], []
    for mix in test_mixes:
        x = feats([mix])[0]
        theta = {k: float(x @ coef[k]) for k in cfg.identified}
        g = data[data.Mix == mix]
        s, _ = solver.solve(theta, g["Time_d"].to_numpy())
        preds.append(pd.DataFrame({"Mix": mix, "Time_d": g["Time_d"].values, "S_obs": g["S_frac"].values,
                                   "S_pred": s}))
        thetas.append({"Mix": mix, **theta})
    return pd.concat(preds), pd.DataFrame(thetas), coef
