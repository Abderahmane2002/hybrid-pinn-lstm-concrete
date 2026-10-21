"""Experiment C (real data): profile likelihoods of the two identified parameters.

For every mix, log10 K_int (resp. log10 D_v) is fixed on a grid and the other parameter
is re-optimised (tabulated disc solver). With Gaussian errors the profile deviance is
N log(SSE / SSE_min); the 95 % interval is where it stays below 3.84. Because the residuals
of a digitised curve are autocorrelated, the interval is also given with the effective
number of points N_eff = N (1 - rho1) / (1 + rho1), rho1 = lag-1 autocorrelation of the
residuals at the optimum (a more honest, wider interval).
Output: results/q1/profiles/{profiles.csv, intervals.csv}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import fit_table, forward_table, load_data, out_dir

GRID_K = np.arange(-21.0, -18.5 + 1e-9, 0.02)
GRID_DV = np.arange(-10.0, -7.5 + 1e-9, 0.01)


def sse(t, s, dv, k):
    return float(np.sum((forward_table(np.array([[dv, k]]), [t])[0] - s) ** 2))


def interval(grid, dev, thr):
    ok = grid[dev <= thr]
    return (ok.min(), ok.max()) if len(ok) else (np.nan, np.nan)


def main():
    out = out_dir("profiles")
    d = load_data()
    prof, ints = [], []
    for mix, g in d.groupby("Mix"):
        t, s = g.Time_d.to_numpy(), g.S_frac.to_numpy()
        (dv0, k0), _ = fit_table(t, s)
        r = forward_table(np.array([[dv0, k0]]), [t])[0] - s
        rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        N = len(t)
        n_eff = max(N * (1 - rho) / (1 + rho), 3.0)
        smin = sse(t, s, dv0, k0)
        pk = np.array([minimize_scalar(lambda dv: sse(t, s, dv, k), bounds=(-10, -7.5), method="bounded",
                                       options={"xatol": 1e-4}).fun for k in GRID_K])
        pd_ = np.array([minimize_scalar(lambda k: sse(t, s, dv, k), bounds=(-21, -18.5), method="bounded",
                                        options={"xatol": 1e-4}).fun for dv in GRID_DV])
        pk, pd_ = np.minimum(pk, np.inf), np.minimum(pd_, np.inf)
        smin = min(smin, pk.min(), pd_.min())
        dev_k, dev_d = N * np.log(pk / smin), N * np.log(pd_ / smin)
        prof += [{"Mix": mix, "param": "log10_K", "value": v, "deviance": dk, "deviance_eff": dk * n_eff / N}
                 for v, dk in zip(GRID_K, dev_k)]
        prof += [{"Mix": mix, "param": "log10_Dv", "value": v, "deviance": dd, "deviance_eff": dd * n_eff / N}
                 for v, dd in zip(GRID_DV, dev_d)]
        kl, kh = interval(GRID_K, dev_k, 3.84)
        kle, khe = interval(GRID_K, dev_k * n_eff / N, 3.84)
        dl, dh = interval(GRID_DV, dev_d, 3.84)
        dle, dhe = interval(GRID_DV, dev_d * n_eff / N, 3.84)
        ints.append({"Mix": mix, "N": N, "rho1": rho, "N_eff": n_eff, "log10_Dv": dv0, "log10_K": k0,
                     "Dv_lo": dl, "Dv_hi": dh, "Dv_lo_eff": dle, "Dv_hi_eff": dhe,
                     "K_lo": kl, "K_hi": kh, "K_lo_eff": kle, "K_hi_eff": khe,
                     "K_interior": bool(kle > -21 and khe < -18.5), "Dv_interior": bool(dle > -10 and dhe < -7.5)})
        print(mix, f"rho1={rho:.2f} N_eff={n_eff:.1f} K in [{kle:.2f}, {khe:.2f}] Dv in [{dle:.3f}, {dhe:.3f}]", flush=True)
    pd.DataFrame(prof).to_csv(out / "profiles.csv", index=False)
    I = pd.DataFrame(ints)
    I["K_width_eff"], I["Dv_width_eff"] = I.K_hi_eff - I.K_lo_eff, I.Dv_hi_eff - I.Dv_lo_eff
    I.to_csv(out / "intervals.csv", index=False)
    print(I[["N", "rho1", "N_eff", "K_width_eff", "Dv_width_eff"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
