"""Are the autocorrelation-corrected 95 % profile intervals calibrated? (SIMULATION)

For every mix, the fitted disc-model curve is taken as the truth, and synthetic replicate
curves are generated at the measurement times by adding AR(1) errors with the standard
deviation and lag-1 autocorrelation of that mix's residuals. Each replicate is analysed exactly
as the experimental curves are: least-squares fit, residual autocorrelation, effective sample
size N_eff = N(1-r1)/(1+r1), profile deviance at the true value of each parameter. The true value
lies inside the 95 % profile interval when the deviance is <= 3.84. Coverage is reported for
  * the uncorrected profile interval (N points treated as independent),
  * the corrected profile interval (N_eff, as in the paper),
  * the Wald interval theta +- 1.96 SE from the Jacobian with independent errors.
A calibration factor c (deviance threshold 3.84 c) that restores 95 % coverage is also given.
Output: results/q1/interval_calibration/{replicates.csv, coverage.csv}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import fit_table, forward_table, load_data, out_dir

N_REP = 100


def sse(t, s, dv, k):
    return float(np.sum((forward_table(np.array([[dv, k]]), [t])[0] - s) ** 2))


def ar1(n, sd, rho, rng):
    e = np.empty(n)
    e[0] = rng.normal(0, sd)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
    return e


def lag1(r):
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


def main():
    out = out_dir("interval_calibration")
    d = load_data()
    rows = []
    for mix, g in d.groupby("Mix"):
        t, s_obs = g.Time_d.to_numpy(), g.S_frac.to_numpy()
        (dv0, k0), _ = fit_table(t, s_obs)
        s0 = forward_table(np.array([[dv0, k0]]), [t])[0]
        r = s_obs - s0
        sd, rho = float(r.std()), lag1(r)
        rng = np.random.default_rng(abs(hash(mix)) % 2 ** 32)
        N = len(t)
        for rep in range(N_REP):
            y = s0 + ar1(N, sd, rho, rng)
            (dv, k), _ = fit_table(t, y, x0=np.array([dv0, k0]))
            smin = sse(t, y, dv, k)
            res = y - forward_table(np.array([[dv, k]]), [t])[0]
            r1 = lag1(res)
            neff = max(N * (1 - r1) / (1 + r1), 3.0)
            sk = minimize_scalar(lambda x: sse(t, y, x, k0), bounds=(-10, -7.5), method="bounded",
                                 options={"xatol": 1e-5}).fun
            sd_ = minimize_scalar(lambda x: sse(t, y, dv0, x), bounds=(-21, -18.5), method="bounded",
                                  options={"xatol": 1e-5}).fun
            smin = min(smin, sk, sd_)
            # Jacobian (forward differences) for the Wald interval
            h = 1e-4
            f0 = forward_table(np.array([[dv, k]]), [t])[0]
            J = np.column_stack([(forward_table(np.array([[dv + h, k]]), [t])[0] - f0) / h,
                                 (forward_table(np.array([[dv, k + h]]), [t])[0] - f0) / h])
            cov = np.linalg.inv(J.T @ J) * smin / (N - 2)
            se = np.sqrt(np.diag(cov))
            rows.append({"Mix": mix, "rep": rep, "N": N, "rho_true": rho, "rho_hat": r1, "N_eff": neff,
                         "dev_K": N * np.log(sk / smin), "dev_Dv": N * np.log(sd_ / smin),
                         "z_Dv": abs(dv - dv0) / se[0], "z_K": abs(k - k0) / se[1]})
        print(mix, f"rho={rho:.2f}", flush=True)
    R = pd.DataFrame(rows)
    R["devc_K"] = R.dev_K * R.N_eff / R.N
    R["devc_Dv"] = R.dev_Dv * R.N_eff / R.N
    R.to_csv(out / "replicates.csv", index=False)
    cov = []
    for label, sub in (("all mixes", R), ("rho < 0.5", R[R.rho_true < 0.5]), ("0.5 <= rho < 0.75", R[(R.rho_true >= 0.5) & (R.rho_true < 0.75)]),
                       ("rho >= 0.75", R[R.rho_true >= 0.75])):
        for par in ("K", "Dv"):
            cov.append({"subset": label, "param": par, "n": len(sub),
                        "coverage_uncorrected": float((sub[f"dev_{par}"] <= 3.84).mean()),
                        "coverage_corrected": float((sub[f"devc_{par}"] <= 3.84).mean()),
                        "coverage_wald_iid": float((sub[f"z_{par}"] <= 1.96).mean()),
                        "calib_factor_corrected": float(np.quantile(sub[f"devc_{par}"], 0.95) / 3.84)})
    C = pd.DataFrame(cov)
    C.to_csv(out / "coverage.csv", index=False)
    pd.set_option("display.width", 200)
    print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
