"""Simulation-calibrated 95 % profile intervals.

The coverage study (exp_interval_calibration.py) shows that the N_eff-corrected profile
intervals cover the true value in about 88-90 % of correlated-error replicates. Multiplying the
deviance threshold 3.84 by the factor c that gives 95 % coverage over all replicates
(separately for each parameter) yields calibrated intervals, computed here from the stored
profiles of every mix. The calibration assumes AR(1) errors with the residual standard deviation
and lag-1 autocorrelation of each curve.
Output: results/q1/profiles/intervals_calibrated.csv
"""
import numpy as np
import pandas as pd

from common import ROOT

Q = ROOT / "results/q1"
P = pd.read_csv(Q / "profiles/profiles.csv")
I = pd.read_csv(Q / "profiles/intervals.csv").set_index("Mix")
C = pd.read_csv(Q / "interval_calibration/coverage.csv")
c = C[C.subset == "all mixes"].set_index("param").calib_factor_corrected
rows = []
for mix, g in P.groupby("Mix"):
    row = {"Mix": mix, "log10_Dv": I.loc[mix, "log10_Dv"], "log10_K": I.loc[mix, "log10_K"]}
    for par, key in (("log10_K", "K"), ("log10_Dv", "Dv")):
        h = g[g.param == par]
        ok = h.value[h.deviance_eff <= 3.84 * c[key]]
        lo, hi = float(ok.min()), float(ok.max())
        bound = (lo <= h.value.min() + 1e-9) or (hi >= h.value.max() - 1e-9)
        row.update({f"{key}_lo_cal": lo, f"{key}_hi_cal": hi, f"{key}_width_cal": hi - lo, f"{key}_at_bound": bound})
    rows.append(row)
R = pd.DataFrame(rows)
R["c_K"], R["c_Dv"] = c["K"], c["Dv"]
R.to_csv(Q / "profiles/intervals_calibrated.csv", index=False)
print(R[["K_width_cal", "Dv_width_cal"]].describe().round(3))
print("intervals reaching the admissible bounds:", int(R.K_at_bound.sum()), "(K),", int(R.Dv_at_bound.sum()), "(Dv)")
