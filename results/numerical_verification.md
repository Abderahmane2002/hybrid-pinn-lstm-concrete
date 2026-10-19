# Numerical verification

## 1. Spatial convergence (BDF solver, rtol 1e-9)

max |S_avg - S_avg,ref| over t = 0.25-90 d; reference: 640 cells

| Mix | 20 cells | 40 cells | 60 cells | 80 cells | 160 cells | observed order (40->80) |
|---|---|---|---|---|---|---|
| C-0.5 | 1.3e-03 | 3.3e-04 | 1.4e-04 | 8.1e-05 | 2.0e-05 | 2.01 |
| C-0.4 | 1.4e-03 | 3.4e-04 | 1.5e-04 | 8.5e-05 | 2.1e-05 | 2.01 |
| F40-0.3 | 1.3e-03 | 3.3e-04 | 1.5e-04 | 8.2e-05 | 2.0e-05 | 2.01 |

## 2. Differentiable solver (linearly implicit Euler, 60 cells, log time grid)

max |S_avg - S_avg,BDF(160 cells)|

| Mix | 60 steps | 120 steps | 240 steps (paper) | 480 steps | 960 steps |
|---|---|---|---|---|---|
| C-0.5 | 1.1e-02 | 5.6e-03 | 2.9e-03 | 1.5e-03 | 7.7e-04 |
| C-0.4 | 1.2e-02 | 5.9e-03 | 3.0e-03 | 1.6e-03 | 8.1e-04 |
| F40-0.3 | 1.1e-02 | 5.4e-03 | 2.8e-03 | 1.4e-03 | 7.5e-04 |

## 3. Global mass balance (BDF solver, 80 cells)

Water lost through the drying face, integrated in time, vs decrease of stored water.

| Mix | t [d] | lost through face | decrease of S_avg L | relative error |
|---|---|---|---|---|
| C-0.5 | 28 | 12.27478 mm | 12.27478 mm | 2.8e-16 |
| C-0.4 | 28 | 11.96592 mm | 11.96592 mm | 4.3e-16 |
| F40-0.3 | 28 | 8.71228 mm | 8.71228 mm | 1.2e-15 |

## 4. Disc model: mesh convergence (BDF, reference 96 x 64 cells)

| Mix | 12 x 8 | 24 x 16 (paper) | 48 x 32 |
|---|---|---|---|
| C-0.5 | 2.5e-03 | 1.2e-03 | 5.9e-04 |
| C-0.4 | 3.4e-03 | 1.3e-03 | 6.4e-04 |
| F40-0.3 | 3.8e-02 | 3.7e-03 | 7.8e-04 |

## 5. Tabulated disc solutions

Grid: 49 x 49 parameter values, log10 D_v in [-10.0, -7.5], log10 K_int in [-21.0, -18.5], 120 times from 0.05 to 110 d; build time 832 s.

Interpolation error against direct BDF solutions at 30 random parameter sets (max over t = 0.5-90 d): median 4.6e-05, max 8.8e-05.
