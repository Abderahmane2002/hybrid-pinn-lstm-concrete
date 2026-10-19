# Identifiability study (classical FD + least squares)

Fits on four representative mixes with three parameter sets
(script: `scripts/identifiability.py`; FD solver, 3 starting points each, n = 6 when not identified).

| Mix | Parameters identified | R² | RMSE | max \|corr\| | cond(J) |
|---|---|---|---|---|---|
| C-0.4 | D_v, α0, K | 0.9908 | 0.0139 | 1.00 (D_v–α0) | 1.7e3 |
| C-0.4 | D_v, α0, n, K | 0.9985 | 0.0055 | 0.99 | 1.7e3 |
| C-0.4 | D_v, K | 0.9910 | 0.0137 | 0.64 | 14 |
| F20-0.3 | D_v, α0, K | 0.9989 | 0.0052 | 1.00 | 1.6e3 |
| F20-0.3 | D_v, α0, n, K | 0.9990 | 0.0049 | 1.00 | 4.8e3 |
| F20-0.3 | D_v, K | 0.9980 | 0.0069 | 0.64 | 10 |
| S40-0.5 | D_v, α0, K | 0.9948 | 0.0109 | 1.00 | 1.7e3 |
| S40-0.5 | D_v, α0, n, K | 0.9993 | 0.0041 | 0.99 | 1.5e3 |
| S40-0.5 | D_v, K | 0.9950 | 0.0107 | 0.66 | 13 |
| F20S20-0.5 | D_v, α0, K | 0.9988 | 0.0050 | 1.00 | 1.0e4 |
| F20S20-0.5 | D_v, α0, n, K | 0.9994 | 0.0035 | 0.99 | 2.0e3 |
| F20S20-0.5 | D_v, K | 0.9948 | 0.0105 | 0.65 | 17 |

Conclusions: D_v and α0 cannot be separated (|ρ| = 1.00); adding n improves the
fit but every pair is correlated above 0.98. Two parameters (D_v, K_int) are
well identified (|ρ| ≈ 0.65, cond ≈ 10–17, relative standard errors < 1%).

Profile of the shared exponent n (6 mixes, D_v and K fitted per mix):

| n | total SSE | mean R² | min R² |
|---|---|---|---|
| 1.5 | 0.01436 | 0.99768 | 0.99613 |
| **2.0** | **0.01341** | **0.99789** | **0.99630** |
| 3.0 | 0.01432 | 0.99765 | 0.99573 |
| 4.0 | 0.01870 | 0.99677 | 0.99377 |
| 6.0 | 0.02727 | 0.99524 | 0.99144 |

n = 2 is used for all mixes.
