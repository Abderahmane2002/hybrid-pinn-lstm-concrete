# Data cleaning report

Raw file: `data/raw/tidy_dataset_correctedV1.csv` (1562 rows, 27 mixes).

## 1. Series copied between mixes

- identical series in S20-0.6, S40-0.6 (raw blocks [42, 43]) -> excluded

## 2. Offset between the long (A) and short (B) digitisations

| Mix | shift A-B [d] | rms [d] | n | decision |
|---|---|---|---|---|
| C-0.3 | +1.07 | 0.14 | 14 | consistent |
| C-0.4 | +0.98 | 0.21 | 13 | consistent |
| C-0.5 | +0.97 | 0.18 | 14 | consistent |
| C-0.6 | -1.90 | 0.46 | 12 | inconsistent |
| F10S10-0.3 | +1.05 | 0.24 | 13 | consistent |
| F10S10-0.4 | +1.06 | 0.18 | 13 | consistent |
| F10S10-0.5 | +1.19 | 0.20 | 13 | consistent |
| F10S10-0.6 | -1.74 | 0.82 | 11 | inconsistent |
| F20-0.3 | +1.14 | 0.27 | 11 | consistent |
| F20-0.4 | +0.98 | 0.11 | 8 | consistent |
| F20-0.5 | +1.10 | 0.16 | 13 | consistent |
| F20-0.6 | -0.02 | 0.35 | 12 | inconsistent |
| F20S20-0.3 | +0.92 | 0.16 | 14 | consistent |
| F20S20-0.4 | +0.94 | 0.22 | 14 | consistent |
| F20S20-0.5 | +1.03 | 0.16 | 14 | consistent |
| F20S20-0.6 | -2.64 | 0.57 | 13 | inconsistent |
| F40-0.3 | +0.87 | 0.21 | 14 | consistent |
| F40-0.4 | +1.14 | 0.25 | 13 | consistent |
| F40-0.5 | +1.14 | 0.22 | 12 | consistent |
| S20-0.3 | +1.11 | 0.24 | 13 | consistent |
| S20-0.4 | +1.12 | 0.31 | 13 | inconsistent |
| S20-0.6 | -3.15 | 0.70 | 11 | inconsistent |
| S40-0.3 | +1.14 | 0.40 | 10 | inconsistent |
| S40-0.4 | +1.13 | 0.29 | 13 | consistent |
| S40-0.5 | +0.97 | 0.16 | 13 | consistent |
| S40-0.6 | -5.14 | 1.20 | 10 | inconsistent |

Median shift over consistent mixes: **+1.05 d** (IQR +0.97 to +1.13 d). It is subtracted from every A series.

## 3. Final curve per mix

| Mix | raw rows | kept | decision |
|---|---|---|---|
| C-0.3 | 58 | 44 | B (<= 14 d) + time-corrected A (> 14 d) |
| C-0.4 | 60 | 46 | B (<= 14 d) + time-corrected A (> 14 d) |
| C-0.5 | 47 | 34 | B (<= 14 d) + time-corrected A (> 14 d) |
| C-0.6 | 53 | 14 | w/b = 0.6: A series unreliable, kept B (0-14 d) only |
| F10S10-0.3 | 63 | 50 | B (<= 14 d) + time-corrected A (> 14 d) |
| F10S10-0.4 | 63 | 50 | B (<= 14 d) + time-corrected A (> 14 d) |
| F10S10-0.5 | 61 | 48 | B (<= 14 d) + time-corrected A (> 14 d) |
| F10S10-0.6 | 52 | 14 | w/b = 0.6: A series unreliable, kept B (0-14 d) only |
| F20-0.3 | 55 | 48 | B (<= 14 d) + time-corrected A (> 14 d) |
| F20-0.4 | 53 | 44 | B (<= 14 d) + time-corrected A (> 14 d) |
| F20-0.5 | 58 | 45 | B (<= 14 d) + time-corrected A (> 14 d) |
| F20-0.6 | 55 | 13 | w/b = 0.6: A series unreliable, kept B (0-14 d) only |
| F20S20-0.3 | 65 | 52 | B (<= 14 d) + time-corrected A (> 14 d) |
| F20S20-0.4 | 70 | 56 | B (<= 14 d) + time-corrected A (> 14 d) |
| F20S20-0.5 | 69 | 55 | B (<= 14 d) + time-corrected A (> 14 d) |
| F20S20-0.6 | 68 | 0 | w/b = 0.6: no reliable series, mix excluded; B series mixes two curves -> dropped |
| F40-0.3 | 52 | 40 | B (<= 14 d) + time-corrected A (> 14 d) |
| F40-0.4 | 62 | 51 | B (<= 14 d) + time-corrected A (> 14 d) |
| F40-0.5 | 66 | 54 | B (<= 14 d) + time-corrected A (> 14 d) |
| F40-0.6 | 50 | 0 | w/b = 0.6: no reliable series, mix excluded |
| S20-0.3 | 62 | 47 | B (<= 14 d) + time-corrected A (> 14 d) |
| S20-0.4 | 54 | 42 | B (<= 14 d) + time-corrected A (> 14 d) |
| S20-0.6 | 50 | 14 | w/b = 0.6: A series unreliable, kept B (0-14 d) only |
| S40-0.3 | 58 | 45 | B (<= 14 d) + time-corrected A (> 14 d) |
| S40-0.4 | 57 | 44 | B (<= 14 d) + time-corrected A (> 14 d) |
| S40-0.5 | 51 | 37 | B (<= 14 d) + time-corrected A (> 14 d) |
| S40-0.6 | 50 | 14 | w/b = 0.6: A series unreliable, kept B (0-14 d) only |

**Result:** 1001 points, 25 mixes. Missing from the raw file or excluded: F20S20-0.6, F40-0.6, S20-0.5.

**Action for the authors:** re-digitise the w/b = 0.6 curves (and S20-0.5) from the figures of Zhuang et al. (2022), or obtain the raw mass-loss data from the authors, then re-run this script.
