# Reproducing the revised manuscript

This folder holds the source of the revised manuscript (`part1.src.md`, `part2.src.md`), its build
scripts (`build/`) and the generated files `Manuscript_Q1_Revised.docx` and `Manuscript_Q1_Revised.pdf`.
Every number, table and figure in the manuscript is produced by the scripts listed below from
the raw data file. The manuscript source contains no typed results: numbers appear as `«key»`
(from `results/q1/q1_numbers.json`) and tables as `⟪TABLE:name⟫` (from `results/q1/tables/`).
The build stops if a key or a table is missing.

## 1. Setup

Tested with Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0, torch 2.14 (CPU), scikit-learn 1.9,
matplotlib 3.11. Building the documents also needs pandoc (≥ 3.1) and, for the PDF, Playwright
with Chromium (`pip install playwright && playwright install chromium`).

```bash
pip install -r requirements.txt scikit-learn playwright
python -m pytest tests/            # solver tests (analytical solution, monotonicity, disc -> slab)
```

Everything runs on one CPU core; no GPU is needed.

## 2. Data

| File | Content |
|---|---|
| `data/raw/tidy_dataset_correctedV1.csv` | drying curves digitised from Zhuang et al. (2022), unchanged |
| `data/processed/drying_clean.csv` | reconstructed curves (one per mix), flag `reliable` (w/b ≤ 0.5) |
| `data/processed/cleaning_report.md` | every reconstruction decision, mix by mix |

Only drying data are available. No absorption measurements or measured moisture profiles were
used, and no synthetic data are mixed with the experimental data: synthetic curves appear only in
the clearly labelled recovery experiments (`results/q1/synthetic`, `results/q1/pen_shrinkage`).

## 3. Original pipeline (unchanged)

```bash
python scripts/clean_data.py
python scripts/run_classical.py --geometry disc
python scripts/build_table.py                         # tabulated disc solutions, ~15 min
bash   scripts/run_all_disc.sh                        # PENN / data-driven network, CV, noise, identifiability
python scripts/run_classical.py --geometry slab
python scripts/run_experiments.py --geometry slab --methods pen mlp --seeds 0 1 2 3 4
python scripts/run_experiments.py --geometry slab --methods pinn pinn_nopde --seeds 0 1 2   # ~25 min per PINN run
python scripts/summarize.py --geometry slab
python scripts/verify_numerics.py
python scripts/validate_geometry.py
python scripts/paper_numbers.py && python scripts/paper_tables.py && python scripts/make_figures.py
```

## 4. Revision experiments (`scripts/q1/`)

| Script | Purpose | Time (1 core) |
|---|---|---|
| `compare_repro.py <clone>` | compares an independent re-run of section 3 with the original results | seconds |
| `series_relation.py` | time shift vs time scaling between the two digitised series | seconds |
| `exp_profiles.py` | profile likelihoods with autocorrelation correction, all mixes | ~10 min |
| `exp_synthetic.py` | recovery of known parameters from synthetic curves (C1 single curve, C2 composition maps) | ~3 min |
| `exp_physics_sensitivity.py` | refits under changed constants, boundary conditions, mesh (direct solver) | ~1.5 h |
| `exp_baselines.py` | 8 composition models under identical hold-out / LOMO / LOFO folds | ~10 min |
| `exp_audit.py` | n-profile without the withheld mix, offset from training mixes, paired mix-level statistics | ~1 min |
| `exp_inner_cv.py` | hyper-parameter check by inner CV on the training mixes only | ~6 min |
| `exp_mlp_icv.py` | data-driven network with the inner-CV setting, all schemes (run before `exp_audit.py`) | ~5 min |
| `exp_robustness.py` | noise, correlated digitisation error, time-origin error, curve-reconstruction variants | ~15 min |
| `exp_data_efficiency.py [truncation]` | number of training mixes; truncated training curves | ~15 min |
| `exp_pen_shrinkage.py` | why the original PENN shrinks K_int (synthetic data only) | ~5 min |
| `exp_pen_r_holdout.py` | revised PENN parameters of all mixes (hold-out models) | seconds |
| `exp_interval_calibration.py` | coverage of the profile intervals under AR(1) errors (100 replicates per mix) | long-running (minutes to an hour on one core) |
| `calibrated_intervals.py` | simulation-calibrated 95 % profile intervals | seconds |
| `curing_age_check.py` | time to 70 % saturation vs the 28-day values of the source | ~1 min |
| `exp_pinn.py` | PINN diagnostics: warm start, fixed classical parameters, λ = 100, synthetic data | ~45 min |
| `q1_numbers.py`, `q1_tables.py`, `q1_figures.py` | numbers, tables and figures of the manuscript | ~1 min |

Shared helpers are in `scripts/q1/common.py`. Outputs go to `results/q1/<experiment>/` and
`figures/q1/`.

## 5. Building the manuscript

```bash
bash paper/q1/build/make_q1.sh        # -> paper/q1/Manuscript_Q1_Revised.docx and .pdf
```

The DOCX keeps one review comment and highlighted fields for the authors; the PDF drops the comment
and renders the equations with KaTeX (bundled in `build/katex`).

## 6. Changes to the package code

`pinn_moisture/diff_solver.py`: `PENTrainer.fit(optimizer="lbfgs")` adds full-batch L-BFGS training,
used by the revised PENN (linear composition map, no weight decay). The default (`"adam"`) is
unchanged, so the original results are reproduced exactly.

## 7. Known limitations

* One dataset, digitised from published figures; curing age and lateral sealing during drying not
  recorded in the data file (28-day curing is supported by `curing_age_check.py`, to be confirmed by
  the authors); w/b = 0.6 and S20-0.5 unusable. The data file holds 1001 points, of which the 69
  flagged w/b = 0.6 points are never used; all results use the remaining 932 points.
* Identified parameters are conditional on fixed constitutive constants (see the sensitivity study).
* The PINN was evaluated in one formulation (hard-constrained similarity ansatz); other PINN designs
  were not tested.
* The tabulated solver covers log10 D_v in [-10, -7.5] and log10 K_int in [-21, -18.5]; parameters
  outside this box require rebuilding the table.
