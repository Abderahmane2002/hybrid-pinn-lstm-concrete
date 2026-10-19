# Paper revision (branch `paper-revision`)

This branch replaces the notebook workflow with a tested Python package that
produces every number, table and figure of the revised manuscript
(`paper/manuscript_v3.docx`). The original notebooks are kept in `notebooks/`.

## Reproducing the paper

```bash
pip install -r requirements.txt
python -m pytest tests/                     # solver checks (analytical, monotonicity, disc -> slab)
python scripts/clean_data.py                # data/processed/drying_clean.csv + cleaning_report.md
python scripts/run_classical.py --geometry disc   # per-mix least-squares fits
python scripts/profile_n.py --geometry disc  # choice of the shape exponent n
python scripts/build_table.py               # tabulated disc solutions (~14 min, once)
bash scripts/run_all_disc.sh                # PENN/MLP x 5 seeds, CV, noise, identifiability
python scripts/run_experiments.py --geometry slab   # PINN comparison (1D slab)
python scripts/verify_numerics.py           # results/numerical_verification.md
python scripts/validate_geometry.py         # results/geometry_check.md (slab vs disc)
python scripts/paper_numbers.py             # results/paper_numbers.json (every number quoted in the text)
python scripts/paper_tables.py              # results/paper_tables/*.md (Tables 3-8, A.1)
python scripts/make_figures.py              # figures/*.png|pdf
bash paper/build/make.sh                    # paper/manuscript_v3.docx (needs pandoc)
```

The manuscript source `paper/manuscript_v3.src.md` contains no hand-typed
results: numbers are written as `«key»` and tables as `⟪TABLE:name⟫`, and the
build stops if any key is missing. Re-running the chain above therefore updates
the paper consistently.

## Model

* Transport: ∂S/∂t = ∇·(D(S)∇S), dual-regime diffusivity of Chen et al. (2022).
* Geometry: the actual 100 × 50 mm disc, axisymmetric, all faces drying
  (`pinn_moisture/axisym.py`). The 1D slab (L = V/A = 12.5 mm) is kept only for
  the PINN comparison; it biases log10 D_v by -0.33 to -0.25 and log10 K_int by
  up to +0.55 decade (`results/geometry_check.md`).
* Identified per mix: θ = (log10 D_v, log10 K_int). α0 = 0.05 and n = 8 are fixed
  (identifiability and n-profile in `results/disc/`).
* Main method: physics-embedded neural network (PENN). Composition -> MLP 3-16-16-2
  -> θ -> disc solver -> S_avg, trained through the solver. For the disc the solver is
  a tabulated solution manifold (49 × 49 parameters × 120 times, bicubic
  interpolation, error ≤ 9e-5; `pinn_moisture/table_solver.py`, built once by
  `scripts/build_table.py` in about 14 min).

## Layout

| Path | Content |
|---|---|
| `pinn_moisture/physics.py` | D(S), boundary/initial conditions, parameter ranges, `config_for(geometry)` |
| `pinn_moisture/fd_solver.py` | 1D finite-volume BDF solver, `make_solver(cfg)` |
| `pinn_moisture/axisym.py` | axisymmetric disc solvers (BDF reference, differentiable LOD) |
| `pinn_moisture/table_solver.py` | tabulated differentiable disc solver |
| `pinn_moisture/diff_solver.py` | differentiable 1D solver, PENN model and trainer |
| `pinn_moisture/inverse_fd.py` | least-squares identification with standard errors and correlations |
| `pinn_moisture/pinn.py` | composition-aware PINN (slab only) |
| `pinn_moisture/baselines.py` | data-driven network, classical fit + composition regression |
| `scripts/` | data cleaning, experiments, CV, verification, numbers, tables, figures |
| `results/disc/`, `results/slab/` | all outputs used in the paper |
| `paper/` | manuscript source, build scripts, generated docx |

## Main results (disc model, six withheld mixes, 278 points)

| Method | R² | RMSE |
|---|---|---|
| PENN (5 seeds) | 0.9931 ± 0.0001 | 0.0137 ± 0.0002 |
| Classical identification + composition regression | 0.9934 | 0.0135 |
| Data-driven network (5 seeds) | 0.9852 ± 0.0012 | 0.0202 ± 0.0008 |
| Reference: classical fit to each withheld mix (uses its data) | 0.9986 | 0.0063 |

* Leave-one-family-out CV: PENN and classical keep R² ≥ 0.965 for every mix; the
  data-driven network drops to 0.22 (F40-0.5) and is below 0.95 for 7 of 20 mixes.
* Noise σ = 0.02 on training data: PENN test RMSE unchanged (0.0137), parameter
  shifts ≤ 0.025 decade; data-driven RMSE 0.0202 -> 0.0223.
* Slab comparison: the PINN fits the training data (RMSE 0.0033) but its parameters
  fail the independent solver check (R² = 0.860; K_int about 1.1 decade too low),
  while the PENN parameters pass it (R² = 0.9998).

## Changes with respect to `notebooks/new_pinn.ipynb`

| Issue in the notebook | Fix |
|---|---|
| Time z-scored while D is in m²/s (PDE dimensionally inconsistent) | dimensionless PDE in ξ = x/L, τ = t/t_ref |
| L = 100 mm, 20 °C / 50% RH | real disc geometry, 50 °C (η_l at 50 °C) |
| Surface condition S0 = 0.8 + φ | S relaxes from 1 to S_e = 0 (oven-dry reference) |
| Initial condition only applied for t < 0.5 d | IC and BC imposed exactly |
| `torch.clamp` on parameters (zero gradient) | smooth sigmoid mapping to bounds |
| Per-mix free parameters ("unseen" mixes were not predictions) | parameters are a function of composition |
| Duplicated / shifted digitisations | `scripts/clean_data.py` (`data/processed/cleaning_report.md`) |
| No seeds, fixed weights | 5 seeds, CV, noise study, solver check |

## Items only the authors can settle

* Corresponding e-mail, X. Wang's CRediT roles, funding, generative-AI declaration, Zenodo DOI.
* Confirm the digitisation source/tool and that the lateral surface was not sealed
  (if sealed: zero-flux lateral face in `axisym.py`, re-run `run_all_disc.sh`).
* Re-digitise the w/b = 0.6 curves and S20-0.5 from Zhuang et al. to recover 8 mixes.
