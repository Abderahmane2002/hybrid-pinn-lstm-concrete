# Paper revision — code restructure (branch `paper-revision`)

This branch replaces the notebook workflow with a small, tested Python package
that produces every number and figure of the revised manuscript from one
command chain. The original notebooks are kept untouched in `notebooks/`.

## Pipeline

```bash
pip install -r requirements.txt
python scripts/clean_data.py              # data/processed/drying_clean.csv + cleaning_report.md
python scripts/run_classical.py           # classical FD + least-squares fit for every mix
python scripts/run_experiments.py         # PINN, PINN without PDE, data-driven MLP x 5 seeds
python scripts/summarize.py               # tables + results/metrics.json
python scripts/make_figures.py            # figures/*.png|pdf
python -m pytest tests/                   # FD solver against the analytical solution
```

## Layout

| Path | Content |
|---|---|
| `pinn_moisture/physics.py` | D(S) of Chen et al. (2022), boundary/initial conditions, parameter ranges |
| `pinn_moisture/fd_solver.py` | finite-volume + BDF solver (classical forward model, verification) |
| `pinn_moisture/inverse_fd.py` | least-squares identification with standard errors and correlations |
| `pinn_moisture/pinn.py` | composition-aware PINN (field network + parameter head) |
| `pinn_moisture/baselines.py` | data-driven network, FD + composition regression |
| `scripts/` | data cleaning, experiments, summary tables, figures |
| `results/` | all outputs (CSV/JSON) used in the paper |
| `figures/` | all paper figures, one style |

## What changed with respect to `notebooks/new_pinn.ipynb` (cell 0)

| Issue in the notebook | Fix |
|---|---|
| Time z-score normalised while D is in m²/s: PDE dimensionally inconsistent (hence D_v ≈ 1e-5) | PDE written in ξ = x/L, τ = t/t_ref with Π = t_ref/L² |
| L = 100 mm, 20 °C/50% RH | Zhuang et al.: 100 × 50 mm discs dried at 50 °C → equivalent slab L = V/A = 12.5 mm, η_l at 50 °C |
| Surface condition S0 = 0.8 + φ (wet surface during drying) | S(0,t) relaxes from 1 to S_e = 0 (oven-dry reference) |
| Initial condition only applied for t < 0.5 d (never: first point at 0.9 d) | IC and surface BC imposed exactly (hard constraint) |
| `torch.clamp` on α0, n, φ: zero gradient, all mixes stuck at bounds | smooth sigmoid mapping to bounds |
| φ only entered the (wrong) BC | removed |
| Per-mix free parameters: "unseen" mixes were not predictions | parameters are a function of composition (parameter head) |
| Duplicated / shifted digitisations in the CSV | `scripts/clean_data.py` (see `data/processed/cleaning_report.md`) |
| Per-sample Python loop, 100 epochs | vectorised full-batch Adam + L-BFGS |
| Fixed λ weights, no seeds | data term scaled by the digitisation noise, PDE residual with similarity scaling, 5 seeds |

## Identifiability

Least-squares fits with the FD solver show that D_v and α0 are perfectly
correlated (|ρ| = 1.00) and n is strongly correlated with both; only two
parameters per mix are identifiable from an average-saturation curve. The model
therefore identifies log10 D_v and log10 K_int per mix, with α0 = 0.05 fixed and
n = 2 shared by all mixes (value minimising the total misfit, see
`results/identifiability.md`).

## Results (six unseen mixes, 278 points; training on 654 points of 14 mixes)

| Method | R² | RMSE | seeds |
|---|---|---|---|
| Physics-embedded network (FV solver + composition MLP) | 0.9935 ± 0.0001 | 0.0133 ± 0.0001 | 5 |
| Classical FD + linear regression of parameters on composition | 0.9938 | 0.0131 | 1 |
| Data-driven network | 0.9852 ± 0.0012 | 0.0202 ± 0.0008 | 5 |
| PINN (composition-aware) | 0.9826 ± 0.0071 | 0.0216 ± 0.0043 | 3 |
| PINN without PDE residual (ablation) | 0.9411 ± 0.0640 | 0.0365 ± 0.0210 | 3 |
| Reference: classical FD fitted to each test mix (uses the test data) | 0.9987 | 0.0059 | 1 |

Physical consistency (FD solver run with the parameters returned by each network, no fitting):

* physics-embedded network: R² = 0.9998, RMSE = 0.0021
* PINN: R² = 0.860, RMSE = 0.061
  (log10 K_int biased by 1.09 decades with respect to the classical fits)

Parameter agreement with the classical per-mix fits (physics-embedded network):
log10 D_v MAE = 0.024 (r = 0.991),
log10 K_int MAE = 0.168 (r = 0.694).

Known limitation of the PINN: it falls into the "trivial dry solution" failure mode
(field collapses to S_e shortly after t = 0) unless the drying-front speed is bounded
(`a_front`, `s_front` in `TrainConfig`). Even with this bound, the PINN reproduces the
data but the PDE is only approximately satisfied in the high-saturation (capillary)
regime, so K_int is not identified reliably. See `scripts/` and `results/` for all runs.
