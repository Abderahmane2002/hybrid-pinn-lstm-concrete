# Scientific revision report

Manuscript: *Learning moisture transport parameters of blended concretes from drying curves: identifiability, generalisation across binders, and a comparison of physics-embedded and physics-informed neural networks* (revised from "Physics-embedded neural network for predicting the moisture transport properties of concrete from mix composition, with a comparison to physics-informed neural networks").

All numbers below were computed by the scripts named in each section (repository branch `q1-revision`, folder `scripts/q1/`, outputs in `results/q1/`). Nothing was changed in the original data, the original manuscript or the original results; the original pipeline still runs unchanged (`pinn_moisture` keeps its default behaviour).

## 0. Summary of the outcome

| Question | Answer from the completed analyses |
|---|---|
| Are the published numbers reproducible? | Yes. An independent re-run from the raw file reproduces 46 of 46 compared quantities within tolerance; the PINN runs are bit-for-bit deterministic. |
| Does the PENN beat classical identification? | No. No statistically detectable difference was found between the PENN and linear two-stage inversion in hold-out, leave-one-mix-out and leave-one-family-out validation (mix-level Wilcoxon p = 0.09 and 0.22; family-level p = 0.375 and 0.469). Family-level 90 % intervals lie within ±0.002 RMSE (margin set in this revision). This is similar accuracy under the conditions studied, not proven equivalence. |
| What does generalise to a new binder family? | The transport model combined with a low-capacity (near-linear) composition map, fitted in two stages or end to end. Gaussian-process, quadratic and data-driven models fail for several mixes (lowest R² 0.20). |
| Where does end-to-end training help? | With short training curves (7-day curves: RMSE 0.0138 vs 0.0165) and, for the regularised original PENN, with very few training mixes (5 mixes: 0.0220 vs 0.0256; the unregularised revised PENN behaves like linear inversion, 0.0275). |
| Are the parameters trustworthy? | D_v: yes (median 95 % interval 0.05 decade wide). K_int: only to a few tenths of a decade (median calibrated 95 % interval 0.42 decade; one-sided for 3 mixes), and conditional on fixed constants (shifts up to 0.4 decade). Mix rankings are robust. |
| Is the PINN result real? | The bias is robust: the K_int bias (≈ 1.1 decade) persists with exact synthetic data, a warm start and λ = 100, and disappears only when the parameters are frozen. The diagnostics strongly suggest, but do not prove, that the soft-constrained formulation is the cause; other PINN designs were not tested. |
| New problem found | The original PENN shrinks K_int between mixes (synthetic slope 0.17). Cause: weight decay plus an under-converged optimiser. A linear map trained with L-BFGS fixes it (slope 0.96) without loss of accuracy. |

## 1. Reproduction of the original results (Stage 1)

* **Problem.** The reported values had not been regenerated independently.
* **Change.** None to the method. A clean clone of the repository was run from the raw data file (all scripts of section 3 of `paper/q1/README.md`) and compared with the original results by `scripts/q1/compare_repro.py`.
* **Results.** 46 of 46 quantities agree within tolerance (Table B.1 of the manuscript; `results/q1/reproducibility/report.md`). Cleaned data are identical; classical parameters agree within 0.001 decade (multi-start randomness); all PENN, data-driven and CV results are identical; the verification files agree to the fifth significant digit. Two PINN runs were initially missing because the first re-run hit a 2-hour limit; they were run separately and reproduce the original values exactly.
* **Effect on findings.** Confirms the original numbers. Manuscript: Appendix B.

## 2. Source verification and curve reconstruction

* **Problem.** The data provenance relied on assumptions. The accepted manuscript of Zhuang et al. (UCL Discovery) shows drying curves at **two curing ages (28 and 56 days)**, which raised the possibility that the two digitised series of each mix are two different curves rather than duplicates.
* **Change.** New test `scripts/q1/series_relation.py`: for every mix, the long series is related to the short one by a pure time shift, a pure time scaling (expected if the curves came from different curing ages, since drying scales with t/D), or both.
* **Results.** Median RMS mismatch (w/b ≤ 0.5): shift 0.005, scaling 0.026, combined fit scale factor 0.98 (IQR 0.97–1.00). The two series are one curve with two time origins; the one-day offset matches the 24 h vacuum saturation. Offsets at the IQR bounds or estimated from training mixes only (1.06 d) change the parameters by ≤ 0.009 decade.
* **Data counts (added after an internal review).** The reconstructed file holds 1,001 points, of which 69 belong to the flagged w/b = 0.6 mixes; every result, original and revised, uses the remaining 932 points (w/b ≤ 0.5). Section 2.3 and Table B.1 now state both counts and which version produced each result.
* **Curing age (added after an internal review).** `curing_age_check.py` compares the time to 70 % saturation (t70) of each reconstructed curve with the 28-day values tabulated in the source: mean t70 1.60, 0.95 and 0.62 d for w/b 0.3, 0.4 and 0.5 against 1.5, 1.0 and 0.8 d; mean absolute deviation 0.11 d with the short-series origin against 1.0 d with the long-series origin. This supports 28 d. Section 5.8 now interprets composition effects as those of 28-day-cured specimens.
* **Effect.** Strengthens the cleaning step; the curing age is supported indirectly and should still be confirmed by the authors (kept as a review comment). Manuscript: Sections 2.3, 2.4, 5.5, 5.8, Fig. 1d.

## 3. Leakage audit of shared choices

* **Problem.** The shape exponent n was chosen from a profile over six mixes, one of which (C-0.5) is a withheld mix; the time offset used all mixes; the network settings could have been tuned on the test mixes.
* **Change.** `exp_audit.py` recomputes the n-profile without C-0.5 and the offset from training mixes; `exp_inner_cv.py` runs an inner leave-one-family-out CV on the 14 training mixes for 9 PENN and 4 data-driven configurations.
* **Results.** Without C-0.5 the misfit at n = 2 is 49 % higher than at n = 8 and only 2 % lower at n = 16: n = 8 is unchanged. Offset from training mixes 1.06 d (was 1.05 d). The original PENN setting ranks 5/9 in inner CV (RMSE 0.0178 vs best 0.0173); the inner-CV choice gives the same hold-out RMSE (0.0138). For the data-driven network the inner CV prefers a ten times stronger weight decay (the original setting ranks last of four). Because judging this choice by its test accuracy (0.0309) would itself use the withheld mixes, the inner-CV setting is now evaluated in every scheme and reported alongside the original one (see Section 6).
* **Effect.** No leakage with an effect on the results. Manuscript: Sections 2.3, 4.2, Appendix C.

## 4. Identifiability: profile likelihoods and synthetic recovery

* **Problem.** Identifiability rested on Jacobian correlations only; the standard errors assumed independent residuals, which digitised curves do not have.
* **Change.** `exp_profiles.py`: profile likelihood of each parameter for all 20 mixes, with the effective sample size N_eff = N(1−ρ₁)/(1+ρ₁). `exp_interval_calibration.py` (added after an internal review): 100 replicate curves per mix with AR(1) errors matched to its residuals, to measure the actual coverage of the intervals; `calibrated_intervals.py` rescales the deviance threshold to 95 % coverage. `exp_synthetic.py` (C1): six known parameter sets, curves generated on a finer mesh, inverted under five conditions.
* **Results.** Median lag-1 autocorrelation 0.63; median N_eff 11 of 46 points. Coverage of nominal 95 % intervals in the simulation: independent-error profile intervals 67 % (K_int) and 64 % (D_v); Jacobian (Wald) intervals 68 % and 64 %; N_eff-corrected profile intervals 90 % and 88 %. Calibration factors for the deviance threshold: 1.46 (K_int) and 1.78 (D_v). Calibrated interval widths: D_v median 0.05 decade (max 0.44); K_int median 0.42; for 3 mixes (F10S10-0.5, F20S20-0.3, F40-0.3) the K_int interval reaches the admissible bound (one-sided identification). Synthetic: exact data recovered within 0.001/0.008 decade, also with 14-day data; noise 0.01 → K_int errors up to 0.25; time-origin error 0.25 d → up to 0.39.
* **Effect.** Weakens the precision claimed for K_int (and shows that the first, N_eff-only correction was still too narrow); strengthens D_v. The manuscript now reports simulation-calibrated, explicitly approximate intervals (Fig. 4, Fig. 10), states that K_int is bounded on one side only for three mixes, and warns against interpreting small K_int differences. Manuscript: Sections 4.2, 5.2, 5.8, 6.2.

## 5. Sensitivity to modelling assumptions

* **Problem.** The constants α₀, n, S_c, S_cap, k, β, P_c0, η, the initial and boundary conditions and the mesh were fixed without quantifying their effect.
* **Change.** `exp_physics_sensitivity.py`: 13 one-at-a-time variants plus a finer mesh, eight representative mixes, direct BDF solver.
* **Results.** See Table 9 and Fig. 12. Constitutive constants shift K_int by up to 0.4 decade and D_v by up to 0.39 decade, nearly uniformly across mixes (rank correlation ≥ 0.88). Mesh refinement changes K_int by ≤ 0.007 decade. A sealed lateral face fits worse by a factor 2.18 in RMSE. P_c0 and η only rescale K_int (exact degeneracy, stated analytically).
* **Effect.** Parameters are now explicitly presented as conditional effective values; composition trends are robust. Manuscript: Sections 3.1, 5.9, 6.3.

## 6. Extended baselines and mix-level statistics

* **Problem.** Only one regression baseline and one data-driven network; method differences were judged from pooled R² values that treat correlated points as independent.
* **Change.** `exp_baselines.py`: OLS, ridge (inner LOO), quadratic+ridge, Gaussian process, PENN (original, 1×8, linear map, revised), data-driven network; identical folds. `exp_audit.py`: per-mix RMSE differences, 10 000-sample bootstrap over mixes, Wilcoxon test.
* **Results.** Hold-out RMSE: OLS 0.0135, PENN 0.0137, revised PENN 0.0132, quadratic 0.0130, GP 0.0204, data-driven 0.0202. LOFO lowest per-mix R²: physics-based linear models ≥ 0.962; GP 0.381, quadratic 0.198, data-driven 0.218. PENN vs OLS: LOMO Δ +0.0006 (95 % CI −0.0012 to +0.0020, p = 0.09); LOFO Δ +0.0005 (p = 0.22). PENN vs data-driven in LOFO: Δ −0.0172, p = 0.001.
* **Added after an internal review.** (i) Binder-family-level uncertainty: per-mix differences averaged within each of the seven families, cluster bootstrap and Wilcoxon test over families. PENN vs OLS: family-level p = 0.375 (LOMO) and 0.469 (LOFO); 90 % intervals −0.0009 to +0.0018 and −0.0003 to +0.0012, within a ±0.002 RMSE margin (about one third of the direct-fit RMSE; set during the revision, therefore descriptive). PENN vs GP in LOFO: cluster interval −0.0281 to −0.0040 but family-level Wilcoxon p = 0.078 (not significant with seven units). (ii) Wording: "equally accurate", "equivalent" and "statistically indistinguishable" were replaced by "no statistically detectable difference under the evaluated conditions". (iii) Inner-CV data-driven baseline (`exp_mlp_icv.py`, weight decay 10⁻³): hold-out RMSE 0.0309, LOMO 0.0326, LOFO 0.0335 (lowest per-mix R² 0.832, 7 mixes below 0.95). The PENN is significantly more accurate than both settings in LOFO; in LOMO only than the inner-CV setting. The original setting stays primary because it was fixed before the analysis, as was the PENN's.
* **Effect.** Contradicts the original framing that the PENN is the better method; the article was restructured around what generalises (transport model + low-capacity map). Manuscript: Sections 4.3, 5.3, 5.4, 6.1, Fig. 6, Tables 4–6.

## 7. Robustness to measurement and processing uncertainty

* **Problem.** Only i.i.d. Gaussian noise had been tested, and only for the PENN and the data-driven network.
* **Change.** `exp_robustness.py`: i.i.d. noise, correlated digitisation-like noise, time-origin errors, four reconstruction variants; classical, PENN, revised PENN, data-driven.
* **Results.** Correlated noise: RMSE 0.0147 (OLS), 0.0142 (PENN), 0.0271 (data-driven). Largest K_int shift under correlated noise: classical per-mix 0.219, PENN 0.009, revised PENN 0.133 decade.
* **Effect.** Physics-based predictions are robust. Per-mix K_int estimates are sensitive to digitisation-like errors; parameters predicted from composition are less so, and the very small shifts of the original PENN partly reflect its compression of K_int (section 9). Manuscript: Section 5.5, Table 7, Fig. 7.

## 8. Data efficiency

* **Problem.** No evidence on whether physics-constrained learning helps with limited data.
* **Change.** `exp_data_efficiency.py`: 5, 8, 11, 14 training mixes × 10 random draws; training curves cut after 3, 4, 7, 14 days.
* **Results.** 14 mixes: OLS 0.0142, PENN 0.0162, revised PENN 0.0159 (median RMSE). 5 mixes: PENN 0.0220 vs OLS 0.0256. 7-day curves: PENN 0.0138, OLS 0.0165, data-driven 0.0314.
* **Effect.** Identifies the genuine, limited advantage of end-to-end training. Manuscript: Section 5.5, Fig. 8, Section 6.1.

## 9. PENN parameter shrinkage and the revised PENN

* **Problem (new, found by synthetic tests).** The original PENN compresses K_int between mixes (real data: correlation with classical estimates 0.48).
* **Change.** `exp_pen_shrinkage.py` (synthetic data only) compared training variants. A first candidate (2×16 network, no weight decay) recovered K_int (slope 0.86) but, evaluated once on the experimental data, over-fitted (hold-out RMSE 0.0187, LOFO lowest R² 0.78); it was rejected and its result is reported. The adopted revision is a linear composition map without weight decay trained to convergence with L-BFGS (`PENTrainer.fit(optimizer="lbfgs")`).
* **Results.** Synthetic K_int slope 0.17 → 0.96; error on withheld mixes 0.088 → 0.022 decade (two-stage: 0.024). Experimental data: hold-out RMSE 0.0132; K_int correlation with classical estimates 0.48 → 0.78; K_int inside the classical 95 % profile interval for 10 → 15 of 20 mixes.
* **Effect.** Improves the physical consistency of the learned parameters at no cost in accuracy; documents a pitfall relevant to any learned-parameter model. Manuscript: Sections 4.3, 5.6, 6.2, Fig. 10.

## 10. PINN reassessment

* **Problem.** The PINN inconsistency could have been an optimisation artefact of one implementation.
* **Change.** `exp_pinn.py`: reference run, warm start at classical parameters, λ_PDE = 100, exact synthetic training data, and parameters frozen at classical values.
* **Results.** K_int bias: reference −1.13, warm start −1.08, λ = 100 −1.03, synthetic −1.13, frozen classical +0.00. Total loss with frozen classical parameters 0.135 vs free 0.135 (same), with larger PDE residual (RMS 0.107 vs 0.048); solver check with frozen classical parameters R² = 0.9988.
* **Effect.** Strengthens the original finding and points to a likely mechanism (a nearly flat loss along K_int; the field appears to fit the residual more easily with a too-low K_int); other PINN designs were not tested, so the mechanism is indicated rather than proven. The manuscript now limits the claim to this formulation and recommends a solver check for any PINN-identified parameter. Manuscript: Section 5.7, Fig. 9, Table D.1, Section 6.2.

## 11. Manuscript

Title, abstract, highlights, introduction (new literature on differentiable solvers, PINN loss landscapes, identifiability, ML for transport indicators), data section (source verification, curing-age caveat, reconstruction test), model (P_c0–K_int degeneracy), methods (tabulated vs direct solver distinguished; two-stage models; revised PENN; mix-level statistics), results reorganised by scientific question (verification, identifiability, prediction, generalisation, robustness and data efficiency, parameter recovery, PINN, interpretation, sensitivity, fields), new discussion (what the PENN adds and does not; accuracy is not identification; limitations; engineering implications), conclusions rewritten from the final evidence, appendices (numerics, reproducibility, hyper-parameter check, PINN diagnostics). 13 figures (7 new), 9 tables + 4 appendix tables, 38 references (7 new, all checked).

## 12. Not done, and why

* **Experimental validation of the parameters** (absorption tests, measured profiles): no such data are available. Stated as the main limitation.
* **Curing age of the digitised curves and lateral sealing**: cannot be read from the data file; the sealing assumption is supported by the fit (Section 5.9) and 28-day curing by the comparison of drying times with the source (Section 2.4), but both should be confirmed by the authors.
* **Rebuilding the table on a finer mesh**: not needed; mesh refinement changes parameters by ≤ 0.007 decade.
* **Other PINN architectures** (e.g. without the similarity ansatz, adaptive sampling, Bayesian PINNs): not tested; the conclusion is restricted to the formulation studied.
* **w/b = 0.6 and S20-0.5**: need re-digitisation by the authors.
