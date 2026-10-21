# Scientific revision report

Manuscript: *Learning moisture transport parameters of blended concretes from drying curves: identifiability, generalisation across binders, and a comparison of physics-embedded and physics-informed neural networks* (revised from "Physics-embedded neural network for predicting the moisture transport properties of concrete from mix composition, with a comparison to physics-informed neural networks").

All numbers below were computed by the scripts named in each section (repository branch `q1-revision`, folder `scripts/q1/`, outputs in `results/q1/`). Nothing was changed in the original data, the original manuscript or the original results; the original pipeline still runs unchanged (`pinn_moisture` keeps its default behaviour).

## 0. Summary of the outcome

| Question | Answer from the completed analyses |
|---|---|
| Are the published numbers reproducible? | Yes. An independent re-run from the raw file reproduces «repro_ok» of «repro_n» compared quantities within tolerance; the PINN runs are bit-for-bit deterministic. |
| Does the PENN beat classical identification? | No. PENN and linear two-stage inversion are statistically indistinguishable in hold-out, leave-one-mix-out and leave-one-family-out validation (paired Wilcoxon p = «pair_lomo_pen_ols_p» and «pair_lofo_pen_ols_p»). |
| What does generalise to a new binder family? | The transport model combined with a low-capacity (near-linear) composition map, fitted in two stages or end to end. Gaussian-process, quadratic and data-driven models fail for several mixes (lowest R² «lofo_min_flex»). |
| Where does end-to-end training help? | With short training curves (7-day curves: RMSE «tr7_pen» vs «tr7_ols») and with very few training mixes (5 mixes: «de5_pen» vs «de5_ols»). |
| Are the parameters trustworthy? | D_v: yes, to ±«prof_dv_med»/2 decade. K_int: only to a few tenths of a decade (median 95 % interval «prof_k_med» decade, up to «prof_k_max»), and conditional on fixed constants (shifts up to «sens_dk_max» decade). Mix rankings are robust. |
| Is the PINN result real? | Yes, and stronger than before: the K_int bias (≈ «sl_pinn_K_bias» decade) persists with exact synthetic data, a warm start and λ = 100; it disappears only when the parameters are frozen. |
| New problem found | The original PENN shrinks K_int between mixes (synthetic slope «shr_slope_paper»). Cause: weight decay plus an under-converged optimiser. A linear map trained with L-BFGS fixes it (slope «shr_slope_r») without loss of accuracy. |

## 1. Reproduction of the original results (Stage 1)

* **Problem.** The reported values had not been regenerated independently.
* **Change.** None to the method. A clean clone of the code was run from the raw data file (`/home/claude/work/repro`, scripts of section 3 of `paper/q1/README.md`) and compared with the original results by `scripts/q1/compare_repro.py`.
* **Results.** «repro_ok» of «repro_n» quantities agree within tolerance (Table B.1 of the manuscript; `results/q1/reproducibility/report.md`). Cleaned data are identical; classical parameters agree within 0.001 decade (multi-start randomness); all PENN, data-driven and CV results are identical; the verification files agree to the fifth significant digit. Two PINN runs were initially missing because the first re-run hit a 2-hour limit; they were run separately and reproduce the original values exactly.
* **Effect on findings.** Confirms the original numbers. Manuscript: Appendix B.

## 2. Source verification and curve reconstruction

* **Problem.** The data provenance relied on assumptions. The accepted manuscript of Zhuang et al. (UCL Discovery) shows drying curves at **two curing ages (28 and 56 days)**, which raised the possibility that the two digitised series of each mix are two different curves rather than duplicates.
* **Change.** New test `scripts/q1/series_relation.py`: for every mix, the long series is related to the short one by a pure time shift, a pure time scaling (expected if the curves came from different curing ages, since drying scales with t/D), or both.
* **Results.** Median RMS mismatch (w/b ≤ 0.5): shift «series_shift_rms», scaling «series_scale_rms», combined fit scale factor «series_affine_l» (IQR «series_affine_lo»–«series_affine_hi»). The two series are one curve with two time origins; the one-day offset matches the 24 h vacuum saturation. Offsets at the IQR bounds or estimated from training mixes only («shift_train» d) change the parameters by ≤ «rb_recon_dk_max» decade.
* **Effect.** Strengthens the cleaning step; the curing age of the digitised figure still has to be confirmed by the authors (kept as a review comment). Manuscript: Section 2.3, Fig. 1d, Section 5.5.

## 3. Leakage audit of shared choices

* **Problem.** The shape exponent n was chosen from a profile over six mixes, one of which (C-0.5) is a withheld mix; the time offset used all mixes; the network settings could have been tuned on the test mixes.
* **Change.** `exp_audit.py` recomputes the n-profile without C-0.5 and the offset from training mixes; `exp_inner_cv.py` runs an inner leave-one-family-out CV on the 14 training mixes for 9 PENN and 4 data-driven configurations.
* **Results.** Without C-0.5 the misfit at n = 2 is «n_gain_2_8_train» % higher than at n = 8 and only «n_gain_8_16_train» % lower at n = 16: n = 8 is unchanged. Offset from training mixes «shift_train» d (was 1.05 d). The original PENN setting ranks «icv_pen_rank»/«icv_pen_n» in inner CV (RMSE «icv_pen_paper» vs best «icv_pen_sel»); the inner-CV choice gives the same hold-out RMSE («icv_pen_test»). For the data-driven network the inner CV prefers stronger weight decay, which is worse on the withheld mixes («icv_mlp_test»), so the baseline was not handicapped.
* **Effect.** No leakage with an effect on the results. Manuscript: Sections 2.3, 4.2, Appendix C.

## 4. Identifiability: profile likelihoods and synthetic recovery

* **Problem.** Identifiability rested on Jacobian correlations only; the standard errors assumed independent residuals, which digitised curves do not have.
* **Change.** `exp_profiles.py`: profile likelihood of each parameter for all 20 mixes, with the effective sample size N_eff = N(1−ρ₁)/(1+ρ₁). `exp_synthetic.py` (C1): six known parameter sets, curves generated on a finer mesh, inverted under five conditions.
* **Results.** Median lag-1 autocorrelation «prof_rho_med»; median N_eff «prof_neff_med» of «prof_n_med» points. 95 % interval widths: D_v median «prof_dv_med» decade (max «prof_dv_max»); K_int median «prof_k_med» (max «prof_k_max», «prof_k_max_mix»); ignoring autocorrelation halves the K_int width («prof_k_unc_med»). Synthetic: exact data recovered within «syn_exact_dv»/«syn_exact_k» decade, also with 14-day data; noise 0.01 → K_int errors up to «syn_iid_k»; time-origin error 0.25 d → up to «syn_t0_k».
* **Effect.** Weakens the precision claimed for K_int; strengthens D_v. The manuscript now reports profile intervals (Fig. 4, Fig. 10) and warns against interpreting K_int differences < 0.3 decade. Manuscript: Sections 4.2, 5.2, 5.8, 6.2.

## 5. Sensitivity to modelling assumptions

* **Problem.** The constants α₀, n, S_c, S_cap, k, β, P_c0, η, the initial and boundary conditions and the mesh were fixed without quantifying their effect.
* **Change.** `exp_physics_sensitivity.py`: 13 one-at-a-time variants plus a finer mesh, eight representative mixes, direct BDF solver.
* **Results.** See Table 9 and Fig. 12. Constitutive constants shift K_int by up to «sens_dk_max» decade and D_v by up to «sens_ddv_max» decade, nearly uniformly across mixes (rank correlation ≥ «sens_rank_min»). Mesh refinement changes K_int by ≤ «sens_mesh_dk» decade. A sealed lateral face fits worse by a factor «ps_sealed_lateral_ratio» in RMSE. P_c0 and η only rescale K_int (exact degeneracy, stated analytically).
* **Effect.** Parameters are now explicitly presented as conditional effective values; composition trends are robust. Manuscript: Sections 3.1, 5.9, 6.3.

## 6. Extended baselines and mix-level statistics

* **Problem.** Only one regression baseline and one data-driven network; method differences were judged from pooled R² values that treat correlated points as independent.
* **Change.** `exp_baselines.py`: OLS, ridge (inner LOO), quadratic+ridge, Gaussian process, PENN (original, 1×8, linear map, revised), data-driven network; identical folds. `exp_audit.py`: per-mix RMSE differences, 10 000-sample bootstrap over mixes, Wilcoxon test.
* **Results.** Hold-out RMSE: OLS «ho_ols_rmse», PENN «ho_pen_rmse», revised PENN «ho_pen_r_rmse», quadratic «ho_quad_rmse», GP «ho_gp_rmse», data-driven «ho_mlp_rmse». LOFO lowest per-mix R²: physics-based linear models ≥ «lofo_min_phys»; GP «lofo_gp_min», quadratic «lofo_quad_min», data-driven «lofo_mlp_min». PENN vs OLS: LOMO Δ «pair_lomo_pen_ols_d» (95 % CI «pair_lomo_pen_ols_lo» to «pair_lomo_pen_ols_hi», p = «pair_lomo_pen_ols_p»); LOFO Δ «pair_lofo_pen_ols_d» (p = «pair_lofo_pen_ols_p»). PENN vs data-driven in LOFO: Δ «pair_lofo_pen_mlp_d», p = «pair_lofo_pen_mlp_p».
* **Effect.** Contradicts the original framing that the PENN is the better method; the article was restructured around what generalises (transport model + low-capacity map). Manuscript: Sections 4.3, 5.3, 5.4, 6.1, Fig. 6, Tables 4–6.

## 7. Robustness to measurement and processing uncertainty

* **Problem.** Only i.i.d. Gaussian noise had been tested, and only for the PENN and the data-driven network.
* **Change.** `exp_robustness.py`: i.i.d. noise, correlated digitisation-like noise, time-origin errors, four reconstruction variants; classical, PENN, revised PENN, data-driven.
* **Results.** Correlated noise: RMSE «rb_corr_001_ols» (OLS), «rb_corr_001_pen» (PENN), «rb_corr_001_mlp» (data-driven). Largest K_int shift under correlated noise: classical per-mix «rb_corr_001_ols_dk», PENN «rb_corr_001_pen_dk», revised PENN «rb_corr_001_pen_r_dk» decade.
* **Effect.** Physics-based predictions are robust. Per-mix K_int estimates are sensitive to digitisation-like errors; parameters predicted from composition are less so, and the very small shifts of the original PENN partly reflect its compression of K_int (section 9). Manuscript: Section 5.5, Table 7, Fig. 7.

## 8. Data efficiency

* **Problem.** No evidence on whether physics-constrained learning helps with limited data.
* **Change.** `exp_data_efficiency.py`: 5, 8, 11, 14 training mixes × 10 random draws; training curves cut after 3, 4, 7, 14 days.
* **Results.** 14 mixes: OLS «de14_ols», PENN «de14_pen», revised PENN «de14_pen_r» (median RMSE). 5 mixes: PENN «de5_pen» vs OLS «de5_ols». 7-day curves: PENN «tr7_pen», OLS «tr7_ols», data-driven «tr7_mlp».
* **Effect.** Identifies the genuine, limited advantage of end-to-end training. Manuscript: Section 5.5, Fig. 8, Section 6.1.

## 9. PENN parameter shrinkage and the revised PENN

* **Problem (new, found by synthetic tests).** The original PENN compresses K_int between mixes (real data: correlation with classical estimates «th_log10_K_corr»).
* **Change.** `exp_pen_shrinkage.py` (synthetic data only) compared training variants. A first candidate (2×16 network, no weight decay) recovered K_int (slope «shr_slope_nowd») but, evaluated once on the experimental data, over-fitted (hold-out RMSE «nowd_holdout_rmse», LOFO lowest R² «nowd_lofo_min»); it was rejected and its result is reported. The adopted revision is a linear composition map without weight decay trained to convergence with L-BFGS (`PENTrainer.fit(optimizer="lbfgs")`).
* **Results.** Synthetic K_int slope «shr_slope_paper» → «shr_slope_r»; error on withheld mixes «shr_maek_paper» → «shr_maek_r» decade (two-stage: «syn2_ols_maek»). Experimental data: hold-out RMSE «ho_pen_r_rmse»; K_int correlation with classical estimates «th_log10_K_corr» → «thr_K_corr»; K_int inside the classical 95 % profile interval for «th_K_inside» → «thr_K_inside» of 20 mixes.
* **Effect.** Improves the physical consistency of the learned parameters at no cost in accuracy; documents a pitfall relevant to any learned-parameter model. Manuscript: Sections 4.3, 5.6, 6.2, Fig. 10.

## 10. PINN reassessment

* **Problem.** The PINN inconsistency could have been an optimisation artefact of one implementation.
* **Change.** `exp_pinn.py`: reference run, warm start at classical parameters, λ_PDE = 100, exact synthetic training data, and parameters frozen at classical values.
* **Results.** K_int bias: reference «pinn_reference_bias», warm start «pinn_warm_start_bias», λ = 100 «pinn_lambda_100_bias», synthetic «pinn_synthetic_bias», frozen classical «pinn_fixed_classical_bias». Total loss with frozen classical parameters «pinn_fixed_classical_total» vs free «pinn_reference_total» (same), with larger PDE residual (RMS «pinn_fixed_classical_pderms» vs «pinn_reference_pderms»); solver check with frozen classical parameters R² = «pinn_fixed_fwd4».
* **Effect.** Strengthens and explains the original finding (flat loss along K_int; the field fits the residual more easily with a too-low K_int). The manuscript now limits the claim to this formulation and recommends a solver check for any PINN-identified parameter. Manuscript: Section 5.7, Fig. 9, Table D.1, Section 6.2.

## 11. Manuscript

Title, abstract, highlights, introduction (new literature on differentiable solvers, PINN loss landscapes, identifiability, ML for transport indicators), data section (source verification, curing-age caveat, reconstruction test), model (P_c0–K_int degeneracy), methods (tabulated vs direct solver distinguished; two-stage models; revised PENN; mix-level statistics), results reorganised by scientific question (verification, identifiability, prediction, generalisation, robustness and data efficiency, parameter recovery, PINN, interpretation, sensitivity, fields), new discussion (what the PENN adds and does not; accuracy is not identification; limitations; engineering implications), conclusions rewritten from the final evidence, appendices (numerics, reproducibility, hyper-parameter check, PINN diagnostics). 13 figures (7 new), 9 tables + 4 appendix tables, 38 references (7 new, all checked).

## 12. Not done, and why

* **Experimental validation of the parameters** (absorption tests, measured profiles): no such data are available. Stated as the main limitation.
* **Curing age of the digitised curves and lateral sealing**: cannot be determined from the data file; the sealing assumption is supported by the fit (Section 5.9), the curing age needs the authors.
* **Rebuilding the table on a finer mesh**: not needed; mesh refinement changes parameters by ≤ «sens_mesh_dk» decade.
* **Other PINN architectures** (e.g. without the similarity ansatz, adaptive sampling, Bayesian PINNs): not tested; the conclusion is restricted to the formulation studied.
* **w/b = 0.6 and S20-0.5**: need re-digitisation by the authors.
