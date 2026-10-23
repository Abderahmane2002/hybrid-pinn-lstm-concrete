# Q1 submission assessment

This assessment was written after all revision analyses were completed. It reads the revised manuscript the way three independent referees would, and it states where the work stands. It contains no acceptance probabilities.

## 1. Scientific novelty (honest view)

What is new:

1. A **practical identifiability analysis** of a vapour–capillary drying model of real disc specimens: profile likelihoods corrected for the autocorrelation of digitised residuals and calibrated by simulation, synthetic recovery tests and a one-at-a-time sensitivity study of every fixed constant. The result is precise: D_v is known within a median 95 % interval of «prof_dv_med» decade, K_int only to a few tenths of a decade (median interval «prof_k_med» decade, one-sided for «prof_k_nbound» mixes) and only conditionally on the constants.
2. A **controlled, statistically sound comparison** of end-to-end (physics-embedded) and two-stage learning of composition → transport parameters, with eight models (plus a second, inner-CV-selected data-driven baseline) under identical folds and mix- and family-level paired statistics. It finds no statistically detectable difference between the two routes on complete data, with family-level 90 % intervals within a ±0.002 RMSE margin set in this revision; this is evidence of similar accuracy under the conditions studied, not proof of equivalence. What generalises to a new binder family is the transport model combined with a low-capacity composition map; flexible maps fail even when they use the transport model.
3. A **diagnostic account of PINN parameter inconsistency** for this problem (the cause is indicated, not proven). The K_int bias (≈ «sl_pinn_K_bias» decade) survives exact synthetic data, a warm start and a ten-fold PDE weight. The PINN loss appears nearly flat along K_int.
4. A documented **pitfall of learned-parameter models**: weight decay and an under-converged optimiser compress a weakly identifiable parameter without affecting predictions. It is revealed only by synthetic recovery and fixed by a converged linear map.

What is not new: differentiable-solver learning itself (Bandai et al. 2024 for soils; differentiable physics in general), and the transport model (Chen et al. 2022). The paper's contribution is methodological and diagnostic, not a new model, and the manuscript says so.

**Verdict on novelty:** sufficient for an applied Q1 journal in building materials or computational civil engineering, *provided the paper is framed as it now is* (identifiability, generalisation and consistency), not as "a new network that beats classical methods". It would not be sufficient for a journal expecting new experimental findings.

## 2. Verified predictive performance

| | Hold-out RMSE (R²) | LOMO RMSE | LOFO RMSE | LOFO lowest per-mix R² |
|---|---|---|---|---|
| Two-stage, linear | «ho_ols_rmse» («ho_ols_r2») | «lomo_ols_rmse» | «lofo_ols_rmse» | «lofo_ols_min» |
| PENN (original) | «ho_pen_rmse» («ho_pen_r2») | «lomo_pen_rmse» | «lofo_pen_rmse» | «lofo_pen_min» |
| PENN (revised) | «ho_pen_r_rmse» («ho_pen_r_r2») | «lomo_pen_r_rmse» | «lofo_pen_r_rmse» | «lofo_pen_r_min» |
| Two-stage, Gaussian process | «ho_gp_rmse» («ho_gp_r2») | «lomo_gp_rmse» | «lofo_gp_rmse» | «lofo_gp_min» |
| Data-driven network (original setting) | «ho_mlp_rmse» («ho_mlp_r2») | «lomo_mlp_rmse» | «lofo_mlp_rmse» | «lofo_mlp_min» |
| Data-driven network (inner-CV setting) | «ho_mlp_icv_rmse» («ho_mlp_icv_r2») | «lomo_mlp_icv_rmse» | «lofo_mlp_icv_rmse» | «lofo_mlp_icv_min» |

All values were recomputed in this revision. The original values were reproduced independently («repro_ok»/«repro_n» quantities).

## 3. Comparison with classical methods

The PENN does **not** outperform classical inversion followed by linear regression. No statistically detectable difference: paired per-mix tests give LOMO p = «pair_lomo_pen_ols_p» and LOFO p = «pair_lofo_pen_ols_p»; with the binder family as the unit, p = «pair_lomo_pen_ols_fp» and «pair_lofo_pen_ols_fp», and the 90 % family-level intervals («pair_lomo_pen_ols_f90lo» to «pair_lomo_pen_ols_f90hi»; «pair_lofo_pen_ols_f90lo» to «pair_lofo_pen_ols_f90hi») lie within ±0.002 RMSE. The manuscript avoids calling the methods equivalent and states the margin was set during the revision. Classical inversion remains the reference for per-mix parameter estimation. The PENN's demonstrated advantages are:

- robustness with short curves («tr7_pen» vs «tr7_ols» RMSE with 7-day curves);
- stability with very few training mixes for the regularised original PENN («de5_pen» vs «de5_ols» with 5 mixes);
- composition-level parameters that are much less sensitive to digitisation errors than per-mix fits;
- convenience: one training run, no per-mix inversion.

The manuscript states this plainly (Sections 5.4, 5.5 and 6.1, and Conclusions 2–4).

## 4. Simulated review

### Reviewer 1: concrete durability specialist

*Major*

- **R1.1** The study uses a single, digitised dataset from one laboratory and one drying condition, and the curing age of the curves (28 or 56 d) was not recorded with the digitised figure. *Status:* partly addressed. Section 2.4 compares the time to 70 % saturation of the reconstructed curves (mean «ca_t70_03», «ca_t70_04», «ca_t70_05» d for w/b 0.3, 0.4, 0.5) with the 28-day values tabulated in the source, which supports 28 d; Section 5.8 states that composition effects are interpreted for that age. The authors should still confirm it from the source.
- **R1.2** There is no independent experimental validation of D_v and K_int: no sorption isotherms, permeability tests or measured moisture profiles. *Status:* disclosed and framed as "consistency, not validation". It remains the main scientific weakness.
- **R1.3** Eight of 28 mixes (w/b = 0.6, S20-0.5) are excluded. The trends therefore cover w/b 0.3–0.5 only. *Status:* disclosed; re-digitisation is recommended.

*Minor*

- **R1.4** Isothermal model at 50 °C, while service conditions differ. Temperature dependence is not addressed. *Status:* stated in Section 6.3.
- **R1.5** The internal saturation fields are presented as useful for durability models, but they are unvalidated. *Status:* now qualified in Section 5.10.

### Reviewer 2: numerical modelling and inverse problems

*Major*

- **R2.1** Are the parameters identifiable? *Status:* addressed with Jacobian analysis, autocorrelation-corrected profile likelihoods and synthetic recovery (Section 5.2, Fig. 4).
- **R2.2** How much do the fixed constants condition the results? *Status:* addressed (Section 5.9, Table 9). Shifts reach «sens_dk_max» decade in K_int, but rankings are preserved (rank correlation ≥ «sens_rank_min»).
- **R2.3** Solver verification and table accuracy. *Status:* addressed (Section 5.1, Appendix A). Mesh refinement moves the parameters by ≤ «sens_mesh_dk» decade.

*Minor*

- **R2.4** Are the corrected intervals calibrated? *Status:* addressed by a correlated-error coverage simulation (Section 4.2). The N_eff correction alone gave 88–90 % coverage, and the intervals are now calibrated to 95 % and labelled approximate. A residual block bootstrap could serve as an independent check if a reviewer asks for one.
- **R2.5** The sealed-lateral-face test supports unsealed drying, but the evidence is indirect. *Status:* stated as supporting, not proving.

### Reviewer 3: scientific machine learning

*Major*

- **R3.1** Novelty relative to differentiable-physics learning (Bandai et al. 2024). *Status:* cited. The contribution is framed as identifiability, generalisation and consistency rather than a new method.
- **R3.2** Fairness of the baselines and hyper-parameter tuning. *Status:* addressed. Eight models run on identical folds. The original PENN and data-driven settings were fixed before this analysis and are kept as primary for both; the inner CV on training mixes confirms the PENN setting but prefers a stronger weight decay for the data-driven network (original setting ranks last of four). That inner-CV setting is now evaluated in all schemes and reported alongside (Section 4.3, Tables 4 and 6, Fig. 6, Appendix C). The LOFO conclusion holds for both settings; the LOMO comparison with the data-driven network depends on the setting, and the text says so. Test accuracy was not used to choose the primary setting.
- **R3.3** Is the PINN comparison fair, and does it generalise? *Status:* addressed by the diagnostic runs (Section 5.7). The conclusion is explicitly restricted to the formulation tested. A reviewer may still ask for a different PINN architecture. *Suggested response:* explain that the diagnostics isolate a loss-landscape property, and offer one additional variant if requested.

*Minor*

- **R3.4** Statistical unit. *Status:* addressed with mix-level paired bootstrap and Wilcoxon tests.
- **R3.5** The revised PENN was introduced after the original results were known. *Status:* the selection was made on synthetic data only and evaluated once on the experimental data. Both versions are reported, and the rejected intermediate variant is reported as well.
- **R3.6** Reproducibility. *Status:* complete. Code, data, an independent re-run and an automated build of every number are provided; a Zenodo DOI is still to be added.

## 5. Remaining limitations

1. One digitised dataset; curing age and lateral sealing not recorded; 8 mixes unusable.
2. No experimental validation of the identified parameters.
3. Parameters conditional on fixed constants (K_int ± a few tenths of a decade from this source alone).
4. One PINN formulation tested.
5. Composition described by three variables only (w/b, fly ash and slag fractions); the effects of curing, binder chemistry and aggregate are not represented.

## 6. Journal recommendation

| Journal | Fit | Comment |
|---|---|---|
| **Journal of Building Engineering** (Elsevier) | **Best fit** | Q1 in Construction & Building Technology and Civil Engineering (JCR quartiles as listed by LetPub, crowd-sourced, year not stated). It published both the source data (Zhuang et al. 2022) and the transport model (Chen et al. 2022). Its scope covers the life cycle of the built environment, including deterioration. Numerical and ML studies on existing data are regularly published. |
| Construction and Building Materials (Elsevier) | Good | Q1 (same source). Its scope explicitly includes laboratory and numerical investigations. Larger readership and somewhat stricter on novelty; the lack of new experiments may be raised. |
| Cement and Concrete Composites (Elsevier) | Weak | Ranked first in Construction & Building Technology. Its scope includes durability and modelling, but it generally expects new materials insight or experiments; a purely re-analytical study on one digitised dataset is likely to be considered insufficient. |
| Computer-Aided Civil and Infrastructure Engineering | Possible, but riskier | It expects a stronger general ML contribution than this paper offers. Scope and quartile not verified in this revision. |

Quartile information should be confirmed through Clarivate JCR or Scopus via the library before submission. The journal websites (sciencedirect.com) could not be accessed from the analysis environment.

## 7. Is the manuscript ready?

**Scientifically:** the analyses that can be done with the available data are complete, internally consistent and reproducible. The conclusions match the evidence, and the limitations are disclosed. The manuscript is **ready for submission to the Journal of Building Engineering or Construction and Building Materials** once the author-only items below are done. It should not be called "validated" in the experimental sense.

**Before submission (authors only):**

1. Confirm the curing age of the digitised curves, the digitisation tool, and whether the lateral surface was sealed during drying (review comment in Section 2.1).
2. Corresponding e-mail, X. Wang's CRediT roles, funding, generative-AI declaration.
3. Archive the code release on Zenodo and add the DOI.
4. Ask Prof. Chen to review Section 3 (his model) and the interpretation of K_int.
5. Optional but valuable: re-digitise w/b = 0.6 and S20-0.5, and obtain the raw mass-loss and absorption data from Zhuang et al., which would allow an independent validation.
