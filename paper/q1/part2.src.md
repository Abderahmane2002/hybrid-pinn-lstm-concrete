
# 5 Results

## 5.1 Numerical verification

The direct solver behaves as expected (Appendix A, Table A.1). For a constant diffusivity it reproduces the analytical series solution of slab drying to within 2 × 10⁻³. Under mesh refinement the slab solution converges at second order, with an error of «fv_err80» in $S_{\text{avg}}$ at 80 cells, and the water lost through the drying face balances the change in stored water to machine precision (relative error «mass_err»). The disc solution on the 24 × 16 mesh differs from a 96 × 64 reference by at most «disc_err», mostly in the first hours of drying, before the first measurement. Refitting eight mixes on a 36 × 24 mesh changed the identified parameters by at most «sens_mesh_ddv» decade for $D_v$ and «sens_mesh_dk» decade for $K_{\text{int}}$, well inside their uncertainty (Section 5.2). The tabulated solver differs from the direct solver by at most «table_err» in $S_{\text{avg}}$, and classical fits obtained with it agree with direct fits to within «table_fit_ddv» decade for $D_v$ and «table_fit_dk» decade for $K_{\text{int}}$. Differences between the learning methods compared below are therefore not caused by numerical approximation.

## 5.2 Identifiability of the transport parameters

With $\alpha_0$ and $n$ fixed, the disc model reproduces every drying curve with $R^2$ between «cl_r2_min» and «cl_r2_max» (median «cl_r2_med», median RMSE «cl_rmse_med»). Freeing more parameters does not help (Table 3). When $D_v$, $\alpha_0$ and $K_{\text{int}}$ are fitted together, $D_v$ and $\alpha_0$ are perfectly correlated ($\lvert\rho\rvert$ = «id_corr3_min») and the condition number of the Jacobian rises to «id_cond3_lo»–«id_cond3_hi»; adding $n$ leaves parameter pairs correlated above «id_corr4_min». With two free parameters the correlation is «id_corr2_lo»–«id_corr2_hi» and the condition number «id_cond2_lo»–«id_cond2_hi». The total misfit decreases with $n$ and levels off beyond $n \approx 8$: on the five training mixes of the profile it is «n_gain_2_8_train»% higher at $n = 2$ and only «n_gain_8_16_train»% lower at $n = 16$, the same pattern as with the withheld mix included. The value $n = 8$ lies in the range reported by Bažant and Najjar ⟦bazant⟧. Because $D_v$ and $\alpha_0$ cannot be separated, the identified $D_v$ is conditional on $\alpha_0 = 0.05$.

⟪TABLE:identifiability⟫

Table: **Table 3.** Jacobian-based identifiability with the disc model ($n = 8$ when not identified): fit quality, largest parameter correlation and condition number.

The Jacobian gives only a local picture. The profile likelihoods (Fig. 4a–c) show a well-defined minimum for both parameters in every mix, with all 95% intervals inside the admissible range. The residuals of the digitised curves are strongly autocorrelated (median lag-1 correlation «prof_rho_med», so that a median of «prof_n_med» points carries the information of about «prof_neff_med» independent ones). With this correction, $D_v$ is determined to within «prof_dv_med» decade (median width of the 95% interval; at most «prof_dv_max») and $K_{\text{int}}$ to within «prof_k_med» decade, but up to «prof_k_max» decade for «prof_k_max_mix». The intervals that ignore autocorrelation are about half as wide (median «prof_k_unc_med» decade for $K_{\text{int}}$) and would overstate the precision. The reference diffusivity is thus a precise quantity, whereas the permeability is known only to a few tenths of a decade from a drying curve.

The synthetic tests confirm this asymmetry (Fig. 4d). Inverting exact curves generated on a finer mesh recovers the true parameters to within «syn_exact_dv» decade for $D_v$ and «syn_exact_k» decade for $K_{\text{int}}$, also when only the first 14 days are available («syn_short_k» decade), so the model is structurally identifiable over the experimental time range. Measurement-like perturbations affect mainly $K_{\text{int}}$: independent noise of 0.01 produces errors of up to «syn_iid_k» decade (against «syn_iid_dv» for $D_v$), correlated noise up to «syn_corr_k», and an error of 0.25 d in the time origin up to «syn_t0_k» decade. $K_{\text{int}}$ controls the capillary stage of the first day, which is described by only a few early points of each curve.

![**Fig. 4.** Identifiability. (a, b) Profile deviance, corrected for residual autocorrelation, of $\log_{10} K_{\text{int}}$ and $\log_{10} D_v$ for four mixes; the dashed line is the 95% threshold. (c) 95% profile intervals of $\log_{10} K_{\text{int}}$ for all mixes, with and without the correction. (d) Recovery errors on synthetic curves with known parameters under five conditions (six parameter sets, three replicates for the noisy conditions).](fig/fig04_identifiability.png){width=6.5in}

## 5.3 Prediction of withheld mixes

Table 4 lists the accuracy on the six withheld mixes and Fig. 5 shows the curves. All methods that use the transport model are close: the PENN reaches $R^2$ = «ho_pen_r2» (RMSE = «ho_pen_rmse» ± «ho_pen_rmse_sd» over five seeds), two-stage inversion with linear regression «ho_ols_r2» (RMSE «ho_ols_rmse»), and the other two-stage and PENN variants lie between RMSE «ho_quad_rmse» and «ho_pen_lin_rmse». The data-driven network is less accurate (RMSE «ho_mlp_rmse»), and so is the Gaussian-process regression (RMSE «ho_gp_rmse»), which extrapolates poorly from 14 mixes. Fitting the model directly to the withheld curves, which uses their data, gives RMSE «ho_fdfit_rmse»; the difference with the predictive methods measures what is lost when the parameters must be inferred from three composition descriptors.

⟪TABLE:holdout⟫

Table: **Table 4.** Accuracy on the six withheld mixes («n_test_pts» points). Networks: mean ± standard deviation over five seeds. ^a^ After a one-off tabulation of the disc solutions («table_time_min» min). ^b^ Sum of the per-mix fits of the training mixes.

⟪TABLE:per_mix⟫

Table: **Table 5.** Coefficient of determination for each withheld mix (networks: mean over five seeds).

The largest differences between methods occur for C-0.5, the most permeable withheld mix and the only plain concrete at w/b = 0.5 (Table 5). The data-driven network underestimates its saturation during the first week ($R^2$ = «mix_mlp_C-0.5»), whereas the physics-based models predict it with $R^2$ ≥ 0.98. In these models, faster drying of a more permeable mix follows from larger parameters, not from a curve shape that must have been seen in training.

![**Fig. 5.** Drying curves of the six withheld mixes: experiment and predictions. Bands show the range over random seeds; $R^2$ refers to the PENN.](fig/fig05_test_curves.png){width=6.5in}

## 5.4 Generalisation across mixes and binder families

Cross-validation over all 20 mixes gives the same ranking (Table 6, Fig. 6). In leave-one-mix-out validation every physics-based model reaches a pooled $R^2$ of 0.991–0.992, and the paired per-mix comparison finds no significant difference between the PENN and linear two-stage inversion (mean RMSE difference «pair_lomo_pen_ols_d», 95% interval «pair_lomo_pen_ols_lo» to «pair_lomo_pen_ols_hi»; Wilcoxon $p$ = «pair_lomo_pen_ols_p»). The Gaussian-process and quadratic regressions are significantly worse ($p$ = «pair_lomo_pen_gp_p» and «pair_lomo_pen_quad_p»).

Withholding a whole binder family separates the models more clearly. Linear two-stage inversion, ridge regression and the PENN variants keep $R^2$ ≥ «lofo_min_phys» for every mix, and again do not differ significantly from one another (PENN against linear inversion: «pair_lofo_pen_ols_d», interval «pair_lofo_pen_ols_lo» to «pair_lofo_pen_ols_hi», $p$ = «pair_lofo_pen_ols_p»). The flexible models fail for several mixes: the lowest per-mix $R^2$ is «lofo_gp_min» for the Gaussian process, «lofo_quad_min» for quadratic regression and «lofo_mlp_min» for the data-driven network, which has «lofo_mlp_nbelow» mixes below $R^2$ = 0.95. The PENN improves on the data-driven network by «pair_lofo_pen_mlp_d» in mean per-mix RMSE (interval «pair_lofo_pen_mlp_lo» to «pair_lofo_pen_mlp_hi», $p$ = «pair_lofo_pen_mlp_p»). The quadratic and Gaussian-process regressions also use the transport model, so the failures are not caused by the absence of physics: they come from composition maps that are too flexible to extrapolate to a binder they have not seen. What generalises is the combination of the transport model with a low-capacity, nearly linear composition map, whether it is fitted in two stages or end to end.

⟪TABLE:cv⟫

Table: **Table 6.** Cross-validation over the 20 mixes (one seed per fold). Lowest $R^2$ and number of mixes below 0.95 refer to per-mix values.

![**Fig. 6.** Model comparison. (a) Pooled RMSE in the three validation schemes. (b) Per-mix $R^2$ when the whole binder family is withheld; values below 0.80 are drawn at the lower edge and counted. (c) Mean per-mix RMSE differences between pairs of methods with 95% bootstrap intervals over mixes; filled symbols: Wilcoxon $p$ < 0.05.](fig/fig06_comparison.png){width=6.5in}

## 5.5 Robustness and data efficiency

**Measurement and processing uncertainty.** Perturbing the training data changes the accuracy on the withheld mixes very little for the physics-based models (Fig. 7a, Table 7). With correlated, digitisation-like errors the RMSE becomes «rb_corr_001_pen» for the PENN and «rb_corr_001_ols» for linear two-stage inversion, against «rb_corr_001_mlp» for the data-driven network; independent noise and time-origin errors have smaller effects. The reconstruction of the curves is not a sensitive choice: moving the time offset to either end of its interquartile range, or estimating it from the training mixes only, changes the classical parameters by at most «rb_recon_ddv_max» decade for $D_v$ and «rb_recon_dk_max» decade for $K_{\text{int}}$. Using only the short (≤ 14 d) or only the long series gives the same accuracy. The parameters, however, react differently. The per-mix classical estimates of $K_{\text{int}}$ move by up to «rb_corr_001_ols_dk» decade under correlated noise and «rb_t0_025_ols_dk» decade under time-origin errors (Fig. 7b), consistent with the profile intervals, whereas parameters predicted from composition move less, because they pool all training curves: by up to «rb_corr_001_pen_r_dk» decade for the revised PENN. The original PENN moves by only «rb_corr_001_pen_dk» decade, but this stability partly reflects the compression of $K_{\text{int}}$ discussed in Section 5.6.

⟪TABLE:robustness⟫

Table: **Table 7.** Robustness: RMSE on the unperturbed withheld curves after perturbing the training data, and largest shift of $\log_{10} K_{\text{int}}$ over the 20 mixes (classical: per-mix fits; PENN: predicted parameters). Noise variants: mean of five replicates.

![**Fig. 7.** Robustness. (a) RMSE on the withheld mixes after perturbing the training data. (b) Largest shift of the identified or predicted $\log_{10} K_{\text{int}}$ relative to the unperturbed reference.](fig/fig07_robustness.png){width=6.5in}

**Number of training mixes.** With 8, 11 or 14 training mixes, linear two-stage inversion and the PENN variants are equivalent; with 14 mixes the median RMSE over ten random draws is «de14_ols» for linear inversion, «de14_pen» for the PENN and «de14_pen_r» for the revised PENN (Fig. 8a,b). With only five mixes, the PENN is more accurate and more stable (median RMSE «de5_pen», worst-mix $R^2$ «de5_pen_worst») than linear inversion («de5_ols», «de5_ols_worst») and ridge regression («de5_ridge»), whose four regression coefficients are poorly constrained by five points. The data-driven network and the Gaussian process are the least accurate at every training size.

**Short drying curves.** The clearest advantage of end-to-end training appears when the training curves are short (Fig. 8c). With curves cut after 7 days («tr7_npts» points instead of «trfull_npts»), the PENN predicts the complete curves of the withheld mixes with RMSE «tr7_pen», against «tr7_ols» for two-stage inversion and «tr7_mlp» for the data-driven network. With curves cut after 3 or 4 days, two-stage inversion can only use the «tr4_nfit» mixes that still have at least three points, and its error rises to «tr4_ols»; the PENN, which uses every point of every training mix, stays at «tr4_pen». Per-mix inversion of a short curve is poorly conditioned, whereas the PENN constrains the parameters of all mixes jointly through the composition map.

![**Fig. 8.** Data efficiency. (a) Median pooled RMSE (band: interquartile range over ten random draws of the training mixes) and (b) median worst-mix $R^2$ against the number of training mixes. (c) RMSE on the complete withheld curves when the training curves are cut after a given drying time.](fig/fig08_efficiency.png){width=6.5in}

## 5.6 End-to-end learning and parameter recovery

Accuracy on withheld curves does not show whether the learned parameters are right. The composition maps were therefore also tested on synthetic curves generated, for the 20 mixes and at their measurement times, from known parameters that depend linearly on composition, with noise of 0.01 added. Linear two-stage inversion recovers the permeability of the withheld mixes with a mean error of «syn2_ols_maek» decade. The PENN in its original setting recovers $D_v$ well but compresses the variation of $K_{\text{int}}$ between mixes: the slope of estimated against true $\log_{10} K_{\text{int}}$ over the training mixes is only «shr_slope_paper», and the error on the withheld mixes is «shr_maek_paper» decade. Training longer with the same weight decay does not help (slope «shr_slope_wd2000»). Removing the weight decay restores the slope («shr_slope_nowd») on synthetic data, but on the experimental data the unregularised 2 × 16 network then over-fits the composition map: its hold-out RMSE rises to «nowd_holdout_rmse» and its lowest per-mix $R^2$ in leave-one-family-out validation falls to «nowd_lofo_min». Weight decay is thus what lets the original PENN generalise, and also what biases its permeabilities.

A linear composition map cannot over-fit in this way. Trained with Adam and a decaying step, it remains under-converged in the weakly sensitive $K_{\text{int}}$ direction (slope «shr_slope_lin_adam»); trained to convergence with L-BFGS and without weight decay, it recovers $K_{\text{int}}$ as well as two-stage inversion (slope «shr_slope_r», mean error «shr_maek_r» decade on the withheld mixes). When the synthetic truth also contains mix-to-mix scatter not explained by composition, the two have similar errors on the withheld mixes («shr_mix_maek_r» and «syn2_mix_ols_maek» decade). This revised PENN was selected on synthetic data only and then evaluated once on the experimental data. Its predictive accuracy is close to that of the original PENN and of linear two-stage inversion (hold-out RMSE «ho_pen_r_rmse»; lowest leave-one-family-out $R^2$ «lofo_pen_r_min»; Tables 4 and 6), and its permeabilities follow the classical estimates more closely (correlation «thr_K_corr», against «th_log10_K_corr» for the original PENN), falling inside the 95% profile interval of the classical estimate for «thr_K_inside» of the 20 mixes (original PENN: «th_K_inside»). A hidden regulariser or an under-converged optimiser can therefore bias a weakly identifiable parameter without any visible effect on the predicted curves; only the synthetic recovery test revealed it.

## 5.7 PENN versus PINN consistency

The PINN was compared with the PENN on the slab model with identical constants, data split and parameter network (Table 8). On the withheld mixes the PINN is less accurate (RMSE «sl_pinn_rmse» against «sl_pen_rmse») and varies more between seeds, although it fits the training data far more closely (training RMSE «sl_pinn_train_rmse» against «sl_pen_train_rmse»). Without its PDE residual it degrades further (RMSE «sl_pinn_nopde_rmse»), so the residual does carry information.

The solver check is decisive (Fig. 9a,b). Run with the PENN parameters, the direct solver reproduces the PENN output with $R^2$ = «sl_fwd_pen_r2». Run with the PINN parameters it does not ($R^2$ = «sl_fwd_pinn_r2»): the PINN reference diffusivities are close to the classical values (mean difference «sl_pinn_Dv_mae» decade), but its permeabilities are about «sl_pinn_K_bias» decade too low.

Four diagnostic runs locate the cause (Fig. 9c,d and Appendix D). Starting the PINN from the classical parameters does not prevent the drift (bias «pinn_warm_start_bias» decade), and a ten times larger PDE weight barely changes it («pinn_lambda_100_bias»). Training on synthetic curves generated by the slab solver from the classical parameters, i.e. without model error or noise, gives the same bias («pinn_synthetic_bias») although the PINN then fits the data to RMSE «pinn_synthetic_rmse». Finally, with the parameters fixed at their classical values, the PINN reaches the same total loss as with free parameters («pinn_fixed_classical_total» against «pinn_reference_total»), with a lower data misfit but a larger PDE residual (RMS «pinn_fixed_classical_pderms» against «pinn_reference_pderms»), and its parameters then pass the solver check ($R^2$ = «pinn_fixed_fwd4»). The PINN loss is therefore nearly flat along $K_{\text{int}}$: the field network finds it easier to satisfy the residual with a too-low permeability, which slows the early capillary stage that the similarity ansatz represents least well, and the data term cannot tell the difference because that stage ends before the first measurement. The inconsistency is a property of the soft-constrained formulation for this problem, not of the data, the initialisation or the weighting.

⟪TABLE:slab⟫

Table: **Table 8.** Controlled comparison on the one-dimensional slab model (six withheld mixes). Solver check: $R^2$ between each model and the direct solver run with the parameters returned by that model.

![**Fig. 9.** PINN consistency on the slab model. (a, b) Average saturation computed by the direct solver with the parameters returned by the PINN and the PENN, against the output of each network. (c) Mean bias of the PINN permeability relative to the classical estimates in the diagnostic runs; a PINN with parameters fixed at the classical values has none. (d) Final loss terms with free and with classical parameters.](fig/fig09_pinn.png){width=6.5in}

## 5.8 Transport parameters and their physical interpretation

Figure 10 compares the classical estimates, with their 95% profile intervals, with the parameters predicted from composition alone. The reference diffusivities agree closely (mean absolute difference «th_log10_Dv_mae» decade for the PENN, «thr_Dv_mae» for the revised PENN). Both parameters increase with w/b. For plain concrete, $D_v$ rises from 10^«C03_Dv»^ to 10^«C05_Dv»^ m²/s between w/b = 0.3 and 0.5, consistent with the coarser and better-connected capillary porosity of high-w/b pastes ⟦mehta,baroghel2007⟧. At a given w/b all SCMs lower $D_v$. The effect is largest for 40% fly ash, whose reference diffusivity at w/b = 0.3 (10^«F4003_Dv»^ m²/s) is «F40_factor» times lower than that of plain concrete, followed by the 20% + 20% ternary blend and 20% fly ash; slag reduces $D_v$ less, and 20% slag least. This ranking agrees with the pore refinement produced by the pozzolanic and latent hydraulic reactions ⟦mehta⟧. The permeabilities increase with w/b as well, but their profile intervals often overlap between neighbouring mixes, so differences of less than about 0.3 decade between individual mixes should not be interpreted.

The diffusivity varies by about an order of magnitude with saturation (Fig. 11), between «D_min_lo» and «D_min_hi» m²/s in the dry, vapour-dominated range and up to «D_max_lo»–«D_max_hi» m²/s near $S$ = «D_Speak», where capillary transport takes over. The position of this transition is set by $S_{\text{cap}}$ and is not identified from the data.

![**Fig. 10.** Transport parameters against w/b: (a) reference diffusivity and (b) intrinsic permeability. Open symbols: classical estimates with 95% profile intervals (corrected for autocorrelation). Filled symbols: predictions from composition by the PENN and the revised PENN (hold-out models).](fig/fig10_parameters.png){width=6.5in}

![**Fig. 11.** Diffusivity predicted by the PENN: (a) vapour and capillary contributions for the withheld mix C-0.4; (b) effect of w/b and fly ash.](fig/fig11_diffusivity.png){width=6.5in}

## 5.9 Sensitivity to the modelling assumptions

The identified parameters are conditional on the fixed constants of Table 2. Changing one constant at a time within plausible bounds and refitting eight representative mixes shows how strong this dependence is (Fig. 12, Table 9). The constants of the vapour branch mainly shift $D_v$ (for example «ps_alpha0_010_ddv» decade on average for $\alpha_0$ = 0.10 and «ps_n_4_ddv» for $n$ = 4), while those of the capillary branch shift $K_{\text{int}}$ ($S_{\text{cap}}$, $\beta$) by up to «sens_dk_max» decade. The fit quality changes by a few per cent at most, except for $n$ = 4 (RMSE ratio «ps_n_4_ratio»), so the data cannot arbitrate between most of these choices. In every case the shifts are nearly uniform across mixes: the rank correlation of the parameters with the reference values is at least «sens_rank_min». Comparisons between mixes, and the composition trends of Section 5.8, are therefore robust, while absolute values carry an uncertainty of a few tenths of a decade from the constitutive assumptions alone.

Two boundary assumptions were also tested. A finite surface exchange (surface time constant 0.1 d) and an initial saturation of 0.97 change the parameters by less than «ps_tau_b_01d_dk» and «ps_S_init_097_dk» decade in $K_{\text{int}}$. A sealed lateral surface, which reduces the problem to one-dimensional drying through the two flat faces, fits the data clearly worse (RMSE «ps_sealed_lateral_ratio» times larger) and requires very different parameters, which supports the assumption that the discs dried on all faces.

⟪TABLE:sensitivity⟫

Table: **Table 9.** Sensitivity of the classical estimates to the modelling assumptions (eight mixes): mean shift of the parameters, ratio of the fit RMSE to the reference, and Spearman rank correlation with the reference estimates.

![**Fig. 12.** Sensitivity of the identified parameters to the modelling assumptions: mean (dot) and range (bar) over eight mixes of the shift of (a) $\log_{10} D_v$ and (b) $\log_{10} K_{\text{int}}$, and (c) ratio of the fit RMSE to the reference model.](fig/fig12_sensitivity.png){width=6.5in}

## 5.10 Internal saturation fields

Because the PENN runs the transport model, every prediction comes with the saturation field inside the specimen (Fig. 13). After one day a dry layer a few millimetres thick has formed along the exposed faces, while the centre is still almost saturated ($S$ = «centre1d_C-0.4» for C-0.4). The centre falls below $S$ = 0.5 after about «tc_C-0.4» days for C-0.4, «tc_S40-0.4» days for S40-0.4 and «tc_F20-0.3» days for F20-0.3. Such fields are what durability models need as initial conditions for chloride ingress or carbonation analyses. They are model predictions, conditional on the assumptions of Section 3, and have not been compared with measured profiles.

![**Fig. 13.** Saturation in one quarter of the disc (drying faces at the top and right) predicted by the PENN for the withheld mixes C-0.4 and F20-0.3 after 1, 7 and 28 days. Contours: $S$ = 0.75 (solid), 0.50 (dashed) and 0.25 (dotted).](fig/fig13_fields.png){width=6.5in}

# 6 Discussion

## 6.1 What the physics-embedded network adds, and what it does not

The results do not support the claim that a physics-embedded network predicts drying more accurately than classical identification. On complete curves, with 8 or more training mixes, the PENN and two-stage inversion with a linear composition map are statistically indistinguishable, and the simpler method is slightly ahead on average. What makes both of them generalise is the same: a verified transport model, which turns composition into a full drying curve through two physically meaningful parameters, and a composition map simple enough to extrapolate. Flexible maps fail when a binder family is withheld whether or not the transport model is used, as the Gaussian-process and quadratic regressions show.

End-to-end training is useful in a narrower but practically relevant situation: when the individual curves do not constrain their parameters well. With curves of a few days, or with very few training mixes, the PENN pools the information of all curves through the composition map and remains accurate where per-mix inversion becomes unstable. It is also convenient: a new mix requires one network evaluation, curves of different length are used as they are, and further descriptors can be added without changing the solver. The price is a tabulated or differentiable solver and, as Section 5.6 showed, the need to check that the training settings do not bias the parameters. We would recommend synthetic recovery tests as a routine part of any such model.

## 6.2 Accuracy is not identification

Three observations in this study separate fitting curves from identifying parameters. First, the classical fits reach $R^2$ > 0.99 for every mix, yet the profile likelihood shows that $K_{\text{int}}$ is uncertain by a few tenths of a decade, up to «prof_k_max» decade, once the autocorrelation of digitised residuals is acknowledged. Second, the original PENN and the revised PENN predict the withheld curves equally well but differ in how faithfully they reproduce the variation of $K_{\text{int}}$ between mixes. Third, the PINN fits the training curves best of all methods and returns the least consistent permeabilities. In each case the predicted average saturation hides an error in a parameter that matters for other uses of the model: $K_{\text{int}}$ governs the capillary regime, which dominates wetting and therefore chloride ingress under cyclic exposure. Parameters intended for durability modelling should be reported with profile intervals and checked by running the identified model in an independent solver.

The PINN result should not be generalised to all PINNs. It holds for this formulation, with hard-constrained boundary and initial conditions and a similarity ansatz, on a stiff nonlinear diffusion problem observed through spatial averages. Our diagnostics show that it is not a matter of initialisation or loss weighting, and the loss-landscape difficulties reported for PINNs ⟦krishnapriyan,rathore⟧ are consistent with it. Other architectures or training strategies may do better, but any PINN used for parameter identification should be checked against an independent solver.

## 6.3 Limitations

The study rests on one published dataset. The curves were digitised from figures, their curing age and the sealing condition of the lateral surface are not recorded in the data file, and eight mixes (w/b = 0.6 and S20-0.5) could not be used. The digitisation errors are autocorrelated and affect $K_{\text{int}}$ in particular. The transport model is isothermal, at the 50 °C of the oven, and uses fixed constitutive constants; transferring the parameters to service temperatures requires a temperature-dependent model. The identified parameters are effective values conditional on these constants and should not be read as intrinsic properties without independent measurements, such as sorption isotherms, permeability tests or measured moisture profiles. No such measurements were available, so the identified parameters have been checked for consistency with the model, not validated experimentally. The absorption tests of the same programme ⟦zhuang⟧ could provide such a validation if the raw data become available. Finally, the internal saturation fields are model predictions.

## 6.4 Engineering implications

For mix design and durability assessment, the study suggests a simple workflow. A drying test of moderate length gives a precise reference diffusivity and an order-of-magnitude permeability for each tested mix. Across a family of mixes, a transport model with a linear composition map, fitted in two stages or end to end, predicts the drying of untested compositions within the binder systems represented, and extrapolates to a new binder system with an accuracy close to that of a direct fit. Flexible machine-learning models trained on the same small database should not be used for such extrapolation. Where only short tests are available, joint end-to-end training is preferable to per-mix inversion. In all cases, the identified parameters inherit the uncertainty of the fixed constitutive constants, which should be stated when they are transferred to service-life models.

# 7 Conclusions

A vapour–capillary moisture transport model of the actual disc specimens was combined with several ways of learning its parameters from mix composition, using a reconstructed public database of 20 plain, binary and ternary concretes. The main conclusions are:

1. Average-saturation curves identify two parameters per mix. The reference diffusivity is determined to within «prof_dv_med» decade; the intrinsic permeability to within «prof_k_med» decade (median 95% intervals corrected for autocorrelated residuals). Both depend on fixed constitutive constants, which shift them by up to «sens_dk_max» decade but preserve the ranking of the mixes.
2. A physics-embedded neural network predicts six withheld mixes with $R^2$ = «ho_pen_r2» (RMSE «ho_pen_rmse»). Classical inversion followed by linear regression on composition is equally accurate (RMSE «ho_ols_rmse»), and the two do not differ significantly in leave-one-mix-out or leave-one-binder-family-out validation.
3. Generalisation to a new binder family requires the transport model and a low-capacity composition map. Gaussian-process, quadratic and data-driven models fail for several mixes (lowest $R^2$ «lofo_min_flex»), whereas linear two-stage inversion and the PENN keep $R^2$ ≥ «lofo_min_phys» for every mix.
4. End-to-end training is more robust than per-mix inversion when the training curves are short (RMSE «tr7_pen» against «tr7_ols» with 7-day curves) or the training mixes few. Weight decay in the network shrinks the permeability towards its mean; synthetic recovery tests revealed this, and removing it restores the composition dependence without changing the predictive accuracy.
5. A PINN trained on the same data fits the curves but returns permeabilities about «sl_pinn_K_bias» decade too low, which fail an independent solver check. The bias persists with exact synthetic data, a warm start and a larger PDE weight, and disappears only when the parameters are fixed: it is a property of the soft-constrained formulation for this problem.

Future work should validate the identified parameters against independent measurements, in particular absorption tests and measured moisture profiles, extend the database beyond one laboratory, and propagate the parameter uncertainty to chloride-ingress predictions ⟦conciatori,yang_bpinn⟧.

# Declarations {.unnumbered}

**CRediT authorship contribution statement.** Abderrahmane Sghuri: Conceptualization, Methodology, Software, Formal analysis, Visualization, Writing – original draft. Xuande Chen: Conceptualization, Supervision, Validation, Writing – review & editing, Funding acquisition. Xiaodong Wang: ⟪U|[…]⟫.

**Data and code availability.** The experimental data are those of Zhuang et al. ⟦zhuang⟧. The digitised and reconstructed data and the code that produces every result, table and figure of this paper are available at https://github.com/Abderahmane2002/hybrid-pinn-lstm-concrete ⟪U|[archive the release on Zenodo and cite its DOI]⟫.

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** ⟪U|[Funding agency and grant number]⟫

**Declaration of generative AI use.** ⟪U|[Elsevier requires authors to disclose the use of generative AI tools in preparing the manuscript or code; state the tool and its purpose here.]⟫

# Appendix A. Numerical verification {.unnumbered}

⟪TABLE:numerics⟫

Table: **Table A.1.** Verification of the solvers: maximum error in $S_{\text{avg}}$ between 0.25 and 90 days for three representative parameter sets, against a fine reference solution (640 cells for the slab solver, a 160-cell BDF solution for the differentiable solver, 96 × 64 cells for the disc).

# Appendix B. Reproducibility of the original results {.unnumbered}

Before any change was made, the complete original pipeline was re-run from a clean copy of the code, starting from the raw digitised file: curve reconstruction, classical fits, tabulation of the disc solutions, all network trainings, cross-validation and verification. Table B.1 compares the regenerated results with those of the original analysis. «repro_ok» of «repro_n» compared quantities agree within the stated tolerances.

⟪TABLE:repro⟫

Table: **Table B.1.** Independent re-run of the original pipeline. Tolerances allow for the random seeds of multi-start optimisation and, for the PINN, for non-deterministic floating-point reductions.

# Appendix C. Hyper-parameter check {.unnumbered}

The settings of the PENN and of the data-driven network were compared with alternatives in an inner leave-one-family-out cross-validation on the 14 training mixes only (Table C.1). For the PENN, the original setting ranks «icv_pen_rank» of «icv_pen_n» configurations, with an inner RMSE («icv_pen_paper») within 0.001 of the best («icv_pen_sel»); the configuration selected by the inner validation gives RMSE «icv_pen_test» on the withheld mixes, the same as the original setting. For the data-driven network the inner validation prefers a stronger weight decay (inner RMSE «icv_mlp_sel» against «icv_mlp_paper»), but that configuration is less accurate on the withheld mixes (RMSE «icv_mlp_test») than the original one. The comparisons in the main text therefore do not favour the PENN through its settings.

⟪TABLE:inner_cv⟫

Table: **Table C.1.** Inner cross-validation (leave one family out within the training mixes): RMSE of every configuration tested.

# Appendix D. PINN diagnostics and training histories {.unnumbered}

⟪TABLE:pinn_diag⟫

Table: **Table D.1.** PINN diagnostic runs (slab model, training mixes, seed 0): final loss terms, training RMSE, PDE residual, solver check and permeability bias relative to the classical slab estimates.

![**Fig. D.1.** Training histories: (a) RMSE of the PENN on the training mixes (five seeds); (b) loss terms of the PINN (median and range over three seeds).](fig/figB1_convergence.png){width=6.5in}

![**Fig. D.2.** Saturation profiles of three withheld mixes on the slab model, computed by the direct solver with the PENN parameters (solid) and by the PINN (dotted). The two differ most during the first two days.](fig/figB2_profiles_slab.png){width=6.5in}

# References {.unnumbered}

⟪REFS⟫
