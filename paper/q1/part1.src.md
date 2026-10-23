---
title: "Learning moisture transport parameters of blended concretes from drying curves: identifiability, generalisation across binders, and a comparison of physics-embedded and physics-informed neural networks"
---

::: {custom-style="Authors"}
Abderrahmane Sghuri^a,b^, Xuande Chen^a,b,\*^, Xiaodong Wang^a^
:::

::: {custom-style="Affiliation"}
^a^ Department of Mathematics, Computer Science and Engineering, Université du Québec à Rimouski (UQAR), Rimouski, QC, Canada\
^b^ Research Center on Concrete Infrastructures (CRIB), Québec, QC, Canada\
\* Corresponding author: ⟪U|[e-mail address]⟫
:::

# Highlights {.unnumbered}

- Drying curves identify only two moisture transport parameters of concrete per mix.
- A solver-embedded network learns them from mix composition with R² = «ho_pen_r2».
- No detectable gain over two-stage inversion; flexible maps fail on new binders.
- Pooling mixes in one model keeps predictions stable when drying curves are short.
- A PINN fits the same data, but its parameters fail an independent solver check.

# Abstract {.unnumbered}

Gravimetric drying tests give the average saturation of concrete specimens, from which moisture transport parameters are usually identified mix by mix. We examine what can be learned across mixes and which modelling choices make the result trustworthy. A vapour–capillary diffusion model of the actual disc specimens was fitted to «n_pts» drying measurements of «n_mixes» plain, binary and ternary concretes from a published database, after correcting a time offset between duplicated digitised series. Profile likelihoods showed that only two parameters per mix are identifiable: the reference diffusivity to within «prof_dv_med» decade and the intrinsic permeability to within «prof_k_med» decade (median 95% intervals). A physics-embedded neural network (PENN), which maps composition to these parameters and is trained through tabulated solutions of the transport equation, predicted six withheld mixes with R² = «ho_pen_r2». Classical inversion followed by linear regression gave similar accuracy (R² = «ho_ols_r2»), with no statistically detectable difference in leave-one-mix-out or leave-one-binder-family-out validation. With a binder family withheld, flexible composition models failed for several mixes (lowest R² «lofo_min_flex»), whereas the linear physics-based models kept R² ≥ «lofo_min_phys». End-to-end training was more robust when the training curves were short. Synthetic tests revealed that weight decay compressed the learned permeabilities; a linear map trained to convergence removed this bias. A physics-informed neural network fitted the curves but returned permeabilities about «sl_pinn_K_bias» decade too low, which diagnostic runs link to its soft-constrained loss. Fixed constitutive constants shift the parameters by up to «sens_dk_max» decade without changing the ranking of the mixes.

**Keywords:** Concrete; Moisture transport; Drying; Supplementary cementitious materials; Inverse analysis; Identifiability; Physics-embedded neural network; Physics-informed neural network

# 1 Introduction

The moisture state of concrete controls most of its deterioration processes. Water carries chlorides to the reinforcement, governs the rate of carbonation, drives freeze–thaw damage and alkali–silica reaction, and its loss causes drying shrinkage ⟦mehta,neville⟧. Service-life models therefore need moisture transport properties, and these depend strongly on the water-to-binder ratio (w/b) and on supplementary cementitious materials (SCMs) such as fly ash and slag, which refine the pore structure ⟦baroghel2007,baroghel1999⟧. Moisture moves partly as liquid under capillary pressure gradients and partly as vapour through the gas-filled pores, and the balance between the two changes with the local degree of saturation ⟦mainguy,chen2022⟧.

The most common laboratory measurement is gravimetric: a specimen is weighed while it dries or absorbs water, which gives its average degree of saturation $S_{\text{avg}}(t)$ ⟦hall,zhuang⟧. The test is inexpensive, but the curve does not show the moisture profile inside the specimen or the material functions that produce it. These are obtained by inverse analysis: a transport model ⟦bazant,kunzel,chen2022,conciatori⟧ is run repeatedly and its parameters are adjusted until the simulated mass loss matches the measurement. Two difficulties are well known. The problem is ill-posed, because different parameter sets can reproduce the same curve ⟦kaipio,tarantola⟧, and the calibration is carried out mix by mix, so a new composition needs a new test and a new inversion. Practical identifiability, i.e. whether the available data constrain each parameter, is rarely examined in this context, although tools such as profile likelihoods are well established in other fields ⟦raue⟧.

Machine learning has been used to relate composition to transport indicators across many mixes, most often for chloride diffusion coefficients ⟦liu2024⟧. These models are purely statistical and do not provide the material functions that a transport simulation needs. Physics-informed neural networks (PINNs) offer a way to combine data and governing equations ⟦raissi,karniadakis,cuomo⟧. A PINN represents the unknown field by a neural network and adds the residual of the governing equation, computed by automatic differentiation ⟦baydin⟧, to the data misfit; unknown coefficients are learned in the same optimisation. PINNs have identified constitutive relations of unsaturated soils ⟦tartakovsky⟧, chloride diffusion coefficients of concrete and alkali-activated mortars ⟦wan2023,wan2025⟧ and moisture diffusivities of stone ⟦huang2025⟧, and neural networks have recovered moisture diffusivities of concrete from synthetic profiles ⟦lee2022⟧. Their training is, however, fragile for stiff transport problems: the optimiser can settle on fields that satisfy the residual almost everywhere while missing the initial condition or the early dynamics ⟦krishnapriyan,wang2021,wang_causal,rathore⟧. In an inverse setting this is easily overlooked, because the network may still fit the measurements closely.

An alternative is to keep a conventional solver and place it inside the learning loop, so that gradients of the data misfit flow through the solver to the quantities being learned ⟦rackauckas,thuerey⟧. Bandai et al. ⟦bandai2024⟧ used this differentiable-solver approach to learn soil water retention and conductivity functions from moisture data. The governing equation then holds to the accuracy of the discretisation. For concrete, where the unknowns are a few parameters per mix, the same idea can be combined with a model of how these parameters depend on composition. Whether such end-to-end learning offers anything over the simpler two-stage route — inverting each mix and regressing the results on composition — has not been tested, and neither has the consistency of PINN-identified moisture parameters with an independent solver.

This paper addresses these questions on a published drying database of plain, binary and ternary concretes ⟦zhuang⟧. Its contributions are:

1. a documented reconstruction of the database, including statistical evidence that its duplicated series are one curve digitised with a time offset;
2. a practical identifiability analysis of a vapour–capillary diffusion model of the real disc specimens, with profile likelihoods that account for correlated residuals, synthetic recovery tests, and the sensitivity of the parameters to fixed constants and boundary conditions;
3. a physics-embedded neural network (PENN) that learns the composition–parameter relationship through tabulated solutions of the transport equation, compared under identical folds with two-stage inversion (four regression models) and a data-driven network, including binder-family extrapolation, mix-level statistics, robustness tests and data-efficiency studies;
4. a controlled comparison with a PINN, with diagnostic runs that locate the origin of its parameter inconsistency.

Section 2 describes the database, Section 3 the transport model and Section 4 the methods. The results are presented in Section 5, discussed in Section 6 and summarised in Section 7.

# 2 Experimental database

## 2.1 Source and test protocol

The data come from the experimental programme of Zhuang et al. ⟦zhuang⟧. Concrete cylinders 100 mm in diameter and 150 mm high were cured sealed at 20 ± 2 °C and 95 ± 5% relative humidity. At the test age, discs 100 mm in diameter and 50 mm thick were cut from the centre of the cylinders, vacuum-saturated for 24 h and dried in an oven at 50 °C, following the conditioning procedure of ASTM C1585 ⟦astm1585⟧. The discs were weighed daily until the mass loss over 24 h fell below 0.1%, and the degree of saturation was plotted against drying time. The source reports drying curves at curing ages of 28 and 56 days ⟦zhuang⟧ and notes that curing age changes the drying behaviour markedly. The curves used here were digitised from the published figures; ⟪C|To be confirmed by the authors: (1) which figure of Zhuang et al. was digitised (Fig. 2, 28 d, or Fig. 3, 56 d) and with which tool. Section 2.4 indicates 28 d; if the 56-day figure was used, Sections 2.4 and 5.8 must be revised. (2) Whether the lateral surface of the discs was sealed during drying. Zhuang et al. state that the sides were sealed with tape for the absorption test, but say nothing about drying. Section 5.9 shows that a sealed lateral face fits the data clearly worse.|the data file does not record their curing age, which is examined in Section 2.4, and the source does not state whether the lateral surface was sealed during drying.⟫

## 2.2 Concrete mixes

The programme covers four w/b ratios (0.3–0.6) and seven binder systems: plain Portland cement (C), 20% and 40% fly ash (F20, F40), 20% and 40% slag (S20, S40), and ternary blends with 10% + 10% and 20% + 20% fly ash and slag (F10S10, F20S20). The proportions are given in Table 1.

⟪TABLE:mixes⟫

Table: **Table 1.** Mix proportions of the concretes (kg/m³), after Zhuang et al. ⟦zhuang⟧. S20-0.5 has no drying curve in the digitised data.

## 2.3 Reconstruction of the drying curves

The digitised file contains «n_raw» points for «n_raw_mixes» mixes. For most mixes it holds two series: a short series with daily points up to 14 days and a long series extending to 40–100 days. Over their common saturation range, the long series is the short series delayed by a nearly constant time (median «shift» d, interquartile range «shift_lo»–«shift_hi» d over the «n_consistent» mixes in which the two agree). Two curves measured on different specimens or at different curing ages would differ by a change of time scale rather than by a shift, since diffusion-controlled drying scales with $t/D$. Fitting both relations to every mix with w/b ≤ 0.5 settles the question: a pure shift reproduces the short series to within «series_shift_rms» in saturation (median RMS), a pure time scaling only to within «series_scale_rms», and a free combination of both returns a scale factor of «series_affine_l» (interquartile range «series_affine_lo»–«series_affine_hi»), i.e. no scaling. The two series are therefore one curve plotted with two time origins; a one-day offset is consistent with time counted from the start of the 24 h vacuum saturation. Each curve was rebuilt from the short series up to 14 days and the time-corrected long series beyond (Fig. 1a,b). The offset estimated from the training mixes alone («shift_train» d) differs from the global value by less than 0.02 d.

The w/b = 0.6 series could not be repaired. The long series of S20-0.6 and S40-0.6 are identical copies, the short series of F20S20-0.6 mixes two curves, and the offsets between the two series are erratic and of the wrong sign (Fig. 1c). The remaining w/b = 0.6 short series dry more slowly at early ages than the corresponding w/b = 0.5 curves, which is implausible, and a classical fit drives the permeability to the lower bound of its range. These mixes were excluded. The reconstructed data file keeps their short series for reference but flags them, so it holds «n_file_pts» points, of which the «n_flag_pts» flagged w/b = 0.6 points are never used. Every analysis in this paper uses the remaining «n_pts» points of «n_mixes» mixes with w/b between 0.3 and 0.5. All steps are scripted, and the effect of the reconstruction choices on the results is quantified in Section 5.5.

## 2.4 Curing age of the digitised curves

Because curing age affects drying markedly, it matters which of the two published sets of curves was digitised. The source gives two quantitative statements that allow a check. After 28 days of curing, the time needed to dry the vacuum-saturated discs to 70% saturation is 1.5, 1.0, 0.8 and 0.6 days for w/b = 0.3, 0.4, 0.5 and 0.6, and the type and content of mineral admixture have little effect. After 56 days, both admixture and w/b have a large effect, with differences of 2–4 days between binders at the same w/b ⟦zhuang⟧. For the digitised curves, the time to 70% saturation obtained from the fitted disc model averages «ca_t70_03», «ca_t70_04» and «ca_t70_05» days for w/b = 0.3, 0.4 and 0.5. These values differ from the 28-day values by «ca_dev_short» day on average, and the spread between binders at a given w/b is at most «ca_spread_max» day. Both observations match the 28-day description and not the 56-day one. Had the time origin of the long series been used instead, the times would exceed the 28-day values by about one day («ca_dev_long» day on average), which independently supports the time origin chosen in Section 2.3. We therefore interpret the database as drying curves after 28 days of curing; the composition effects identified below refer to that age and may be larger at later ages.

![**Fig. 1.** Reconstruction of the drying curves. (a) C-0.4 and (b) F40-0.4: raw short and long series and the rebuilt curve. (c) S40-0.6: inconsistent series, mix excluded. (d) Median RMS mismatch between the two series of each mix (w/b ≤ 0.5) when they are related by a time shift, a time scaling, or both.](fig/fig01_data.png){width=6.5in}

# 3 Moisture transport model

## 3.1 Governing equation and diffusivity

Once hydration has largely stopped, sources and sinks of water can be neglected ⟦baroghel2007⟧, and moisture conservation reads

$$\frac{\partial S}{\partial t} = \nabla\cdot\left[ D(S;\boldsymbol{\theta})\,\nabla S \right] \qquad\qquad (1)$$

where $S \in [0,1]$ is the degree of saturation and $D$ [m²/s] the effective moisture diffusivity. We use the dual-regime diffusivity of Chen et al. ⟦chen2022⟧, which blends a capillary and a vapour contribution:

$$D(S;\boldsymbol{\theta}) = f(S)\,D_{\text{cap}}(S) + \left[1 - f(S)\right] D_{\text{vap}}(S) \qquad\qquad (2)$$

$$D_{\text{vap}}(S) = D_v \left[ \alpha_0 + \frac{1-\alpha_0}{1 + \left(\dfrac{1-S}{1-S_c}\right)^{n}} \right] \qquad\qquad (3)$$

$$D_{\text{cap}}(S) = \frac{K_{\text{int}}}{\eta_l}\left|\frac{\partial P_c}{\partial S}\right|, \qquad \frac{\partial P_c}{\partial S} = -\beta\,P_{c0}\,S^{-\beta-1} \qquad\qquad (4)$$

$$f(S) = \frac{1}{1 + \exp\left[-k\,(S - S_{\text{cap}})\right]} \qquad\qquad (5)$$

Equation (3) has the sigmoidal form of Bažant and Najjar ⟦bazant⟧, with $D_v$ the reference diffusivity, $\alpha_0$ the ratio of minimum to maximum vapour diffusivity, $n$ a shape exponent and $S_c$ the transition saturation. In Eq. (4), $K_{\text{int}}$ is the intrinsic permeability, $\eta_l$ the viscosity of water, $P_{c0}$ a reference capillary pressure and $\beta$ an exponent. The switch (5) makes vapour transport dominant at low saturation and capillary transport dominant at high saturation. The fixed constants are listed in Table 2. Since $D_{\text{cap}}$ depends on $K_{\text{int}}$ only through $K_{\text{int}} P_{c0}/\eta_l$, only this product is identifiable: a different choice of $P_{c0}$ or $\eta_l$ shifts $\log_{10} K_{\text{int}}$ by a known constant and leaves every prediction unchanged.

| Symbol | Meaning | Value | Source |
|:------|:----------------------------|:----------:|:------:|
| $R$, $H$ | Disc radius and half-thickness | 50 mm, 25 mm | ⟦zhuang⟧ |
| $T$ | Drying temperature | 50 °C | ⟦zhuang⟧ |
| $\eta_l$ | Viscosity of water at 50 °C | 5.47 × 10⁻⁴ Pa·s | – |
| $P_{c0}$ | Reference capillary pressure | 5 MPa | ⟦chen2022⟧ |
| $\beta$ | Capillary pressure exponent | 2.5 | ⟦chen2022⟧ |
| $S_c$ | Transition saturation in Eq. (3) | 0.80 | ⟦chen2022⟧ |
| $S_{\text{cap}}$, $k$ | Switching saturation and steepness | 0.75, 50 | ⟦chen2022⟧ |
| $\alpha_0$ | Minimum-to-maximum diffusivity ratio | 0.05 | Section 5.2 |
| $n$ | Shape exponent, shared by all mixes | 8 | Section 5.2 |

Table: **Table 2.** Fixed constants of the transport model. Their influence on the identified parameters is quantified in Section 5.9.

## 3.2 Specimen geometry and boundary conditions

With the axis and the mid-plane as planes of symmetry, Eq. (1) is solved on $0 \le r \le R$, $0 \le z \le H$ with

$$S(r,z,0) = 1, \qquad S(R,z,t) = S(r,H,t) = 0, \qquad \frac{\partial S}{\partial r}(0,z,t) = \frac{\partial S}{\partial z}(r,0,t) = 0 \qquad\qquad (6)$$

The specimen starts saturated, and the exposed surfaces are assumed to reach the oven-dry state immediately, which defines $S = 0$. The quantity compared with the measurements is the volume average

$$S_{\text{avg}}(t) = \frac{2}{R^2 H}\int_0^H\!\!\int_0^R S(r,z,t)\,r\,\mathrm{d}r\,\mathrm{d}z \qquad\qquad (7)$$

A one-dimensional slab whose half-thickness equals the volume-to-surface ratio, $L = V/A = 12.5$ mm, is a common simplification. It reproduces the early drying of the disc but not its later stages, because the edges dry from two sides: with the same parameters, the slab curve departs from the disc curve by up to «geo_maxdiff» in $S_{\text{avg}}$. A slab model fitted to the same data returns parameters biased by «geo_dDv_lo» to «geo_dDv_hi» decade in $D_v$ and up to +«geo_dK_hi» decade in $K_{\text{int}}$ (Fig. 2). The disc geometry is therefore used for every result on transport parameters; the slab is kept only for the controlled comparison with the PINN (Section 5.7), where both methods use the same simplified model.

![**Fig. 2.** Disc model and one-dimensional slab. (a, b) Measured average saturation, disc model fitted to the data, and slab model ($L = V/A$) run with the same parameters. (c) Saturation in one quarter of the disc after 7 days for C-0.4; the edge dries from two sides.](fig/fig02_geometry.png){width=6.5in}

# 4 Methods

## 4.1 Numerical solution and verification

Equation (1) is discretised with a cell-centred finite-volume scheme on a mesh refined towards the drying faces (24 × 16 cells in $(r, z)$ for the disc, 60–80 cells for the slab) and integrated in time with an implicit BDF scheme ⟦hairer⟧ at a relative tolerance of 10⁻⁶. This direct solver is the reference model of the study. It was verified against the analytical solution for constant diffusivity, by mesh and time-step refinement, by a global water balance, and by checking that a disc of very large radius reproduces the slab solution (Section 5.1, Appendix A).

Learning requires many evaluations of the model and gradients with respect to its parameters. Because only two parameters are identified (Section 5.2), the disc problem was solved once with the direct solver on a 49 × 49 grid of $(\log_{10} D_v, \log_{10} K_{\text{int}})$ covering $[-10, -7.5] \times [-21, -18.5]$, at 120 logarithmically spaced times between 0.05 and 110 days. Between grid points, $S_{\text{avg}}$ is interpolated with bicubic convolution ⟦keys⟧ in the parameters and linearly in log-time. The interpolant is differentiable and every value it returns is a combination of solutions of Eq. (1). We refer to it as the tabulated solver; it is used for training and, for speed, in the extensive validation experiments, after checking that it reproduces the direct solver to within «table_err» in $S_{\text{avg}}$ and the classical parameter estimates to within «table_fit_dk» decade. For the slab, a differentiable version of the finite-volume scheme (linearly implicit Euler, 240 logarithmic time steps) is used instead.

## 4.2 Classical inversion and identifiability

For each mix, $\boldsymbol{\theta} = [\log_{10} D_v, \log_{10} K_{\text{int}}]$ is estimated by nonlinear least squares (trust-region reflective algorithm, several starting points). Practical identifiability was assessed in three ways. First, the Jacobian at the optimum gives parameter correlations and a condition number, computed for two, three and four free parameters. Second, profile likelihoods ⟦raue⟧ were computed for every mix: one parameter is fixed on a grid and the other re-optimised, and the 95% interval is the range where the deviance $N_{\text{eff}} \ln(\text{SSE}/\text{SSE}_{\min})$ stays below $3.84\,c$. Digitised curves have strongly autocorrelated residuals, so the number of points $N$ was replaced by $N_{\text{eff}} = N(1-\rho_1)/(1+\rho_1)$, with $\rho_1$ the lag-1 autocorrelation of the residuals. Because this correction is itself approximate, its coverage was checked by simulation: for every mix, 100 replicate curves were generated from the fitted model with AR(1) errors having the residual standard deviation and lag-1 autocorrelation of that curve, and each replicate was analysed exactly as the data. The factor $c$ is the value that gives 95% coverage over all replicates, determined separately for each parameter. The resulting intervals are approximate: they assume AR(1) errors. Third, synthetic curves with known parameters, generated on a finer mesh (36 × 24) than the one used for inversion, were inverted with and without perturbations (Section 5.2). The shared exponent $n$ was selected by profiling the total misfit of five training mixes; including or excluding the withheld mix C-0.5 does not change the choice.

## 4.3 Models linking composition and transport parameters

The composition of a mix is described by $\mathbf{c} = [(\text{w/b} - 0.3)/0.3,\ \text{FA}/0.4,\ \text{SL}/0.4]$, where FA and SL are the fly ash and slag mass fractions of the binder. Two families of models predict $\boldsymbol{\theta}(\mathbf{c})$ for a mix that was not tested.

**Two-stage inversion.** The classical estimates of the training mixes are regressed on $\mathbf{c}$, and the transport model is run with the regressed parameters. Four regressions were compared: ordinary least squares (OLS, linear), ridge regression with the penalty selected by leave-one-out cross-validation on the training mixes, quadratic features with the same ridge penalty, and Gaussian-process (GP) regression with an anisotropic squared-exponential kernel and a noise term, fitted by maximum marginal likelihood ⟦rasmussen⟧.

**Physics-embedded neural network.** In the PENN (Fig. 3), a small network $g_{\mathbf{w}}$ maps $\mathbf{c}$ to the parameters, mapped smoothly into the admissible range,

$$\boldsymbol{\theta}(\mathbf{c}) = \boldsymbol{\theta}_{\min} + (\boldsymbol{\theta}_{\max} - \boldsymbol{\theta}_{\min}) \odot \sigma\!\left(g_{\mathbf{w}}(\mathbf{c})\right) \qquad\qquad (8)$$

with $\sigma$ the logistic function. The parameters are passed to the tabulated solver and the weights minimise the misfit over all training points,

$$\mathcal{L}(\mathbf{w}) = \frac{1}{N}\sum_{j=1}^{N}\left[S_{\text{avg}}\!\left(t_j;\boldsymbol{\theta}(\mathbf{c}_j)\right) - S_{\text{avg},\exp}(t_j)\right]^2 \qquad\qquad (9)$$

with gradients obtained by automatic differentiation through the solver ⟦rackauckas,thuerey⟧. No per-mix inversion is needed: the network is trained directly on the drying curves of all training mixes at once. The reference PENN has two hidden layers of 16 tanh units; a reduced network (one layer of 8 units) and a linear map ($g_{\mathbf{w}}$ affine in $\mathbf{c}$) were also tested. Training uses Adam ⟦kingma⟧ (learning rate 0.02, decay 0.995 per iteration, weight decay 10⁻⁴) for 400 full-batch iterations, after which the training error no longer changes (Fig. D.1a). A revised PENN, introduced in Section 5.6, uses a linear map without weight decay trained to convergence with full-batch L-BFGS.

**Data-driven network.** A network with four hidden layers of 48 tanh units maps $(\sqrt{t/t_{\text{ref}}}, \mathbf{c})$, $t_{\text{ref}} = 100$ d, directly to $S_{\text{avg}}$ (Adam, 4000 iterations, weight decay 10⁻⁵).

**Choice of settings.** The settings of the PENN and of the data-driven network given above were fixed in the original study, before this analysis, and are retained for both model classes so that neither is tuned more than the other. To check them without using the withheld mixes, every setting was also compared in an inner leave-one-family-out cross-validation restricted to the training mixes (Appendix C). For the PENN the original setting lies within 0.001 RMSE of the best configuration in the inner validation. For the data-driven network it prefers a ten times stronger weight decay (10⁻³); this inner-CV setting is therefore reported as a second data-driven baseline in every validation scheme, and conclusions about the data-driven approach are drawn only where both settings agree. Neither the choice of the primary setting nor the interpretation uses the accuracy on the withheld mixes.

![**Fig. 3.** Physics-embedded neural network. The parameter network predicts the transport parameters from the composition; the tabulated solver returns the average saturation; the data misfit is back-propagated through the solver to the network weights. In the two-stage route, the parameters are instead identified mix by mix and then regressed on composition.](fig/fig03_workflow.png){width=6.5in}

## 4.4 Physics-informed neural network

The PINN was set up on the slab version of the problem ($L = 12.5$ mm, $n = 2$ from the corresponding profile) with the same parameter network (8). A field network $N$ with four hidden layers of 48 tanh units takes $(\xi, s, \mathbf{c})$ as input, with $\xi = x/L$ and $s = \sqrt{t/t_{\text{ref}}}$. The boundary condition at the drying face and the initial condition are built into the field,

$$S = 1 - \exp\left(-\frac{\xi}{s} A\right), \qquad A = a(s) + \text{softplus}(N) \qquad\qquad (10)$$

so that $S \to 1$ when $t \to 0$ and $S = 0$ at $\xi = 0$. The loss combines the integral data misfit, the dimensionless PDE residual and the symmetry condition at $\xi = 1$,

$$\mathcal{L} = \sigma_d^{-2}\,\mathcal{L}_{\text{data}} + \lambda_{\text{PDE}}\,\mathcal{L}_{\text{PDE}} + \lambda_{\text{sym}}\,\mathcal{L}_{\text{sym}} \qquad\qquad (11)$$

with $\sigma_d = 0.01$, $\lambda_{\text{PDE}} = 10$ and $\lambda_{\text{sym}} = 1$. The residual is multiplied by $t/t_{\text{ref}}$, which keeps it bounded at early times, and half of the 3000 collocation points per iteration are sampled in the similarity variable $\xi/s$ so that the drying front is resolved at every time. Training uses 3000 Adam iterations followed by 1000 L-BFGS iterations (Fig. D.1b). Without the term $a(s) = 10^{-3} + 0.1\exp[-(s/0.2)^2]$, which bounds the speed of the drying front at early times, the PINN converged in every trial to a trivial field that dries almost instantly, a known failure mode for stiff transport problems ⟦krishnapriyan,wang_causal⟧ that causal weighting of the residual ⟦wang_causal⟧ did not remove.

## 4.5 Validation protocol

1. **Hold-out test.** Six mixes covering four binder families and three w/b ratios (C-0.4, C-0.5, F10S10-0.3, F20-0.3, F20S20-0.4 and S40-0.4; «n_test_pts» points) were withheld; the other «n_train_mixes» mixes («n_train_pts» points) were used for training. Networks were trained with five random seeds (three for the PINN).
2. **Cross-validation.** Leave-one-mix-out (LOMO, 20 folds) and leave-one-binder-family-out (LOFO, 7 folds), one seed per fold. Because the points of one curve are not independent, the mix is the statistical unit: methods are compared through per-mix RMSE differences, with 95% bootstrap intervals (10 000 resamples of the mixes ⟦efron⟧) and Wilcoxon signed-rank tests. Because mixes of the same binder family share most of their composition, the comparison was repeated with the binder family as the unit: per-mix differences are averaged within each of the seven families, and the family means are resampled (cluster bootstrap) and tested with the Wilcoxon test; with seven families the smallest attainable two-sided $p$ is 0.016. A non-significant difference does not show that two methods are equally accurate; to judge practical equivalence we use a margin of ±0.002 in per-mix RMSE, about one third of the residual RMSE of direct per-mix fits («ho_fdfit_rmse») and below the digitisation uncertainty. Two methods are called practically equivalent when the 90% family-level interval of their difference lies within this margin (two one-sided tests at the 5% level). The margin was set during this revision, after the main comparisons had been run, so this criterion is descriptive rather than a pre-registered test.
3. **Robustness.** The training data were perturbed by independent noise (standard deviation 0.01 and 0.02), by correlated digitisation-like errors (per-curve offset plus AR(1) noise), by random time-origin errors (0.25 d), and by alternative reconstructions of the curves (offset at the bounds of its interquartile range, short or long series only). Predictions were always scored against the unperturbed withheld curves.
4. **Data efficiency.** Models were trained on 5, 8, 11 or 14 randomly drawn mixes (10 draws each, identical for all methods) and tested on the remaining ones, and on training curves truncated after 3, 4, 7 or 14 days.
5. **Solver check.** The parameters returned by each model were inserted, without fitting, into the direct solver, and the resulting curves were compared with the model output.

Accuracy is reported as the coefficient of determination and the root-mean-square error,

$$R^2 = 1 - \frac{\sum_j (y_j - \hat{y}_j)^2}{\sum_j (y_j - \bar{y})^2}, \qquad \text{RMSE} = \sqrt{\frac{1}{N}\sum_j (y_j - \hat{y}_j)^2} \qquad\qquad (12)$$

pooled over all points of a test set, and per mix. All computations ran on a single CPU core.
