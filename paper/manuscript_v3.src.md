---
title: "Physics-embedded neural network for predicting the moisture transport properties of concrete from mix composition, with a comparison to physics-informed neural networks"
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

- A small network maps concrete mix composition to moisture transport parameters.
- It is trained through a transport solver, so its predictions obey the governing PDE.
- Withheld mixes are predicted with R² = «ho_pen_r2», even with a binder family left out.
- A PINN fits the same data, but its parameters fail an independent solver check.
- Real disc geometry, identifiability and data errors are handled; the code is open.

# Abstract {.unnumbered}

Drying tests on concrete usually record only the specimen mass, i.e. the average degree of saturation. Recovering transport properties from such curves is an ill-posed inverse problem, normally solved one mix at a time. We propose a physics-embedded neural network (PENN) in which a small network maps the mix composition (water-to-binder ratio, fly ash and slag contents) to the parameters of a nonlinear vapour–capillary diffusivity, and a solver of the moisture transport equation in the actual disc geometry converts these parameters into drying curves. Because the network is trained through the solver, every prediction satisfies the governing equation and a new mix needs only its composition. The method was tested on «n_pts» drying measurements of «n_mixes» plain, binary and ternary concretes after correcting duplicated and time-shifted series in the source data. Only two parameters per mix, the reference diffusivity and the intrinsic permeability, proved identifiable. On six withheld mixes the PENN reached R² = «ho_pen_r2» (RMSE = «ho_pen_rmse»), a «ho_gain_mlp»% lower error than a purely data-driven network and the same accuracy as classical identification combined with composition regression. When a whole binder family was withheld, the physics-based models kept R² ≥ «cv_lofo_fdreg_min» for every mix, whereas the data-driven network dropped to «cv_lofo_mlp_min». A physics-informed neural network (PINN) trained on the same data reproduced the curves, but its parameters did not reproduce them in an independent solver (R² = «sl_fwd_pinn_r2»). Embedding the solver, rather than penalising a residual, yields transferable and physically consistent transport parameters from routine gravimetric tests.

**Keywords:** Concrete; Moisture transport; Drying; Supplementary cementitious materials; Inverse analysis; Physics-embedded neural network; Physics-informed neural network

# 1 Introduction

Most deterioration processes in concrete depend on its moisture state. Water carries chlorides to the reinforcement, controls the rate of carbonation, drives freeze–thaw damage and alkali–silica reaction, and its loss causes drying shrinkage ⟦mehta,neville⟧. Service-life models therefore need the moisture transport properties of the concrete in question, and these depend strongly on the water-to-binder ratio (w/b) and on supplementary cementitious materials (SCMs) such as fly ash and slag, which refine the pore structure ⟦baroghel2007,baroghel1999⟧. Moisture moves partly as liquid driven by capillary pressure and partly as vapour diffusing through the gas-filled pores, and the balance between the two changes with the local degree of saturation ⟦mainguy,chen2022⟧.

The simplest and most common laboratory measurement is gravimetric: a specimen is weighed while it dries or absorbs water, giving its average degree of saturation $S_{\text{avg}}(t)$ ⟦hall,zhuang⟧. The test is cheap and robust, but the curve says nothing directly about the moisture profile inside the specimen or about the material functions that produce it. These are usually obtained by inverse analysis: a numerical model ⟦bazant,kunzel,chen2022,conciatori⟧ is run repeatedly and its parameters are adjusted until the simulated mass loss matches the measurement. Two difficulties are well known. The problem is ill-posed, since different parameter sets can reproduce the same curve ⟦kaipio,tarantola⟧, and the calibration is carried out mix by mix, so that a new composition requires a new test and a new inverse analysis.

Machine learning offers a way to link composition and transport behaviour across many mixes. Physics-informed neural networks (PINNs) are the most widely used option ⟦raissi,karniadakis,cuomo⟧. A PINN represents the unknown field with a neural network and adds the residual of the governing equation, computed by automatic differentiation ⟦baydin⟧, to the data misfit; unknown coefficients can be learned in the same optimisation. PINNs have been used to identify constitutive relations of unsaturated soils ⟦tartakovsky⟧ and chloride diffusion coefficients of concrete and alkali-activated mortars ⟦wan2023,wan2025⟧, and neural networks have recovered moisture diffusivities of concrete from synthetic profiles ⟦lee2022⟧. Their training, however, is known to be fragile for stiff transport problems: the optimiser can settle on fields that satisfy the residual almost everywhere while missing the initial condition or the early-time dynamics ⟦krishnapriyan,wang2021,wang_causal⟧. In an inverse setting this is hard to notice, because the network may still fit the measurements well.

A different route is to keep a conventional numerical solver and place it inside the learning loop. In this differentiable-physics approach, the network predicts parameters or closures, the solver integrates the equations, and gradients flow back through the solver ⟦rackauckas,thuerey⟧. The governing equation then holds to the accuracy of the discretisation, and the network only learns what is genuinely unknown. We are not aware of applications of this idea to moisture transport in cementitious materials, nor of studies that test PINN-identified moisture parameters against an independent solver.

This paper develops such a physics-embedded neural network (PENN) for the drying of concrete and evaluates it against classical identification, a purely data-driven network and a PINN. The contributions are:

1. a cleaned, documented version of a published drying database, with an analysis of which transport parameters can actually be identified from average-saturation curves;
2. a PENN that predicts these parameters from mix composition and is trained through a solver of the transport equation in the real disc geometry;
3. a validation on withheld mixes, by leave-one-mix-out and leave-one-binder-family-out cross-validation and under added measurement noise;
4. a controlled comparison with a PINN, including an independent check of the identified parameters, which shows why the two approaches differ.

The database and its cleaning are described in Section 2, the transport model and its identifiability in Section 3, and the numerical and learning methods in Section 4. Section 5 presents the results, Section 6 discusses them and Section 7 concludes.

# 2 Experimental database

## 2.1 Source and test protocol

The data come from the experimental programme of Zhuang et al. ⟦zhuang⟧ on the drying and capillary absorption of concretes containing fly ash and slag. Discs 100 mm in diameter and 50 mm thick were cut from the central part of 100 mm × 150 mm cylinders, vacuum-saturated for 24 h and dried in an oven at 50 °C, following the conditioning procedure of ASTM C1585 ⟦astm1585⟧. The discs were weighed daily until the mass change over 24 h fell below 0.1%, and the degree of saturation was expressed relative to the saturated and oven-dry states. ⟪C|To confirm by the authors: (1) the drying curves were digitised from the figures of Zhuang et al. (name the tool, e.g. WebPlotDigitizer); (2) the lateral surface was not sealed during drying, as assumed by the disc model. If it was sealed, set the lateral face to zero flux (one line in pinn_moisture/axisym.py) and re-run scripts/run_all_disc.sh.|The drying curves used here were digitised from the figures of that publication.⟫

## 2.2 Concrete mixes

The programme covers four w/b ratios (0.3–0.6) and seven binder systems: plain Portland cement (C), 20% and 40% fly ash (F20, F40), 20% and 40% slag (S20, S40), and ternary blends with 10% + 10% and 20% + 20% fly ash and slag (F10S10, F20S20). The proportions are given in Table 1.

| Mix | w/b | Water | Cement | Fly ash | Slag |
|:----------|:----:|:----:|:----:|:----:|:----:|
| C-0.3 | 0.3 | 179 | 596 | – | – |
| C-0.4 | 0.4 | 204 | 509 | – | – |
| C-0.5 | 0.5 | 226 | 451 | – | – |
| C-0.6 | 0.6 | 236 | 393 | – | – |
| F20-0.3 | 0.3 | 173 | 461 | 115 | – |
| F20-0.4 | 0.4 | 200 | 399 | 100 | – |
| F20-0.5 | 0.5 | 222 | 355 | 89 | – |
| F20-0.6 | 0.6 | 233 | 311 | 78 | – |
| F40-0.3 | 0.3 | 167 | 334 | 223 | – |
| F40-0.4 | 0.4 | 197 | 296 | 197 | – |
| F40-0.5 | 0.5 | 217 | 260 | 174 | – |
| F40-0.6 | 0.6 | 231 | 231 | 154 | – |
| S20-0.3 | 0.3 | 176 | 470 | – | 118 |
| S20-0.4 | 0.4 | 205 | 409 | – | 102 |
| S20-0.5 | 0.5 | 227 | 363 | – | 91 |
| S20-0.6 | 0.6 | 237 | 316 | – | 79 |
| S40-0.3 | 0.3 | 174 | 348 | – | 232 |
| S40-0.4 | 0.4 | 205 | 307 | – | 205 |
| S40-0.5 | 0.5 | 223 | 268 | – | 178 |
| S40-0.6 | 0.6 | 237 | 237 | – | 158 |
| F10S10-0.3 | 0.3 | 175 | 466 | 58 | 58 |
| F10S10-0.4 | 0.4 | 202 | 403 | 50 | 50 |
| F10S10-0.5 | 0.5 | 223 | 357 | 45 | 45 |
| F10S10-0.6 | 0.6 | 236 | 315 | 39 | 39 |
| F20S20-0.3 | 0.3 | 170 | 341 | 114 | 114 |
| F20S20-0.4 | 0.4 | 196 | 295 | 98 | 98 |
| F20S20-0.5 | 0.5 | 224 | 269 | 90 | 90 |
| F20S20-0.6 | 0.6 | 241 | 241 | 80 | 80 |

Table: **Table 1.** Mix proportions of the concretes (kg/m³), after Zhuang et al. ⟦zhuang⟧.

## 2.3 Data cleaning

The digitised file contained «n_raw» points for «n_raw_mixes» mixes; S20-0.5 was missing. For most mixes it held two digitisations of the same drying curve: a short series with daily points up to 14 days and a long series extending to 40–100 days. Matching equal saturation values showed that the long series was shifted in time by «shift» days (interquartile range «shift_lo»–«shift_hi» days over the «n_consistent» mixes in which the two series agreed). Only the short series is consistent with drying from $t = 0$, since $1 - S_{\text{avg}}$ grows as $\sqrt{t}$ at early ages. Each curve was therefore rebuilt from the short series up to 14 days and the time-corrected long series beyond, which removed the duplicated points (Fig. 1a,b).

The w/b = 0.6 series could not be repaired. The long series of S20-0.6 and S40-0.6 are identical copies, the short series of F20S20-0.6 mixes two curves, and the time offsets between the two series are erratic (Fig. 1c). The remaining w/b = 0.6 curves also dry more slowly at early ages than the corresponding w/b = 0.5 curves, which is implausible, and a classical fit of them drives the permeability to the lower bound of its admissible range. These mixes were excluded. The database used below contains «n_pts» points for «n_mixes» mixes with w/b between 0.3 and 0.5. Every step is scripted and reported in the code repository.

![**Fig. 1.** Raw digitised series and cleaned curves. (a) C-0.4 and (b) F40-0.4: the long series is the short series shifted by about one day. (c) S40-0.6: the two series are inconsistent and the mix is excluded.](fig3/fig_data_cleaning.png){width=6.5in}

# 3 Moisture transport model

## 3.1 Governing equation and diffusivity

Once hydration has largely stopped, sources and sinks of water can be neglected ⟦baroghel2007⟧, and moisture conservation reads

$$\frac{\partial S}{\partial t} = \nabla\cdot\left[ D(S;\boldsymbol{\theta})\,\nabla S \right] \qquad\qquad (1)$$

where $S \in [0,1]$ is the degree of saturation and $D$ [m²/s] the effective moisture diffusivity. We use the dual-regime diffusivity of Chen et al. ⟦chen2022⟧, which blends a capillary and a vapour contribution:

$$D(S;\boldsymbol{\theta}) = f(S)\,D_{\text{cap}}(S) + \left[1 - f(S)\right] D_{\text{vap}}(S) \qquad\qquad (2)$$

$$D_{\text{vap}}(S) = D_v \left[ \alpha_0 + \frac{1-\alpha_0}{1 + \left(\dfrac{1-S}{1-S_c}\right)^{n}} \right] \qquad\qquad (3)$$

$$D_{\text{cap}}(S) = \frac{K_{\text{int}}}{\eta_l}\left|\frac{\partial P_c}{\partial S}\right|, \qquad \frac{\partial P_c}{\partial S} = -\beta\,P_{c0}\,S^{-\beta-1} \qquad\qquad (4)$$

$$f(S) = \frac{1}{1 + \exp\left[-k\,(S - S_{\text{cap}})\right]} \qquad\qquad (5)$$

Equation (3) has the sigmoidal form proposed by Bažant and Najjar ⟦bazant⟧, with $D_v$ the reference diffusivity, $\alpha_0$ the ratio of minimum to maximum diffusivity, $n$ a shape exponent and $S_c$ the transition saturation. In Eq. (4), $K_{\text{int}}$ is the intrinsic permeability, $\eta_l$ the viscosity of water, $P_{c0}$ a reference capillary pressure and $\beta$ an exponent. The switching function (5) makes vapour transport dominant at low saturation and capillary transport dominant at high saturation. The fixed constants are listed in Table 2.

| Symbol | Meaning | Value | Source |
|:------|:----------------------------|:----------:|:------:|
| $R$, $H$ | Disc radius and half-thickness | 50 mm, 25 mm | ⟦zhuang⟧ |
| $T$ | Drying temperature | 50 °C | ⟦zhuang⟧ |
| $\eta_l$ | Viscosity of water at 50 °C | 5.47 × 10⁻⁴ Pa·s | – |
| $P_{c0}$ | Reference capillary pressure | 5 MPa | ⟦chen2022⟧ |
| $\beta$ | Capillary pressure exponent | 2.5 | ⟦chen2022⟧ |
| $S_c$ | Transition saturation in Eq. (3) | 0.80 | ⟦chen2022⟧ |
| $S_{\text{cap}}$, $k$ | Switching saturation and steepness | 0.75, 50 | ⟦chen2022⟧ |
| $\alpha_0$ | Minimum-to-maximum diffusivity ratio | 0.05 | Section 3.3 |
| $n$ | Shape exponent, shared by all mixes | 8 | Section 3.3 |

Table: **Table 2.** Fixed constants of the transport model.

## 3.2 Specimen geometry and boundary conditions

The discs dry through their two flat faces and their lateral surface. Using the axis and the mid-plane as planes of symmetry, Eq. (1) is solved on $0 \le r \le R$, $0 \le z \le H$ with

$$S(r,z,0) = 1, \qquad S(R,z,t) = S(r,H,t) = 0, \qquad \frac{\partial S}{\partial r}(0,z,t) = \frac{\partial S}{\partial z}(r,0,t) = 0 \qquad\qquad (6)$$

The specimen starts fully saturated, and the exposed surfaces are assumed to reach the oven-dry state immediately, which defines $S = 0$. The quantity compared with the measurements is the volume average

$$S_{\text{avg}}(t) = \frac{2}{R^2 H}\int_0^H\!\!\int_0^R S(r,z,t)\,r\,\mathrm{d}r\,\mathrm{d}z \qquad\qquad (7)$$

A one-dimensional slab whose half-thickness equals the volume-to-surface ratio, $L = V/A = 12.5$ mm, is a common simplification. It reproduces the early-time drying rate of the disc but not its later stages, because the corners of the disc dry from two sides. With the same parameters, the slab curve departs from the disc curve by up to «geo_maxdiff» in $S_{\text{avg}}$ (Fig. 2). A slab model can still be fitted to the disc curves (RMSE ≤ «geo_refit_rmse_hi»), but it then returns parameters that are biased by «geo_dDv_lo» to «geo_dDv_hi» decades in $D_v$ and «geo_dK_lo» to +«geo_dK_hi» decades in $K_{\text{int}}$. The disc geometry is therefore used for all results on transport parameters. The slab is kept only for the controlled comparison with the PINN in Section 5.5, where both methods use the same simplified model.

![**Fig. 2.** Disc model and one-dimensional slab. (a, b) Measured average saturation, disc model fitted to the data, and slab model ($L = V/A$) run with the same parameters. (c) Saturation in the disc (one quarter of the cross-section) after 7 days for C-0.4; the corner dries from two sides.](fig3/fig_geometry.png){width=6.5in}

## 3.3 Identifiability

An average-saturation curve cannot determine many parameters at once. We fitted the disc model to four representative mixes with three parameter sets (Table 3). When $D_v$, $\alpha_0$ and $K_{\text{int}}$ are fitted together, $D_v$ and $\alpha_0$ are perfectly correlated ($\lvert\rho\rvert$ = «id_corr3_min») and the Jacobian is ill-conditioned (condition number «id_cond3_lo» to «id_cond3_hi»). Adding $n$ hardly improves the fit and leaves parameter pairs correlated above «id_corr4_min». With $\alpha_0$ and $n$ fixed, $D_v$ and $K_{\text{int}}$ are well determined: their correlation is «id_corr2_lo»–«id_corr2_hi» and the condition number «id_cond2_lo»–«id_cond2_hi». We therefore identify $\boldsymbol{\theta} = [\log_{10} D_v, \log_{10} K_{\text{int}}]$ for each mix. Because $D_v$ and $\alpha_0$ cannot be separated, the identified $D_v$ is conditional on $\alpha_0 = 0.05$.

The shared exponent $n$ was chosen by profiling: $D_v$ and $K_{\text{int}}$ were fitted to six mixes for $n$ between 1.5 and 16. The total misfit decreases with $n$ and levels off beyond $n \approx 8$; it is «n_gain_2_8»% lower at $n = 8$ than at $n = 2$, and only «n_gain_8_16»% lower at $n = 16$. We use $n = 8$, which lies in the range 6–16 reported by Bažant and Najjar ⟦bazant⟧.

⟪TABLE:identifiability⟫

Table: **Table 3.** Identifiability study with the disc model ($n = 8$ when not identified): fit quality, largest parameter correlation and condition number of the Jacobian.

# 4 Methods

## 4.1 Numerical solution and verification

Equation (1) is discretised with a cell-centred finite-volume scheme on a mesh refined towards the drying faces. For the disc we use 24 × 16 cells in $(r, z)$; for the slab, 60–80 cells. The resulting ordinary differential equations are integrated with an implicit BDF scheme ⟦hairer⟧ at a relative tolerance of 10⁻⁶. The solvers were verified in four ways (Appendix A, Table A.1). For a constant diffusivity, the slab solver reproduces the analytical series solution of slab drying to within 2 × 10⁻³. Under mesh refinement, the slab solution converges at second order, with an error of «fv_err80» in $S_{\text{avg}}$ at 80 cells. The disc solution on the 24 × 16 mesh is within «disc_err» of a 96 × 64 reference. Finally, the water lost through the drying face balances the change in stored water to machine precision (relative error «mass_err»). A further test checks that a disc of very large radius reproduces the slab solution.

## 4.2 Classical identification

In the classical approach, $\boldsymbol{\theta}$ is fitted to each mix separately by nonlinear least squares, using a trust-region reflective algorithm with three starting points. Standard errors and correlations are estimated from the Jacobian at the optimum. To predict a mix that was not tested, the identified parameters of the available mixes are regressed linearly on composition, and the solver is run with the regressed parameters. We refer to this as classical identification with composition regression.

## 4.3 Physics-embedded neural network

The PENN is shown in Fig. 3. A multilayer perceptron $g_{\mathbf{w}}$ with two hidden layers of 16 tanh units maps the composition vector $\mathbf{c} = [(\text{w/b} - 0.3)/0.3,\ \text{FA}/0.4,\ \text{SL}/0.4]$ to the transport parameters. Here FA and SL are the fly ash and slag mass fractions of the binder. The output is mapped smoothly into the admissible range,

$$\boldsymbol{\theta}(\mathbf{c}) = \boldsymbol{\theta}_{\min} + (\boldsymbol{\theta}_{\max} - \boldsymbol{\theta}_{\min}) \odot \sigma\!\left(g_{\mathbf{w}}(\mathbf{c})\right) \qquad\qquad (8)$$

where $\sigma$ is the logistic function, $\log_{10} D_v \in [-10, -7.5]$ and $\log_{10} K_{\text{int}} \in [-21, -18.5]$. The parameters are passed to the transport solver, and the network weights minimise the misfit on all training points:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{N}\sum_{j=1}^{N}\left[S_{\text{avg}}\!\left(t_j;\boldsymbol{\theta}(\mathbf{c}_j)\right) - S_{\text{avg},\exp}(t_j)\right]^2 \qquad\qquad (9)$$

Gradients of Eq. (9) with respect to $\mathbf{w}$ are obtained by automatic differentiation through the solver ⟦rackauckas,thuerey⟧. The solver therefore has to be differentiable. For the slab, the finite-volume scheme was rewritten with tensor operations and integrated by a linearly implicit Euler method on 240 logarithmically spaced time steps; its error is about «diff_err240» in $S_{\text{avg}}$. For the disc, a differentiable version of the same kind costs about 5 s per training iteration. Since only two parameters are identified, we instead solved the disc problem once with the BDF solver on a 49 × 49 grid of $(\log_{10} D_v, \log_{10} K_{\text{int}})$ over the admissible range, at 120 times between 0.05 and 110 days (about «table_time_min» min of computation). Between grid points the solution is interpolated with bicubic convolution ⟦keys⟧, which is differentiable. Each table entry is a solution of Eq. (1), and the interpolated curves differ from direct solutions by at most «table_err» in $S_{\text{avg}}$, which is well below the discretisation error. Training uses Adam ⟦kingma⟧ (learning rate 0.02, decay 0.995 per iteration, weight decay 10⁻⁴) for 400 full-batch iterations, after which the training error no longer changes (Fig. B.1a).

![**Fig. 3.** Physics-embedded neural network. The parameter network predicts the transport parameters from the composition; the solver returns the saturation field and its volume average; the data misfit is back-propagated through the solver to the network weights.](fig3/fig_workflow.png){width=6.5in}

## 4.4 Physics-informed neural network

The PINN was set up on the slab version of the problem ($L = 12.5$ mm, $n = 2$ from the corresponding profile) and uses the same parameter network (8). A field network $N$ with four hidden layers of 48 tanh units takes $(\xi, s, \mathbf{c})$ as input, with $\xi = x/L$, $s = \sqrt{t/t_{\text{ref}}}$ and $t_{\text{ref}} = 100$ days. The boundary condition at the drying face and the initial condition are built into the field:

$$S = 1 - \exp\left(-\frac{\xi}{s} A\right), \qquad A = a(s) + \text{softplus}(N) \qquad\qquad (10)$$

Since $\xi/s$ is the diffusive similarity variable, $S$ tends to 1 when $t \to 0$ and equals 0 at $\xi = 0$. The loss combines the integral data misfit, the dimensionless PDE residual and the symmetry condition at $\xi = 1$:

$$\mathcal{L} = \sigma_d^{-2}\,\mathcal{L}_{\text{data}} + \lambda_{\text{PDE}}\,\mathcal{L}_{\text{PDE}} + \lambda_{\text{sym}}\,\mathcal{L}_{\text{sym}} \qquad\qquad (11)$$

with $\sigma_d = 0.01$, $\lambda_{\text{PDE}} = 10$ and $\lambda_{\text{sym}} = 1$. The residual is multiplied by $t/t_{\text{ref}}$, which keeps it bounded at early times, and half of the 3000 collocation points per iteration are sampled in the similarity variable so that the drying front is resolved at every time. Training uses 3000 Adam iterations followed by 1000 L-BFGS iterations (Fig. B.1b).

Without the term $a(s)$ in Eq. (10), the PINN converged in every trial to a trivial field that dries almost instantly after $t = 0$. That field satisfies the residual nearly everywhere, which is a known failure mode for stiff transport problems ⟦krishnapriyan,wang_causal⟧, and causal weighting of the residual ⟦wang_causal⟧ did not remove it. The lower bound $a(s) = 10^{-3} + 0.1\exp[-(s/0.2)^2]$ limits the speed of the drying front at early times and prevents the collapse. It is used for all PINN results.

## 4.5 Baseline and validation protocol

The data-driven baseline is a network with four hidden layers of 48 tanh units that maps $(\sqrt{t/t_{\text{ref}}}, \mathbf{c})$ directly to $S_{\text{avg}}$. It is trained with Adam for 4000 iterations.

Four validation steps were carried out.

1. **Hold-out test.** Six mixes covering four binder families and three w/b ratios (C-0.4, C-0.5, F10S10-0.3, F20-0.3, F20S20-0.4 and S40-0.4; «n_test_pts» points) were withheld from training and from the composition regression. The other «n_train_mixes» mixes («n_train_pts» points) were used for training. Neural models were trained with five random seeds (three for the PINN), and results are given as mean ± standard deviation.
2. **Cross-validation.** Leave-one-mix-out over the 20 mixes, and leave-one-binder-family-out, in which all mixes of one of the seven binder systems are predicted from the other six. Each fold was trained once (one seed), since the hold-out test showed a negligible spread between seeds.
3. **Noise robustness.** Gaussian noise with a standard deviation of 0.01 or 0.02 was added to the training data, and the models were retrained.
4. **Solver check.** The parameters returned by each model were inserted, without any fitting, into the independent BDF solver, and the resulting curves were compared with the model output.

Accuracy is measured by the coefficient of determination and the root-mean-square error,

$$R^2 = 1 - \frac{\sum_j (y_j - \hat{y}_j)^2}{\sum_j (y_j - \bar{y})^2}, \qquad \text{RMSE} = \sqrt{\frac{1}{N}\sum_j (y_j - \hat{y}_j)^2} \qquad\qquad (12)$$

All computations ran on a single CPU core.

# 5 Results

## 5.1 Classical identification

With $D_v$ and $K_{\text{int}}$ as the only free parameters, the disc model reproduces every drying curve with $R^2$ between «cl_r2_min» and «cl_r2_max» (median «cl_r2_med», median RMSE «cl_rmse_med»), at about «cl_time» s per mix. The standard errors are at most «cl_se_Dv_max» decade for $D_v$ and «cl_se_K_max» decade for $K_{\text{int}}$. The reference diffusivities range from 10^«cl_Dv_lo»^ to 10^«cl_Dv_hi»^ m²/s, and the permeabilities from 10^«cl_K_lo»^ to 10^«cl_K_hi»^ m², which is within the range reported for ordinary and high-performance cementitious materials ⟦baroghel1999,mainguy⟧. These per-mix values serve as the reference in what follows.

## 5.2 Prediction of withheld mixes

Table 4 summarises the hold-out test, Table 5 gives the result for each withheld mix, and Fig. 4 shows the curves. The PENN reaches $R^2$ = «ho_pen_r2» ± «ho_pen_r2_sd» and RMSE = «ho_pen_rmse» ± «ho_pen_rmse_sd». Its error is «ho_gain_mlp»% lower than that of the data-driven network (RMSE «ho_mlp_rmse») and essentially equal to that of classical identification with composition regression (RMSE «ho_fdreg_rmse»). That equality is expected, since both methods solve the same equation and differ only in how the parameters are linked to composition. Fitting the solver directly to the withheld curves, which uses their data, gives RMSE = «ho_fdfit_rmse». The difference with the predictive methods is the information lost when the parameters must be inferred from three composition descriptors.

⟪TABLE:holdout⟫

Table: **Table 4.** Accuracy on the six withheld mixes («n_test_pts» points). ^a^ After a one-off tabulation of the disc solutions («table_time_min» min). ^b^ Sum of the per-mix fits of the training mixes.

⟪TABLE:per_mix⟫

Table: **Table 5.** Coefficient of determination for each withheld mix (mean over seeds).

The largest difference between the methods is for C-0.5, the most permeable of the withheld mixes and the only plain concrete at w/b = 0.5. The data-driven network underestimates its saturation during the first week ($R^2$ = «mix_mlp_C-0.5»), whereas the PENN predicts it with $R^2$ = «mix_pen_C-0.5». For the physics-based models, faster drying of a more permeable mix follows from larger parameters, not from a curve shape that must have been seen during training. The residuals of the PENN are largest in the first days of drying (Fig. 5), when the mass changes fastest and digitisation errors weigh most.

![**Fig. 4.** Drying curves of the six withheld mixes: experiment and predictions. Bands show the range over random seeds; $R^2$ refers to the PENN.](fig3/fig_test_curves.png){width=6.5in}

![**Fig. 5.** PENN predictions: (a) predicted against measured average saturation for training and withheld mixes; (b) residuals against drying time.](fig3/fig_parity.png){width=6.5in}

## 5.3 Cross-validation

A single split can be favourable by chance, so all mixes were also predicted in turn (Table 6, Fig. 6). In leave-one-mix-out cross-validation the three methods remain close: the PENN gives a pooled $R^2$ of «cv_lomo_pen_r2», classical identification with regression «cv_lomo_fdreg_r2», and the data-driven network «cv_lomo_mlp_r2», and no mix falls below $R^2$ = 0.95. The picture changes when a whole binder family is withheld, which forces the models to extrapolate in SCM type or content. The two physics-based methods keep $R^2$ ≥ «cv_lofo_fdreg_min» for every mix, with pooled RMSE values of «cv_lofo_pen_rmse» (PENN) and «cv_lofo_fdreg_rmse» (classical identification). The data-driven network, by contrast, falls below 0.95 for «cv_lofo_mlp_nbad» of the 20 mixes, down to $R^2$ = «cv_lofo_mlp_min» for «cv_lofo_mlp_worst_mix», and its pooled RMSE («cv_lofo_mlp_rmse») is more than twice that of the PENN («cv_lofo_pen_rmse»). A composition–curve mapping learned without physics interpolates well but does not transfer to binders it has not seen.

⟪TABLE:cv⟫

Table: **Table 6.** Cross-validation over all 20 mixes.

![**Fig. 6.** Coefficient of determination of each mix when it is predicted (a) from the other 19 mixes and (b) without any mix of its binder family. Values below 0.80 are plotted at the lower edge and labelled with their actual value.](fig3/fig_cv.png){width=6.5in}

## 5.4 Robustness to measurement noise

Adding Gaussian noise to the training data leaves the PENN predictions practically unchanged (Table 7). With a standard deviation of 0.02, which is about three times the median RMSE of the classical fits, the test RMSE remains «noise02_pen_rmse» and the identified parameters move by at most «noise02_dDv» decade for $D_v$ and «noise02_dK» decade for $K_{\text{int}}$. The error of the data-driven network grows from «ho_mlp_rmse» to «noise02_mlp_rmse». The solver acts as a strong regulariser: noise can only shift two parameters per mix, and a smooth composition map averages it out.

⟪TABLE:noise⟫

Table: **Table 7.** Effect of noise added to the training data (five seeds; parameter shifts relative to noise-free training, maximum over the 20 mixes).

## 5.5 Comparison with a physics-informed neural network

The PINN comparison uses the slab version of the problem, for which both approaches were implemented with identical constants, data split and parameter network (Table 8). On the withheld mixes the PINN is less accurate than the PENN (RMSE «sl_pinn_rmse» against «sl_pen_rmse») and varies more between seeds. At the same time it fits the training data far more closely (training RMSE «sl_pinn_train_rmse» against «sl_pen_train_rmse»). Its flexible field can follow the training curves more closely than the transport equation allows, and this extra freedom does not carry over to new mixes. Without its PDE residual the PINN degrades further (RMSE «sl_pinn_nopde_rmse»), so the residual does carry information.

⟪TABLE:slab⟫

Table: **Table 8.** Controlled comparison on the one-dimensional slab model (six withheld mixes). The solver check gives the $R^2$ between each model and the independent BDF solver run with the parameters returned by that model.

The solver check is more revealing (Fig. 7). Run with the parameters of the PENN, the BDF solver reproduces the PENN curves with $R^2$ = «sl_fwd_pen_r2». With the parameters of the PINN it does not ($R^2$ = «sl_fwd_pinn_r2», RMSE = «sl_fwd_pinn_rmse»), and on the withheld mixes it matches the data only with $R^2$ = «sl_fwd_pinn_data_test_r2». The reference diffusivities of the PINN are close to the classical values (mean difference «sl_pinn_Dv_mae» decade), but its permeabilities are about «sl_pinn_K_bias» decade too low. $K_{\text{int}}$ controls the capillary regime, which prevails in the specimen during the first day of drying, before the first measurement and where the PDE residual is hardest to enforce. The PINN can therefore reproduce the measured curves with a field that does not obey the transport equation early on; the corresponding profiles are compared in Fig. B.2. Because the PENN integrates the equation from $t = 0$, this kind of inconsistency cannot arise.

![**Fig. 7.** Solver check on the slab model: average saturation computed by the independent BDF solver with the parameters returned by (a) the PINN and (b) the PENN, plotted against the output of each model.](fig3/fig_forward.png){width=6.5in}

## 5.6 Transport parameters and effect of composition

Figure 8 compares the parameters predicted by the PENN from composition alone with the classical per-mix fits. The reference diffusivities agree closely (mean absolute difference «th_log10_Dv_mae» decade, $r$ = «th_log10_Dv_corr»). The permeabilities follow the same overall trend with w/b but over a narrower range (10^«th_log10_K_lo»^ to 10^«th_log10_K_hi»^ m²) and with a weaker mix-to-mix correlation ($r$ = «th_log10_K_corr»). $K_{\text{int}}$ is the less constrained of the two parameters, and the composition map smooths its scatter. The spread across random seeds is negligible («th_log10_Dv_sd» decade for $D_v$, «th_log10_K_sd» for $K_{\text{int}}$).

Both parameters increase with w/b. For plain concrete, $D_v$ rises from 10^«C03_Dv»^ to 10^«C05_Dv»^ m²/s between w/b = 0.3 and 0.5, consistent with the coarser and better connected capillary porosity of high-w/b pastes ⟦mehta,baroghel2007⟧. At a given w/b, all SCMs lower $D_v$. The effect is largest for 40% fly ash, whose reference diffusivity at w/b = 0.3 (10^«F4003_Dv»^ m²/s) is «F40_factor» times lower than that of plain concrete, followed by the 20% + 20% ternary blend and 20% fly ash; slag reduces $D_v$ less, and 20% slag least. This ranking agrees with the pore refinement produced by the pozzolanic and latent hydraulic reactions ⟦mehta⟧.

The diffusivity varies by about an order of magnitude with saturation (Fig. 9). It lies between «D_min_lo» and «D_min_hi» m²/s in the dry, vapour-dominated range and reaches «D_max_lo» to «D_max_hi» m²/s near $S$ = «D_Speak», where capillary transport takes over. The position of the transition is set by $S_{\text{cap}}$ in the model of Chen et al. ⟦chen2022⟧ and is not identified from the data.

![**Fig. 8.** Transport parameters against w/b: (a) reference diffusivity $D_v$ and (b) intrinsic permeability $K_{\text{int}}$. Filled symbols: PENN predictions from composition (error bars: standard deviation over seeds). Open symbols: classical per-mix fits (error bars: standard errors).](fig3/fig_parameters.png){width=6.5in}

![**Fig. 9.** Diffusivity predicted by the PENN: (a) vapour and capillary contributions for the withheld mix C-0.4; (b) effect of w/b and fly ash.](fig3/fig_diffusivity.png){width=6.5in}

## 5.7 Internal saturation fields

Since the PENN runs the transport model, the saturation field inside the disc comes with every prediction (Fig. 10). After one day, a dry layer a few millimetres thick has formed along the exposed faces, while the centre is still almost saturated (S = «centre1d_C-0.4» for C-0.4). Drying then proceeds from the edge and the faces towards the centre, and the corner region dries first. The centre falls below S = 0.5 after about «tc_C-0.4» days for C-0.4, «tc_S40-0.4» days for S40-0.4 and «tc_F20-0.3» days for F20-0.3, which shows how strongly fly ash slows internal drying even when the average curves look similar at early ages. Such fields are what durability models need as initial conditions for chloride ingress or carbonation, and they are obtained here from a mass-loss test.

![**Fig. 10.** Saturation in the disc (one quarter of the cross-section; drying faces at the top and right) predicted by the PENN for the withheld mixes C-0.4 and F20-0.3 after 1, 7 and 28 days. Contours: $S$ = 0.75 (solid), 0.50 (dashed) and 0.25 (dotted).](fig3/fig_fields.png){width=6.5in}

# 6 Discussion

## 6.1 Embedding versus penalising the physics

The results separate two things that are easily conflated: fitting drying curves and identifying physically meaningful parameters. All the methods fit the training data, and those that contain the physics also predict new mixes well. Only the methods that actually solve the transport equation, however, return parameters that reproduce the data when used in an independent model. In a PINN the governing equation is a soft constraint, weighted against the data and weakest where the solution is steepest; for drying concrete this is the early capillary stage, which is precisely where the permeability acts. A PINN can be a useful surrogate of the curves, but its parameters should not be reported without a solver check such as Fig. 7. We would recommend this check for any study that identifies material parameters with PINNs.

The PENN matches the accuracy of classical identification while removing its main practical drawback. A single model covers the whole composition space, a new mix requires one evaluation of the network instead of a new test and inverse analysis, and the composition map can be extended to further descriptors (curing age, porosity, binder chemistry) without changing the solver. Where it differs most from a purely data-driven model is extrapolation: when a binder family is missing from the training data, the solver constrains the prediction and the error stays small, whereas the data-driven network can fail badly. The tabulated solver also makes training very fast. The two-parameter structure that permits tabulation is a consequence of the identifiability analysis rather than an arbitrary simplification.

## 6.2 Limitations

The identified parameters are effective values of an isothermal model with fixed constants. In particular, $D_v$ is conditional on $\alpha_0$, and the regime transition is imposed through $S_{\text{cap}}$; the parameters should not be read as intrinsic properties without independent measurements. Only two parameters per mix can be identified from average-saturation curves, so richer descriptions of the pore structure would need additional data such as sorption isotherms or internal humidity measurements. The database comes from one laboratory and one drying condition, the curves were digitised from published figures, and the w/b = 0.6 mixes had to be excluded. Finally, the internal saturation fields are model predictions and have not been compared with direct measurements, for example by gamma-ray attenuation or NMR.

# 7 Conclusions

A physics-embedded neural network was developed to predict the moisture transport parameters of concrete from mix composition and was compared with classical identification, a data-driven network and a physics-informed neural network. The main conclusions are:

1. Average-saturation curves identify only two transport parameters per mix, the reference diffusivity and the intrinsic permeability. The specimen geometry must be modelled explicitly: an equivalent slab biases these parameters by up to «geo_dK_hi» decade.
2. Trained through a transport solver, the PENN predicts the drying of withheld mixes with $R^2$ = «ho_pen_r2», a «ho_gain_mlp»% lower error than a data-driven network, and matches classical identification with composition regression. It remains accurate when a whole binder family is withheld ($R^2$ ≥ «cv_lofo_pen_min» for every mix), whereas the data-driven network does not.
3. The PENN is insensitive to noise in the training data, and its parameters satisfy the transport equation by construction. A PINN trained on the same data fits the curves, but its parameters fail an independent solver check ($R^2$ = «sl_fwd_pinn_r2»), with permeabilities about one order of magnitude too low.
4. The predicted parameters increase with w/b and decrease with SCM content, most strongly for 40% fly ash, in agreement with the known effect of these materials on pore structure. The model also yields the internal saturation field, which can serve as input to durability models.

The absorption tests available for the same mixes offer an independent validation of the identified parameters and will be used in future work, together with wetting–drying cycles, uncertainty quantification ⟦yang_bpinn⟧ and coupling with chloride transport ⟦conciatori⟧.

# Declarations {.unnumbered}

**CRediT authorship contribution statement.** Abderrahmane Sghuri: Conceptualization, Methodology, Software, Formal analysis, Visualization, Writing – original draft. Xuande Chen: Conceptualization, Supervision, Validation, Writing – review & editing, Funding acquisition. Xiaodong Wang: ⟪U|[…]⟫.

**Data and code availability.** The experimental data are those of Zhuang et al. ⟦zhuang⟧. The digitised and cleaned data and the code that produces every result, table and figure of this paper are available at https://github.com/Abderahmane2002/hybrid-pinn-lstm-concrete ⟪U|[archive a release on Zenodo and cite its DOI]⟫.

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** ⟪U|[Funding agency and grant number]⟫

**Declaration of generative AI use.** ⟪U|[Elsevier requires authors to disclose the use of generative AI tools in preparing the manuscript or code; state the tool and its purpose here.]⟫

# Appendix A. Numerical verification {.unnumbered}

⟪TABLE:numerics⟫

Table: **Table A.1.** Verification of the solvers: maximum error in $S_{\text{avg}}$ between 0.25 and 90 days for three representative parameter sets, against a fine reference solution (640 cells for the slab solver, a 160-cell BDF solution for the differentiable solver, 96 × 64 cells for the disc).

# Appendix B. Training histories and PINN profiles {.unnumbered}

![**Fig. B.1.** Training histories: (a) RMSE of the PENN on the training mixes (five seeds); (b) loss terms of the PINN (median and range over three seeds).](fig3/fig_convergence.png){width=6.5in}

![**Fig. B.2.** Saturation profiles of three withheld mixes on the slab model, computed by the BDF solver with the PENN parameters (solid) and by the PINN (dotted). The two differ most during the first two days.](fig3/fig_profiles_slab.png){width=6.5in}

# References {.unnumbered}

⟪REFS⟫
