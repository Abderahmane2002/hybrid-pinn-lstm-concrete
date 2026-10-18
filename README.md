# Hybrid PINN for Moisture Transport in Concrete

> **Revision branch (`paper-revision`).** The code used in the revised manuscript is the
> `pinn_moisture/` package and the `scripts/` pipeline (cleaned data, finite-volume solver,
> physics-embedded network, PINN and baselines). See `REVISION_NOTES.md` for the commands,
> the list of fixes with respect to the notebooks, and the results. The notebooks are kept
> unchanged for reference.


Physics-Informed Neural Network framework for modeling moisture transport in cementitious materials using nonlinear diffusion and experimental saturation measurements.

> This repository contains a PINN-based approach for reconstructing moisture fields and identifying transport parameters in concrete using global saturation data.

---

## Overview

Moisture transport in concrete is governed by nonlinear diffusion mechanisms combining vapor diffusion and capillary transport.

This project develops a **Physics-Informed Neural Network (PINN)** capable of:
- Reconstructing the spatio-temporal saturation field $S(x,t)$
- Learning material transport parameters directly from experimental data
- Incorporating physical constraints through PDE residuals
- Modeling nonlinear diffusion mechanisms in cementitious materials

---

## Repository Structure
```text
hybrid-pinn-lstm-concrete/
│
├── notebooks/
│   └── new_pinn.ipynb        # Main PINN implementation
│
├── plots/                    # Generated figures and visualization results
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore
```

The main implementation is located in `notebooks/new_pinn.ipynb` and contains:
- PINN architecture
- Physics-informed loss formulation
- Nonlinear diffusion model (vapor + capillary transport)
- Training procedure
- Model evaluation and visualization
- Reconstruction of saturation profiles $S(x,t)$

---

## Physical Model

Moisture transport in concrete is modeled using a nonlinear diffusion equation:

$$\frac{\partial S}{\partial t} = \frac{\partial}{\partial x}\left(D(S)\frac{\partial S}{\partial x}\right)$$

where:
- $S(x,t)$ is the saturation field
- $D(S)$ is the nonlinear diffusion coefficient

The diffusion coefficient combines two mechanisms:
- **Vapor transport** $D_v(S)$
- **Capillary transport** $D_c(S)$

A smooth transition between both regimes is modeled as:

$$D(S) = w(S)\,D_c(S) + (1 - w(S))\,D_v(S)$$

---

## Main Features

- Physics-Informed Neural Network for nonlinear diffusion
- Learnable material parameters for each concrete mix
- Vapor + capillary transport modeling
- Integral loss based on average saturation measurements
- Boundary condition and initial condition constraints
- Reconstruction of internal saturation profiles
- Automatic visualization of results

---

## Installation

Clone the repository:
```bash
git clone https://github.com/Abderahmane2002/hybrid-pinn-lstm-concrete.git
cd hybrid-pinn-lstm-concrete
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

Open the main notebook and run the cells sequentially:
```bash
notebooks/new_pinn.ipynb
```

This will:
1. Load the dataset
2. Initialize the PINN model
3. Train the model
4. Reconstruct saturation profiles
5. Visualize diffusion mechanisms
6. Evaluate prediction performance

> If no dataset is provided, the code can generate synthetic data for testing.

---

## Outputs

The framework produces:
- Training loss evolution
- Learned diffusion coefficients $D(S)$
- Reconstructed saturation profiles $S(x,t)$
- Comparison between predicted and experimental saturation
- $R^2$ performance metrics

Generated figures are stored in the `plots/` directory.

---

## Requirements
```
torch
numpy
pandas
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Author

**Abderrahmane Sghuri**  
Master's Research – Civil Engineering  
Université du Québec à Rimouski (UQAR)

*Research topic: Physics-Informed Neural Networks for modeling moisture transport in concrete.*

---

## License

This repository is provided for academic and research purposes.
