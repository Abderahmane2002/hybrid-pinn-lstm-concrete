# Hybrid PINN-LSTM Concrete Moisture Model

This repository contains a Physics-Informed Neural Network (PINN) framework developed for modeling moisture transport in concrete using global saturation measurements. The model combines data-driven learning and physical constraints to reconstruct the spatio-temporal saturation field inside concrete specimens.

## Overview

The code is designed to:

- learn the moisture evolution in concrete from experimental data
- reconstruct local saturation profiles \( S(x,t) \) from average saturation measurements
- estimate mix-dependent material parameters
- incorporate physical constraints through a nonlinear diffusion PDE
- account for both vapor diffusion and capillary transport mechanisms

The framework is especially intended for research applications related to concrete durability, hygro-thermal transport, and inverse identification of transport parameters.

## Main Features

- Physics-Informed Neural Network (PINN) for 1D moisture transport
- Nonlinear diffusion model with:
  - vapor transport
  - capillary transport
  - sigmoid-based transition between mechanisms
- Learnable material parameters for each concrete mix:
  - vapor diffusion coefficient `Dv`
  - tortuosity parameter `alpha0`
  - pore shape parameter `nShape`
  - porosity `phi`
- Integral loss based on average saturation measurements
- Boundary condition and initial condition constraints
- Automatic plotting of:
  - training losses
  - learned material parameters
  - diffusion coefficient curves
  - saturation profiles
  - prediction vs experimental data

## Repository Structure

```text
hybrid-pinn-lstm-concrete/
│
├── new_pinn.py
├── tidy_dataset_correctedV1.csv
├── requirements.txt
├── README.md
└── concrete_moisture_pinn_model.pth