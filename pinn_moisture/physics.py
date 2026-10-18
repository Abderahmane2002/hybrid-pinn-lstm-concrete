"""Physical model: nonlinear moisture diffusion in drying concrete.

Governing equation (1D, isothermal, no sources)

    dS/dt = d/dx [ D(S; theta) dS/dx ],   x in [0, L],  t > 0

with the dual-regime diffusivity of Chen et al. (2022):

    D(S)     = f(S) D_cap(S) + (1 - f(S)) D_vap(S)
    D_vap(S) = D_v [ alpha0 + (1 - alpha0) / (1 + ((1 - S)/(1 - S_c))^n) ]
    D_cap(S) = beta * Pc0 * S^(-beta-1) * K_int / eta_l
    f(S)     = sigmoid(k (S - S_cap))

Boundary / initial conditions (oven drying of a vacuum-saturated disc):

    S(x, 0) = S_init = 1
    S(0, t) = S_b(t) = S_e + (S_init - S_e) exp(-t / tau_b)   (drying face)
    dS/dx(L, t) = 0                                           (symmetry plane)

All functions accept numpy arrays or torch tensors (``xp`` = numpy or torch).
Units: x [m], t [day], D [m^2/s] (converted with SECONDS_PER_DAY where needed).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

SECONDS_PER_DAY = 86400.0


@dataclass
class PhysicsConfig:
    # geometry: 100 mm x 50 mm disc dried on all faces -> equivalent slab
    # half-thickness L = V / A_exposed = 12.5 mm (set 0.025 if the lateral face was sealed)
    L: float = 0.0125            # [m]
    t_ref: float = 100.0         # [day] time scale used for non-dimensionalisation
    # boundary / initial conditions
    S_init: float = 1.0          # vacuum-saturated specimen
    S_e: float = 0.0             # surface saturation in the oven (oven-dry reference)
    tau_b: float = 1e-4          # [day] surface reaches S_e almost instantly (Dirichlet); FD regularisation only
    # fixed constants of the diffusivity (Chen et al. 2022 / original code)
    S_c: float = 0.8
    S_cap: float = 0.75
    k_switch: float = 50.0
    beta: float = 2.5
    Pc0: float = 5.0e6           # [Pa]
    eta_l: float = 5.47e-4       # [Pa s] water at 50 C
    S_cap_floor: float = 0.3     # D_cap only evaluated for S >= floor (f(S) ~ 1e-10 below)
    # admissible ranges of the identified parameters (mapped smoothly, no hard clamp)
    log10_Dv_range: tuple = (-12.0, -6.0)     # [m^2/s]
    alpha0_range: tuple = (0.001, 0.5)
    n_range: tuple = (1.0, 16.0)
    log10_K_range: tuple = (-23.0, -18.0)     # [m^2]
    # Identified per mix: log10 D_v and log10 K_int. Identifiability study
    # (scripts/identifiability.py): D_v and alpha0 are perfectly correlated (|rho| = 1.00),
    # so alpha0 is fixed; n is shared by all mixes and fixed at the value minimising the
    # total misfit of the classical fits (n = 2).
    identified: tuple = ("log10_Dv", "log10_K")
    defaults: dict = field(default_factory=lambda: {
        "log10_Dv": -9.0, "alpha0": 0.05, "n": 2.0, "log10_K": -19.77})  # K = 1.7e-20 m^2

    def to_dict(self):
        return asdict(self)


def _backend(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return torch
    except ImportError:
        pass
    return np


def surface_saturation(t_day, cfg: PhysicsConfig):
    xp = _backend(t_day)
    return cfg.S_e + (cfg.S_init - cfg.S_e) * xp.exp(-t_day / cfg.tau_b)


def D_vap(S, Dv, alpha0, n, cfg: PhysicsConfig):
    xp = _backend(S)
    S = xp.clip(S, 0.0, 1.0) if xp is np else S.clamp(0.0, 1.0)
    r = (1.0 - S) / (1.0 - cfg.S_c)
    return Dv * (alpha0 + (1.0 - alpha0) / (1.0 + r ** n))


def D_cap(S, K, cfg: PhysicsConfig):
    xp = _backend(S)
    S = xp.clip(S, cfg.S_cap_floor, 1.0) if xp is np else S.clamp(cfg.S_cap_floor, 1.0)
    return cfg.beta * cfg.Pc0 * S ** (-cfg.beta - 1.0) * K / cfg.eta_l


def switch(S, cfg: PhysicsConfig):
    xp = _backend(S)
    z = cfg.k_switch * (S - cfg.S_cap)
    return 1.0 / (1.0 + xp.exp(-z))


def diffusivity(S, Dv, alpha0, n, K, cfg: PhysicsConfig):
    """Effective moisture diffusivity D(S) [m^2/s]."""
    f = switch(S, cfg)
    return f * D_cap(S, K, cfg) + (1.0 - f) * D_vap(S, Dv, alpha0, n, cfg)


def theta_to_physical(theta: dict, cfg: PhysicsConfig) -> dict:
    """Complete a (possibly partial) parameter dict with defaults; return Dv, alpha0, n, K."""
    p = dict(cfg.defaults)
    p.update(theta)
    return {"Dv": 10.0 ** p["log10_Dv"], "alpha0": p["alpha0"], "n": p["n"], "K": 10.0 ** p["log10_K"]}


PARAM_RANGES = {"log10_Dv": "log10_Dv_range", "alpha0": "alpha0_range", "n": "n_range", "log10_K": "log10_K_range"}


def param_bounds(cfg: PhysicsConfig):
    return [getattr(cfg, PARAM_RANGES[k]) for k in cfg.identified]
