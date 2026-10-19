"""The FD solver must reproduce the analytical solution for constant diffusivity."""
import numpy as np

from pinn_moisture.fd_solver import FDSolver
from pinn_moisture.physics import PhysicsConfig, SECONDS_PER_DAY


def analytical_slab(t_day, D, L, n_terms=200):
    """Average saturation of a slab (half-thickness L), S=1 initially, S=0 at the surface."""
    k = np.arange(n_terms)[:, None]
    lam = (2 * k + 1) * np.pi / (2 * L)
    return (8 / ((2 * k + 1) ** 2 * np.pi ** 2) * np.exp(-lam ** 2 * D * t_day * SECONDS_PER_DAY)).sum(0)


def test_constant_diffusivity_matches_series_solution():
    cfg = PhysicsConfig(tau_b=1e-4)
    D = 2e-10
    solver = FDSolver(cfg, n_cells=120, D_fn=lambda S, p: np.full_like(S, D))
    t = np.array([0.5, 1, 2, 5, 10, 20, 50])
    s_fd, _ = solver.solve({}, t)
    s_ex = analytical_slab(t, D, cfg.L)
    assert np.max(np.abs(s_fd - s_ex)) < 2e-3, (s_fd, s_ex)


def test_mass_balance_monotone_drying():
    cfg = PhysicsConfig()
    s, S = FDSolver(cfg).solve({}, np.linspace(0.5, 60, 30))
    assert np.all(np.diff(s) <= 1e-9)            # average saturation never increases
    assert np.all(np.diff(S, axis=0) >= -1e-6)   # profiles increase from the surface inwards


if __name__ == "__main__":
    test_constant_diffusivity_matches_series_solution()
    test_mass_balance_monotone_drying()
    print("ok")


def test_differentiable_solver_matches_reference():
    import torch
    from pinn_moisture.diff_solver import TorchFVSolver
    from pinn_moisture.pinn import DTYPE
    cfg = PhysicsConfig()
    th = {"log10_Dv": -9.2, "log10_K": -19.6}
    t = np.array([0.5, 2, 7, 20, 60])
    ref, _ = FDSolver(cfg, n_cells=120).solve(th, t, rtol=1e-8, atol=1e-10)
    s = TorchFVSolver(cfg)
    q = s.interp(s.run({k: torch.tensor([v], dtype=DTYPE) for k, v in th.items()}),
                 torch.tensor(t[None, :], dtype=DTYPE)).numpy()[0]
    assert np.max(np.abs(q - ref)) < 5e-3


def test_disc_reduces_to_slab_when_lateral_face_is_far():
    """A very wide disc dried through its flat faces only behaves like a slab of half-thickness H."""
    from pinn_moisture.axisym import AxisymBDF
    from pinn_moisture.physics import config_for
    cfg = config_for("disc", R_disc=2.0, H_disc=0.0125)
    th = {"log10_Dv": -8.7, "log10_K": -19.6}
    t = np.array([1.0, 5.0, 20.0])
    s2, _ = AxisymBDF(cfg, nr=8, nz=24, ratio=1.0).solve(th, t)
    cfg_s = config_for("disc")
    cfg_s.geometry, cfg_s.L = "slab", 0.0125
    s1, _ = FDSolver(cfg_s, n_cells=24, ratio=1.0).solve(th, t)
    # the lateral face of a 2 m disc affects < 1% of the volume
    assert np.max(np.abs(s2 - s1)) < 0.01
