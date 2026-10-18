"""Composition-aware physics-informed neural network for drying concrete.

Two networks are trained jointly:

* the *field network* N(xi, sqrt(tau), c) represents the saturation field of any
  mix of composition c = (w/b, fly-ash fraction, slag fraction);
* the *parameter head* g(c) returns the transport parameters theta(c)
  (log10 D_v, log10 K_int) of that mix.

Because the parameters are a function of composition and not free per-mix
variables, a mix that is withheld from training is a genuine prediction.

Hard constraints (similarity ansatz). With s = sqrt(tau) the field is written as

    S(xi, t) = S_e + (S_init - S_e) * (1 - exp(-(xi / s) * A)),   A = a_min + a_scale * softplus(N) > 0,

so that S(0, t) = S_e for t > 0 and S -> S_init when t -> 0 (xi / s -> infinity):
the initial condition is built in through the diffusive similarity variable xi/sqrt(tau)
instead of being imposed at a single instant, which prevents the known PINN failure
in which the field "forgets" the initial condition before the first collocation time.
Only the symmetry condition at xi = 1, the PDE residual and the integral data
constraint are imposed through the loss.

Dimensionless PDE: xi = x/L, tau = t/t_ref,
    dS/dtau = Pi * d/dxi [ D(S) dS/dxi ],   Pi = t_ref[s] / L^2.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn

from .physics import PhysicsConfig, SECONDS_PER_DAY, diffusivity, param_bounds, theta_to_physical

DTYPE = torch.float64


@dataclass
class TrainConfig:
    hidden: int = 48
    layers: int = 4
    head_hidden: int = 16
    a_scale: float = 1.0
    a_min: float = 1e-3
    a_front: float = 0.1             # ~4.6 / (2 sqrt(Pi D_max)), D_max = 1e-8 m2/s
    s_front: float = 0.2             # sqrt(t/t_ref) after which the bound is relaxed (~4 d)
    n_quad: int = 32                 # Gauss-Legendre points for the integral constraint
    n_colloc: int = 3000             # PDE collocation points per iteration
    n_neumann: int = 256
    tau_min: float = 1e-6            # smallest collocation time (~9 s for t_ref = 100 d)
    eta_max: float = 40.0            # range of the similarity variable xi/sqrt(tau) sampled
    res_weight_exp: float = 1.0      # residual x tau: regular as tau -> 0 in similarity coordinates
    sigma_data: float = 0.01         # data scale (digitisation noise level)
    lambda_pde: float = 10.0
    lambda_neu: float = 1.0
    adam_iters: int = 3000
    lr: float = 2e-3
    lr_decay: float = 0.9993         # per iteration
    lbfgs_iters: int = 1000
    head_weight_decay: float = 1e-4
    seed: int = 0
    use_pde: bool = True
    causal_eps: float = 0.0          # >0: causal weighting of the residual over time bins (Wang et al. 2022)
    causal_bins: int = 32

    def to_dict(self):
        return asdict(self)


def composition_features(df):
    """Map (w/b, FA fraction, SL fraction) to [0, 1]."""
    return np.stack([(df["wb"].to_numpy() - 0.3) / 0.3,
                     df["FA_frac"].to_numpy() / 0.4,
                     df["SL_frac"].to_numpy() / 0.4], axis=-1)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, layers):
        super().__init__()
        mods, d = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        mods.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*mods)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ParameterHead(nn.Module):
    """theta(c), mapped smoothly into the admissible box (no hard clamp)."""

    def __init__(self, pcfg: PhysicsConfig, hidden: int):
        super().__init__()
        lo, hi = np.array(param_bounds(pcfg)).T
        self.register_buffer("lo", torch.tensor(lo, dtype=DTYPE))
        self.register_buffer("hi", torch.tensor(hi, dtype=DTYPE))
        self.mlp = MLP(3, len(lo), hidden, 2)
        # start at the default parameter values
        z0 = np.array([pcfg.defaults[k] for k in pcfg.identified])
        p0 = (z0 - lo) / (hi - lo)
        with torch.no_grad():
            self.mlp.net[-1].bias.copy_(torch.tensor(np.log(p0 / (1 - p0)), dtype=DTYPE))
            self.mlp.net[-1].weight.mul_(0.1)
        self.names = list(pcfg.identified)

    def forward(self, c):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.mlp(c))


class CompositionPINN(nn.Module):
    def __init__(self, pcfg: PhysicsConfig, tcfg: TrainConfig):
        super().__init__()
        self.pcfg, self.tcfg = pcfg, tcfg
        self.field = MLP(5, 1, tcfg.hidden, tcfg.layers)
        self.head = ParameterHead(pcfg, tcfg.head_hidden)
        self.Pi = pcfg.t_ref * SECONDS_PER_DAY / pcfg.L ** 2

    # ---- saturation field ---------------------------------------------------
    def S(self, xi, tau, c):
        """xi, tau: (N,) tensors; c: (N, 3)."""
        p = self.pcfg
        s = torch.sqrt(tau)
        inp = torch.cat([xi[:, None] * 2 - 1, s[:, None] * 2 - 1, c * 2 - 1], dim=1)
        # lower bound on A = bound on the speed of the drying front while it propagates
        # (x_f <= 2 sqrt(D_max t)); relaxed once the front has crossed the specimen (s > s_front)
        a_lo = self.tcfg.a_min + self.tcfg.a_front * torch.exp(-(s / self.tcfg.s_front) ** 2)
        A = a_lo + self.tcfg.a_scale * nn.functional.softplus(self.field(inp)[:, 0])
        return p.S_e + (p.S_init - p.S_e) * (1.0 - torch.exp(-(xi / s) * A))

    def S_avg(self, tau, c, n_quad=None):
        """Spatial average of S using Gauss-Legendre quadrature on xi = eta^2."""
        n_quad = n_quad or self.tcfg.n_quad
        eta, w = np.polynomial.legendre.leggauss(n_quad)
        eta = torch.tensor(0.5 * (eta + 1), dtype=DTYPE)
        w = torch.tensor(0.5 * w, dtype=DTYPE)
        xi = eta ** 2
        jac = 2 * eta
        nT = tau.shape[0]
        S = self.S(xi.repeat(nT), tau.repeat_interleave(n_quad), c.repeat_interleave(n_quad, 0))
        return (S.view(nT, n_quad) * (w * jac)[None, :]).sum(1)

    # ---- physics ----------------------------------------------------------------
    def D(self, S, theta):
        th = {k: theta[:, i] for i, k in enumerate(self.head.names)}
        pp = theta_to_physical(th, self.pcfg)
        return diffusivity(S, pp["Dv"], pp["alpha0"], pp["n"], pp["K"], self.pcfg)

    def pde_residual(self, xi, tau, c):
        xi = xi.clone().requires_grad_(True)
        tau = tau.clone().requires_grad_(True)
        S = self.S(xi, tau, c)
        S_tau, S_xi = torch.autograd.grad(S.sum(), (tau, xi), create_graph=True)
        flux = self.D(S, self.head(c)) * S_xi
        flux_xi = torch.autograd.grad(flux.sum(), xi, create_graph=True)[0]
        return S_tau - self.Pi * flux_xi

    def neumann_residual(self, tau, c):
        xi = torch.ones_like(tau).requires_grad_(True)
        S = self.S(xi, tau, c)
        return torch.autograd.grad(S.sum(), xi, create_graph=True)[0]


class Trainer:
    def __init__(self, train_df, pcfg: PhysicsConfig, tcfg: TrainConfig):
        torch.manual_seed(tcfg.seed)
        np.random.seed(tcfg.seed)
        self.pcfg, self.tcfg = pcfg, tcfg
        self.model = CompositionPINN(pcfg, tcfg).to(DTYPE)
        self.tau = torch.tensor(train_df["Time_d"].to_numpy() / pcfg.t_ref, dtype=DTYPE)
        self.c = torch.tensor(composition_features(train_df), dtype=DTYPE)
        self.s_obs = torch.tensor(train_df["S_frac"].to_numpy(), dtype=DTYPE)
        mixes = train_df.drop_duplicates("Mix")
        self.c_mix = torch.tensor(composition_features(mixes), dtype=DTYPE)
        self.t_max = float(train_df["Time_d"].max() / pcfg.t_ref)
        self.history = []
        self.gen = torch.Generator().manual_seed(tcfg.seed)

    def sample_collocation(self):
        """Half of the points uniform in (xi^0.5, sqrt(tau)); half in similarity variables
        (eta = xi / sqrt(tau), log-uniform sqrt(tau)) so that the moving drying front is
        sampled at every time, including the earliest ones."""
        tc = self.tcfg
        n = tc.n_colloc
        n1 = n // 2
        g = self.gen
        k = torch.randint(0, len(self.c_mix), (n,), generator=g)
        s_lo, s_hi = math.sqrt(tc.tau_min), math.sqrt(self.t_max)
        s1 = s_lo + torch.rand(n1, generator=g, dtype=DTYPE) * (s_hi - s_lo)
        xi1 = torch.rand(n1, generator=g, dtype=DTYPE) ** 2
        s2 = torch.exp(math.log(s_lo) + torch.rand(n - n1, generator=g, dtype=DTYPE) * (math.log(s_hi) - math.log(s_lo)))
        eta = torch.rand(n - n1, generator=g, dtype=DTYPE) * tc.eta_max
        xi2 = torch.clamp(eta * s2, max=1.0)
        xi, tau = torch.cat([xi1, xi2]), torch.cat([s1, s2]) ** 2
        kn = torch.randint(0, len(self.c_mix), (tc.n_neumann,), generator=g)
        un = torch.rand(tc.n_neumann, generator=g, dtype=DTYPE)
        tau_n = (s_lo + un * (s_hi - s_lo)) ** 2
        return xi, tau, self.c_mix[k], tau_n, self.c_mix[kn]

    def losses(self, colloc):
        m, tc = self.model, self.tcfg
        s_pred = m.S_avg(self.tau, self.c)
        l_data = torch.mean((s_pred - self.s_obs) ** 2) / tc.sigma_data ** 2
        xi, tau, c, tau_n, c_n = colloc
        if tc.use_pde:
            # tau * r stays O(1) when tau -> 0 (it is the PDE written in the similarity variables)
            r = m.pde_residual(xi, tau, c) * tau ** tc.res_weight_exp
            if tc.causal_eps > 0:
                # residual per log-time bin; later bins only count once earlier ones are resolved
                ls = torch.log(torch.sqrt(tau))
                lo, hi = math.log(math.sqrt(tc.tau_min)), math.log(math.sqrt(self.t_max))
                b = torch.clamp(((ls - lo) / (hi - lo) * tc.causal_bins).long(), 0, tc.causal_bins - 1)
                Lb = torch.zeros(tc.causal_bins, dtype=DTYPE).index_add(0, b, r ** 2)
                cnt = torch.zeros(tc.causal_bins, dtype=DTYPE).index_add(0, b, torch.ones_like(r)).clamp(min=1)
                Lb = Lb / cnt
                w = torch.exp(-tc.causal_eps * torch.cumsum(torch.cat([Lb.new_zeros(1), Lb[:-1]]), 0)).detach()
                self.causal_min_w = float(w.min())
                l_pde = torch.mean(w * Lb)
            else:
                l_pde = torch.mean(r ** 2)
        else:
            l_pde = torch.zeros((), dtype=DTYPE)
        l_neu = torch.mean(m.neumann_residual(tau_n, c_n) ** 2)
        total = l_data + tc.lambda_pde * l_pde + tc.lambda_neu * l_neu
        return total, {"data": l_data.item(), "pde": l_pde.item(), "neu": l_neu.item(),
                       "rmse": math.sqrt(l_data.item()) * tc.sigma_data}

    def fit(self, log_every=250, verbose=True):
        m, tc = self.model, self.tcfg
        opt = torch.optim.Adam([
            {"params": m.field.parameters()},
            {"params": m.head.parameters(), "weight_decay": tc.head_weight_decay}], lr=tc.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=tc.lr_decay)
        t0 = time.time()
        for it in range(tc.adam_iters):
            colloc = self.sample_collocation()
            opt.zero_grad()
            loss, parts = self.losses(colloc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            if it % log_every == 0 or it == tc.adam_iters - 1:
                parts.update(it=it, total=loss.item(), stage="adam", time_s=time.time() - t0)
                self.history.append(parts)
                if verbose:
                    print(f"[adam {it:5d}] total={loss.item():.3e} data={parts['data']:.3e} "
                          f"pde={parts['pde']:.3e} neu={parts['neu']:.3e} rmse={parts['rmse']:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        if tc.lbfgs_iters > 0:
            colloc = self.sample_collocation()          # fixed set for L-BFGS
            opt2 = torch.optim.LBFGS(m.parameters(), lr=1.0, max_iter=tc.lbfgs_iters,
                                     history_size=50, line_search_fn="strong_wolfe")

            def closure():
                opt2.zero_grad()
                loss, _ = self.losses(colloc)
                loss.backward()
                return loss
            opt2.step(closure)
            loss, parts = self.losses(colloc)
            parts.update(it=tc.adam_iters + tc.lbfgs_iters, total=loss.item(), stage="lbfgs",
                         time_s=time.time() - t0)
            self.history.append(parts)
            if verbose:
                print(f"[lbfgs] total={loss.item():.3e} data={parts['data']:.3e} pde={parts['pde']:.3e} "
                      f"neu={parts['neu']:.3e} rmse={parts['rmse']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        self.train_time = time.time() - t0
        return self

    # ---- inference -----------------------------------------------------------------
    @torch.no_grad()
    def predict_avg(self, df):
        tau = torch.tensor(df["Time_d"].to_numpy() / self.pcfg.t_ref, dtype=DTYPE)
        c = torch.tensor(composition_features(df), dtype=DTYPE)
        return self.model.S_avg(tau, c, n_quad=64).numpy()

    @torch.no_grad()
    def predict_theta(self, df):
        mixes = df.drop_duplicates("Mix")
        c = torch.tensor(composition_features(mixes), dtype=DTYPE)
        th = self.model.head(c).numpy()
        out = mixes[["Mix"]].reset_index(drop=True).copy()
        for i, k in enumerate(self.model.head.names):
            out[k] = th[:, i]
        return out

    @torch.no_grad()
    def predict_profile(self, mix_row, x_frac, t_days):
        c = torch.tensor(composition_features(mix_row.iloc[[0]]), dtype=DTYPE)
        xi = torch.tensor(x_frac, dtype=DTYPE)
        out = []
        for t in t_days:
            tau = torch.full_like(xi, t / self.pcfg.t_ref)
            out.append(self.model.S(xi, tau, c.expand(len(xi), 3)).numpy())
        return np.array(out)

    @torch.no_grad()
    def pde_residual_stats(self, n=4096):
        xi, tau, c, *_ = self.sample_collocation()
        with torch.enable_grad():
            r = self.model.pde_residual(xi[:n], tau[:n], c[:n])
        return float(torch.sqrt(torch.mean((r * tau[:n] ** self.tcfg.res_weight_exp) ** 2)))
