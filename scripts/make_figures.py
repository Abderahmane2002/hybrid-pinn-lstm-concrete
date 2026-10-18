"""Generate every figure of the paper from the files in results/ (one style, one source).

python scripts/make_figures.py  ->  figures/*.png (300 dpi) and figures/*.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.fd_solver import FDSolver  # noqa: E402
from pinn_moisture.physics import (PhysicsConfig, D_cap, D_vap, diffusivity, switch,  # noqa: E402
                                   theta_to_physical)
from pinn_moisture.pinn import CompositionPINN, TrainConfig, Trainer  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
TEST = ["C-0.4", "C-0.5", "F10S10-0.3", "F20-0.3", "F20S20-0.4", "S40-0.4"]
C = {"pen": "#2a78d6", "pinn": "#6250d6", "mlp": "#eb6834", "fd": "#1baf7a", "data": "#2b2b2b", "muted": "#8a8984",
     "grid": "#e4e3df", "B": "#2a78d6", "A": "#eb6834"}
MARK = {"C": "o", "F20": "s", "F40": "D", "S20": "^", "S40": "v", "F10S10": "P", "F20S20": "X"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter):
    ax.text(-0.02, 1.03, f"({letter})", transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def family(mix):
    return mix.split("-")[0]


def runs(method):
    out = {}
    for d in sorted((RES / "runs").glob(f"{method}_seed*")):
        if (d / "pred.csv").exists():
            out[int(d.name.split("seed")[1])] = d
    return out


def load_trainer(run_dir, data):
    meta = json.loads((run_dir / "meta.json").read_text())
    tcfg = TrainConfig(**meta["train"])
    tr = Trainer(data[~data.Mix.isin(TEST)], PhysicsConfig(), tcfg)  # data: reliable mixes only
    tr.model.load_state_dict(torch.load(run_dir / "model.pt"))
    return tr


# ---------------------------------------------------------------------------------------
def fig_data_cleaning(raw, clean):
    raw = raw.copy()
    raw["block"] = (raw.Mix != raw.Mix.shift()).cumsum()
    t_end = raw.groupby("block").Time_d.transform("max")
    raw["series"] = np.where(t_end <= 20, "B", "A")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, mix, letter in zip(axes, ["C-0.4", "F40-0.4", "S40-0.6"], "abc"):
        r = raw[raw.Mix == mix]
        for s, lab in (("A", "raw long series"), ("B", "raw short series")):
            g = r[r.series == s].sort_values("Time_d")
            ax.plot(g.Time_d, g.S_frac, "o", ms=3.2, mfc="none", mew=0.8, color=C[s], label=lab)
        cl = clean[clean.Mix == mix]
        ax.plot(cl.Time_d, cl.S_frac, "-", color=C["data"], lw=1.0, label="cleaned curve")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 1)
        ax.set_title(mix if cl.reliable.all() else f"{mix} (excluded)")
        ax.set_xlabel("Drying time (d)")
        panel(ax, letter)
    axes[0].set_ylabel(r"Average saturation $S_\mathrm{avg}$")
    axes[0].legend(loc="upper right")
    save(fig, "fig_data_cleaning")


def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    hs = [pd.read_csv(d / "history.csv") for d in runs("pen").values()]
    for h in hs:
        ax.plot(h.it, np.sqrt(h.mse), "-", color=C["pen"], lw=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel(r"RMSE on $S_\mathrm{avg}$ (training mixes)")
    ax.set_title("Physics-embedded network (5 seeds)")
    panel(ax, "a")
    ax = axes[1]
    styles = {"data": (C["pen"], r"$\mathcal{L}_\mathrm{data}$"), "pde": (C["mlp"], r"$\mathcal{L}_\mathrm{PDE}$"),
              "neu": (C["fd"], r"$\mathcal{L}_\mathrm{sym}$")}
    r = runs("pinn")
    for k, (col, lab) in styles.items():
        hs = [pd.read_csv(d / "history.csv") for d in r.values()]
        hs = [h[h.stage == "adam"] for h in hs]
        Y = np.array([h[k].values for h in hs])
        ax.plot(hs[0].it.values, np.median(Y, 0), "-", color=col, label=lab)
        ax.fill_between(hs[0].it.values, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("Loss term")
    ax.set_title("PINN (3 seeds)")
    ax.legend(loc="upper right", ncol=3)
    panel(ax, "b")
    save(fig, "fig_convergence")


def fig_test_curves(data):
    preds = {m: {s_: pd.read_csv(d / "pred.csv") for s_, d in runs(m).items()} for m in ("pen", "pinn", "mlp")}
    fdreg = pd.read_csv(RES / "fdreg_pred.csv")
    pm = pd.read_csv(RES / "table_test_mixes.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    for ax, mix, letter in zip(axes.flat, TEST, "abcdef"):
        g = data[data.Mix == mix]
        ax.plot(g.Time_d, g.S_frac, "o", ms=3.2, mfc="white", mec=C["data"], mew=0.8, label="Experiment", zorder=5)
        for m, lab, ls in (("pen", "Physics-embedded net", "-"), ("pinn", "PINN", "-."), ("mlp", "Data-driven net", "--")):
            P = np.array([p[p.Mix == mix].S_pred.values for p in preds[m].values()])
            ax.fill_between(g.Time_d, P.min(0), P.max(0), color=C[m], alpha=0.15, lw=0)
            ax.plot(g.Time_d, np.median(P, 0), ls, color=C[m], lw=1.3, label=lab)
        f = fdreg[fdreg.Mix == mix]
        ax.plot(f.Time_d, f.S_pred, ":", color=C["fd"], lw=1.6, label="FD + regression")
        r2 = pm[(pm.method == "pen") & (pm.Mix == mix)].r2_mean.values[0]
        ax.text(0.97, 0.95, f"{mix}\n$R^2$ = {r2:.3f}", transform=ax.transAxes, ha="right", va="top")
        ax.set_xlim(0, min(60, g.Time_d.max() * 1.05))
        ax.set_ylim(0, 0.8)
        panel(ax, letter)
    for ax in axes[1]:
        ax.set_xlabel("Drying time (d)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$S_\mathrm{avg}$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig_test_curves")


def fig_parity():
    pen = {s_: pd.read_csv(d / "pred.csv") for s_, d in runs("pen").items()}
    p = pen[min(pen)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for split, col, mk, lab in (("train", C["muted"], "o", "Training mixes"), ("test", C["pen"], "s", "Unseen mixes")):
        g = p[p.split == split]
        ax.plot(g.S_obs, g.S_pred, mk, ms=3, mfc="none", mew=0.7, color=col, label=lab)
    ax.plot([0, 1], [0, 1], "-", color=C["data"], lw=0.8)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Measured $S_\mathrm{avg}$")
    ax.set_ylabel(r"Predicted $S_\mathrm{avg}$")
    ax.legend(loc="upper left")
    panel(ax, "a")
    ax = axes[1]
    for split, col, mk in (("train", C["muted"], "o"), ("test", C["pen"], "s")):
        g = p[p.split == split]
        ax.plot(g.Time_d, g.S_pred - g.S_obs, mk, ms=3, mfc="none", mew=0.7, color=col)
    ax.axhline(0, color=C["data"], lw=0.8)
    ax.set_xlabel("Drying time (d)")
    ax.set_ylabel(r"Residual $S_\mathrm{pred}-S_\mathrm{obs}$")
    ax.set_xlim(0, 100)
    panel(ax, "b")
    save(fig, "fig_parity")


def fig_parameters():
    th = pd.read_csv(RES / "theta_all.csv")
    th["wb"] = th.Mix.str.split("-").str[1].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    labels = {"log10_Dv": r"$\log_{10} D_v$ (m$^2$/s)", "log10_K": r"$\log_{10} K_\mathrm{int}$ (m$^2$)"}
    for ax, k, letter in zip(axes, ["log10_Dv", "log10_K"], "ab"):
        for fam, mk in MARK.items():
            g = th[th.Mix.str.split("-").str[0] == fam]
            if not len(g):
                continue
            jitter = (list(MARK).index(fam) - 3) * 0.007
            ax.errorbar(g.wb + jitter, g[f"{k}_pen"], yerr=g[f"{k}_pen_sd"], fmt=mk, ms=4.5, color=C["pen"],
                        mfc=C["pen"], mec="white", mew=0.5, elinewidth=0.7, capsize=0)
            ax.errorbar(g.wb + jitter, g[f"{k}_fd"], yerr=g[f"se_{k}"], fmt=mk, ms=4.5, mfc="none", mec=C["fd"],
                        mew=0.9, ecolor=C["fd"], elinewidth=0.7, capsize=0)
        ax.set_xlabel("w/b")
        ax.set_ylabel(labels[k])
        ax.set_xticks([0.3, 0.4, 0.5])
        panel(ax, letter)
    h = [plt.Line2D([], [], marker=m, ls="", color=C["muted"], label=f) for f, m in MARK.items()]
    h += [plt.Line2D([], [], marker="o", ls="", color=C["pen"], label="Physics-embedded net (from composition)"),
          plt.Line2D([], [], marker="o", ls="", mfc="none", mec=C["fd"], label="Classical FD fit (per mix)")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12), handletextpad=0.2,
               columnspacing=0.8)
    save(fig, "fig_parameters")


def fig_diffusivity():
    th = pd.read_csv(RES / "theta_all.csv").set_index("Mix")
    cfg = PhysicsConfig()
    S = np.linspace(0.005, 1.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    mix = "C-0.4"
    p = theta_to_physical({k: th.loc[mix, f"{k}_pen"] for k in cfg.identified}, cfg)
    f = switch(S, cfg)
    ax.plot(S, (1 - f) * D_vap(S, p["Dv"], p["alpha0"], p["n"], cfg), "--", color=C["mlp"], label=r"$(1-f)\,D_\mathrm{vap}$")
    ax.plot(S, f * D_cap(S, p["K"], cfg), "--", color=C["fd"], label=r"$f\,D_\mathrm{cap}$")
    ax.plot(S, diffusivity(S, p["Dv"], p["alpha0"], p["n"], p["K"], cfg), "-", color=C["data"], label=r"$D(S)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e-8)
    ax.set_xlabel("Saturation $S$")
    ax.set_ylabel(r"Diffusivity (m$^2$/s)")
    ax.set_title(f"{mix} (unseen mix, predicted)")
    ax.legend(loc="upper left")
    panel(ax, "a")
    ax = axes[1]
    for mix, col in (("C-0.3", C["pen"]), ("C-0.5", C["mlp"]), ("F40-0.3", C["fd"])):
        p = theta_to_physical({k: th.loc[mix, f"{k}_pen"] for k in cfg.identified}, cfg)
        ax.plot(S, diffusivity(S, p["Dv"], p["alpha0"], p["n"], p["K"], cfg), "-", color=col, label=mix)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e-8)
    ax.set_xlabel("Saturation $S$")
    ax.set_ylabel(r"$D(S)$ (m$^2$/s)")
    ax.legend(loc="upper left")
    panel(ax, "b")
    save(fig, "fig_diffusivity")


def fig_profiles(data):
    th = pd.read_csv(RES / "theta_all.csv").set_index("Mix")
    pinn_dirs = runs("pinn")
    tr = load_trainer(pinn_dirs[min(pinn_dirs)], data) if pinn_dirs else None
    cfg = PhysicsConfig()
    solver = FDSolver(cfg, n_cells=120)
    times = [0.5, 2, 7, 28]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    mixes = ["C-0.4", "F20-0.3", "S40-0.4"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, mix, letter in zip(axes, mixes, "abc"):
        xf = solver.mesh.centers / cfg.L
        _, S_fd = solver.solve({k: float(th.loc[mix, f"{k}_pen"]) for k in cfg.identified}, times)
        S_p = tr.predict_profile(data[data.Mix == mix], xf, times) if tr else None
        for i, (t, col) in enumerate(zip(times, cols)):
            ax.plot(xf * cfg.L * 1e3, S_fd[:, i], "-", color=col, label=f"{t:g} d")
            if S_p is not None:
                ax.plot(xf * cfg.L * 1e3, S_p[i], ":", color=col, lw=1.1)
        ax.set_title(mix + (" (unseen)" if mix in TEST else ""))
        ax.set_xlabel("Depth from drying face (mm)")
        ax.set_xlim(0, cfg.L * 1e3)
        ax.set_ylim(0, 1.02)
        panel(ax, letter)
    axes[0].set_ylabel("Local saturation $S(x,t)$")
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=C["data"], ls="-", label="Physics-embedded net"),
          plt.Line2D([], [], color=C["data"], ls=":", lw=1.1, label="PINN")]
    axes[2].legend(handles=h, loc="lower right", ncol=2)
    save(fig, "fig_profiles")


def fig_forward():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, meth, title, letter in ((axes[0], "pinn", "PINN", "a"), (axes[1], "pen", "Physics-embedded net", "b")):
        f = RES / f"forward_verification_{meth}.csv"
        if not f.exists():
            continue
        fv = pd.read_csv(f)
        for test, col, mk, lab in ((False, C["muted"], "o", "Training mixes"), (True, C[meth], "s", "Unseen mixes")):
            g = fv[fv.test == test]
            ax.plot(g.S_model, g.S_fd_model_theta, mk, ms=3, mfc="none", mew=0.7, color=col, label=lab)
        ax.plot([0, 1], [0, 1], "-", color=C["data"], lw=0.8)
        m = json.loads((RES / "metrics.json").read_text())[f"forward_{meth}_fd_vs_model_all"]
        ax.text(0.97, 0.05, f"$R^2$ = {m['r2']:.3f}\nRMSE = {m['rmse']:.3f}", transform=ax.transAxes, ha="right")
        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, 0.85)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(r"Network $S_\mathrm{avg}$")
        ax.set_ylabel("FD solver with the network parameters")
        ax.legend(loc="upper left")
        panel(ax, letter)
    save(fig, "fig_forward")


if __name__ == "__main__":
    data = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
    raw = pd.read_csv(ROOT / "data/raw/tidy_dataset_correctedV1.csv")
    fig_data_cleaning(raw, data)
    data = data[data.reliable].reset_index(drop=True)
    if runs("pen"):
        fig_convergence()
        fig_test_curves(data)
        fig_parity()
        fig_parameters()
        fig_diffusivity()
        fig_profiles(data)
        fig_forward()
