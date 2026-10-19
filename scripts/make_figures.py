"""Generate every figure of the paper from the files in results/ (one style, one source).

Main results use the disc model (results/disc); the controlled comparison with the
PINN uses the 1D slab (results/slab).

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
from pinn_moisture.fd_solver import FDSolver, make_solver  # noqa: E402
from pinn_moisture.physics import (D_cap, D_vap, config_for, diffusivity, switch,  # noqa: E402
                                   theta_to_physical)
from pinn_moisture.pinn import TrainConfig, Trainer  # noqa: E402

DISC, SLAB, FIG = ROOT / "results/disc", ROOT / "results/slab", ROOT / "figures"
FIG.mkdir(exist_ok=True)
TEST = ["C-0.4", "C-0.5", "F10S10-0.3", "F20-0.3", "F20S20-0.4", "S40-0.4"]
C = {"pen": "#2a78d6", "pinn": "#6250d6", "mlp": "#eb6834", "fd": "#1baf7a", "data": "#2b2b2b",
     "muted": "#8a8984", "grid": "#e4e3df", "B": "#2a78d6", "A": "#eb6834"}
MARK = {"C": "o", "F20": "s", "F40": "D", "S20": "^", "S40": "v", "F10S10": "P", "F20S20": "X"}
LABEL = {"pen": "PENN", "fdreg": "FD + regression", "mlp": "Data-driven net", "pinn": "PINN"}
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


def runs(res, method):
    out = {}
    for d in sorted((res / "runs").glob(f"{method}_seed*")):
        if (d / "pred.csv").exists() and "noise" not in d.name:
            out[int(d.name.split("seed")[1])] = d
    return out


def load_pinn(run_dir, data):
    meta = json.loads((run_dir / "meta.json").read_text())
    tr = Trainer(data[~data.Mix.isin(TEST)], config_for("slab"), TrainConfig(**meta["train"]))
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


def fig_workflow():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=col + "1f", ec=col, lw=1.0))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontweight="bold", fontsize=7.4,
                linespacing=1.1)
        ax.text(x + w / 2, y + h / 2 - 3.0, body, ha="center", va="center", fontsize=7.2, linespacing=1.35)

    def arrow(x0, y0, x1, y1, col=C["data"], ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0,
                                     color=col, ls=ls, connectionstyle=f"arc3,rad={rad}"))

    box(1, 20, 15, 16, "Mix\ncomposition", "w/b\nfly-ash fraction\nslag fraction", C["muted"])
    box(21, 20, 17, 16, "Parameter\nnetwork", "$\\theta = g_w(\\mathbf{c})$\nMLP 3-16-16-2\n"
        "$\\log D_v,\\ \\log K_\\mathrm{int}$", C["pen"])
    box(43, 20, 17, 16, "Diffusivity\nmodel", "$D(S;\\theta)$\nvapour + capillary\n(Chen et al.)", C["fd"])
    box(65, 20, 17, 16, "Transport\nsolver", "$\\partial_t S = \\nabla\\cdot(D\\,\\nabla S)$\ndisc, all faces\n"
        "drying", C["fd"])
    box(85, 20, 14, 16, "Volume\naverage", "$S_\\mathrm{avg}(t)$", C["muted"])
    box(68, 1, 31, 12, "Data misfit\n(training mixes)",
        "$\\mathcal{L}=\\langle (S_\\mathrm{avg}-S_\\mathrm{avg,exp})^2\\rangle$", C["mlp"])
    for x0, x1 in ((16.8, 20.2), (38.8, 42.2), (60.8, 64.2), (82.8, 84.2)):
        arrow(x0, 28, x1, 28)
    arrow(92, 19.2, 92, 13.6)
    arrow(67.2, 7, 29.5, 19.2, col=C["mlp"], ls="--", rad=-0.25)
    ax.text(22, 3.0, "gradients by automatic\ndifferentiation through the solver", color=C["mlp"], fontsize=7.2,
            ha="center")
    save(fig, "fig_workflow")


def fig_geometry(data):
    """Disc model vs 1D slab: average saturation and a saturation map."""
    cfg_d = config_for("disc")
    cfg_s = config_for("disc")
    cfg_s.geometry = "slab"                       # same constants, equivalent slab L = V/A
    cl_d = pd.read_csv(DISC / "classical_fits.csv").set_index("Mix")
    disc, slab = make_solver(cfg_d), FDSolver(cfg_s, n_cells=80)
    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.42)
    for k, mix in enumerate(["C-0.4", "F40-0.3"]):
        ax = fig.add_subplot(gs[0, k])
        g = data[data.Mix == mix]
        t = np.geomspace(0.05, g.Time_d.max(), 120)
        th_d = {q: float(cl_d.loc[mix, q]) for q in cfg_d.identified}
        ax.plot(g.Time_d, g.S_frac, "o", ms=3, mfc="white", mec=C["data"], mew=0.8, label="Experiment")
        ax.plot(t, disc.solve(th_d, t)[0], "-", color=C["pen"], label="Disc model (fitted)")
        ax.plot(t, slab.solve(th_d, t)[0], "--", color=C["mlp"], lw=1.2, label="1D slab, same parameters")
        ax.set_xscale("log")
        ax.set_xlim(0.1, 100)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Drying time (d)")
        ax.set_title(mix)
        panel(ax, "ab"[k])
        if k == 0:
            ax.set_ylabel(r"$S_\mathrm{avg}$")
            handles, labels = ax.get_legend_handles_labels()
    ax = fig.add_subplot(gs[0, 2])
    mix = "C-0.4"
    th_d = {q: float(cl_d.loc[mix, q]) for q in cfg_d.identified}
    from pinn_moisture.axisym import AxisymBDF
    sv = AxisymBDF(cfg_d, nr=60, nz=30, ratio=1.06)
    _, S = sv.solve(th_d, np.array([7.0]), return_fields=True)
    r, z = sv.m.rc * 1e3, sv.m.zc * 1e3
    im = ax.pcolormesh(r, z, S[:, :, 0].T, cmap="Blues", vmin=0, vmax=1, shading="gouraud")
    ax.contour(r, z, S[:, :, 0].T, levels=[0.2, 0.4, 0.6], colors="white", linewidths=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("r (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(f"{mix}, t = 7 d")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("S")
    panel(ax, "c")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.36, -0.02))
    save(fig, "fig_geometry")


def fig_test_curves(data):
    preds = {m: {s_: pd.read_csv(d / "pred.csv") for s_, d in runs(DISC, m).items()} for m in ("pen", "mlp")}
    fdreg = pd.read_csv(DISC / "fdreg_pred.csv")
    pm = pd.read_csv(DISC / "table_test_mixes.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    for ax, mix, letter in zip(axes.flat, TEST, "abcdef"):
        g = data[data.Mix == mix]
        ax.plot(g.Time_d, g.S_frac, "o", ms=3.2, mfc="white", mec=C["data"], mew=0.8, label="Experiment", zorder=5)
        for m, ls in (("pen", "-"), ("mlp", "--")):
            P = np.array([p[p.Mix == mix].S_pred.values for p in preds[m].values()])
            ax.fill_between(g.Time_d, P.min(0), P.max(0), color=C[m], alpha=0.15, lw=0)
            ax.plot(g.Time_d, np.median(P, 0), ls, color=C[m], lw=1.3, label=LABEL[m])
        f = fdreg[fdreg.Mix == mix]
        ax.plot(f.Time_d, f.S_pred, ":", color=C["fd"], lw=1.6, label=LABEL["fdreg"])
        r2 = pm[(pm.method == "pen") & (pm.Mix == mix)].r2_mean.values[0]
        ax.text(0.97, 0.95, f"{mix}\nPENN $R^2$ = {r2:.3f}", transform=ax.transAxes, ha="right", va="top")
        ax.set_xlim(0, min(60, g.Time_d.max() * 1.05))
        ax.set_ylim(0, 0.8)
        panel(ax, letter)
    for ax in axes[1]:
        ax.set_xlabel("Drying time (d)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$S_\mathrm{avg}$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig_test_curves")


def fig_parity():
    pen = {s_: pd.read_csv(d / "pred.csv") for s_, d in runs(DISC, "pen").items()}
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


def fig_cv():
    pm = pd.read_csv(DISC / "cv_per_mix.csv")
    mixes = sorted(pm.Mix.unique(), key=lambda m: (list(MARK).index(m.split("-")[0]), m))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    for ax, scheme, title, letter in ((axes[0], "lomo", "Leave one mix out", "a"),
                                      (axes[1], "lofo", "Leave one binder family out", "b")):
        x = np.arange(len(mixes))
        for j, m in enumerate(("pen", "fdreg", "mlp")):
            g = pm[(pm.scheme == scheme) & (pm.method == m)].set_index("Mix").reindex(mixes)
            y = g.r2.to_numpy()
            yc = np.clip(y, 0.80, None)
            ax.plot(x + (j - 1) * 0.22, yc, "o^s"[j], ms=4.2, color=C["fd" if m == "fdreg" else m],
                    mfc=C["fd" if m == "fdreg" else m] if m == "pen" else "white", mew=0.9, ls="", label=LABEL[m])
            for xi, yi in zip(x[y < 0.80], y[y < 0.80]):
                ax.annotate(f"{yi:.2f}", (xi + (j - 1) * 0.22, 0.80), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=6.5, color=C[m if m != "fdreg" else "fd"],
                            arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C[m if m != "fdreg" else "fd"]))
        ax.axhline(0.95, color=C["muted"], lw=0.7, ls="--")
        ax.set_ylim(0.79, 1.005)
        ax.set_ylabel("$R^2$ of held-out mix")
        ax.set_title(title, loc="left", x=0.02, y=0.02, fontsize=8.5)
        panel(ax, letter)
    axes[1].set_xticks(np.arange(len(mixes)))
    axes[1].set_xticklabels(mixes, rotation=60, ha="right")
    axes[0].legend(loc="lower right", ncol=3)
    save(fig, "fig_cv")


def fig_forward():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(wspace=0.4)
    m = json.loads((SLAB / "metrics.json").read_text())
    for ax, meth, title, letter in ((axes[0], "pinn", "PINN", "a"), (axes[1], "pen", "PENN", "b")):
        fv = pd.read_csv(SLAB / f"forward_verification_{meth}.csv")
        for test, col, mk, lab in ((False, C["muted"], "o", "Training mixes"), (True, C[meth], "s", "Unseen mixes")):
            g = fv[fv.test == test]
            ax.plot(g.S_model, g.S_fd_model_theta, mk, ms=3, mfc="none", mew=0.7, color=col, label=lab)
        ax.plot([0, 1], [0, 1], "-", color=C["data"], lw=0.8)
        q = m[f"forward_{meth}_fd_vs_model_all"]
        ax.text(0.97, 0.05, f"$R^2$ = {q['r2']:.3f}\nRMSE = {q['rmse']:.3f}", transform=ax.transAxes, ha="right")
        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, 0.85)
        ax.set_aspect("equal")
        ax.set_title(title + " (1D slab)")
        ax.set_xlabel(r"Network $S_\mathrm{avg}$")
        ax.set_ylabel("BDF solver with the network parameters")
        ax.legend(loc="upper left")
        panel(ax, letter)
    save(fig, "fig_forward")


def fig_parameters():
    th = pd.read_csv(DISC / "theta_all.csv")
    th["wb"] = th.Mix.str.split("-").str[1].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(wspace=0.38, bottom=0.30)
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
    h += [plt.Line2D([], [], marker="o", ls="", color=C["pen"], label="PENN (predicted from composition)"),
          plt.Line2D([], [], marker="o", ls="", mfc="none", mec=C["fd"], label="Classical fit (per mix)")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), handletextpad=0.2,
               columnspacing=0.8)
    save(fig, "fig_parameters")


def fig_diffusivity():
    th = pd.read_csv(DISC / "theta_all.csv").set_index("Mix")
    cfg = config_for("disc")
    S = np.linspace(0.005, 1.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    fig.subplots_adjust(wspace=0.35)
    ax = axes[0]
    mix = "C-0.4"
    p = theta_to_physical({k: th.loc[mix, f"{k}_pen"] for k in cfg.identified}, cfg)
    f = switch(S, cfg)
    ax.plot(S, (1 - f) * D_vap(S, p["Dv"], p["alpha0"], p["n"], cfg), "--", color=C["mlp"],
            label=r"$(1-f)\,D_\mathrm{vap}$")
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


def fig_fields():
    """Saturation maps of an unseen mix predicted by the PENN (disc model)."""
    th = pd.read_csv(DISC / "theta_all.csv").set_index("Mix")
    cfg = config_for("disc")
    from pinn_moisture.axisym import AxisymBDF
    sv = AxisymBDF(cfg, nr=60, nz=30, ratio=1.06)
    times = [1.0, 7.0, 28.0]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=True, sharey=True)
    for row, mix in enumerate(["C-0.4", "F20-0.3"]):
        th_m = {k: float(th.loc[mix, f"{k}_pen"]) for k in cfg.identified}
        _, S = sv.solve(th_m, np.array(times), return_fields=True)
        r, z = sv.m.rc * 1e3, sv.m.zc * 1e3
        for col, t in enumerate(times):
            ax = axes[row, col]
            im = ax.pcolormesh(r, z, S[:, :, col].T, cmap="Blues", vmin=0, vmax=1, shading="gouraud")
            cs = ax.contour(r, z, S[:, :, col].T, levels=[0.25, 0.5, 0.75], colors="white", linewidths=0.6)
            ax.clabel(cs, fontsize=5.5, fmt="%.2f", inline_spacing=2)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_title(f"{mix}, t = {t:g} d", fontsize=8)
            if row == 1:
                ax.set_xlabel("r (mm)")
            if col == 0:
                ax.set_ylabel("z (mm)")
            panel(ax, "abcdef"[row * 3 + col])
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Saturation S")
    save(fig, "fig_fields")


def fig_profiles_slab(data):
    """Appendix: PINN vs PENN profiles on the 1D slab."""
    th = pd.read_csv(SLAB / "theta_all.csv").set_index("Mix")
    pinn_dirs = runs(SLAB, "pinn")
    tr = load_pinn(pinn_dirs[min(pinn_dirs)], data)
    cfg = config_for("slab")
    solver = FDSolver(cfg, n_cells=120)
    times = [0.5, 2, 7, 28]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, mix, letter in zip(axes, ["C-0.4", "F20-0.3", "S40-0.4"], "abc"):
        xf = solver.mesh.centers / cfg.L
        _, S_fd = solver.solve({k: float(th.loc[mix, f"{k}_pen"]) for k in cfg.identified}, times)
        S_p = tr.predict_profile(data[data.Mix == mix], xf, times)
        for i, (t, col) in enumerate(zip(times, cols)):
            ax.plot(xf * cfg.L * 1e3, S_fd[:, i], "-", color=col, label=f"{t:g} d")
            ax.plot(xf * cfg.L * 1e3, S_p[i], ":", color=col, lw=1.1)
        ax.set_title(mix + " (unseen)")
        ax.set_xlabel("Depth (mm)")
        ax.set_xlim(0, cfg.L * 1e3)
        ax.set_ylim(0, 1.02)
        panel(ax, letter)
    axes[0].set_ylabel("Local saturation $S(x,t)$")
    h, _ = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=C["data"], ls="-", label="PENN"),
          plt.Line2D([], [], color=C["data"], ls=":", lw=1.1, label="PINN")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_profiles_slab")


def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.35)
    ax = axes[0]
    for d in runs(DISC, "pen").values():
        h = pd.read_csv(d / "history.csv")
        ax.plot(h.it, np.sqrt(h.mse), "-", color=C["pen"], lw=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel(r"RMSE on $S_\mathrm{avg}$ (training mixes)")
    ax.set_title("PENN, disc model (5 seeds)")
    panel(ax, "a")
    ax = axes[1]
    styles = {"data": (C["pen"], r"$\mathcal{L}_\mathrm{data}$"), "pde": (C["mlp"], r"$\mathcal{L}_\mathrm{PDE}$"),
              "neu": (C["fd"], r"$\mathcal{L}_\mathrm{sym}$")}
    r = runs(SLAB, "pinn")
    for k, (col, lab) in styles.items():
        hs = [pd.read_csv(d / "history.csv") for d in r.values()]
        hs = [h[h.stage == "adam"] for h in hs]
        Y = np.array([h[k].values for h in hs])
        ax.plot(hs[0].it.values, np.median(Y, 0), "-", color=col, label=lab)
        ax.fill_between(hs[0].it.values, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("Loss term")
    ax.set_title("PINN, 1D slab (3 seeds)")
    ax.legend(loc="upper right", ncol=3)
    panel(ax, "b")
    save(fig, "fig_convergence")


if __name__ == "__main__":
    data = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
    raw = pd.read_csv(ROOT / "data/raw/tidy_dataset_correctedV1.csv")
    only = set(sys.argv[1:])
    todo = {
        "fig_workflow": lambda: fig_workflow(),
        "fig_data_cleaning": lambda: fig_data_cleaning(raw, data),
        "fig_geometry": lambda: fig_geometry(data[data.reliable]),
        "fig_test_curves": lambda: fig_test_curves(data[data.reliable]),
        "fig_parity": lambda: fig_parity(),
        "fig_cv": lambda: fig_cv(),
        "fig_forward": lambda: fig_forward(),
        "fig_parameters": lambda: fig_parameters(),
        "fig_diffusivity": lambda: fig_diffusivity(),
        "fig_fields": lambda: fig_fields(),
        "fig_profiles_slab": lambda: fig_profiles_slab(data[data.reliable].reset_index(drop=True)),
        "fig_convergence": lambda: fig_convergence(),
    }
    for name, fn in todo.items():
        if not only or name in only:
            fn()
