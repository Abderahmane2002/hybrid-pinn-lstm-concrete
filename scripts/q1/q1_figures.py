"""Figures of the revised manuscript (figures/q1/*.png|pdf), drawn from results/q1 and results/disc|slab.

usage: python scripts/q1/q1_figures.py [fig names...]
Figures re-used unchanged from scripts/make_figures.py are copied under their new numbers.
Colours follow one fixed entity order (validated for colour-vision deficiency); every
series also carries its own marker so identity never depends on colour alone.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_figures as mf  # noqa: E402  (style, palette, helpers)

ROOT = HERE.parents[1]
Q = ROOT / "results/q1"
OUT = ROOT / "figures/q1"
OUT.mkdir(parents=True, exist_ok=True)
C, panel = mf.C, mf.panel
TEST = mf.TEST
# fixed entity -> colour / marker (validated order: blue, aqua, orange, violet, yellow, magenta)
COL = {"pen": "#2a78d6", "ols": "#1baf7a", "mlp": "#eb6834", "pen_r": "#6250d6", "gp": "#eda100", "quad": "#e87ba4",
       "ridge": "#0d8a5f", "pen_lin": "#7fb2ea", "pen_s": "#2a78d6"}
MK = {"pen": "o", "ols": "s", "mlp": "^", "pen_r": "D", "gp": "v", "quad": "P", "ridge": "X", "pen_lin": "h", "pen_s": "*"}
NAME = {"pen": "PENN", "pen_r": "PENN, revised", "pen_lin": "PENN, linear map", "pen_s": "PENN, 1 × 8",
        "ols": "Two-stage, linear", "ridge": "Two-stage, ridge", "quad": "Two-stage, quadratic",
        "gp": "Two-stage, Gaussian process", "mlp": "Data-driven network"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def copy(src, dst):
    for ext in ("png", "pdf"):
        shutil.copy(ROOT / f"figures/{src}.{ext}", OUT / f"{dst}.{ext}")
    print("copied", src, "->", dst)


# ---------------------------------------------------------------------------------------------
def fig01_data():
    raw = pd.read_csv(ROOT / "data/raw/tidy_dataset_correctedV1.csv")
    clean = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
    raw["block"] = (raw.Mix != raw.Mix.shift()).cumsum()
    raw["series"] = np.where(raw.groupby("block").Time_d.transform("max") <= 20, "B", "A")
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.55, wspace=0.3)
    for i, (mix, letter) in enumerate(zip(["C-0.4", "F40-0.4", "S40-0.6"], "abc")):
        ax = fig.add_subplot(gs[0, i])
        r = raw[raw.Mix == mix]
        for s, lab, mk in (("A", "raw long series", "o"), ("B", "raw short series", "s")):
            g = r[r.series == s].sort_values("Time_d")
            ax.plot(g.Time_d, g.S_frac, mk, ms=3.0, mfc="none", mew=0.8, color=C[s], label=lab)
        cl = clean[clean.Mix == mix]
        if cl.reliable.all():
            ax.plot(cl.Time_d, cl.S_frac, "-", color=C["data"], lw=1.0, label="rebuilt curve")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 1)
        ax.set_title(mix if cl.reliable.all() else f"{mix} (excluded)")
        ax.set_xlabel("Drying time (d)")
        if i == 0:
            ax.set_ylabel(r"$S_\mathrm{avg}$")
            ax.legend(loc="upper right")
        panel(ax, letter)
    ax = fig.add_subplot(gs[1, :])
    sr = pd.read_csv(Q / "series_relation.csv")
    sr = sr[sr.wb < 0.55].sort_values(["wb", "Mix"]).reset_index(drop=True)
    x = np.arange(len(sr))
    for col, lab, mk, c in (("shift_rms_S", "time shift", "o", COL["pen"]), ("scale_rms_S", "time scaling", "s", COL["mlp"]),
                            ("affine_rms_S", "shift + scaling", "x", C["data"])):
        ax.plot(x, sr[col], mk, ms=4.5 if mk != "x" else 4, mfc="none" if mk != "x" else None, mew=0.9, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(sr.Mix, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("RMS mismatch in $S$")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("Relation between the two digitised series of each mix (w/b ≤ 0.5)")
    panel(ax, "d")
    save(fig, "fig01_data")


def fig04_identifiability():
    P = pd.read_csv(Q / "profiles/profiles.csv")
    I = pd.read_csv(Q / "profiles/intervals.csv").set_index("Mix")
    C1 = pd.read_csv(Q / "synthetic/c1.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    fig.subplots_adjust(hspace=0.45, wspace=0.32)
    show = [("S20-0.3", "o"), ("C-0.4", "s"), ("F40-0.3", "^"), ("F20S20-0.3", "D")]
    cols = [COL["pen"], COL["ols"], COL["mlp"], COL["pen_r"]]
    for ax, par, letter, lab in ((axes[0, 0], "log10_K", "a", r"$\log_{10} K_\mathrm{int}$ (m$^2$)"),
                                 (axes[0, 1], "log10_Dv", "b", r"$\log_{10} D_v$ (m$^2$/s)")):
        for (mix, mk), c in zip(show, cols):
            g = P[(P.Mix == mix) & (P.param == par)]
            ax.plot(g.value, g.deviance_eff, "-", color=c, lw=1.2, label=f"{mix} ($N_{{eff}}$ = {I.loc[mix, 'N_eff']:.0f})")
            k = g.deviance_eff.idxmin()
            ax.plot(g.value[k], 0, mk, color=c, ms=4)
        ax.axhline(3.84, color=C["muted"], ls="--", lw=0.8)
        ax.text(0.02, 3.84, " 95 %", color=C["muted"], va="bottom", fontsize=7, transform=ax.get_yaxis_transform())
        ax.set_ylim(0, 12)
        if par == "log10_K":
            ax.set_xlim(-21, -18.5)
        else:
            ax.set_xlim(-9.5, -8.4)
        ax.set_xlabel(lab)
        ax.set_ylabel("Profile deviance (corrected)")
        panel(ax, letter)
    axes[0, 0].legend(loc="upper right", fontsize=6.5)
    ax = axes[1, 0]
    I = I.reset_index()
    I["wb"] = I.Mix.str.split("-").str[1].astype(float)
    I = I.sort_values(["wb", "log10_K"]).reset_index(drop=True)
    y = np.arange(len(I))
    ax.hlines(y, I.K_lo_eff, I.K_hi_eff, color=COL["ols"], lw=2.2, label="corrected 95 % interval")
    ax.hlines(y, I.K_lo, I.K_hi, color=C["data"], lw=0.9, label="uncorrected")
    ax.plot(I.log10_K, y, "|", color=C["data"], ms=7)
    ax.set_yticks(y)
    ax.set_yticklabels(I.Mix, fontsize=6.3)
    ax.set_xlabel(r"$\log_{10} K_\mathrm{int}$ (m$^2$)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.grid(axis="y", visible=False)
    panel(ax, "c")
    ax = axes[1, 1]
    conds = [("exact", "exact"), ("short_14d", r"$t \leq$ 14 d"), ("iid_0.01", "noise 0.01"),
             ("corr_0.01", "correlated 0.01"), ("t0_0.25", "time origin 0.25 d")]
    for j, (cnd, lab) in enumerate(conds):
        g = C1[C1.cond == cnd]
        ax.plot(np.full(len(g), j) - 0.12 + np.random.default_rng(j).uniform(-0.05, 0.05, len(g)), g.err_Dv.abs(), "o",
                ms=3.5, mfc="none", color=COL["pen"], label=r"$|\Delta \log_{10} D_v|$" if j == 0 else None)
        ax.plot(np.full(len(g), j) + 0.12 + np.random.default_rng(j + 9).uniform(-0.05, 0.05, len(g)), g.err_K.abs(), "s",
                ms=3.5, mfc="none", color=COL["mlp"], label=r"$|\Delta \log_{10} K_\mathrm{int}|$" if j == 0 else None)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[1] for c in conds], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Recovery error (decades)")
    ax.set_title("Synthetic curves, known parameters")
    ax.legend(loc="upper left", fontsize=7)
    panel(ax, "d")
    save(fig, "fig04_identifiability")


def fig06_comparison():
    pooled = pd.read_csv(Q / "baselines/pooled.csv")
    pm = pd.read_csv(Q / "baselines/permix.csv")
    paired = pd.read_csv(Q / "audit/paired.csv")
    meths = ["pen", "pen_r", "ols", "gp", "quad", "mlp"]
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.5, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    schemes = [("holdout", "Hold-out"), ("lomo", "Leave one\nmix out"), ("lofo", "Leave one\nfamily out")]
    for i, m in enumerate(meths):
        v = [pooled[(pooled.scheme == s) & (pooled.method == m)].rmse.iloc[0] for s, _ in schemes]
        ax.plot(np.arange(3) + (i - 2.5) * 0.11, v, MK[m], color=COL[m], ms=5, mfc=COL[m], mec="white", mew=0.4,
                label=NAME[m], ls="")
    ax.set_xticks(range(3))
    ax.set_xticklabels([s[1] for s in schemes])
    ax.set_ylabel("Pooled RMSE")
    ax.set_ylim(0, None)
    panel(ax, "a")
    ax.legend(loc="upper left", fontsize=6.5, ncol=1)
    ax = fig.add_subplot(gs[0, 1])
    lofo = pm[(pm.scheme == "lofo") & (pm.seed == 0)]
    for i, m in enumerate(meths):
        g = lofo[lofo.method == m]
        y = np.clip(g.r2.to_numpy(), 0.80, None)
        x = np.full(len(g), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(g))
        ax.plot(x, y, MK[m], color=COL[m], ms=3.6, mfc="none", mew=0.8)
        nlow = int((g.r2 < 0.80).sum())
        if nlow:
            ax.text(i, 0.795, f"{nlow} < 0.80\n(min {g.r2.min():.2f})", ha="center", va="top", fontsize=6.2, color=C["data"])
    ax.axhline(0.95, color=C["muted"], ls="--", lw=0.8)
    ax.set_ylim(0.74, 1.005)
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels([NAME[m].replace("Two-stage, ", "").replace(", revised", "\nrevised").replace("Data-driven network", "Data-\ndriven")
                        .replace("Gaussian process", "GP") for m in meths], fontsize=6.8)
    ax.set_ylabel(r"$R^2$ per mix (family withheld)")
    panel(ax, "b")
    ax = fig.add_subplot(gs[1, :])
    pairs = [("pen", "ols"), ("pen_r", "ols"), ("pen", "pen_r"), ("pen", "mlp"), ("ols", "mlp"), ("pen", "gp"),
             ("ols", "gp"), ("pen", "quad"), ("ols", "quad")]
    yl = []
    for j, (a, b) in enumerate(pairs):
        for k, (sch, off, mk, c) in enumerate((("lomo", -0.15, "o", C["data"]), ("lofo", 0.15, "s", COL["pen"]))):
            r = paired[(paired.scheme == sch) & (paired.a == a) & (paired.b == b)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(r.mean_diff, j + off, xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]], fmt=mk, color=c,
                        mfc="white" if r.wilcoxon_p > 0.05 else c, ms=4.5, elinewidth=1.0, capsize=0,
                        label=("leave one mix out" if sch == "lomo" else "leave one family out") if j == 0 else None)
        yl.append(f"{NAME[a]}  vs  {NAME[b]}")
    ax.axvline(0, color=C["muted"], lw=0.8)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(yl, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean per-mix RMSE difference (first − second), with 95 % bootstrap interval; filled: Wilcoxon p < 0.05")
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(axis="y", visible=False)
    panel(ax, "c")
    save(fig, "fig06_comparison")


def fig08_efficiency():
    S = pd.read_csv(Q / "data_efficiency/summary.csv")
    T = pd.read_csv(Q / "data_efficiency/truncation.csv")
    meths = ["pen", "pen_r", "ols", "gp", "mlp"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    fig.subplots_adjust(wspace=0.42)
    for m in meths:
        g = S[S.method == m].sort_values("k")
        if not len(g):
            continue
        axes[0].plot(g.k, g.rmse_med, "-", marker=MK[m], color=COL[m], ms=4, label=NAME[m])
        axes[0].fill_between(g.k, g.rmse_q25, g.rmse_q75, color=COL[m], alpha=0.10, lw=0)
        axes[1].plot(g.k, g.r2_worst_med, "-", marker=MK[m], color=COL[m], ms=4)
    axes[0].set_xlabel("Training mixes")
    axes[0].set_ylabel("Pooled RMSE (median, IQR)")
    axes[1].set_xlabel("Training mixes")
    axes[1].set_ylabel(r"Worst-mix $R^2$ (median of draws)")
    axes[1].set_ylim(0.5, 1.0)
    for ax in axes[:2]:
        ax.set_xticks([5, 8, 11, 14])
    T = T.copy()
    T["c"] = T.cutoff_d.replace({"full": "100"}).astype(float)
    for m in ["pen", "pen_r", "ols", "mlp"]:
        g = T[T.method == m].sort_values("c")
        if len(g):
            axes[2].plot(g.c, g.rmse, "-", marker=MK[m], color=COL[m], ms=4)
    axes[2].set_xscale("log")
    axes[2].set_xticks([3, 4, 7, 14, 100])
    axes[2].set_xticklabels(["3", "4", "7", "14", "all"])
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Training curves cut at (d)")
    axes[2].set_ylabel("RMSE, withheld mixes")
    for ax, l in zip(axes, "abc"):
        panel(ax, l)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=7)
    save(fig, "fig08_efficiency")


def fig07_robustness():
    S = pd.read_csv(Q / "robustness/summary.csv")
    order = [("clean", "reference"), ("iid_0.01", "noise 0.01"), ("iid_0.02", "noise 0.02"),
             ("corr_0.01", "correlated 0.01"), ("t0_0.25", "time origin 0.25 d"), ("shift_0.97", "offset 0.97 d"),
             ("shift_1.13", "offset 1.13 d"), ("shift_train", "offset, training mixes"), ("B_only", "short series only"),
             ("A_only", "long series only")]
    meths = ["pen", "pen_r", "ols", "mlp"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    y = np.arange(len(order))
    for i, m in enumerate(meths):
        g = S[S.method == m].set_index("variant")
        if not len(g):
            continue
        v = [g.rmse.get(k, np.nan) for k, _ in order]
        axes[0].plot(v, y + (i - 1.5) * 0.15, MK[m], color=COL[m], ms=4.5, mfc=COL[m], mec="white", mew=0.4, label=NAME[m], ls="")
        if m != "mlp":
            dk = [g.max_dK.get(k, np.nan) for k, _ in order]
            lab = NAME[m] if m != "ols" else "Classical fits (per mix)"
            axes[1].plot(dk, y + (i - 1.5) * 0.15, MK[m], color=COL[m], ms=4.5, mfc=COL[m], mec="white", mew=0.4, label=lab, ls="")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([o[1] for o in order], fontsize=7.2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE, withheld mixes")
    axes[0].legend(loc="lower right", fontsize=6.6)
    axes[1].set_xscale("log")
    axes[1].set_xlim(1e-4, 1)
    axes[1].set_xlabel(r"Largest shift of $\log_{10} K_\mathrm{int}$ (decades)")
    axes[1].legend(loc="lower right", fontsize=6.6)
    for ax, l in zip(axes, "ab"):
        ax.grid(axis="y", visible=False)
        panel(ax, l)
    save(fig, "fig07_robustness")


def fig09_pinn():
    import json
    R = pd.read_csv(Q / "pinn/runs.csv").set_index("run")
    m = json.loads((ROOT / "results/slab/metrics.json").read_text())
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.38)
    for (meth, title, letter), pos in zip((("pinn", "PINN", "a"), ("pen", "PENN", "b")), (gs[0, 0], gs[0, 1])):
        ax = fig.add_subplot(pos)
        fv = pd.read_csv(ROOT / f"results/slab/forward_verification_{meth}.csv")
        for test, col, mk, lab in ((False, C["muted"], "o", "Training mixes"), (True, COL["pen"] if meth == "pen" else C["pinn"], "s", "Withheld mixes")):
            g = fv[fv.test == test]
            ax.plot(g.S_model, g.S_fd_model_theta, mk, ms=2.8, mfc="none", mew=0.7, color=col, label=lab)
        ax.plot([0, 1], [0, 1], "-", color=C["data"], lw=0.8)
        q = m[f"forward_{meth}_fd_vs_model_all"]
        ax.text(0.97, 0.05, f"$R^2$ = {q['r2']:.4f}\nRMSE = {q['rmse']:.3f}", transform=ax.transAxes, ha="right")
        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, 0.85)
        ax.set_aspect("equal")
        ax.set_title(title + " (1D slab)")
        ax.set_xlabel(r"Network $S_\mathrm{avg}$")
        ax.set_ylabel("Direct solver, network parameters")
        ax.legend(loc="upper left", fontsize=7)
        panel(ax, letter)
    runs = [("reference", "PINN as in the paper"), ("warm_start", "started at classical parameters"),
            ("lambda_100", r"PDE weight $\lambda$ = 100"), ("synthetic", "synthetic data (known truth)"),
            ("fixed_classical", "parameters fixed at classical values")]
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(runs))
    ax.barh(y, [R.loc[r, "bias_K_vs_classical"] for r, _ in runs], color=[C["pinn"]] * 4 + [COL["ols"]], height=0.55)
    for j, (r, _) in enumerate(runs):
        b = R.loc[r, "bias_K_vs_classical"]
        ax.text(b - 0.03 if b < -0.1 else 0.03, j, f"{b:+.2f}", va="center", ha="right" if b < -0.1 else "left", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in runs], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=C["data"], lw=0.8)
    ax.set_xlim(-1.4, 0.3)
    ax.set_xlabel(r"Mean bias of $\log_{10} K_\mathrm{int}$ vs classical (decades)")
    ax.grid(axis="y", visible=False)
    panel(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    terms = [("loss_data", r"data, $\sigma_d^{-2}\mathcal{L}_\mathrm{data}$"), ("pde10", r"physics, $\lambda\mathcal{L}_\mathrm{PDE}$"),
             ("loss_total", "total")]
    R["pde10"] = 10 * R.loss_pde
    for k, (run, lab, c, hatch) in enumerate((("reference", "free parameters", C["pinn"], None),
                                               ("fixed_classical", "classical parameters", COL["ols"], "//"))):
        v = [R.loc[run, t] for t, _ in terms]
        ax.bar(np.arange(3) + (k - 0.5) * 0.36, v, width=0.34, color=c, hatch=hatch, edgecolor="white", label=lab)
    ax.set_xticks(range(3))
    ax.set_xticklabels([t[1] for t in terms], fontsize=7)
    ax.set_ylabel("Final loss (training mixes)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="x", visible=False)
    panel(ax, "d")
    save(fig, "fig09_pinn")


def fig10_parameters():
    th = pd.read_csv(ROOT / "results/disc/theta_all.csv").set_index("Mix")
    I = pd.read_csv(Q / "profiles/intervals.csv").set_index("Mix")
    # parameter maps over all 20 mixes from the hold-out models: PENN (results/disc runs) and revised PENN
    pr = pd.read_csv(Q / "baselines/pen_r_all_theta.csv").groupby("Mix")[["log10_Dv", "log10_K"]].agg(["mean", "std"])
    th["wb"] = [float(m.split("-")[1]) for m in th.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    fig.subplots_adjust(wspace=0.38, bottom=0.31)
    labels = {"log10_Dv": r"$\log_{10} D_v$ (m$^2$/s)", "log10_K": r"$\log_{10} K_\mathrm{int}$ (m$^2$)"}
    for ax, k, letter in zip(axes, ["log10_Dv", "log10_K"], "ab"):
        kk = "Dv" if k == "log10_Dv" else "K"
        for fam, mk in mf.MARK.items():
            g = th[[m.rsplit("-", 1)[0] == fam for m in th.index]]
            if not len(g):
                continue
            j = (list(mf.MARK).index(fam) - 3) * 0.008
            lo, hi = I.loc[g.index, f"{kk}_lo_eff"], I.loc[g.index, f"{kk}_hi_eff"]
            ax.errorbar(g.wb + j - 0.0025, g[f"{k}_fd"], yerr=[g[f"{k}_fd"] - lo, hi - g[f"{k}_fd"]], fmt=mk, ms=4.3,
                        mfc="none", mec=COL["ols"], mew=0.9, ecolor=COL["ols"], elinewidth=0.7, capsize=0)
            ax.plot(g.wb + j, g[f"{k}_pen"], mk, ms=4.3, color=COL["pen"], mec="white", mew=0.4)
            ax.plot(g.wb + j + 0.0025, pr.loc[g.index, (k, "mean")], mk, ms=4.3, color=COL["pen_r"], mec="white", mew=0.4)
        ax.set_xlabel("w/b")
        ax.set_ylabel(labels[k])
        ax.set_xticks([0.3, 0.4, 0.5])
        panel(ax, letter)
    h = [plt.Line2D([], [], marker=m, ls="", color=C["muted"], label=f) for f, m in mf.MARK.items()]
    h += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=COL["ols"], label="Classical fit, 95 % profile interval"),
          plt.Line2D([], [], marker="o", ls="", color=COL["pen"], label="PENN"),
          plt.Line2D([], [], marker="o", ls="", color=COL["pen_r"], label="PENN, revised")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), handletextpad=0.2, columnspacing=0.8)
    save(fig, "fig10_parameters")


def fig12_sensitivity():
    S = pd.read_csv(Q / "physics_sensitivity/summary.csv")
    F = pd.read_csv(Q / "physics_sensitivity/fits.csv")
    order = [("mesh_36x24", "finer mesh (36 × 24)"), ("sealed_lateral", "lateral face sealed"),
             ("alpha0_0.02", r"$\alpha_0$ = 0.02"), ("alpha0_0.10", r"$\alpha_0$ = 0.10"), ("n_4", "$n$ = 4"), ("n_16", "$n$ = 16"),
             ("S_c_0.75", "$S_c$ = 0.75"), ("S_cap_0.70", r"$S_\mathrm{cap}$ = 0.70"), ("S_cap_0.80", r"$S_\mathrm{cap}$ = 0.80"),
             ("k_25", "$k$ = 25"), ("beta_2.0", r"$\beta$ = 2.0"), ("beta_3.0", r"$\beta$ = 3.0"),
             ("S_init_0.97", r"$S_\mathrm{init}$ = 0.97"), ("tau_b_0.1d", r"surface exchange, $\tau_b$ = 0.1 d")]
    order = [o for o in order if o[0] in set(S.variant)]
    base = F[F.variant == "base"].set_index("Mix")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.9), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.75]})
    fig.subplots_adjust(wspace=0.08)
    y = np.arange(len(order))
    for ax, par, lab in ((axes[0], "log10_Dv", r"$\Delta\log_{10} D_v$"), (axes[1], "log10_K", r"$\Delta\log_{10} K_\mathrm{int}$")):
        for j, (v, _) in enumerate(order):
            g = F[F.variant == v].set_index("Mix")
            d = (g[par] - base.loc[g.index, par]).to_numpy()
            ax.plot([d.min(), d.max()], [j, j], "-", color=COL["pen"], lw=1.6, alpha=0.45)
            ax.plot(d.mean(), j, "o", color=COL["pen"], ms=4)
        ax.axvline(0, color=C["data"], lw=0.8)
        ax.set_xlabel(lab + " (decades)")
        ax.grid(axis="y", visible=False)
    for j, (v, _) in enumerate(order):
        r = S[S.variant == v].iloc[0]
        axes[2].plot(r.rmse_ratio, j, "s", color=COL["mlp"], ms=4)
    axes[2].axvline(1, color=C["data"], lw=0.8)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("RMSE ratio to base")
    axes[2].grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([o[1] for o in order], fontsize=7)
    axes[0].invert_yaxis()
    for ax, l in zip(axes, "abc"):
        panel(ax, l)
    save(fig, "fig12_sensitivity")


REUSED = {"fig02_geometry": "fig_geometry", "fig03_workflow": "fig_workflow", "fig05_test_curves": "fig_test_curves",
          "fig11_diffusivity": "fig_diffusivity", "fig13_fields": "fig_fields", "figB1_convergence": "fig_convergence",
          "figB2_profiles_slab": "fig_profiles_slab"}

if __name__ == "__main__":
    only = set(sys.argv[1:])
    todo = {"fig01_data": fig01_data, "fig04_identifiability": fig04_identifiability, "fig06_comparison": fig06_comparison,
            "fig08_efficiency": fig08_efficiency, "fig07_robustness": fig07_robustness, "fig09_pinn": fig09_pinn,
            "fig10_parameters": fig10_parameters, "fig12_sensitivity": fig12_sensitivity}
    for name, fn in todo.items():
        if not only or name in only:
            fn()
    for dst, src in REUSED.items():
        if not only or dst in only:
            copy(src, dst)
