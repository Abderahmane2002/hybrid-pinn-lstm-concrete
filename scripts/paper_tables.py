"""Write the result tables of the manuscript as pandoc pipe tables (results/paper_tables/*.md).

The manuscript includes them by name, so the tables always match the result files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, D, S = ROOT / "results", ROOT / "results/disc", ROOT / "results/slab"
OUT = R / "paper_tables"
TEST = ["C-0.4", "C-0.5", "F10S10-0.3", "F20-0.3", "F20S20-0.4", "S40-0.4"]
SYM = {"D_v": "$D_v$", "α0": "$\\alpha_0$", "n": "$n$", "K_int": "$K_{\\text{int}}$"}


def pm(mean, sd, fmt="{:.4f}"):
    return fmt.format(mean) if (sd is None or np.isnan(sd)) else f"{fmt.format(mean)} ± {fmt.format(sd)}"


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")


def identifiability():
    d = pd.read_csv(D / "identifiability.csv")
    lines = ["| Mix | Identified | $R^2$ | RMSE | max $\\lvert\\rho\\rvert$ | cond($\\mathbf{J}$) |",
             "|:------|:--------------------|:----:|:----:|:----:|:----:|"]
    for r in d.itertuples():
        ident = ", ".join(SYM[s.strip()] for s in r.identified.split(","))
        e = int(np.floor(np.log10(r.cond)))
        cond = f"{r.cond:.0f}" if r.cond < 100 else f"{r.cond / 10 ** e:.1f} × 10^{e}^"
        lines.append(f"| {r.Mix} | {ident} | {r.r2:.4f} | {r.rmse:.4f} | {r.max_corr:.2f} | {cond} |")
    write("identifiability", lines)


def holdout():
    m = json.loads((D / "metrics.json").read_text())["methods_test"]
    rows = [("pen", "Physics-embedded neural network (PENN)", f"{m['pen']['time_s']:.0f} s^a^"),
            ("fdreg", "Classical identification + regression on composition", f"{m['fdreg']['time_s']:.0f} s^b^"),
            ("mlp", "Data-driven network", f"{m['mlp']['time_s']:.0f} s"),
            ("fdfit", "*Reference: classical fit to each withheld mix*", f"*{m['fdfit']['time_s']:.0f} s per mix*")]
    lines = ["| Method | $R^2$ | RMSE | Training time |", "|:------------------------------------|:--------------:|:--------------:|:-----------:|"]
    for k, lab, cost in rows:
        r = m[k]
        sd = r["r2_sd"] if r["n_seeds"] > 1 else None
        sdr = r["rmse_sd"] if r["n_seeds"] > 1 else None
        r2, rm = pm(r["r2_mean"], sd), pm(r["rmse_mean"], sdr)
        if k == "fdfit":
            r2, rm = f"*{r2}*", f"*{rm}*"
        lines.append(f"| {lab} | {r2} | {rm} | {cost} |")
    write("holdout", lines)


def per_mix():
    t = pd.read_csv(D / "table_test_mixes.csv")
    lines = ["| Mix | PENN | Classical + regression | Data-driven |", "|:------|:----:|:----:|:----:|"]
    for mix in TEST:
        v = [t[(t.method == k) & (t.Mix == mix)].r2_mean.values[0] for k in ("pen", "fdreg", "mlp")]
        lines.append(f"| {mix} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    write("per_mix", lines)


def cv():
    s = pd.read_csv(D / "cv_summary.csv").set_index(["scheme", "method"])
    lab = {"pen": "PENN", "fdreg": "Classical + regression", "mlp": "Data-driven"}
    lines = ["| Scheme | Method | $R^2$ (pooled) | RMSE | Lowest $R^2$ of a mix | Mixes with $R^2$ < 0.95 |",
             "|:--------------------|:--------------------|:----------:|:--------:|:-----------:|:-----------:|"]
    for sch, name in (("lomo", "Leave one mix out (20 folds)"), ("lofo", "Leave one family out (7 folds)")):
        for k in ("pen", "fdreg", "mlp"):
            r = s.loc[(sch, k)]
            lines.append(f"| {name if k == 'pen' else ''} | {lab[k]} | {r.r2:.4f} | {r.rmse:.4f} | "
                         f"{r.r2_mix_min:.3f} | {int(r['n_mix_r2_below_0.95'])} |")
    write("cv", lines)


def noise():
    th0 = pd.concat([pd.read_csv(d / "theta.csv") for d in sorted((D / "runs").glob("pen_seed?"))])
    th0 = th0.groupby("Mix")[["log10_Dv", "log10_K"]].mean()
    lines = ["| Noise added to training data | PENN RMSE | Data-driven RMSE | max shift of $\\log_{10} D_v$ | "
             "max shift of $\\log_{10} K_{\\text{int}}$ |", "|:-----------:|:--------------:|:--------------:|:-----------:|:-----------:|"]
    for lev in ("0", "0.01", "0.02"):
        vals = {}
        for meth in ("pen", "mlp"):
            pat = f"{meth}_seed?" if lev == "0" else f"{meth}_seed?_noise{lev}"
            rm = []
            for d in sorted((D / "runs").glob(pat)):
                p = pd.read_csv(d / "pred.csv")
                p = p[p.split == "test"]
                rm.append(np.sqrt(np.mean((p.S_pred - p.S_obs) ** 2)))
            vals[meth] = (np.mean(rm), np.std(rm, ddof=1))
        if lev == "0":
            dd, dk = "–", "–"
        else:
            th = pd.concat([pd.read_csv(d / "theta.csv") for d in sorted((D / "runs").glob(f"pen_seed?_noise{lev}"))])
            th = th.groupby("Mix")[["log10_Dv", "log10_K"]].mean()
            dd = f"{np.max(np.abs(th.log10_Dv - th0.log10_Dv)):.3f}"
            dk = f"{np.max(np.abs(th.log10_K - th0.log10_K)):.3f}"
        sd_txt = "none" if lev == "0" else f"σ = {lev}"
        lines.append(f"| {sd_txt} | {pm(*vals['pen'])} | {pm(*vals['mlp'])} | {dd} | {dk} |")
    write("noise", lines)


def slab():
    ms = json.loads((S / "metrics.json").read_text())
    t, tr = ms["methods_test"], ms["methods_train"]
    rows = [("pen", "PENN (differentiable solver)"), ("fdreg", "Classical + regression"),
            ("pinn", "PINN"), ("pinn_nopde", "PINN without PDE residual"), ("mlp", "Data-driven network")]
    lines = ["| Method | $R^2$ (withheld) | RMSE (withheld) | RMSE (training) | Solver check $R^2$ |",
             "|:---------------------------|:--------------:|:--------------:|:----------:|:----------:|"]
    for k, lab in rows:
        r = t[k]
        sd = r["r2_sd"] if r["n_seeds"] > 1 else None
        sdr = r["rmse_sd"] if r["n_seeds"] > 1 else None
        trn = f"{tr[k]['rmse']:.4f}" if k in tr else "–"
        fwd = ms.get(f"forward_{k}_fd_vs_model_all")
        fwd = (f"{fwd['r2']:.4f}" if fwd["r2"] > 0.99 else f"{fwd['r2']:.3f}") if fwd else ("exact" if k == "fdreg" else "–")
        lines.append(f"| {lab} | {pm(r['r2_mean'], sd)} | {pm(r['rmse_mean'], sdr)} | {trn} | {fwd} |")
    write("slab", lines)


def numerics():
    nv = (R / "numerical_verification.md").read_text()
    sec = lambda k: [l for l in nv.split(f"## {k}.")[1].split("##")[0].splitlines()  # noqa: E731
                     if l.startswith("| C-") or l.startswith("| F")]
    e1 = [[float(x) for x in l.split("|")[2:7]] for l in sec(1)]
    e2 = [[float(x) for x in l.split("|")[2:7]] for l in sec(2)]
    e4 = [[float(x) for x in l.split("|")[2:5]] for l in sec(4)]
    mb = [float(l.split("|")[5]) for l in sec(3)]
    f = lambda v: f"{v:.1e}".replace("e-0", " × 10^−").replace("e-", " × 10^−") + "^"  # noqa: E731
    lines = ["| Check | Setting | Max. error in $S_{\\text{avg}}$ |", "|:-----------------------|:--------------------|:----:|"]
    lines += [f"| 1D finite-volume solver (BDF) | {n} cells | {f(max(r[i] for r in e1))} |"
              for i, n in enumerate([20, 40, 60, 80, 160])]
    lines += [f"| 1D differentiable solver | {n} time steps | {f(max(r[i] for r in e2))} |"
              for i, n in enumerate([60, 120, 240, 480, 960])]
    lines += [f"| Disc model (BDF) | {n} cells | {f(max(r[i] for r in e4))} |"
              for i, n in enumerate(["12 × 8", "24 × 16", "48 × 32"])]
    lines += [f"| Global mass balance, 1D | 28 days | {f(max(mb))} (relative) |"]
    write("numerics", lines)


if __name__ == "__main__":
    identifiability()
    holdout()
    per_mix()
    cv()
    noise()
    slab()
    numerics()
    for p in sorted(OUT.glob("*.md")):
        print(f"== {p.name}\n{p.read_text()}")
