"""Tables of the revised manuscript as pandoc pipe tables (results/q1/tables/*.md).

The manuscript includes them by name, so every table is generated from the result files.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from common import ROOT, TEST_MIXES

Q = ROOT / "results/q1"
OUT = Q / "tables"
OUT.mkdir(parents=True, exist_ok=True)
NAME = {"pen": "PENN", "pen_r": "PENN, revised (linear map, L-BFGS)", "pen_lin": "PENN, linear composition map",
        "pen_s": "PENN, 1 × 8 units", "ols": "Two-stage, linear (OLS)", "ridge": "Two-stage, ridge",
        "quad": "Two-stage, quadratic", "gp": "Two-stage, Gaussian process", "mlp": "Data-driven network"}
ORDER = ["ols", "ridge", "quad", "gp", "pen", "pen_r", "pen_lin", "pen_s", "mlp"]


def write(name, lines):
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")


def pm(m, s, f="{:.4f}"):
    return f.format(m) if s is None or not np.isfinite(s) else f"{f.format(m)} ± {f.format(s)}"


def mixes():
    raw = pd.read_csv(ROOT / "data/raw/tidy_dataset_correctedV1.csv").drop_duplicates("Mix")
    rows = {r.Mix: (r["w/b"], r.Water, r.Cement, r.Fly_ash, r.Slag) for _, r in raw.iterrows()}
    rows["S20-0.5"] = (0.5, 227, 363, 0, 91)              # from Table 1 of the original manuscript (no drying data)
    order = ["C", "F20", "F40", "S20", "S40", "F10S10", "F20S20"]
    keys = sorted(rows, key=lambda m: (order.index(m.rsplit("-", 1)[0]), float(m.rsplit("-", 1)[1])))
    clean = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
    used = set(clean[clean.reliable].Mix)
    L = ["| Mix | w/b | Water | Cement | Fly ash | Slag | Used |", "|:----------|:----:|:----:|:----:|:----:|:----:|:----:|"]
    for m in keys:
        wb, w, c, fa, sl = rows[m]
        use = ("withheld" if m in TEST_MIXES else "training") if m in used else "–"
        L.append(f"| {m} | {wb:.1f} | {w:.0f} | {c:.0f} | {fa:.0f} | {sl:.0f} | {use} |".replace("| 0 | ", "| – | ")
                 .replace(" | 0 | ", " | – | "))
    write("mixes", L)


def holdout():
    P = pd.read_csv(Q / "baselines/pooled.csv").set_index(["scheme", "method"])
    T = pd.read_csv(Q / "baselines/timing.csv").set_index(["scheme", "method"])
    m0 = json.loads((ROOT / "results/disc/metrics.json").read_text())["methods_test"]
    L = ["| Method | $R^2$ | RMSE | Training time |", "|:------------------------------------|:--------------:|:--------------:|:-----------:|"]
    for m in ORDER:
        r = P.loc[("holdout", m)]
        if m == "mlp":
            t = f"{m0['mlp']['time_s']:.0f} s"
        elif m.startswith("pen"):
            tt = T.loc[('holdout', m), 'time_s']
            t = (f"{tt:.0f} s^a^" if tt >= 1 else "< 1 s^a^")
        else:
            t = f"{m0['fdreg']['time_s']:.0f} s^b^"
        L.append(f"| {NAME[m]} | {pm(r.r2, r.r2_sd)} | {pm(r.rmse, r.rmse_sd)} | {t} |")
    r = m0["fdfit"]
    L.append(f"| *Reference: classical fit to each withheld mix* | *{r['r2_mean']:.4f}* | *{r['rmse_mean']:.4f}* | *{r['time_s']:.0f} s per mix* |")
    write("holdout", L)


def per_mix():
    pmx = pd.read_csv(Q / "baselines/permix.csv")
    pmx = pmx[pmx.scheme == "holdout"].groupby(["method", "Mix"]).r2.mean()
    meths = ["ols", "pen", "pen_r", "gp", "mlp"]
    L = ["| Mix | " + " | ".join(NAME[m].replace(" (no weight decay)", "").replace(" (OLS)", "") for m in meths) + " |",
         "|:------|" + ":----:|" * len(meths)]
    for mix in TEST_MIXES:
        L.append(f"| {mix} | " + " | ".join(f"{pmx.loc[(m, mix)]:.3f}" for m in meths) + " |")
    write("per_mix", L)


def cv():
    P = pd.read_csv(Q / "baselines/pooled.csv").set_index(["scheme", "method"])
    L = ["| Method | LOMO $R^2$ | LOMO RMSE | LOMO lowest $R^2$ | LOFO $R^2$ | LOFO RMSE | LOFO lowest $R^2$ | LOFO mixes < 0.95 |",
         "|:------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|"]
    for m in ORDER:
        a, b = P.loc[("lomo", m)], P.loc[("lofo", m)]
        L.append(f"| {NAME[m]} | {a.r2:.4f} | {a.rmse:.4f} | {a.r2_mix_min:.3f} | {b.r2:.4f} | {b.rmse:.4f} | "
                 f"{b.r2_mix_min:.3f} | {b.n_below:.0f} |")
    write("cv", L)


def robustness():
    S = pd.read_csv(Q / "robustness/summary.csv").set_index(["variant", "method"])
    order = [("clean", "None (reference)"), ("iid_0.01", "Independent noise, s.d. 0.01"), ("iid_0.02", "Independent noise, s.d. 0.02"),
             ("corr_0.01", "Correlated noise, s.d. 0.01"), ("t0_0.25", "Time-origin error, s.d. 0.25 d"),
             ("shift_0.97", "Offset 0.97 d"), ("shift_1.13", "Offset 1.13 d"), ("shift_train", "Offset from training mixes"),
             ("B_only", "Short series only"), ("A_only", "Long series only")]
    L = ["| Perturbation of the training data | RMSE linear | RMSE PENN | RMSE PENN rev. | RMSE data-driven | "
         "max Δlog K classical | max Δlog K PENN | max Δlog K PENN rev. |",
         "|:------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|"]
    for v, lab in order:
        def g(m, c, f="{:.4f}"):
            try:
                x = S.loc[(v, m), c]
                return "–" if not np.isfinite(x) or (v == "clean" and c == "max_dK") else f.format(x)
            except KeyError:
                return "–"
        L.append(f"| {lab} | {g('ols', 'rmse')} | {g('pen', 'rmse')} | {g('pen_r', 'rmse')} | {g('mlp', 'rmse')} | "
                 f"{g('ols', 'max_dK', '{:.3f}')} | {g('pen', 'max_dK', '{:.3f}')} | {g('pen_r', 'max_dK', '{:.3f}')} |")
    write("robustness", L)


def sensitivity():
    p = Q / "physics_sensitivity/summary.csv"
    if not p.exists():
        return
    S = pd.read_csv(p).set_index("variant")
    lab = {"mesh_36x24": "Finer mesh (36 × 24)", "sealed_lateral": "Lateral surface sealed (1D, L = 25 mm)",
           "alpha0_0.02": "$\\alpha_0$ = 0.02 (0.05)", "alpha0_0.10": "$\\alpha_0$ = 0.10 (0.05)",
           "n_4": "$n$ = 4 (8)", "n_16": "$n$ = 16 (8)", "S_c_0.75": "$S_c$ = 0.75 (0.80)",
           "S_cap_0.70": "$S_{\\text{cap}}$ = 0.70 (0.75)", "S_cap_0.80": "$S_{\\text{cap}}$ = 0.80 (0.75)",
           "k_25": "$k$ = 25 (50)", "beta_2.0": "$\\beta$ = 2.0 (2.5)", "beta_3.0": "$\\beta$ = 3.0 (2.5)",
           "S_init_0.97": "$S_{\\text{init}}$ = 0.97 (1)", "tau_b_0.1d": "Surface exchange, $\\tau_b$ = 0.1 d (10⁻⁴ d)"}
    L = ["| Change (reference value) | mean Δlog $D_v$ | mean Δlog $K_{\\text{int}}$ | max \\|Δlog $K_{\\text{int}}$\\| | RMSE ratio | rank corr. $D_v$ | rank corr. $K_{\\text{int}}$ |",
         "|:------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|"]
    for v, l in lab.items():
        if v in S.index:
            r = S.loc[v]
            L.append(f"| {l} | {r.dDv_mean:+.3f} | {r.dK_mean:+.3f} | {r.dK_maxabs:.3f} | {r.rmse_ratio:.2f} | {r.rank_Dv:.2f} | "
                     f"{r.rank_K:.2f} |".replace("+-", "−").replace("| -", "| −"))
    L.append("| $P_{c0}$ or $\\eta_l$ × c | 0 | −log₁₀ c (exact) | – | 1.00 | 1.00 | 1.00 |")
    write("sensitivity", L)


def inner_cv():
    R = pd.read_csv(Q / "inner_cv/inner.csv")
    L = ["| Model | Configuration | Inner RMSE |", "|:------------|:------------------------------|:----:|"]
    for r in R.itertuples():
        if r.kind == "pen":
            c = f"{int(r.layers)} × {int(r.hidden)} tanh, {int(r.iters)} iterations"
            paper = int(r.hidden) == 16 and int(r.iters) == 400
            name = "PENN"
        else:
            c = f"{int(r.layers)} × {int(r.hidden)} tanh, weight decay {r.weight_decay:g}"
            paper = int(r.hidden) == 48 and r.weight_decay == 1e-5
            name = "Data-driven network"
        L.append(f"| {name} | {c}{' (original setting)' if paper else ''} | {r.inner_rmse:.4f} |")
    write("inner_cv", L)


def pinn_diag():
    R = pd.read_csv(Q / "pinn/runs.csv").set_index("run")
    lab = [("reference", "As in the comparison"), ("warm_start", "Parameters started at classical values"),
           ("lambda_100", "PDE weight λ = 100"), ("synthetic", "Synthetic training data (known parameters)"),
           ("fixed_classical", "Parameters fixed at classical values")]
    L = ["| Run | Total loss | Data term | PDE term | Training RMSE | PDE residual RMS | Solver check $R^2$ | Bias of log $K_{\\text{int}}$ |",
         "|:------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|"]
    for k, l in lab:
        r = R.loc[k]
        L.append(f"| {l} | {r.loss_total:.3f} | {r.loss_data:.3f} | {r.loss_pde:.4f} | {r.train_rmse:.4f} | {r.pde_rms:.3f} | "
                 f"{r.fwd_vs_model_r2:.3f} | {r.bias_K_vs_classical:+.2f} |".replace("-", "−"))
    write("pinn_diag", L)


def repro():
    p = Q / "reproducibility/comparison.csv"
    if not p.exists():
        return
    R = pd.read_csv(p)
    L = ["| Quantity | Original | Re-run | Status |", "|:----------------------------------------|:----:|:----:|:----:|"]
    for r in R.itertuples():
        f = (lambda x: f"{x:.4g}")
        L.append(f"| {r.item} | {f(r.manuscript)} | {f(r.reproduced)} | {r.status} |")
    write("repro", L)


if __name__ == "__main__":
    for f in ("identifiability", "slab", "numerics"):
        shutil.copy(ROOT / f"results/paper_tables/{f}.md", OUT / f"{f}.md")
    mixes(); holdout(); per_mix(); cv(); robustness(); sensitivity(); inner_cv(); pinn_diag(); repro()
    for p in sorted(OUT.glob("*.md")):
        print("==", p.name, len(p.read_text().splitlines()), "lines")
