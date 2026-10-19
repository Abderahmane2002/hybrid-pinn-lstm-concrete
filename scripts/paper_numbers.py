"""Collect every number quoted in the manuscript from the result files.

python scripts/paper_numbers.py  ->  results/paper_numbers.json (flat dict, formatted strings)

The manuscript source refers to these keys as «key»; the build script substitutes them
and refuses to build if a key is missing, so the text cannot drift from the results.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_moisture.axisym import AxisymBDF  # noqa: E402
from pinn_moisture.physics import config_for, diffusivity, theta_to_physical  # noqa: E402

R, D, S = ROOT / "results", ROOT / "results/disc", ROOT / "results/slab"
TEST = ["C-0.4", "C-0.5", "F10S10-0.3", "F20-0.3", "F20S20-0.4", "S40-0.4"]
N = {}


def put(key, value, fmt="{:.4f}"):
    N[key] = fmt.format(value) if not isinstance(value, str) else value


def pct(a, b):
    return 100 * (a - b) / a


def main():
    raw = pd.read_csv(ROOT / "data/raw/tidy_dataset_correctedV1.csv")
    clean = pd.read_csv(ROOT / "data/processed/drying_clean.csv")
    rel = clean[clean.reliable]
    put("n_raw", len(raw), "{:,}")
    put("n_raw_mixes", raw.Mix.nunique(), "{}")
    put("n_pts", len(rel), "{:,}")
    put("n_mixes", rel.Mix.nunique(), "{}")
    put("n_test_pts", rel.Mix.isin(TEST).sum(), "{}")
    put("n_train_pts", (~rel.Mix.isin(TEST)).sum(), "{}")
    put("n_train_mixes", rel[~rel.Mix.isin(TEST)].Mix.nunique(), "{}")
    rep = (ROOT / "data/processed/cleaning_report.md").read_text()
    m = re.search(r"\*\*([+-][\d.]+) d\*\* \(IQR ([+-][\d.]+) to ([+-][\d.]+) d\)", rep)
    put("shift", float(m.group(1)), "{:.2f}")
    put("shift_lo", float(m.group(2)), "{:.2f}")
    put("shift_hi", float(m.group(3)), "{:.2f}")
    put("n_consistent", rep.count("| consistent |"), "{}")

    # ---- numerical verification (parse the markdown tables) ----------------------------
    nv = (R / "numerical_verification.md").read_text()
    sec = lambda k: nv.split(f"## {k}.")[1].split("##")[0]  # noqa: E731
    rows = [l for l in sec(1).splitlines() if l.startswith("| C-") or l.startswith("| F")]
    e80 = [float(l.split("|")[5]) for l in rows]
    order = [float(l.split("|")[7]) for l in rows]
    put("fv_err80", max(e80), "{:.0e}")
    put("fv_order", np.mean(order), "{:.1f}")
    rows = [l for l in sec(2).splitlines() if l.startswith("| C-") or l.startswith("| F")]
    put("diff_err240", max(float(l.split("|")[4]) for l in rows), "{:.0e}")
    rows = [l for l in sec(3).splitlines() if l.startswith("| C-") or l.startswith("| F")]
    put("mass_err", max(float(l.split("|")[5]) for l in rows), "{:.0e}")
    rows = [l for l in sec(4).splitlines() if l.startswith("| C-") or l.startswith("| F")]
    put("disc_err", max(float(l.split("|")[3]) for l in rows), "{:.0e}")
    tc = (D / "table_check.md").read_text()
    put("table_err", float(re.search(r"max ([\d.e+-]+)\.", tc).group(1)), "{:.0e}")
    put("table_time_min", float(re.search(r"build time (\d+) s", tc).group(1)) / 60, "{:.0f}")

    # ---- geometry check ------------------------------------------------------------------
    g = pd.read_csv(R / "geometry_check.csv")
    put("geo_maxdiff", g.max_diff.max(), "{:.2f}")
    put("geo_dDv_lo", g.dlog10_Dv.min(), "{:.2f}")
    put("geo_dDv_hi", g.dlog10_Dv.max(), "{:.2f}")
    put("geo_dK_lo", g.dlog10_K.min(), "{:.2f}")
    put("geo_dK_hi", g.dlog10_K.max(), "{:.2f}")
    put("geo_refit_rmse_hi", g.refit_rmse.max(), "{:.3f}")

    # ---- n profile -----------------------------------------------------------------------
    pn = pd.read_csv(D / "profile_n_runs.csv")
    pn["sse"] = pn.rmse ** 2 * pn.N
    sse = pn.groupby("n").sse.sum()
    put("n_gain_8_16", 100 * (sse[8.0] - sse[16.0]) / sse[8.0], "{:.0f}")
    put("n_gain_2_8", 100 * (sse[2.0] - sse[8.0]) / sse[2.0], "{:.0f}")

    # ---- identifiability (disc) ------------------------------------------------------------
    idf = pd.read_csv(D / "identifiability.csv")
    two = idf[idf.n_par == 2]
    three = idf[idf.n_par == 3]
    four = idf[idf.n_par == 4]
    put("id_corr3_min", three.max_corr.min(), "{:.2f}")
    put("id_cond3_lo", three.cond.min(), "{:.0e}")
    put("id_cond3_hi", three.cond.max(), "{:.0e}")
    put("id_corr4_min", four.max_corr.min(), "{:.2f}")
    put("id_corr2_hi", two.max_corr.max(), "{:.2f}")
    put("id_corr2_lo", two.max_corr.min(), "{:.2f}")
    put("id_cond2_lo", two.cond.min(), "{:.0f}")
    put("id_cond2_hi", two.cond.max(), "{:.0f}")

    # ---- classical identification (disc) ----------------------------------------------------
    cl = pd.read_csv(D / "classical_fits.csv")
    put("cl_r2_min", cl.r2.min())
    put("cl_r2_max", cl.r2.max())
    put("cl_r2_med", cl.r2.median())
    put("cl_rmse_med", cl.rmse.median(), "{:.4f}")
    put("cl_time", cl.time_s.mean(), "{:.0f}")
    put("cl_relse_max", 100 * np.max(np.abs(np.r_[cl.se_log10_Dv / 1, cl.se_log10_K / 1])) * np.log(10), "{:.0f}")
    put("cl_se_Dv_max", cl.se_log10_Dv.max(), "{:.3f}")
    put("cl_se_K_max", cl.se_log10_K.max(), "{:.3f}")
    put("cl_Dv_lo", cl.log10_Dv.min(), "{:.2f}")
    put("cl_Dv_hi", cl.log10_Dv.max(), "{:.2f}")
    put("cl_K_lo", cl.log10_K.min(), "{:.2f}")
    put("cl_K_hi", cl.log10_K.max(), "{:.2f}")

    # ---- hold-out (disc) -------------------------------------------------------------------
    md = json.loads((D / "metrics.json").read_text())
    t = md["methods_test"]
    for k in ("pen", "fdreg", "mlp", "fdfit"):
        put(f"ho_{k}_r2", t[k]["r2_mean"])
        put(f"ho_{k}_rmse", t[k]["rmse_mean"])
        if t[k]["n_seeds"] > 1:
            put(f"ho_{k}_r2_sd", t[k]["r2_sd"])
            put(f"ho_{k}_rmse_sd", t[k]["rmse_sd"])
    put("ho_gain_mlp", pct(t["mlp"]["rmse_mean"], t["pen"]["rmse_mean"]), "{:.0f}")
    put("ho_pen_time", t["pen"]["time_s"], "{:.0f}")
    put("ho_mlp_time", t["mlp"]["time_s"], "{:.0f}")
    put("ho_fdreg_time", t["fdreg"]["time_s"], "{:.0f}")
    put("ho_pen_train_rmse", md["methods_train"]["pen"]["rmse"], "{:.4f}")
    tm = pd.read_csv(D / "table_test_mixes.csv")
    for meth in ("pen", "fdreg", "mlp"):
        for mix in TEST:
            put(f"mix_{meth}_{mix}", tm[(tm.method == meth) & (tm.Mix == mix)].r2_mean.values[0], "{:.3f}")
    pen = tm[tm.method == "pen"].set_index("Mix").r2_mean
    put("ho_pen_mix_min", pen.min(), "{:.3f}")
    put("ho_pen_mix_min_name", pen.idxmin())
    put("ho_pen_mix_max", pen.max(), "{:.3f}")
    put("ho_pen_mix_max_name", pen.idxmax())
    put("fwd_pen_rmse", md["forward_pen_fd_vs_model_all"]["rmse"], "{:.0e}")
    for k in ("log10_Dv", "log10_K"):
        put(f"th_{k}_mae", md[f"theta_pen_vs_fd_{k}_mae"], "{:.3f}")
        put(f"th_{k}_corr", md[f"theta_pen_vs_fd_{k}_corr"], "{:.2f}")
        put(f"th_{k}_sd", md[f"theta_pen_{k}_seed_sd_mean"], "{:.3f}")
        put(f"th_{k}_lo", md[f"theta_pen_{k}_range"][0], "{:.2f}")
        put(f"th_{k}_hi", md[f"theta_pen_{k}_range"][1], "{:.2f}")

    # ---- cross-validation (disc) -------------------------------------------------------------
    cv = pd.read_csv(D / "cv_summary.csv").set_index(["scheme", "method"])
    for sch in ("lomo", "lofo"):
        for meth in ("pen", "fdreg", "mlp"):
            r = cv.loc[(sch, meth)]
            put(f"cv_{sch}_{meth}_r2", r.r2)
            put(f"cv_{sch}_{meth}_rmse", r.rmse)
            put(f"cv_{sch}_{meth}_min", r.r2_mix_min, "{:.2f}" if r.r2_mix_min < 0.9 else "{:.3f}")
            put(f"cv_{sch}_{meth}_nbad", int(r["n_mix_r2_below_0.95"]), "{}")
    cpm = pd.read_csv(D / "cv_per_mix.csv")
    worst = cpm[(cpm.scheme == "lofo") & (cpm.method == "mlp")].sort_values("r2").iloc[0]
    put("cv_lofo_mlp_worst_mix", worst.Mix)
    put("cv_lofo_pen_gain", pct(cv.loc[("lofo", "mlp")].rmse, cv.loc[("lofo", "pen")].rmse), "{:.0f}")

    # ---- noise robustness ----------------------------------------------------------------------
    th0 = pd.concat([pd.read_csv(d / "theta.csv") for d in sorted((D / "runs").glob("pen_seed?"))])
    th0m = th0.groupby("Mix")[["log10_Dv", "log10_K"]].mean()
    for lev in ("0.01", "0.02"):
        rmse = {}
        for meth in ("pen", "mlp"):
            vals = []
            for d in sorted((D / "runs").glob(f"{meth}_seed?_noise{lev}")):
                p = pd.read_csv(d / "pred.csv")
                p = p[p.split == "test"]
                vals.append(np.sqrt(np.mean((p.S_pred - p.S_obs) ** 2)))
            rmse[meth] = np.mean(vals)
            put(f"noise{lev[2:]}_{meth}_rmse", rmse[meth])
        thn = pd.concat([pd.read_csv(d / "theta.csv") for d in sorted((D / "runs").glob(f"pen_seed?_noise{lev}"))])
        thn = thn.groupby("Mix")[["log10_Dv", "log10_K"]].mean()
        put(f"noise{lev[2:]}_dDv", np.max(np.abs(thn.log10_Dv - th0m.log10_Dv)), "{:.3f}")
        put(f"noise{lev[2:]}_dK", np.max(np.abs(thn.log10_K - th0m.log10_K)), "{:.3f}")

    # ---- slab comparison with the PINN ------------------------------------------------------------
    ms = json.loads((S / "metrics.json").read_text())
    t = ms["methods_test"]
    for k in ("pen", "fdreg", "mlp", "pinn", "pinn_nopde"):
        put(f"sl_{k}_r2", t[k]["r2_mean"])
        put(f"sl_{k}_rmse", t[k]["rmse_mean"])
        if t[k]["n_seeds"] > 1:
            put(f"sl_{k}_r2_sd", t[k]["r2_sd"])
            put(f"sl_{k}_rmse_sd", t[k]["rmse_sd"])
    put("sl_pinn_train_rmse", ms["methods_train"]["pinn"]["rmse"])
    put("sl_pen_train_rmse", ms["methods_train"]["pen"]["rmse"])
    put("sl_pinn_time", t["pinn"]["time_s"], "{:.0f}")
    put("sl_pen_time", t["pen"]["time_s"], "{:.0f}")
    for meth in ("pen", "pinn"):
        put(f"sl_fwd_{meth}_r2", ms[f"forward_{meth}_fd_vs_model_all"]["r2"], "{:.3f}" if meth == "pinn" else "{:.4f}")
        put(f"sl_fwd_{meth}_rmse", ms[f"forward_{meth}_fd_vs_model_all"]["rmse"], "{:.3f}")
    put("sl_fwd_pinn_data_test_r2", ms["forward_pinn_fd_vs_data_test"]["r2"], "{:.2f}")
    put("sl_pinn_K_bias", ms["theta_pinn_vs_fd_log10_K_mae"], "{:.1f}")
    put("sl_pinn_Dv_mae", ms["theta_pinn_vs_fd_log10_Dv_mae"], "{:.2f}")
    tms = pd.read_csv(S / "table_test_mixes.csv")
    put("sl_pinn_C05", tms[(tms.method == "pinn") & (tms.Mix == "C-0.5")].r2_mean.values[0], "{:.2f}")
    put("sl_mlp_C05", tms[(tms.method == "mlp") & (tms.Mix == "C-0.5")].r2_mean.values[0], "{:.2f}")

    # ---- physical results (disc, PENN parameters) -----------------------------------------------------
    th = pd.read_csv(D / "theta_all.csv").set_index("Mix")
    cfg = config_for("disc")
    put("C03_Dv", th.loc["C-0.3", "log10_Dv_pen"], "{:.2f}")
    put("C05_Dv", th.loc["C-0.5", "log10_Dv_pen"], "{:.2f}")
    put("F4003_Dv", th.loc["F40-0.3", "log10_Dv_pen"], "{:.2f}")
    put("F40_factor", 10 ** (th.loc["C-0.3", "log10_Dv_pen"] - th.loc["F40-0.3", "log10_Dv_pen"]), "{:.1f}")
    Sg = np.linspace(0, 1, 401)
    dmin, dmax, smax, cross = [], [], [], []
    for mix in th.index:
        p = theta_to_physical({k: th.loc[mix, f"{k}_pen"] for k in cfg.identified}, cfg)
        Dd = diffusivity(Sg, p["Dv"], p["alpha0"], p["n"], p["K"], cfg)
        dmin.append(Dd.min())
        dmax.append(Dd.max())
        smax.append(Sg[np.argmax(Dd)])
    put("D_min_lo", min(dmin), "{:.0e}")
    put("D_min_hi", max(dmin), "{:.0e}")
    put("D_max_lo", min(dmax), "{:.0e}")
    put("D_max_hi", max(dmax), "{:.0e}")
    put("D_Speak", np.median(smax), "{:.2f}")
    # internal fields: time for the specimen centre (r = 0, z = 0) to fall below S = 0.5
    sv = AxisymBDF(cfg, nr=48, nz=32, ratio=1.055)
    tt = np.geomspace(0.1, 100, 300)
    for mix in ("C-0.4", "S40-0.4", "F20-0.3"):
        _, F = sv.solve({k: float(th.loc[mix, f"{k}_pen"]) for k in cfg.identified}, tt, return_fields=True)
        centre = F[0, 0, :]
        put(f"tc_{mix}", tt[np.argmax(centre < 0.5)], "{:.0f}")
        i1 = np.argmin(np.abs(tt - 1.0))
        put(f"centre1d_{mix}", centre[i1], "{:.2f}")

    (R / "paper_numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))
    print(json.dumps(N, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
