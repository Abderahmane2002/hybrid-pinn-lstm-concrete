"""Every number quoted in the revised manuscript, computed from the result files.

Merges results/paper_numbers.json (original pipeline) with the revision experiments
(results/q1/*) into results/q1/q1_numbers.json. The manuscript source refers to these
values as «key»; the build stops if a key is missing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ROOT, TEST_MIXES

Q = ROOT / "results/q1"
N = json.loads((ROOT / "results/paper_numbers.json").read_text())


def put(k, v, fmt="{:.4f}"):
    N[k] = fmt.format(v) if not isinstance(v, str) else v


def main():
    # ---- data reconstruction
    sr = pd.read_csv(Q / "series_relation.csv")
    s5 = sr[sr.wb < 0.55]
    put("series_shift_rms", s5.shift_rms_S.median(), "{:.3f}")
    put("series_scale_rms", s5.scale_rms_S.median(), "{:.3f}")
    put("series_affine_l", s5.affine_l.median(), "{:.2f}")
    put("series_affine_lo", s5.affine_l.quantile(.25), "{:.2f}")
    put("series_affine_hi", s5.affine_l.quantile(.75), "{:.2f}")
    put("series_scale_factor", s5.scale_rms_S.median() / s5.shift_rms_S.median(), "{:.0f}")
    put("shift_train", float((Q / "robustness/shift_train.txt").read_text()), "{:.2f}")
    # ---- identifiability: profiles and synthetic recovery
    I = pd.read_csv(Q / "profiles/intervals.csv")
    put("prof_dv_med", I.Dv_width_eff.median(), "{:.2f}")
    put("prof_k_med", I.K_width_eff.median(), "{:.2f}")
    put("prof_k_max", I.K_width_eff.max(), "{:.1f}")
    put("prof_k_max_mix", I.loc[I.K_width_eff.idxmax(), "Mix"])
    put("prof_dv_max", I.Dv_width_eff.max(), "{:.2f}")
    put("prof_rho_med", I.rho1.median(), "{:.2f}")
    put("prof_neff_med", I.N_eff.median(), "{:.0f}")
    put("prof_n_med", I.N.median(), "{:.0f}")
    put("prof_k_unc_med", (I.K_hi - I.K_lo).median(), "{:.2f}")
    put("prof_all_interior", "all" if (I.K_interior & I.Dv_interior).all() else "not all")
    c1 = pd.read_csv(Q / "synthetic/summary_c1.csv").set_index("cond")
    put("syn_exact_dv", c1.loc["exact", "maxabs_Dv"], "{:.3f}")
    put("syn_exact_k", c1.loc["exact", "maxabs_K"], "{:.3f}")
    put("syn_short_k", c1.loc["short_14d", "maxabs_K"], "{:.3f}")
    put("syn_iid_dv", c1.loc["iid_0.01", "maxabs_Dv"], "{:.3f}")
    put("syn_iid_k", c1.loc["iid_0.01", "maxabs_K"], "{:.2f}")
    put("syn_corr_k", c1.loc["corr_0.01", "maxabs_K"], "{:.2f}")
    put("syn_t0_k", c1.loc["t0_0.25", "maxabs_K"], "{:.2f}")
    put("syn_t0_dv", c1.loc["t0_0.25", "maxabs_Dv"], "{:.3f}")
    pr = pd.read_csv(Q / "audit/profile_n_train.csv")
    t = pr[pr.set == "training mixes only"].set_index("n")
    put("n_gain_2_8_train", 100 * (t.loc[2.0, "sse"] / t.loc[8.0, "sse"] - 1), "{:.0f}")
    put("n_gain_8_16_train", 100 * (1 - t.loc[16.0, "sse"] / t.loc[8.0, "sse"]), "{:.0f}")
    # table vs direct fits
    tf = pd.read_csv(Q / "baselines/classical_table_fits.csv").set_index("Mix")
    cl = pd.read_csv(ROOT / "results/disc/classical_fits.csv").set_index("Mix").loc[tf.index]
    put("table_fit_dk", (tf.log10_K - cl.log10_K).abs().max(), "{:.3f}")
    put("table_fit_ddv", (tf.log10_Dv - cl.log10_Dv).abs().max(), "{:.4f}")
    # ---- model comparison
    P = pd.read_csv(Q / "baselines/pooled.csv").set_index(["scheme", "method"])
    for sch in ("holdout", "lomo", "lofo"):
        for m in ("pen", "pen_r", "pen_lin", "pen_s", "ols", "ridge", "quad", "gp", "mlp"):
            if (sch, m) in P.index:
                r = P.loc[(sch, m)]
                tag = {"holdout": "ho", "lomo": "lomo", "lofo": "lofo"}[sch]
                put(f"{tag}_{m}_rmse", r.rmse)
                put(f"{tag}_{m}_r2", r.r2)
                put(f"{tag}_{m}_min", r.r2_mix_min, "{:.3f}")
                put(f"{tag}_{m}_nbelow", r.n_below, "{:.0f}")
                if sch == "holdout" and np.isfinite(r.rmse_sd):
                    put(f"ho_{m}_rmse_sd", r.rmse_sd)
                    put(f"ho_{m}_r2_sd", r.r2_sd)
    flex = [P.loc[("lofo", m), "r2_mix_min"] for m in ("gp", "quad", "mlp")]
    phys = [P.loc[("lofo", m), "r2_mix_min"] for m in ("ols", "ridge", "pen", "pen_r", "pen_lin")]
    put("lofo_min_flex", min(flex), "{:.2f}")
    put("lofo_max_flex", max(flex), "{:.2f}")
    put("lofo_min_phys", min(phys), "{:.3f}")
    pa = pd.read_csv(Q / "audit/paired.csv")

    def pair(sch, a, b):
        return pa[(pa.scheme == sch) & (pa.a == a) & (pa.b == b)].iloc[0]
    for sch in ("lomo", "lofo"):
        for a, b in (("pen", "ols"), ("pen_r", "ols"), ("pen", "mlp"), ("ols", "mlp"), ("pen", "gp"), ("pen", "quad"),
                     ("pen", "pen_r"), ("ols", "gp"), ("ols", "quad")):
            r = pair(sch, a, b)
            k = f"pair_{sch}_{a}_{b}"
            put(k + "_d", r.mean_diff, "{:+.4f}")
            put(k + "_lo", r.ci_lo, "{:+.4f}")
            put(k + "_hi", r.ci_hi, "{:+.4f}")
            put(k + "_p", r.wilcoxon_p, "{:.2f}" if r.wilcoxon_p >= 0.01 else "{:.3f}")
            put(k + "_wins", r.a_better_in, "{:.0f}")
    # ---- PENN shrinkage (synthetic)
    sh = pd.read_csv(Q / "pen_shrinkage/summary.csv")
    g = sh[(sh.truth == "truth_lin") & (sh.split == "train")].set_index("variant")
    put("shr_slope_paper", g.loc["paper (2x16, 400 it, wd 1e-4)", "slope_K"], "{:.2f}")
    put("shr_slope_r", g.loc["linear, L-BFGS, wd 0 (revised)", "slope_K"], "{:.2f}")
    put("shr_slope_nowd", g.loc["2x16, 2000 it, wd 0", "slope_K"], "{:.2f}")
    put("shr_slope_lin_adam", g.loc["linear, 4000 it, wd 0, lr 0.05", "slope_K"], "{:.2f}")
    put("shr_slope_wd2000", g.loc["2x16, 2000 it, wd 1e-4", "slope_K"], "{:.2f}")
    gt = sh[(sh.truth == "truth_lin") & (sh.split == "test")].set_index("variant")
    put("shr_maek_paper", gt.loc["paper (2x16, 400 it, wd 1e-4)", "mae_K"], "{:.3f}")
    put("shr_maek_r", gt.loc["linear, L-BFGS, wd 0 (revised)", "mae_K"], "{:.3f}")
    put("shr_maek_nowd", gt.loc["2x16, 2000 it, wd 0", "mae_K"], "{:.3f}")
    gm = sh[(sh.truth == "truth_mix") & (sh.split == "test")].set_index("variant")
    put("shr_mix_maek_r", gm.loc["linear, L-BFGS, wd 0 (revised)", "mae_K"], "{:.3f}")
    put("shr_mix_maek_paper", gm.loc["paper (2x16, 400 it, wd 1e-4)", "mae_K"], "{:.3f}")
    # superseded exploratory variant (2x16, no weight decay) evaluated on the experimental data
    for line in (Q / "pen_shrinkage/nowd_2x16_experimental.txt").read_text().splitlines():
        p = line.split()
        if len(p) > 8 and p[1] == "pen_r":
            put(f"nowd_{p[0]}_rmse", float(p[4]))
            put(f"nowd_{p[0]}_min", float(p[7]), "{:.2f}")
    c2 = pd.read_csv(Q / "synthetic/summary_c2.csv")
    r = c2[(c2.truth == "truth_lin") & (c2.method == "classical+ols") & (c2.split == "test")].iloc[0]
    put("syn2_ols_maek", r.mae_K, "{:.3f}")
    r = c2[(c2.truth == "truth_lin") & (c2.method == "classical+ols") & (c2.split == "train")].iloc[0]
    put("syn2_ols_slope_r", r.r_K, "{:.2f}")
    r = c2[(c2.truth == "truth_mix") & (c2.method == "classical+ols") & (c2.split == "test")].iloc[0]
    put("syn2_mix_ols_maek", r.mae_K, "{:.3f}")
    put("syn2_pen_r_train", c2[(c2.truth == "truth_lin") & (c2.method == "pen") & (c2.split == "train")].r_K.iloc[0], "{:.2f}")
    # real-data parameter agreement, revised PENN vs classical
    th = pd.read_csv(ROOT / "results/disc/theta_all.csv").set_index("Mix")
    rr = pd.read_csv(Q / "baselines/pen_r_all_theta.csv").groupby("Mix")[["log10_Dv", "log10_K"]].mean().loc[th.index]
    put("thr_K_corr", np.corrcoef(rr.log10_K, th.log10_K_fd)[0, 1], "{:.2f}")
    put("thr_K_mae", (rr.log10_K - th.log10_K_fd).abs().mean(), "{:.3f}")
    put("thr_Dv_mae", (rr.log10_Dv - th.log10_Dv_fd).abs().mean(), "{:.3f}")
    put("thr_K_lo", rr.log10_K.min(), "{:.2f}")
    put("thr_K_hi", rr.log10_K.max(), "{:.2f}")
    I2 = pd.read_csv(Q / "profiles/intervals.csv").set_index("Mix").loc[th.index]
    inside = ((rr.log10_K >= I2.K_lo_eff - 1e-9) & (rr.log10_K <= I2.K_hi_eff + 1e-9))
    put("thr_K_inside", int(inside.sum()), "{:.0f}")
    inside0 = ((th.log10_K_pen >= I2.K_lo_eff) & (th.log10_K_pen <= I2.K_hi_eff))
    put("th_K_inside", int(inside0.sum()), "{:.0f}")
    put("pen_r_time", pd.read_csv(Q / "baselines/pen_r_all_theta.csv").train_time_s.mean(), "{:.0f}")
    # ---- data efficiency and truncation
    S = pd.read_csv(Q / "data_efficiency/summary.csv").set_index(["k", "method"])
    for k in (5, 8, 14):
        for m in ("pen", "pen_r", "ols", "ridge", "gp", "mlp"):
            if (k, m) in S.index:
                put(f"de{k}_{m}", S.loc[(k, m), "rmse_med"])
                put(f"de{k}_{m}_worst", S.loc[(k, m), "r2_worst_med"], "{:.2f}")
    T = pd.read_csv(Q / "data_efficiency/truncation.csv")
    T["c"] = T.cutoff_d.astype(str)
    for c, tag in (("3.0", "3"), ("4.0", "4"), ("7.0", "7"), ("14.0", "14"), ("full", "full")):
        for m in ("pen", "pen_r", "ols", "mlp"):
            r = T[(T.c == c) & (T.method == m)]
            if len(r):
                put(f"tr{tag}_{m}", r.rmse.iloc[0])
        r = T[(T.c == c) & (T.method == "ols")]
        put(f"tr{tag}_nfit", r.n_mix_fit.iloc[0], "{:.0f}")
        put(f"tr{tag}_npts", r.n_train_pts.iloc[0], "{:.0f}")
    put("trunc_c", "7")
    N["trunc_pen"], N["trunc_ols"] = N["tr7_pen"], N["tr7_ols"]
    # ---- robustness
    R = pd.read_csv(Q / "robustness/summary.csv").set_index(["variant", "method"])
    for v in R.index.get_level_values(0).unique():
        for m in ("pen", "pen_r", "ols", "mlp"):
            if (v, m) in R.index:
                tag = v.replace(".", "")
                put(f"rb_{tag}_{m}", R.loc[(v, m), "rmse"])
                if np.isfinite(R.loc[(v, m), "max_dK"]):
                    put(f"rb_{tag}_{m}_dk", R.loc[(v, m), "max_dK"], "{:.3f}")
                    put(f"rb_{tag}_{m}_ddv", R.loc[(v, m), "max_dDv"], "{:.3f}")
    put("rb_recon_dk_max", max(R.loc[(v, "ols"), "max_dK"] for v in ("shift_0.97", "shift_1.13", "shift_train")), "{:.3f}")
    put("rb_recon_ddv_max", max(R.loc[(v, "ols"), "max_dDv"] for v in ("shift_0.97", "shift_1.13", "shift_train")), "{:.3f}")
    # ---- inner CV
    ic = pd.read_csv(Q / "inner_cv/selected.csv").set_index("kind")
    put("icv_pen_rank", ic.loc["pen", "inner_rank_of_paper"], "{:.0f}")
    put("icv_pen_n", ic.loc["pen", "n_configs"], "{:.0f}")
    put("icv_pen_sel", ic.loc["pen", "sel_inner_rmse"], "{:.4f}")
    put("icv_pen_paper", ic.loc["pen", "paper_inner_rmse"], "{:.4f}")
    put("icv_pen_test", ic.loc["pen", "test_rmse_selected"], "{:.4f}")
    put("icv_mlp_sel", ic.loc["mlp", "sel_inner_rmse"], "{:.4f}")
    put("icv_mlp_paper", ic.loc["mlp", "paper_inner_rmse"], "{:.4f}")
    put("icv_mlp_test", ic.loc["mlp", "test_rmse_selected"], "{:.4f}")
    # ---- PINN diagnostics
    pn = pd.read_csv(Q / "pinn/runs.csv").set_index("run")
    for r in pn.index:
        put(f"pinn_{r}_bias", pn.loc[r, "bias_K_vs_classical"], "{:+.2f}")
        put(f"pinn_{r}_fwd", pn.loc[r, "fwd_vs_model_r2"], "{:.3f}")
        put(f"pinn_{r}_total", pn.loc[r, "loss_total"], "{:.3f}")
        put(f"pinn_{r}_data", pn.loc[r, "loss_data"], "{:.3f}")
        put(f"pinn_{r}_pde", pn.loc[r, "loss_pde"], "{:.4f}")
        put(f"pinn_{r}_pderms", pn.loc[r, "pde_rms"], "{:.3f}")
        put(f"pinn_{r}_rmse", pn.loc[r, "train_rmse"], "{:.4f}")
    put("pinn_fixed_fwd4", pn.loc["fixed_classical", "fwd_vs_model_r2"], "{:.4f}")
    # ---- physics sensitivity
    ps = Q / "physics_sensitivity/summary.csv"
    if ps.exists():
        S = pd.read_csv(ps).set_index("variant")
        for v in S.index:
            tag = v.replace(".", "")
            put(f"ps_{tag}_ddv", S.loc[v, "dDv_mean"], "{:+.2f}")
            put(f"ps_{tag}_dk", S.loc[v, "dK_mean"], "{:+.2f}")
            put(f"ps_{tag}_ratio", S.loc[v, "rmse_ratio"], "{:.2f}")
            put(f"ps_{tag}_rankk", S.loc[v, "rank_K"], "{:.2f}")
            put(f"ps_{tag}_rankdv", S.loc[v, "rank_Dv"], "{:.2f}")
        con = S.drop(index=[v for v in ("base", "mesh_36x24", "sealed_lateral") if v in S.index])
        put("sens_dk_max", con.dK_maxabs.max(), "{:.1f}")
        put("sens_ddv_max", con.dDv_maxabs.max(), "{:.2f}")
        put("sens_rank_min", min(con.rank_K.min(), con.rank_Dv.min()), "{:.2f}")
        put("sens_rankk_min", con.rank_K.min(), "{:.2f}")
        put("sens_rankdv_min", con.rank_Dv.min(), "{:.2f}")
        put("sens_mesh_dk", S.loc["mesh_36x24", "dK_maxabs"], "{:.3f}")
        put("sens_mesh_ddv", S.loc["mesh_36x24", "dDv_maxabs"], "{:.3f}")
    # ---- reproducibility
    rp = Q / "reproducibility/comparison.csv"
    if rp.exists():
        R = pd.read_csv(rp)
        put("repro_n", len(R), "{:.0f}")
        put("repro_ok", int((R.status == "reproduced").sum()), "{:.0f}")
    (Q / "q1_numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))
    print(len(N), "keys")


if __name__ == "__main__":
    main()
