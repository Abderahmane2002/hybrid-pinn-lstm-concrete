"""Are the two digitised series of a mix the same curve (time offset) or two different
curves (e.g. two curing ages, which would differ by a time *scaling* for diffusion)?

For each mix with both series we fit, over the overlapping saturation range,
  shift : A(t) = B(t - d)
  scale : A(t) = B(t / l)
  affine: A(t) = B((t - d) / l)
and report the rms misfit in time and in saturation.
Output: results/q1/series_relation.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q1"; OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data/raw/tidy_dataset_correctedV1.csv")
df["block"] = (df.Mix != df.Mix.shift()).cumsum()
df["series"] = np.where(df.groupby("block").Time_d.transform("max") <= 20, "B", "A")
rows = []
for mix, g in df.groupby("Mix"):
    if set(g.series) != {"A", "B"} or g.block.nunique() != 2:
        continue
    a = g[g.series == "A"].sort_values("Time_d"); b = g[g.series == "B"].sort_values("Time_d")
    ta, sa = a.Time_d.to_numpy(), np.minimum.accumulate(a.S_frac.to_numpy())
    tb, sb = b.Time_d.to_numpy(), b.S_frac.to_numpy()
    # keep B points that are monotone (drop the reversed blocks of the corrupted w/b = 0.6 series)
    ok = (sb <= sa.max()) & (sb >= sa.min()) & (np.r_[True, np.diff(sb) <= 0.02])
    tb, sb = tb[ok], sb[ok]
    if len(tb) < 6:
        continue
    def model(p, kind):
        d, l = (p[0], 1.0) if kind == "shift" else (0.0, p[0]) if kind == "scale" else (p[0], p[1])
        return np.interp(-sb, -sa, ta), tb * l + d      # time at which A reaches S_B, vs predicted
    res = {}
    for kind, p0 in (("shift", [1.0]), ("scale", [1.5]), ("affine", [1.0, 1.0])):
        f = lambda p: np.subtract(*model(p, kind))
        r = least_squares(f, p0)
        tA_obs, tA_mod = model(r.x, kind)
        # saturation misfit: A evaluated at predicted times vs B
        s_mod = np.interp(tA_mod, ta, sa)
        res[kind] = (r.x, np.sqrt(np.mean(f(r.x) ** 2)), np.sqrt(np.mean((s_mod - sb) ** 2)))
    rows.append(dict(Mix=mix, wb=g["w/b"].iloc[0], n=len(tb),
                     shift_d=res["shift"][0][0], shift_rms_t=res["shift"][1], shift_rms_S=res["shift"][2],
                     scale_l=res["scale"][0][0], scale_rms_t=res["scale"][1], scale_rms_S=res["scale"][2],
                     affine_d=res["affine"][0][0], affine_l=res["affine"][0][1],
                     affine_rms_t=res["affine"][1], affine_rms_S=res["affine"][2]))
out = pd.DataFrame(rows).sort_values(["wb", "Mix"])
out.to_csv(OUT / "series_relation.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
c = out[out.wb < 0.6]
print("\nw/b <= 0.5: median rms(S) shift %.4f  scale %.4f  affine %.4f; affine scale factor median %.3f (IQR %.3f-%.3f)"
      % (c.shift_rms_S.median(), c.scale_rms_S.median(), c.affine_rms_S.median(),
         c.affine_l.median(), c.affine_l.quantile(.25), c.affine_l.quantile(.75)))
