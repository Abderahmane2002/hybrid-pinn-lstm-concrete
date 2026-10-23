"""Which curing age (28 or 56 days) do the digitised drying curves correspond to?

Zhuang et al. (2022) give two quantitative statements about the time needed to dry the
vacuum-saturated discs to 70 % saturation (t70):
  * 28-day curing: t70 = 1.5, 1.0, 0.8 and 0.6 d for w/b = 0.3, 0.4, 0.5, 0.6, and "the type
    and the content of mineral admixture have little effect";
  * 56-day curing: "both mineral admixture and W/B ratio have great effects", with "a drying time
    difference of 2-4 d between concrete with the same W/B ratio but different types or contents
    of mineral admixtures".
For every reliable mix t70 is computed from the fitted disc model (the measured curves start
near 70 % saturation) under the two possible time origins of the digitised data: the origin of
the short series (used in the paper) and that of the long series (+1.05 d). The spread of t70
between binders at the same w/b does not depend on the origin.
Output: results/q1/curing_age/t70.csv, summary.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ROOT, forward_table, out_dir

ZHUANG_28 = {0.3: 1.5, 0.4: 1.0, 0.5: 0.8}


def main():
    out = out_dir("curing_age")
    cl = pd.read_csv(ROOT / "results/disc/classical_fits.csv")
    shift = 1.05
    t = np.geomspace(0.05, 20, 4000)
    rows = []
    for r in cl.itertuples():
        s = forward_table(np.array([[r.log10_Dv, r.log10_K]]), [t])[0]
        t70 = float(np.interp(-0.70, -s, t))
        rows.append({"Mix": r.Mix, "wb": float(r.Mix.split("-")[1]), "binder": r.Mix.rsplit("-", 1)[0],
                     "t70_short_origin": t70, "t70_long_origin": t70 + shift})
    T = pd.DataFrame(rows)
    T.to_csv(out / "t70.csv", index=False)
    S = T.groupby("wb").agg(t70_mean=("t70_short_origin", "mean"), t70_min=("t70_short_origin", "min"),
                            t70_max=("t70_short_origin", "max"), n=("Mix", "size")).reset_index()
    S["spread_d"] = S.t70_max - S.t70_min
    S["t70_mean_long_origin"] = S.t70_mean + shift
    S["zhuang_28d"] = S.wb.map(ZHUANG_28)
    S.to_csv(out / "summary.csv", index=False)
    res = {"max_spread_d": float(S.spread_d.max()), "shift_d": shift,
           "mean_abs_dev_short_origin": float((S.t70_mean - S.zhuang_28d).abs().mean()),
           "mean_abs_dev_long_origin": float((S.t70_mean_long_origin - S.zhuang_28d).abs().mean())}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    pd.set_option("display.width", 200)
    print(T.sort_values(["wb", "t70_short_origin"]).round(2).to_string(index=False))
    print(S.round(2).to_string(index=False))
    print(res)


if __name__ == "__main__":
    main()
