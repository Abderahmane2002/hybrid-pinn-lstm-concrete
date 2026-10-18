"""Build the cleaned drying dataset used in the paper.

Input : data/raw/tidy_dataset_correctedV1.csv  (curves digitised from Zhuang et al., 2022)
Output: data/processed/drying_clean.csv
        data/processed/cleaning_report.md

What the raw file contains
--------------------------
For most mixes the raw file holds two digitisations of the *same* drying curve:

* series "B": daily points between ~1 and 14 days;
* series "A": the full curve up to 40-100 days, whose time axis is offset by
  about +1.1 days (A(t + 1.1 d) == B(t) within digitisation noise).

The B series is consistent with drying starting at t = 0 (1 - S grows like
sqrt(t)); the A series is not. We therefore subtract the offset from A, keep B
for t <= 14 d and A only for t > 14 d, so that every mix is one curve without
duplicated points.

For w/b = 0.6 the two series are *not* consistent (A of S20-0.6 and S40-0.6 are
identical copies, the B series of F20S20-0.6 mixes two curves, and the
estimated offsets are erratic). Until these curves are re-digitised, only the
clean B series (0-14 d) are kept for w/b = 0.6.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B_MAX_T = 20.0          # a series ending before 20 d is a "B" (short, daily) series
MAX_SHIFT_RMS = 0.30    # [d] accept the A-vs-B time alignment only below this rms


def split_series(df: pd.DataFrame) -> pd.DataFrame:
    """Label contiguous raw blocks of the same mix as separate series."""
    df = df.copy()
    df["raw_row"] = np.arange(len(df))
    df["block"] = (df["Mix"] != df["Mix"].shift()).cumsum()
    t_end = df.groupby("block")["Time_d"].transform("max")
    df["series"] = np.where(t_end <= B_MAX_T, "B", "A")
    return df


def estimate_shift(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, int]:
    """Time shift so that A(t_B + shift) = B(t_B); returns (shift, rms, n)."""
    a = a.sort_values("Time_d")
    b = b.sort_values("Time_d")
    s_env = np.minimum.accumulate(a["S_frac"].to_numpy())   # monotone envelope
    t_a = a["Time_d"].to_numpy()
    inside = (b["S_frac"] <= s_env.max()) & (b["S_frac"] >= s_env.min())
    t_b = b.loc[inside, "Time_d"].to_numpy()
    s_b = b.loc[inside, "S_frac"].to_numpy()
    if len(t_b) < 4:
        return np.nan, np.nan, len(t_b)
    t_match = np.interp(-s_b, -s_env, t_a)
    shift = float(np.median(t_match - t_b))
    rms = float(np.sqrt(np.mean((t_match - t_b - shift) ** 2)))
    return shift, rms, len(t_b)


def main(raw: Path, out_dir: Path) -> None:
    df = split_series(pd.read_csv(raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    composition = df.groupby("Mix")[["w/b", "Water", "Cement", "Fly_ash", "Slag"]].first()

    report = ["# Data cleaning report", "",
              f"Raw file: `{raw.relative_to(ROOT)}` ({len(df)} rows, {df['Mix'].nunique()} mixes).", ""]

    # 1. duplicated series between different mixes
    report += ["## 1. Series copied between mixes", ""]
    sig = df.groupby("block").apply(lambda g: tuple(np.round(g["S_frac"].to_numpy(), 5)))
    dup_blocks = set()
    for s, blocks in sig.groupby(sig).groups.items():
        if len(blocks) > 1:
            mixes = [df.loc[df.block == b, "Mix"].iloc[0] for b in blocks]
            report.append(f"- identical series in {', '.join(mixes)} (raw blocks {list(blocks)}) -> excluded")
            dup_blocks.update(blocks)
    report.append("")

    # 2. time offset between the A and B digitisations
    report += ["## 2. Offset between the long (A) and short (B) digitisations", "",
               "| Mix | shift A-B [d] | rms [d] | n | decision |", "|---|---|---|---|---|"]
    shifts = {}
    for mix, g in df.groupby("Mix"):
        a, b = g[g.series == "A"], g[g.series == "B"]
        if len(a) and len(b):
            sh, rms, n = estimate_shift(a, b)
            ok = np.isfinite(sh) and rms < MAX_SHIFT_RMS and composition.loc[mix, "w/b"] < 0.55
            shifts[mix] = (sh, rms, ok)
            report.append(f"| {mix} | {sh:+.2f} | {rms:.2f} | {n} | {'consistent' if ok else 'inconsistent'} |")
    good = [v[0] for v in shifts.values() if v[2]]
    global_shift = float(np.median(good))
    report += ["", f"Median shift over consistent mixes: **{global_shift:+.2f} d** "
               f"(IQR {np.percentile(good, 25):+.2f} to {np.percentile(good, 75):+.2f} d). "
               "It is subtracted from every A series.", ""]

    # 3. assemble one curve per mix
    rows, decisions = [], []
    for mix, g in df.groupby("Mix"):
        wb = composition.loc[mix, "w/b"]
        a = g[(g.series == "A") & ~g.block.isin(dup_blocks)].copy()
        b = g[g.series == "B"].copy()
        a["Time_d"] = a["Time_d"] - global_shift
        mixed_b = len(b) and (np.sum(np.diff(b.sort_values("Time_d")["S_frac"].to_numpy()) > 0.01) > 1)
        if mixed_b:
            b = b.iloc[0:0]
        if wb >= 0.55:
            parts = [b]
            note = "w/b = 0.6: A series unreliable, kept B (0-14 d) only" if len(b) else \
                   "w/b = 0.6: no reliable series, mix excluded"
        else:
            parts = [b, a[a.Time_d > (b.Time_d.max() if len(b) else 0.0)]]
            note = "B (<= 14 d) + time-corrected A (> 14 d)" if len(b) else "time-corrected A only"
        if mixed_b:
            note += "; B series mixes two curves -> dropped"
        cur = pd.concat(parts).sort_values("Time_d")
        cur = cur[cur.Time_d > 0]
        decisions.append(f"| {mix} | {len(g)} | {len(cur)} | {note} |")
        rows.append(cur)
    clean = pd.concat(rows)
    clean = clean[["Mix", "Time_d", "S_frac", "w/b", "Water", "Cement", "Fly_ash", "Slag", "series", "raw_row"]]
    clean = clean.rename(columns={"w/b": "wb"})
    binder = clean["Cement"] + clean["Fly_ash"] + clean["Slag"]
    clean["FA_frac"] = clean["Fly_ash"] / binder
    clean["SL_frac"] = clean["Slag"] / binder
    # w/b = 0.6 short series are kept for reference but flagged: their early drying is slower
    # than at w/b = 0.5 and the classical fit drives K_int to its lower bound (inconsistent data)
    clean["reliable"] = clean["wb"] < 0.55
    clean = clean.sort_values(["Mix", "Time_d"]).reset_index(drop=True)
    clean.to_csv(out_dir / "drying_clean.csv", index=False)

    report += ["## 3. Final curve per mix", "", "| Mix | raw rows | kept | decision |", "|---|---|---|---|"] + decisions
    expected = {f"{p}-{w}" for p in ["C", "F20", "F40", "S20", "S40", "F10S10", "F20S20"] for w in ["0.3", "0.4", "0.5", "0.6"]}
    missing = sorted(expected - set(clean["Mix"]))
    report += ["", f"**Result:** {len(clean)} points, {clean['Mix'].nunique()} mixes. "
               f"Missing from the raw file or excluded: {', '.join(missing) if missing else 'none'}.",
               "", "**Action for the authors:** re-digitise the w/b = 0.6 curves (and S20-0.5) from the figures of "
               "Zhuang et al. (2022), or obtain the raw mass-loss data from the authors, then re-run this script."]
    (out_dir / "cleaning_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/tidy_dataset_correctedV1.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed")
    args = ap.parse_args()
    main(args.raw, args.out)
