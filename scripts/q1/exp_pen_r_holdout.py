"""Revised PENN (linear composition map, no weight decay, L-BFGS) on the hold-out split, 5 seeds:
parameters of all 20 mixes and per-mix accuracy on the withheld ones, for the parameter
figure and the per-mix table. Output: results/q1/baselines/pen_r_all_theta.csv"""
import pandas as pd
import torch
from common import TEST_MIXES, disc_cfg, load_data, mix_table, out_dir
from pinn_moisture.diff_solver import PENTrainer
torch.set_num_threads(1)
d = load_data()
train = d[~d.Mix.isin(TEST_MIXES)]
rows = []
for s in range(5):
    tr = PENTrainer(train, disc_cfg(), seed=s, iters=500, weight_decay=0.0, hidden=1, layers=0).fit(verbose=False, optimizer="lbfgs")
    rows.append(tr.predict_theta(mix_table(d)).assign(seed=s, train_time_s=tr.train_time))
    print("seed", s, f"{tr.train_time:.1f} s", flush=True)
pd.concat(rows).to_csv(out_dir("baselines") / "pen_r_all_theta.csv", index=False)
