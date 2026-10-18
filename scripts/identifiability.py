"""Identifiability study: fit 2, 3 and 4 parameters per mix with the FD solver and report correlations."""
import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
from pinn_moisture.physics import PhysicsConfig
from pinn_moisture.inverse_fd import fit_mix
d=pd.read_csv(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "data/processed/drying_clean.csv"))
mixes=['C-0.4','F20-0.3','S40-0.5','F20S20-0.5']
def run(args):
    mix,ident=args
    g=d[d.Mix==mix]; cfg=PhysicsConfig(identified=ident); cfg.defaults=dict(cfg.defaults, n=6.0)  # value used in the study
    r=fit_mix(g.Time_d.values,g.S_frac.values,cfg,n_starts=3)
    return mix,ident,r
jobs=[(m,i) for m in mixes for i in [("log10_Dv","alpha0","log10_K"),("log10_Dv","alpha0","n","log10_K"),("log10_Dv","log10_K")]]
with Pool(2) as p:
    for mix,ident,r in p.imap(run,jobs):
        print(mix,len(ident),'R2=%.4f rmse=%.4f cond=%.1e t=%.0fs'%(r['r2'],r['rmse'],r['cond'],r['time_s']),
              {k:round(v,3) for k,v in r['theta'].items()},'se',{k:round(v,3) for k,v in r['se'].items()},'bound',r['at_bound'])
        print('   corr',np.round(np.array(r['corr']),2).tolist())
