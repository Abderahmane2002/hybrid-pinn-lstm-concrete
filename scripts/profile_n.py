"""Profile the shared shape exponent n: fit (D_v, K) per mix for several fixed n."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
from pinn_moisture.physics import PhysicsConfig
from pinn_moisture.inverse_fd import fit_mix
d=pd.read_csv(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "data/processed/drying_clean.csv"))
mixes=['C-0.3','C-0.5','F40-0.4','S20-0.4','F10S10-0.5','F20S20-0.3']
def run(a):
    mix,n=a; g=d[d.Mix==mix]
    cfg=PhysicsConfig(identified=("log10_Dv","log10_K")); cfg.defaults=dict(cfg.defaults,n=n)
    r=fit_mix(g.Time_d.values,g.S_frac.values,cfg,n_starts=2); return mix,n,r['rmse'],r['r2'],len(g)
res=[]
with Pool(2) as p:
    for out in p.imap(run,[(m,n) for n in [1.5,2.0,3.0,4.0,6.0] for m in mixes]): res.append(out); print(out,flush=True)
df=pd.DataFrame(res,columns=['mix','n','rmse','r2','N']); df['sse']=df.rmse**2*df.N
print(df.groupby('n').agg(sse=('sse','sum'),r2_mean=('r2','mean'),r2_min=('r2','min')))
