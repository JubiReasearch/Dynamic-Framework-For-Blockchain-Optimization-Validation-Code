import sys,json;sys.path.insert(0,'.')
import numpy as np
from model import *
REF=np.ones(4)
def dist(A,B): return np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1))
def spacing(G):
    if len(G)<3: return float('nan')
    d=np.abs(G[:,None,:]-G[None,:,:]).sum(-1); np.fill_diagonal(d,np.inf); m=d.min(1)
    return float(m.std(ddof=1)/m.mean())
rows=[]
for seed in range(30):
    rng=np.random.default_rng(1000+seed)
    env=(rng.uniform(.3,.8),rng.uniform(.1,.5),rng.uniform(.05,.35))
    Xn,Gn=nsga2(env,80,30,np.random.default_rng(seed))
    Xw,Gw=weighted_sum(env,128,np.random.default_rng(seed))
    Xb,Gb=biobjective(env,80,30,np.random.default_rng(seed))
    Xr,Gr=nsga2(env,200,120,np.random.default_rng(10_000+seed))
    pool=np.vstack([Gr,Gn,Gw,Gb]); ref=pool[nondominated(pool)]
    r={'seed':seed,'env':env}
    for name,G in (('nsga2',Gn),('weighted_sum',Gw),('bi_objective',Gb)):
        nd=G[nondominated(G)]
        dom=dominates_matrix(np.vstack([ref,G]))[:len(ref),len(ref):].any(0)
        r[name]=dict(n=len(G),n_nd=int(len(nd)),hv=hv(nd,REF),gd=float(dist(nd,ref).min(1).mean()),igd=float(dist(ref,nd).min(1).mean()),
                     spacing=spacing(nd),dominated_by_ref=float(dom.mean()))
    r['ref_n']=int(len(ref))
    rows.append(r); print(seed,flush=True)
json.dump(rows,open('pareto_raw.json','w'))
def col(m,k):return np.array([r[m][k] for r in rows])
out={'n_runs':30,'methods':{}}
for m in ('nsga2','weighted_sum','bi_objective'):
    out['methods'][m]={k:[float(col(m,k).mean()),float(col(m,k).std(ddof=1))] for k in ('n','n_nd','hv','gd','igd','spacing','dominated_by_ref')}
out['ref_n']=[float(np.mean([r['ref_n'] for r in rows]))]
out['tests']={}
for other in ('weighted_sum','bi_objective'):
    for k in ('hv','igd','gd'):
        out['tests'][f'{k}:nsga2-vs-{other}']=paired_report(col('nsga2',k),col(other,k))
json.dump(out,open('pareto.json','w'),indent=1)
print(json.dumps(out['methods'],indent=0)[:1800]);
for k,v in out['tests'].items():print(k,{a:round(b,4) if isinstance(b,float) else b for a,b in v.items() if a in('t','p','dz','mean_diff','shapiro_p','wilcoxon_p')},[round(c,4) for c in v['ci']])
