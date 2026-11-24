import sys,json;sys.path.insert(0,'.')
import numpy as np
from scipy.optimize import minimize
from model import *
W=np.array([.25,.25,.25,.25]); GAMMA=0.55; ETA=0.08; NP=10; NC=5; T=60
def U4(x,env): return float(objectives(x,env)[0]@W)
def env_path(rng):
    lam=np.zeros(T);al=np.zeros(T);ch=np.zeros(T);l,a,c=.55,.2,.15
    for t in range(T):
        l=.55+.8*(l-.55)+rng.normal(0,.04);a=.2+.8*(a-.2)+rng.normal(0,.03);c=.15+.8*(c-.15)+rng.normal(0,.02)
        lam[t],al[t],ch[t]=l,a,c
    for _ in range(2):                    # attack episodes and demand spikes
        s=rng.integers(5,T-20);al[s:s+15]=np.maximum(al[s:s+15],.7)
        s=rng.integers(5,T-20);lam[s:s+15]=np.maximum(lam[s:s+15],.9)
    return np.clip(lam,0,1),np.clip(al,0,.95),np.clip(ch,0,.9),
def static_design():
    rng=np.random.default_rng(7);envs=[(rng.uniform(.3,.9),rng.uniform(.1,.7),rng.uniform(.05,.35)) for _ in range(300)]
    f=lambda x:-np.mean([U4(x,e) for e in envs]); best=None
    for k in range(8):
        r=minimize(f,np.random.default_rng(k).random(3),bounds=[(0,1)]*3,method='SLSQP')
        if best is None or r.fun<best.fun:best=r
    return best.x
def mpc_step(x,env):
    def unroll(u):
        U=u.reshape(NC,3);xs=[];cur=x.copy()
        for k in range(NP):
            cur=cur+(U[k] if k<NC else 0);xs.append(cur.copy())
        return np.array(xs)
    def cost(u):
        xs=unroll(u);return -sum(U4(p,env) for p in xs)+0.5*np.sum(u**2)
    cons=[{'type':'ineq','fun':lambda u:np.array([objectives(p,env)[0][1]-GAMMA for p in unroll(u)])},
          {'type':'ineq','fun':lambda u:np.concatenate([unroll(u).ravel(),1-unroll(u).ravel()])}]
    r=minimize(cost,np.zeros(NC*3),bounds=[(-ETA,ETA)]*(NC*3),constraints=cons,method='SLSQP',options={'maxiter':40})
    step=np.clip(r.x[:3],-ETA,ETA);return np.clip(x+step,0,1)
xs0=static_design();print('static design',xs0,flush=True)
def run(seed):
    rng=np.random.default_rng(500+seed);lam,al,ch=env_path(rng)
    xm=xs0.copy();rec={'static':[],'mpc':[]}
    for t in range(T):
        env=(lam[t],al[t],ch[t]);xm=mpc_step(xm,env)
        rec['mpc'].append(objectives(xm,env)[0]);rec['static'].append(objectives(xs0,env)[0])
    r={'seed':seed}
    for m in rec:
        A=np.array(rec[m]);r[m]=dict(mean=A.mean(0).tolist(),U4=float(A.mean(0)@W),viol=float((A[:,1]<GAMMA).mean()),
             atk_S=float(A[al>=.65,1].mean()),spike_C=float(A[lam>=.85,2].mean()))
    return r
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(2) as p: rows=p.map(run,range(30))
json.dump(rows,open('mpc_raw.json','w'))
out={'static_x':xs0.tolist(),'T':T,'Np':NP,'Nc':NC,'eta':ETA,'gamma':GAMMA,'weights':W.tolist(),'tests':{}}
get=lambda m,f:np.array([f(r[m]) for r in rows])
out['tests']['U4']=paired_report(get('mpc',lambda d:d['U4']),get('static',lambda d:d['U4']))
for i,nm in enumerate(('D','S','C','E')):
    out['tests'][nm]=paired_report(get('mpc',lambda d:d['mean'][i]),get('static',lambda d:d['mean'][i]))
out['tests']['attack_S']=paired_report(get('mpc',lambda d:d['atk_S']),get('static',lambda d:d['atk_S']))
out['tests']['spike_C']=paired_report(get('mpc',lambda d:d['spike_C']),get('static',lambda d:d['spike_C']))
out['tests']['violation']=paired_report(get('mpc',lambda d:d['viol']),get('static',lambda d:d['viol']))
out['viol_mean']={m:float(get(m,lambda d:d['viol']).mean()) for m in ('mpc','static')}
json.dump(out,open('mpc.json','w'),indent=1)
