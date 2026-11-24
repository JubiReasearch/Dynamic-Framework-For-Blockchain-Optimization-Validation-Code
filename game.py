import sys,json;sys.path.insert(0,'.')
import numpy as np
from scipy import stats
from model import objectives,paired_report
g=np.linspace(0,1,36);XX=np.array(np.meshgrid(g,g,g,indexing='ij')).reshape(3,-1).T
CC=0.6
TEMP=0.02
def payoff_grid(env,n):
    F=objectives(XX,env); s=np.linspace(0,1,n)
    U1=np.zeros((n,n));U2=np.zeros((n,n));K=np.zeros((n,n),int)
    for i,ws in enumerate(s):
        W=np.array([[.25,ws,wc,.25] for wc in s]);W=W/W.sum(1,keepdims=True)
        Z=(F@W.T)/TEMP;Z=Z-Z.max(0);P=np.exp(Z);P/=P.sum(0);Fs=P.T@F     # entropy-regularised (softmax) choice of operating point
        U1[i]=Fs[:,1]+.5*Fs[:,0]-CC*ws**2;U2[i]=Fs[:,2]+.5*Fs[:,3]-CC*s**2
    return s,U1,U2
def pure_ne(U1,U2):
    br1=U1.argmax(0)            # for each wc column, best ws index
    br2=U2.argmax(1)            # for each ws row, best wc index
    return [(i,j) for j in range(U1.shape[1]) for i in [br1[j]] if br2[i]==j],br1,br2
out={}
env0=(.55,.2,.15)
s,U1,U2=payoff_grid(env0,101)
ne,br1,br2=pure_ne(U1,U2)
out['main']={'env':env0,'n_ne':len(ne),'ne':[(float(s[i]),float(s[j])) for i,j in ne]}
i,j=ne[0]
out['main']['payoffs_at_ne']=[float(U1[i,j]),float(U2[i,j])]
# unilateral deviation gain on the grid (should be <= 0)
out['main']['max_dev_gain']=[float(max(0,(U1[:,j]-U1[i,j]).max())),float(max(0,(U2[i,:]-U2[i,j]).max()))]
out['main']['payoff_at_corners']={'(0,0)':[float(U1[0,0]),float(U2[0,0])],'(1,1)':[float(U1[-1,-1]),float(U2[-1,-1])]}
out['main']['payoff_range']=[float(min(U1.min(),U2.min())),float(max(U1.max(),U2.max()))]
# concavity along own strategy (unimodality check)
def unimodal(v):
    d=np.sign(np.diff(v));d=d[d!=0];return int((np.diff(d)!=0).sum()<=1)
out['main']['unimodal_rows']=float(np.mean([unimodal(U1[:,k]) for k in range(101)]+[unimodal(U2[k,:]) for k in range(101)]))
# slopes of best responses (contraction): BR1(wc)=s[br1], BR2(ws)=s[br2]
b1=s[br1];b2=s[br2]
def wslope(b):
    k=10;out_=np.zeros(len(b))
    for m in range(len(b)):
        lo,hi=max(0,m-k),min(len(b)-1,m+k);out_[m]=abs(b[hi]-b[lo])/(s[hi]-s[lo])
    return out_
sl1=wslope(b1);sl2=wslope(b2)
out['main']['br_slope_at_ne']=[float(sl1[j]),float(sl2[i])];out['main']['contraction_factor']=float(sl1[j]*sl2[i])
out['main']['br_slope_max']=[float(sl1.max()),float(sl2.max())]
# best-response dynamics from 30 random starts (sequential), steps to a fixed point
rng=np.random.default_rng(3);steps=[];dist=[]
for k in range(30):
    a,b=rng.integers(0,101,2)
    for it in range(1,500):
        a2=br1[b];b2_=br2[a2]
        if a2==a and b2_==b:break
        a,b=a2,b2_
    steps.append(it);dist.append(float(np.hypot(s[a]-s[i],s[b]-s[j])))
out['main']['brd_steps']=[float(np.mean(steps)),float(np.std(steps,ddof=1)),int(max(steps))]
out['main']['brd_final_dist_max']=float(max(dist));out['main']['brd_converged_to_ne']=int(sum(d<1e-9 for d in dist))
# 30 environments
rows=[]
for k in range(30):
    rng=np.random.default_rng(1000+k);env=(rng.uniform(.3,.8),rng.uniform(.1,.5),rng.uniform(.05,.35))
    s2,V1,V2=payoff_grid(env,51);ne2,_,_=pure_ne(V1,V2)
    G1=V1.max(0,keepdims=True)-V1;G2=V2.max(1,keepdims=True)-V2;EPS=np.maximum(G1,G2)
    a,b=np.unravel_index(EPS.argmin(),EPS.shape);eps=float(EPS[a,b])
    rows.append(dict(env=env,eps=eps,n_ne=len(ne2),ne=(float(s2[a]),float(s2[b])),u_ne=[float(V1[a,b]),float(V2[a,b])],
        u00=[float(V1[0,0]),float(V2[0,0])],u11=[float(V1[-1,-1]),float(V2[-1,-1])]))
json.dump(rows,open('game_raw.json','w'))
out['envs']={'n':30,'exact_grid_ne_share':float(np.mean([r['n_ne']>=1 for r in rows])),'eps_max':float(max(r['eps'] for r in rows)),'eps_mean':float(np.mean([r['eps'] for r in rows])),
  'ne_ws':[float(np.mean([r['ne'][0] for r in rows])),float(np.std([r['ne'][0] for r in rows],ddof=1))],
  'ne_wc':[float(np.mean([r['ne'][1] for r in rows])),float(np.std([r['ne'][1] for r in rows],ddof=1))]}
al=[r['env'][1] for r in rows];lam=[r['env'][0] for r in rows]
rs=stats.spearmanr(al,[r['ne'][0] for r in rows]);rc=stats.spearmanr(lam,[r['ne'][1] for r in rows])
out['envs']['spearman_alpha_ws']=[float(rs.statistic),float(rs.pvalue)];out['envs']['spearman_lambda_wc']=[float(rc.statistic),float(rc.pvalue)]
out['tests']={}
for nm,key in (('corner_00','u00'),('corner_11','u11')):
    for ag in (0,1):
        out['tests'][f'agent{ag+1}_NE_vs_{nm}']=paired_report([r['u_ne'][ag] for r in rows],[r[key][ag] for r in rows])
    out['tests'][f'joint_NE_vs_{nm}']=paired_report([sum(r['u_ne']) for r in rows],[sum(r[key]) for r in rows])
# consensus update on interoperable chains
cs=[];ratio=[]
for k in range(30):
    rng=np.random.default_rng(2000+k);n=5
    A=(rng.random((n,n))<.5).astype(float);A=np.triu(A,1);A=A+A.T
    for i in range(n-1):A[i,i+1]=A[i+1,i]=1                # connected path backbone
    L=np.diag(A.sum(1))-A;mu=0.9/np.linalg.eigvalsh(L).max()
    P=np.eye(n)-mu*L;lam2=np.sort(np.abs(np.linalg.eigvals(P)))[-2]
    th=rng.random((n,3));mean=th.mean(0);e0=np.linalg.norm(th-mean)
    it=0
    while np.linalg.norm(th-mean)>1e-8*e0 and it<20000:th=P@th;it+=1
    cs.append(it);ratio.append(float(np.log(1e-8)/np.log(lam2)/it))
    assert np.allclose(th,mean,atol=1e-6*max(1,e0))
out['consensus']={'graphs':30,'chains':5,'iters':[float(np.mean(cs)),float(np.std(cs,ddof=1))],'theory_over_observed':[float(np.mean(ratio)),float(np.std(ratio,ddof=1))],'all_converged_to_average':True}
# consensus ADMM (Boyd): min sum 0.5 (x_i - c_i)^2 s.t. x_i = z
its=[]
for k in range(30):
    rng=np.random.default_rng(3000+k);n=5;c=rng.random(n)*10;rho=1.0;x=np.zeros(n);u=np.zeros(n);z=0.0
    for it in range(1,5000):
        x=(c+rho*(z-u))/(1+rho);zn=(x+u).mean();u=u+x-zn
        r=np.linalg.norm(x-zn);sd=rho*np.sqrt(n)*abs(zn-z);z=zn
        if r<1e-6 and sd<1e-6:break
    assert abs(z-c.mean())<1e-5
    its.append(it)
out['admm']={'runs':30,'agents':5,'rho':1.0,'iters':[float(np.mean(its)),float(np.std(its,ddof=1))],'max_iters':int(max(its)),'converged_to_global_optimum':True}
json.dump(out,open('game.json','w'),indent=1)
print(json.dumps({k:out[k] for k in ('main','envs','consensus','admm')},indent=0)[:2500])
for k,v in out['tests'].items():print(k,{a:round(b,4) for a,b in v.items() if a in('t','p','dz','mean_diff')})
