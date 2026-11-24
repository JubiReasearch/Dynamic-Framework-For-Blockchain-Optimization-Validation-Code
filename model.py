import numpy as np
from scipy import stats
# ---- Quadrilemma instantiation (maximisation, all objectives in [0,1]) ----
# x = (b, v, tau) in [0,1]^3 : block size, validator-set size, block interval (normalised)
# env = (lam, alpha, chi)    : load, adversarial activity, validator churn
def objectives(X, env):
    X = np.clip(np.atleast_2d(X), 0, 1); lam, al, chi = env
    b, v, t = X[:, 0], X[:, 1], X[:, 2]
    D = (1 - 0.5 * chi) * (0.05 + 0.95 * v) ** 0.7 * (1 - 0.5 * b)
    S = 1 - al * np.exp(-2.5 * (0.45 * t + 0.55 * v))
    cap = 1.4 * (0.02 + 0.98 * b) ** 0.6 * (1 - 0.75 * t) * (1 - 0.3 * v)   # capacity relative to demand
    C = np.minimum(1.0, cap / (0.25 + 0.9 * lam))
    en = 0.45 * v + 0.55 * b * (1 - t)          # normalised energy per block e/e_max
    E = 1 - 0.5 * en - 0.5 * (0.8 * en)          # beta = 0.8 e, rho1 = rho2 = 0.5
    return np.stack([D, S, C, E], axis=1)

def dominates_matrix(G):
    # G minimisation; returns boolean matrix dom[i,j] = i dominates j
    le = (G[:, None, :] <= G[None, :, :]).all(-1)
    lt = (G[:, None, :] < G[None, :, :]).any(-1)
    return le & lt

def nondominated(G):
    dom = dominates_matrix(G)
    return ~dom.any(0)

def hv(P, ref):
    P = np.asarray(P); d = P.shape[1]
    if len(P) == 0: return 0.0
    if d == 1: return float(ref[0] - P[:, 0].min())
    if d == 2:
        P = P[np.argsort(P[:, 0])]; vol = 0.0; best = ref[1]
        for x, y in P:
            if y < best:
                vol += (ref[0] - x) * (best - y); best = y
        return float(vol)
    P = P[np.argsort(P[:, -1])]; vol = 0.0
    for i in range(len(P)):
        top = P[i + 1, -1] if i + 1 < len(P) else ref[-1]
        h = top - P[i, -1]
        if h <= 0: continue
        sub = P[:i + 1, :-1]
        sub = sub[nondominated(sub)]
        vol += h * hv(sub, ref[:-1])
    return float(vol)

def nsga2(env, pop=80, gens=30, rng=None, nvar=3, cols=None):
    rng = rng or np.random.default_rng(0)
    def ev(X):
        F = objectives(X, env)
        return 1 - (F if cols is None else F[:, cols])
    X = rng.random((pop, nvar)); G = ev(X)
    def rank_crowd(G):
        n = len(G); rank = np.zeros(n, int); remaining = np.arange(n); r = 0
        crowd = np.zeros(n)
        dom = dominates_matrix(G)
        cnt = dom.sum(0).astype(int); alive = np.ones(n, bool)
        while alive.any():
            front = np.where(alive & (cnt == 0))[0]
            rank[front] = r
            for f in front:
                cnt -= dom[f]
            alive[front] = False
            cnt[front] = -1
            # crowding within front
            if len(front) > 2:
                for m in range(G.shape[1]):
                    o = front[np.argsort(G[front, m])]
                    rngm = G[o[-1], m] - G[o[0], m] or 1.0
                    crowd[o[0]] = crowd[o[-1]] = np.inf
                    crowd[o[1:-1]] += (G[o[2:], m] - G[o[:-2], m]) / rngm
            else:
                crowd[front] = np.inf
            r += 1
        return rank, crowd
    rank, crowd = rank_crowd(G)
    for _ in range(gens):
        def tour():
            a, b = rng.integers(0, pop, pop), rng.integers(0, pop, pop)
            win = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b])), a, b)
            return win
        p1, p2 = X[tour()], X[tour()]
        u = rng.random(p1.shape); eta = 15
        beta = np.where(u <= .5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
        c = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        mut = rng.random(c.shape) < 1 / nvar; u = rng.random(c.shape); em = 20
        dl = np.where(u < .5, (2 * u) ** (1 / (em + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (em + 1)))
        c = np.clip(c + mut * dl, 0, 1)
        XA = np.vstack([X, c]); GA = np.vstack([G, ev(c)])
        rk, cr = rank_crowd(GA)
        order = np.lexsort((-cr, rk))[:pop]
        X, G = XA[order], GA[order]; rank, crowd = rank_crowd(G)
    return X, G

def weighted_sum(env, nw=128, rng=None):
    from scipy.optimize import minimize
    rng = rng or np.random.default_rng(0)
    W = rng.dirichlet(np.ones(4), nw); out = []
    for w in W:
        best = None
        for _ in range(3):
            x0 = rng.random(3)
            r = minimize(lambda x: -(objectives(x, env)[0] @ w), x0, bounds=[(0, 1)] * 3, method='SLSQP')
            if best is None or r.fun < best.fun: best = r
        out.append(np.clip(best.x, 0, 1))
    X = np.array(out); return X, 1 - objectives(X, env)

def biobjective(env, pop=80, gens=30, rng=None):
    X, _ = nsga2(env, pop, gens, rng, cols=[1, 2])   # optimise (S, C) only
    return X, 1 - objectives(X, env)

def paired_report(a, b, alpha=0.05):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    t, p = stats.ttest_rel(a, b)
    sd = d.std(ddof=1); dz = d.mean() / sd if sd > 0 else float('nan')
    se = sd / np.sqrt(n); h = stats.t.ppf(1 - alpha / 2, n - 1) * se
    sw = stats.shapiro(d)
    try: w = stats.wilcoxon(d)
    except ValueError: w = None
    return dict(n=n, df=n - 1, mean_a=float(np.mean(a)), mean_b=float(np.mean(b)), mean_diff=float(d.mean()), sd_diff=float(sd),
                t=float(t), p=float(p), dz=float(dz), ci=[float(d.mean() - h), float(d.mean() + h)],
                shapiro_W=float(sw.statistic), shapiro_p=float(sw.pvalue),
                wilcoxon_p=(float(w.pvalue) if w else None))
