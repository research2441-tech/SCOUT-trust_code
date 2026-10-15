"""Trace-driven entity-level world (Section V-B).

Devices sit on a time-varying topology, users own devices and open sessions on them, and every
window each device emits flows. The number of flows whose benign-conformity value u falls at or
below the screening level q is drawn with the empirical probabilities measured on REAL held-out
flows (benign test flows; flows of the family left out of training), which is distributionally
identical to resampling individual flows from those pools because every entity-level method consumes
only per-window flow counts. Every random draw is generated from a generator keyed on (world seed,
window) for the whole population, so all methods face the same world (common random numbers); only
the consequences of their own actions differ."""
import numpy as np, networkx as nx
from scipy.sparse import csr_matrix

T_DEFAULT = 160


def make_topology(kind, N, rng):
    s = int(rng.integers(1 << 30))
    if kind == 'BA':
        G = nx.barabasi_albert_graph(N, 2, seed=s)
    elif kind == 'WS':
        G = nx.connected_watts_strogatz_graph(N, 6, 0.1, seed=s)
    elif kind == 'HIER':          # gateway backbone (ring + chords) with device clusters behind each gateway
        ng = max(4, N // 25); G = nx.cycle_graph(ng)
        for _ in range(ng // 2):
            a, b = rng.choice(ng, 2, replace=False); G.add_edge(int(a), int(b))
        cl = {}
        for v in range(ng, N):
            g = int(rng.integers(ng)); G.add_edge(v, g); cl.setdefault(g, []).append(v)
        for g, mem in cl.items():
            for v in mem:
                for w_ in mem:
                    if v < w_ and rng.random() < 2.0 / max(len(mem), 1):
                        G.add_edge(v, w_)
    else:
        raise ValueError(kind)
    return G


class Scenario:
    def __init__(self, N=500, T=T_DEFAULT, kind='BA', churn=0.01, hetero_frac=0.10, burst_frac=0.05,
                 rate_scale=1.0, q=0.01, worm_p=0.04, onset=(25, 90), attack_scale=1.0):
        sc = N / 500.0 * attack_scale
        self.N, self.T, self.kind = N, T, kind
        self.n_low = max(2, round(6 * sc)); self.n_onoff = max(2, round(6 * sc))
        self.n_hop = max(1, round(3 * sc)); self.n_flood = max(1, round(2 * sc)); self.n_worm = 1
        self.n_gw = 1 if N >= 200 else 0
        self.churn, self.hetero_frac, self.burst_frac = churn, hetero_frac, burst_frac
        self.rate_scale, self.q, self.worm_p, self.onset = rate_scale, q, worm_p, onset


def flag_prob(u, q):
    return float((u <= q).mean())


class World:
    def __init__(self, sc, pools, seed, test_rate_scale=1.0):
        self.sc = sc; N, T = sc.N, sc.T; self.N, self.T, self.seed = N, T, seed
        rng = np.random.default_rng(seed)
        self.G0 = make_topology(sc.kind, N, rng)
        self.rate = 6.0 * sc.rate_scale * rng.lognormal(0, 0.5, N) / np.exp(0.125)
        gam = np.ones(N); het = rng.random(N) < sc.hetero_frac; gam[het] = rng.uniform(1.2, 1.8, het.sum())
        self.gam = gam
        ub, ua = pools['benign'], pools['attack']; q = sc.q
        self.pb = np.array([flag_prob(ub, q ** (1.0 / g)) for g in gam])     # u^g <= q  <=>  u <= q^(1/g)
        self.pb_base, self.pa = flag_prob(ub, q), flag_prob(ua, q)
        self.bursty = rng.random(N) < sc.burst_frac
        deg = np.array([d for _, d in sorted(self.G0.degree())]); self.deg0 = deg
        self.svc = np.zeros(N, bool); self.svc[np.argsort(-deg)[:max(2, N // 20)]] = True
        U = max(2, N // 4); own = rng.integers(0, U, N); own[rng.permutation(N)[:U]] = np.arange(U)
        self.own, self.U0 = own, U
        self.n_ins = sc.n_hop; self.U = U + self.n_ins
        cand = rng.permutation(np.where(~self.svc)[0]); k = 0
        def take(n):
            nonlocal k
            out = cand[k:k + n]; k += n; return out
        svc_nodes = rng.permutation(np.where(self.svc)[0])
        self.gw = svc_nodes[:sc.n_gw]                               # compromised gateway / service node
        self.low = take(sc.n_low); self.onoff = take(sc.n_onoff); self.flood = take(sc.n_flood); self.worm0 = take(sc.n_worm)
        lo, hi = sc.onset
        self.onset = np.full(N, np.inf)
        for grp in (self.low, self.onoff, self.flood, self.worm0, self.gw):
            self.onset[grp] = rng.integers(lo, hi, len(grp))
        self.mrel = np.zeros(N)                                   # malicious-to-benign flow ratio when active
        f = rng.uniform(0.05, 0.30, len(self.low)); self.mrel[self.low] = f / (1 - f)
        self.mrel[self.onoff] = 1.0; self.mrel[self.worm0] = 0.25; self.mrel[self.gw] = 0.25
        self.kind = np.array(['benign'] * N, dtype=object)
        for nm, grp in (('low', self.low), ('onoff', self.onoff), ('flood', self.flood), ('worm', self.worm0), ('gw', self.gw)):
            self.kind[grp] = nm
        self.ins_onset = rng.integers(lo, hi, self.n_ins).astype(float)
        home = rng.choice(np.where(~self.svc)[0], self.n_ins, replace=False); self.ins_home = home
        self.ins_reach = []
        for h in home:
            d = nx.single_source_shortest_path_length(self.G0, int(h), cutoff=2)
            self.ins_reach.append(np.array(sorted(v for v in d if not self.svc[v]), dtype=np.int64))
        self.on_sched = np.zeros((T, N), bool)
        for i in np.r_[self.onoff, self.flood]:
            t = int(self.onset[i])
            while t < T:
                L = int(rng.integers(3, 7)); self.on_sched[t:t + L, i] = True
                t += L + int(rng.integers(10, 30))
        self.burst_on = np.zeros((T, N), bool)
        for i in np.where(self.bursty)[0]:
            for t0 in np.where(rng.random(T) < 0.03)[0]:
                self.burst_on[t0:t0 + int(rng.integers(2, 5)), i] = True
        self.topo = self._topologies()

    def _topologies(self):
        G = self.G0.copy(); out = []
        for t in range(self.T):
            r = np.random.default_rng([self.seed, 7, t])
            m = int(round(self.sc.churn * G.number_of_edges())); E = list(G.edges())
            for j in (r.choice(len(E), m, replace=False) if m else []):
                a, b = E[j]
                if G.degree(a) > 1 and G.degree(b) > 1 and G.has_edge(a, b):
                    G.remove_edge(a, b)
                v = int(r.integers(self.N)); nb = list(G.neighbors(v))
                if nb:
                    w_ = nb[int(r.integers(len(nb)))]
                    nb2 = [x for x in G.neighbors(w_) if x != v and not G.has_edge(v, x)]
                    if nb2:
                        G.add_edge(v, nb2[int(r.integers(len(nb2)))])
            out.append(np.array(sorted(G.edges()), dtype=np.int64).reshape(-1, 2))
        return out

    def draws(self, t):
        """All stochastic quantities of window t for the whole population, independent of defender actions."""
        r = np.random.default_rng([self.seed, 11, t]); N = self.N
        lam = self.rate * r.gamma(5.0, 0.2, N)
        lam = lam * np.where(self.burst_on[t], 3.0, 1.0) * (r.random(N) >= 0.03)
        nb = r.poisson(lam); kb = r.binomial(nb, self.pb)
        nm = r.poisson(self.rate * self.mrel); km = r.binomial(nm, self.pa)
        nw = r.poisson(self.rate * 0.25); kw = r.binomial(nw, self.pa)          # emission of a worm-infected device
        nf = r.poisson(10.0 * self.rate); kf = r.binomial(nf, self.pb)
        hop = r.random(self.n_ins); nv = r.poisson(4.0 * r.gamma(4.0, 0.25, self.n_ins))
        nvm = r.binomial(nv, 0.5)
        kv = r.binomial(nvm, self.pa) + r.binomial(nv - nvm, self.pb_base)
        kvb = r.binomial(nv, self.pb_base)                                    # same account before it turns
        worm_u = r.random(len(self.topo[t]) * 2)
        return dict(nb=nb, kb=kb, nm=nm, km=km, nw=nw, kw=kw, nf=nf, kf=kf, hop=hop, nv=nv, nvm=nvm, kv=kv, kvb=kvb, worm_u=worm_u)


def adjacency(E, N, active):
    keep = active[E[:, 0]] & active[E[:, 1]]
    a, b = E[keep, 0], E[keep, 1]
    return csr_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(N, N))
