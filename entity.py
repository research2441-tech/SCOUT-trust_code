"""Entity-level experiments (E2-E9): validation-only tuning on known families, then open-loop trust
quality and closed-loop containment on worlds that replay the held-out (unknown) family."""
import os
import sys, json, time, itertools, numpy as np, xgboost as xgb
from world import World, Scenario
from trust import NoDefense, EWMA, BetaRep, Asym, CUSUM, BlameDiff, Learned, SCOUT
from sim import simulate, ThresholdPolicy, UtilityPolicy

RES = os.environ.get('SCOUT_RESULTS', 'results')
EPS_FIR = 0.02
NEVER = ThresholdPolicy(np.inf)

ABL = {'SCOUT': {}, 'SCOUT-noDormant': dict(dormant=False), 'SCOUT-noContagion': dict(contagion='none'),
       'SCOUT-blame': dict(contagion='blame'), 'SCOUT-blameRec': dict(contagion='blame_rec'), 'SCOUT-noAttribution': dict(attribution=False),
       'SCOUT-noVolume': dict(volume=False), 'SCOUT-noEnrol': dict(enrol=False), 'SCOUT-noUtility': {},
       'SCOUT-noTemper': dict(omega=1.0)}
BASE = ['EWMA', 'Beta', 'Asym', 'CUSUM', 'BetaDiff', 'Shiryaev', 'LearnedRisk']


def load_pools(ds, held, seed, score='clf'):
    z = np.load(f'{RES}/pool_{ds}_{held}_{seed}.npz')
    known = sorted({k.split('|')[1] for k in z.files if k.startswith('val|')} - {'benign'})
    rng = np.random.default_rng(seed)
    m = min(len(z[f'val|{f}|{score}']) for f in known); m = max(m, 200)
    kv = np.concatenate([rng.choice(z[f'val|{f}|{score}'], m) for f in known])
    return (dict(benign=z[f'val|benign|{score}'], attack=kv),
            dict(benign=z[f'test|benign|{score}'], attack=z[f'test|{held}|{score}']))


def make(name, W, P, model=None):
    if name == 'None': return NoDefense(W, P)
    if name == 'EWMA': return EWMA(W, P)
    if name == 'Beta': return BetaRep(W, P)
    if name == 'Asym': return Asym(W, P)
    if name == 'CUSUM': return CUSUM(W, P)
    if name == 'BetaDiff': return BlameDiff(W, P)
    if name == 'LearnedRisk': return Learned(W, P, model)
    if name == 'Shiryaev':
        m = SCOUT(W, P, dormant=False, contagion='none', attribution=False, volume=False, enrol=False, omega=1.0)
        m.name = 'Shiryaev'; return m
    m = SCOUT(W, P, **ABL[name]); m.name = name
    return m


def policy_kind(name):
    return 'util' if name.startswith('SCOUT') and name != 'SCOUT-noUtility' else 'thr'


def mkpol(kind, v, vu=None):
    return UtilityPolicy(v, vu) if kind == 'util' else ThresholdPolicy(v, vu)


def to_op(name, worlds, P, model, x):
    if policy_kind(name) == 'thr' and make(name, worlds[0], P, model).prob:
        return float(1 / (1 + np.exp(-x)))
    return float(np.exp(x))


OFF = {'util': 0.0, 'thr': np.inf}


def val_rates(name, worlds, P, v, vu, model):
    k = policy_kind(name); o = [simulate(W, make(name, W, P, model), mkpol(k, v, vu)) for W in worlds]
    return float(np.mean([r['dev_FIR'] for r in o])), float(np.mean([r['usr_FSR'] for r in o]))


def bisect(name, worlds, P, model, f, steps):
    kind = policy_kind(name)
    if kind == 'util':
        lo, hi = np.log(1e-3), np.log(3000.0)                       # false isolation increases with kappa
    else:
        prob = make(name, worlds[0], P, model).prob
        lo, hi = (-7.0, 14.0) if prob else (np.log(0.5), np.log(400.0))   # logit / log scale
    best = None
    for _ in range(steps):
        mid = 0.5 * (lo + hi); val = f(to_op(name, worlds, P, model, mid))
        if kind == 'util':
            if val <= EPS_FIR: best, lo = mid, mid
            else: hi = mid
        else:
            if val <= EPS_FIR: best, hi = mid, mid
            else: lo = mid
    if best is None:
        best = lo if kind == 'util' else hi
    return to_op(name, worlds, P, model, best)


def tune(name, worlds, P, model=None, steps=7):
    """Most sensitive operating points whose mean validation false-isolation rate (devices) and false-suspension
    rate (user accounts) each stay within EPS_FIR; devices first with user actions disabled, then users."""
    kind = policy_kind(name)
    vd = bisect(name, worlds, P, model, lambda v: val_rates(name, worlds, P, v, OFF[kind], model)[0], steps)
    vu = bisect(name, worlds, P, model, lambda v: val_rates(name, worlds, P, vd, v, model)[1], steps)
    return (vd, vu)


def select_filter(worlds, P):
    """Calibration-targeted selection of the filter constants: minimum open-loop log-loss of the compromise
    posterior on validation worlds (known families only)."""
    grid = list(itertools.product([0.02, 0.05, 0.15], [0.3, 0.6, 1.0], [4.0, 8.0], [4.0, 8.0]))
    best, bl, table = None, np.inf, []
    for r, om, c, psi in grid:
        ll = []
        for W in worlds:
            m = SCOUT(W, dict(P, r=r, omega=om, clip=c)); m.psi = psi
            out = simulate(W, m, NEVER, record=True)
            rd = np.array([x['rd'] for x in out['recs']][20:]); y = np.array([x['comp'] for x in out['recs']][20:])
            p = np.clip(rd, 1e-4, 1 - 1e-4)
            ll.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        table.append(dict(r=r, omega=om, clip=c, psi=psi, logloss=float(np.mean(ll))))
        if np.mean(ll) < bl:
            bl, best = float(np.mean(ll)), dict(r=r, omega=om, clip=c, psi=psi)
    return best, table


def train_learned(worlds, P):
    X, Y = [], []
    for W in worlds:
        m = Learned(W, P); U0 = W.U0
        orig = m.step
        feats = []
        def step(t, o, m=m):
            Fd, Fu = m.feats(o); feats.append((t, Fd, Fu)); return np.zeros(m.N), np.zeros(m.U)
        m.step = step
        simulate(W, m, NEVER)
        for t, Fd, Fu in feats:
            if t < 20: continue
            yd = W.onset <= t; yu = np.zeros(W.U, bool); yu[U0:] = W.ins_onset <= t
            X.append(np.r_[Fd, Fu]); Y.append(np.r_[yd, yu])
    X, Y = np.concatenate(X), np.concatenate(Y).astype(int)
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, n_jobs=1,
                            tree_method='hist', eval_metric='logloss')
    return clf.fit(X, Y)


def apply_psi(m, P):
    if isinstance(m, SCOUT) and m.name != 'Shiryaev':
        m.psi = P.get('psi', 4.0)
    return m


def run_fold(ds, held, seed, kinds=('BA', 'WS', 'HIER'), vkinds=('BA', 'HIER'), N=500, Nval=300, methods=None, tag='main', extra=None):
    t0 = time.time(); extra = extra or {}
    vpool, tpool = load_pools(ds, held, seed, extra.get('score', 'clf'))
    vworlds = [World(Scenario(N=Nval, kind=k, **extra.get('val_sc', {})), vpool, seed=1000 + 37 * seed + i) for i, k in enumerate(vkinds)]
    P = dict(q0_global=vworlds[0].pb_base)
    filt, table = select_filter(vworlds, P); P.update(filt)
    model = train_learned(vworlds, P)
    methods = methods or (['None'] + BASE + list(ABL))
    ops = {}
    for name in methods:
        if name == 'None':
            ops[name] = (np.inf, np.inf); continue
        ops[name] = tune(name, vworlds, P, model)
    tworlds = [World(Scenario(N=N, kind=k, **extra.get('test_sc', {})), tpool, seed=2000 + 101 * seed + i) for i, k in enumerate(kinds)]
    rows = []
    for W in tworlds:
        for name in methods:
            m = apply_psi(make(name, W, P, model), P)
            r = simulate(W, m, mkpol(policy_kind(name), *ops[name]) if name != 'None' else NEVER)
            m2 = apply_psi(make(name, W, P, model), P)
            q = simulate(W, m2, NEVER)                    # open loop: trust quality without actions
            r.update({f'ol_{k}': v for k, v in q.items() if k in ('auroc_dev', 'auroc_usr', 'brier_dev', 'ece_dev', 'tier')})
            r.update(dict(dataset=ds, held=held, seed=seed, kind=W.sc.kind, N=W.N, op=ops[name], tag=tag,
                          pb=W.pb_base, pa=W.pa))
            rows.append(r)
    with open(f'{RES}/entity_{tag}.jsonl', 'a') as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + '\n')
    with open(f'{RES}/filter_{tag}.jsonl', 'a') as f:
        f.write(json.dumps(dict(dataset=ds, held=held, seed=seed, selected=filt, table=table, ops=ops)) + '\n')
    print(ds, held, seed, tag, 'filter', filt, '%.0fs' % (time.time() - t0), flush=True)
    return rows


# psi must reach every SCOUT instance built inside tuning as well
_make = make
def make(name, W, P, model=None):          # noqa: F811
    return apply_psi(_make(name, W, P, model), P)


if __name__ == '__main__':
    ds = sys.argv[1]; part = int(sys.argv[2]); nparts = int(sys.argv[3])
    from data import LOADERS
    fams = {'TON_IoT': ['backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss'],
            'CICIoT2023': ['BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-based']}[ds]
    jobs = [(h, s) for s in (0, 1) for h in fams]
    for j, (h, s) in enumerate(jobs):
        if j % nparts == part:
            run_fold(ds, h, s)
