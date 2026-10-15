"""Supplementary experiments reusing the validation-selected filter constants and operating points of the main
runs (no retuning on test conditions): scale (E5), robustness to benign drift / churn / heterogeneity (E4),
on-off duty-cycle sweep (E9), sensitivity of SCOUT constants (E3), open-loop collateral of propagation, runtime (E6)."""
import os
import sys, json, time, numpy as np
from world import World, Scenario
from trust import SCOUT, BetaRep
from sim import simulate, ThresholdPolicy, UtilityPolicy
import entity as E

RES = os.environ.get('SCOUT_RESULTS', 'results')
SUB = {'TON_IoT': ['ransomware', 'scanning', 'xss'], 'CICIoT2023': ['BruteForce', 'Recon', 'Spoofing']}
METH = ['Beta', 'Asym', 'BetaDiff', 'Shiryaev', 'LearnedRisk', 'SCOUT', 'SCOUT-noUtility', 'SCOUT-noDormant']


def restore(ds, held, seed):
    rec = [json.loads(l) for l in open(f'{RES}/filter_main.jsonl')]
    rec = [r for r in rec if r['dataset'] == ds and r['held'] == held and r['seed'] == seed][0]
    vpool, tpool = E.load_pools(ds, held, seed)
    vworlds = [World(Scenario(N=300, kind=k), vpool, seed=1000 + 37 * seed + i) for i, k in enumerate(('BA', 'HIER'))]
    P = dict(q0_global=vworlds[0].pb_base); P.update(rec['selected'])
    model = E.train_learned(vworlds, P)
    return P, {k: tuple(v) for k, v in rec['ops'].items()}, model, tpool


def run_world(W, P, ops, model, methods, tag, meta):
    rows = []
    for name in methods:
        m = E.make(name, W, P, model); t0 = time.time()
        r = simulate(W, m, E.mkpol(E.policy_kind(name), *ops[name]))
        r['sec_per_window'] = (time.time() - t0) / W.T
        r.update(meta); r.update(dict(tag=tag, method=name, N=W.N, kind=W.sc.kind))
        rows.append(r)
    with open(f'{RES}/supp_{tag}.jsonl', 'a') as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + '\n')


def scale(ds, held, seed=0):
    P, ops, model, tp = restore(ds, held, seed)
    for N in (120, 250, 1000):
        for i, k in enumerate(('BA', 'WS', 'HIER')):
            W = World(Scenario(N=N, kind=k), tp, seed=5000 + 13 * N + i)
            run_world(W, P, ops, model, METH, 'scale', dict(dataset=ds, held=held, seed=seed))


def robust(ds, held, seed=0):
    P, ops, model, tp = restore(ds, held, seed)
    conds = [('rate0.5', dict(rate_scale=0.5)), ('rate2', dict(rate_scale=2.0)), ('churn0', dict(churn=0.0)),
             ('churn5', dict(churn=0.05)), ('hetero30', dict(hetero_frac=0.30)), ('burst20', dict(burst_frac=0.20))]
    for cname, kw in conds:
        for i, k in enumerate(('BA', 'WS', 'HIER')):
            W = World(Scenario(N=500, kind=k, **kw), tp, seed=6000 + i)
            run_world(W, P, ops, model, METH, 'robust', dict(dataset=ds, held=held, seed=seed, cond=cname))


def duty(ds, held, seed=0):
    """All attackers are on-off devices with 4-window bursts separated by a fixed dormant gap G."""
    P, ops, model, tp = restore(ds, held, seed)
    for G in (5, 10, 20, 40, 80):
        for i, k in enumerate(('BA', 'WS', 'HIER')):
            W = World(Scenario(N=500, kind=k), tp, seed=7000 + i)
            W.on_sched[:] = False
            for j in W.onoff:
                t = int(W.onset[j])
                while t < W.T:
                    W.on_sched[t:t + 4, j] = True; t += 4 + G
            # neutralise the other attack types so the sweep isolates the on-off adversary
            for grp in (W.low, W.flood, W.worm0, W.gw):
                W.onset[grp] = np.inf; W.kind[grp] = 'benign'; W.mrel[grp] = 0
            W.n_ins = 0
            run_world(W, P, ops, model, ['Beta', 'Asym', 'EWMA', 'Shiryaev', 'SCOUT-noDormant', 'SCOUT', 'LearnedRisk'],
                      'duty', dict(dataset=ds, held=held, seed=seed, gap=G))


def sens(ds, held, seed=0):
    """Open-loop trust quality as the filter constants move away from the validation choice, plus the
    inflation of benign neighbours of compromised devices under scrutiny (contagion) vs blame propagation."""
    P, ops, model, tp = restore(ds, held, seed)
    out = []
    for i, k in enumerate(('BA', 'WS', 'HIER')):
        W = World(Scenario(N=500, kind=k, worm_p=0.0), tp, seed=8000 + i)
        grid = [('r', v) for v in (0.01, 0.05, 0.15, 0.3)] + [('hmax', v) for v in (0.0, 0.01, 0.03, 0.1, 0.3)] + \
               [('omega', v) for v in (0.3, 0.6, 1.0)] + [('contagion', 'blame'), ('contagion', 'none'), ('q', 0.005), ('q', 0.02)]
        for par, v in grid:
            kw = {par: v}
            m = SCOUT(W, P, **kw) if par != 'omega' else SCOUT(W, P, omega=v)
            m.psi = P.get('psi', 4.0)
            if par == 'q':
                Wq = World(Scenario(N=500, kind=k, q=v, worm_p=0.0), tp, seed=8000 + i); m = SCOUT(Wq, P); m.psi = P.get('psi', 4.0)
                r = simulate(Wq, m, E.NEVER, record=True)
            else:
                r = simulate(W, m, E.NEVER, record=True)
            rd = np.array([x['rd'] for x in r['recs']])
            mx = rd[20:].max(0)
            comp_dev = np.isfinite(r['onset_final']); A0 = np.zeros(W.N, bool)
            for a, b in W.G0.edges():
                if comp_dev[a]: A0[b] = True
                if comp_dev[b]: A0[a] = True
            nbr_benign = A0 & ~comp_dev; far_benign = ~A0 & ~comp_dev
            on = r['onset_final']
            fin = np.isfinite(on)
            delay = []
            for j in np.where(fin & (on < W.T - 5))[0]:
                hit = np.where(rd[int(on[j]):, j] >= 0.5)[0]
                delay.append(hit[0] if len(hit) else W.T - on[j])
            out.append(dict(dataset=ds, held=held, kind=k, param=par, value=v, auroc=r['auroc_dev'], ece=r.get('ece_dev'),
                            brier=r.get('brier_dev'), nbr_max=float(mx[nbr_benign].mean()), far_max=float(mx[far_benign].mean()),
                            nbr_hi=float((mx[nbr_benign] >= 0.5).mean()), far_hi=float((mx[far_benign] >= 0.5).mean()),
                            delay_med=float(np.median(delay))))
    with open(f'{RES}/supp_sens.jsonl', 'a') as f:
        for r in out:
            f.write(json.dumps(r, default=float) + '\n')


def budget(ds, held, seed=0):
    """Retune SCOUT and the strongest baselines at three false-isolation budgets."""
    vpool, tp = E.load_pools(ds, held, seed)
    P, ops, model, _ = restore(ds, held, seed)
    vworlds = [World(Scenario(N=300, kind=k), vpool, seed=1000 + 37 * seed + i) for i, k in enumerate(('BA', 'HIER'))]
    for eps in (0.01, 0.05):
        E.EPS_FIR = eps
        ops2 = {n: E.tune(n, vworlds, P, model) for n in ('Beta', 'Shiryaev', 'SCOUT', 'LearnedRisk')}
        for i, k in enumerate(('BA', 'WS', 'HIER')):
            W = World(Scenario(N=500, kind=k), tp, seed=2000 + 101 * seed + i)
            run_world(W, P, ops2, model, list(ops2), 'budget', dict(dataset=ds, held=held, seed=seed, eps=eps))
    E.EPS_FIR = 0.02


def relia(ds, held, seed=0):
    """Open-loop reliability bins of the device compromise probability."""
    P, ops, model, tp = restore(ds, held, seed)
    out = []
    for i, k in enumerate(('BA', 'WS', 'HIER')):
        W = World(Scenario(N=500, kind=k, worm_p=0.0), tp, seed=2000 + 101 * seed + i)
        for name in ('EWMA', 'Beta', 'Asym', 'BetaDiff', 'Shiryaev', 'LearnedRisk', 'SCOUT'):
            r = simulate(W, E.make(name, W, P, model), E.NEVER, record=True)
            rd = np.array([x['rd'] for x in r['recs']][20:]).ravel(); y = np.array([x['comp'] for x in r['recs']][20:]).ravel()
            p = np.clip(rd, 0, 1); b = np.minimum((p * 10).astype(int), 9)
            out.append(dict(dataset=ds, held=held, kind=k, method=name, n=np.bincount(b, minlength=10).tolist(),
                            pos=np.bincount(b, y.astype(float), 10).tolist(), psum=np.bincount(b, p, 10).tolist()))
    with open(f'{RES}/supp_relia.jsonl', 'a') as f:
        for r in out:
            f.write(json.dumps(r, default=float) + '\n')


if __name__ == '__main__':
    what = sys.argv[1]; part = int(sys.argv[2]); nparts = int(sys.argv[3])
    jobs = [(ds, h) for ds in SUB for h in SUB[ds]]
    for j, (ds, h) in enumerate(jobs):
        if j % nparts == part:
            {'scale': scale, 'robust': robust, 'duty': duty, 'sens': sens, 'budget': budget, 'relia': relia}[what](ds, h)
            print(what, ds, h, 'done', flush=True)
