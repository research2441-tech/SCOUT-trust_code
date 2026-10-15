"""Second pass over every main fold (same tuned constants): (i) open-loop trust quality on the same test worlds with
lateral spread switched off, so that prevalence stays at the seeded adversaries when no action is ever taken;
(ii) the blame-diffusion ablation, tuned and evaluated like every other method."""
import os
import sys, json, numpy as np
from world import World, Scenario
from sim import simulate
import entity as E
import supp

RES = os.environ.get('SCOUT_RESULTS', 'results')


def fold(ds, held, seed):
    P, ops, model, tp = supp.restore(ds, held, seed)
    vpool, _ = E.load_pools(ds, held, seed)
    vworlds = [World(Scenario(N=300, kind=k), vpool, seed=1000 + 37 * seed + i) for i, k in enumerate(('BA', 'HIER'))]
    ops['SCOUT-blame'] = E.tune('SCOUT-blame', vworlds, P, model)
    rows_cl, rows_ol = [], []
    for i, k in enumerate(('BA', 'WS', 'HIER')):
        W = World(Scenario(N=500, kind=k), tp, seed=2000 + 101 * seed + i)
        r = simulate(W, E.make('SCOUT-blame', W, P, model), E.mkpol('util', *ops['SCOUT-blame']))
        r.update(dict(dataset=ds, held=held, seed=seed, kind=k, N=500, op=ops['SCOUT-blame'], tag='main', pb=W.pb_base, pa=W.pa))
        rows_cl.append(r)
        W0 = World(Scenario(N=500, kind=k, worm_p=0.0), tp, seed=2000 + 101 * seed + i)
        for name in ['None'] + E.BASE + list(E.ABL):
            q = simulate(W0, E.make(name, W0, P, model), E.NEVER)
            rows_ol.append(dict(dataset=ds, held=held, seed=seed, kind=k, method=name,
                                **{k2: q.get(k2) for k2 in ('auroc_dev', 'auroc_usr', 'brier_dev', 'ece_dev', 'tier')}))
    with open(f'{RES}/entity_blame.jsonl', 'a') as f:
        for r in rows_cl: f.write(json.dumps(r, default=float) + '\n')
    with open(f'{RES}/entity_ol.jsonl', 'a') as f:
        for r in rows_ol: f.write(json.dumps(r, default=float) + '\n')
    print('rerun', ds, held, seed, flush=True)


if __name__ == '__main__':
    part, nparts = int(sys.argv[1]), int(sys.argv[2])
    jobs = [(json.loads(l)['dataset'], json.loads(l)['held'], json.loads(l)['seed']) for l in open(f'{RES}/filter_main.jsonl')]
    for j, (d, h, s) in enumerate(jobs):
        if j % nparts == part:
            fold(d, h, s)
