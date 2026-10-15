"""Closed-loop simulation of a trust method plus containment policy on a World (Section V-B, V-D)."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score
from world import adjacency

R_REVIEW = 12          # windows until an isolated entity is forensically reviewed
PHI_R = 0.3            # share of traffic that a restricted entity may still send


class ThresholdPolicy:
    def __init__(self, theta, theta_u=None, ratio=0.6):
        self.th, self.ratio = theta, ratio
        self.thu = theta if theta_u is None else theta_u

    def __call__(self, rd, ru, ctx):
        iso = rd >= self.th; res = (rd >= self.ratio * self.th) & ~iso
        sus = ru >= self.thu; stp = (ru >= self.ratio * self.thu) & ~sus
        return iso, np.zeros_like(iso), res, sus, stp


class UtilityPolicy:
    """Bayes action under expected loss kappa*pi*H(a) + (1-pi)*D(a) on the current topology."""
    def __init__(self, kappa, kappa_u=None, eta=0.1):
        self.k, self.eta = kappa, eta
        self.ku = kappa if kappa_u is None else kappa_u

    def __call__(self, rd, ru, ctx):
        N = len(rd); A = ctx['A']; deg = np.asarray(A.sum(1)).ravel()
        e = 1.0 + np.log1p(deg)                                      # lateral exposure of a compromised node
        k, pi = self.k, rd
        C = np.c_[k * pi * e,                                        # allow
                  k * pi * PHI_R * e + (1 - pi) * (1 - PHI_R),       # restrict
                  k * pi * self.eta * e + (1 - pi),                  # egress isolation (keeps transit)
                  (1 - pi)]                                          # quarantine (transit loss added below)
        cand = np.where((pi > 0.02) & (C.argmin(1) == 3))[0]
        if len(cand):
            cand = cand[articulation(A, cand)]
        if len(cand):
            tau = transit_loss(A, cand, ctx['svc'], ctx['active'])
            C[cand, 3] = (1 - pi[cand]) * (1 + tau)
        a = C.argmin(1)
        iso_q, iso_e, res = a == 3, a == 2, a == 1
        pu = ru
        ku = self.ku
        Cu = np.c_[ku * pu, ku * pu * PHI_R + (1 - pu) * (1 - PHI_R), (1 - pu)]
        au = Cu.argmin(1)
        return iso_q, iso_e, res, au == 2, au == 1


def served_mask(A, svc, active):
    n, lab = connected_components(A, directed=False)
    ok = np.zeros(n, bool); ok[lab[svc & active]] = True
    return ok[lab] & active


def articulation(A, cand):
    import networkx as nx
    G = nx.from_scipy_sparse_array(A)
    ap = set(nx.articulation_points(G))
    return np.array([i in ap for i in cand], bool)


def transit_loss(A, cand, svc, active):
    """Benign demand (other than the node's own) that would lose every route to a service node."""
    from scipy.sparse import diags
    base = served_mask(A, svc, active); out = np.zeros(len(cand))
    for j, i in enumerate(cand):
        act = active.copy(); act[i] = False
        Dm = diags(act.astype(float)); A2 = (Dm @ A @ Dm).tocsr()
        s2 = served_mask(A2, svc, act)
        lost = base & ~s2; lost[i] = False
        out[j] = float(lost.sum())
    return out


def simulate(W, method, policy, record=False, review=R_REVIEW):
    N, T, U, U0 = W.N, W.T, W.U, W.U0
    onset = W.onset.copy(); kind = W.kind.copy(); mrel = W.mrel.copy()
    ins_on = W.ins_onset
    Q = np.zeros(N, bool); E = np.zeros(N, bool); Rr = np.zeros(N, bool)
    S = np.zeros(U, bool); St = np.zeros(U, bool)
    perm_d = np.zeros(N, bool); perm_u = np.zeros(U, bool)
    due_d = {}; due_u = {}
    t_cont_d = np.full(N, np.inf); t_cont_u = np.full(U, np.inf)
    false_iso = np.zeros(N, bool); false_iso_events = 0; false_sus = np.zeros(U, bool)
    ben_gen = ben_del = mal_del = mal_gen = 0.0
    collat = 0.0; lcc = []; res_benign = 0.0; ben_dev_windows = 0.0
    recs = [] if record else None
    rate_mean = W.rate.mean()
    demand = W.rate / rate_mean
    lab_d, lab_u, sc_d, sc_u = [], [], [], []
    tiers = np.zeros((3, 2))
    for t in range(T):
        comp = onset <= t
        act_mal = comp & ((kind == 'low') | (kind == 'worm') | (kind == 'gw') | (((kind == 'onoff') | (kind == 'flood')) & W.on_sched[t]))
        d = W.draws(t)
        active = ~Q
        A = adjacency(W.topo[t], N, active)
        served = served_mask(A, W.svc, active)
        # ---- emissions
        nb, kb = d['nb'] * active, d['kb'] * active
        mal_dev = act_mal & active & (kind != 'flood')
        isw = kind == 'worm'
        nm = np.where(isw, d['nw'], d['nm']) * mal_dev; km = np.where(isw, d['kw'], d['km']) * mal_dev
        fl = act_mal & active & (kind == 'flood')
        nf, kf = d['nf'] * fl, d['kf'] * fl
        own = W.own
        cell_dev = [np.arange(N)]; cell_usr = [own]; cell_n = [nb + nm + nf]; cell_k = [kb + km + kf]
        vis_mal = 0.0; vis_del = 0.0
        for x in range(W.n_ins):
            u = U0 + x
            if S[u] or perm_u[u]:
                continue
            turned = t >= ins_on[x]
            if turned:
                reach = W.ins_reach[x]; dv = int(reach[min(int(d['hop'][x] * len(reach)), len(reach) - 1)])
            else:
                dv = int(W.ins_home[x])
            if not Q[dv]:
                cell_dev.append(np.array([dv])); cell_usr.append(np.array([u])); cell_n.append(np.array([d['nv'][x]]))
                cell_k.append(np.array([d['kv'][x] if turned else d['kvb'][x]]))
                if turned:
                    g = 0.0 if E[dv] else (PHI_R if (St[u] or Rr[dv]) else 1.0)
                    vis_mal += d['nvm'][x]; vis_del += d['nvm'][x] * g
        cd = np.concatenate(cell_dev); cu = np.concatenate(cell_usr)
        cn = np.concatenate(cell_n).astype(float); ck = np.concatenate(cell_k).astype(float)
        n_dev = np.bincount(cd, cn, N); k_dev = np.bincount(cd, ck, N)
        n_usr = np.bincount(cu, cn, U); k_usr = np.bincount(cu, ck, U)
        # ---- accounting of benign service and residual attack traffic
        gdev = np.where(Q | E, 0.0, np.where(Rr, PHI_R, 1.0))
        own_s = S[own]
        f = gdev * np.where(own_s, 0.5, 1.0) * np.where(served, 1.0, 0.5)
        benign_dev = ~comp
        ben_gen += (d['nb'] * benign_dev).sum(); ben_del += (d['nb'] * f * benign_dev).sum()
        mal_gen += (nm + nf).sum() + vis_mal
        mal_del += ((nm + nf) * gdev).sum() + vis_del
        never = np.isinf(W.onset) & (kind == 'benign')
        collat += (never & ~served & ~Q & ~E).sum()
        res_benign += (Rr & never).sum(); ben_dev_windows += never.sum()
        ncomp, labs = connected_components(A, directed=False)
        lcc.append(np.bincount(labs[active]).max() / N if active.any() else 0)
        # ---- trust update
        o = dict(n_dev=n_dev, k_dev=k_dev, n_usr=n_usr, k_usr=k_usr, cell_dev=cd, cell_usr=cu, cell_n=cn,
                 cell_k=ck, A=A)
        rd, ru = method.step(t, o)
        obs = active & ~E
        if t >= 20:
            lab_d.append(comp[obs]); sc_d.append(rd[obs])
            uo = ~S & ~perm_u
            truth_u = np.zeros(U, bool); truth_u[U0:] = t >= ins_on
            lab_u.append(truth_u[uo]); sc_u.append(ru[uo])
        if record:
            recs.append(dict(t=t, rd=rd.copy(), comp=comp.copy(), Q=Q.copy(), E=E.copy(), R=Rr.copy(), ell=getattr(method, 'last_ell_d', None)))
        # ---- policy (effective from the next window)
        ctx = dict(A=A, svc=W.svc, active=active, demand=getattr(method, 'mu', W.rate) / rate_mean)
        iso_q, iso_e, res, sus, stp = policy(rd, ru, ctx)
        if t < 20:
            iso_q[:] = iso_e[:] = res[:] = sus[:] = stp[:] = False
        new_q = iso_q & ~Q & ~E & ~perm_d; new_e = iso_e & ~Q & ~E & ~perm_d
        for i in np.where(new_q | new_e)[0]:
            if comp[i]:
                t_cont_d[i] = min(t_cont_d[i], t)
            else:
                false_iso[i] = True; false_iso_events += 1
            due_d[i] = (t + review, bool(comp[i]))
        Q |= new_q; E |= new_e
        Rr = res & ~Q & ~E
        new_s = sus & ~S & ~perm_u
        for u in np.where(new_s)[0]:
            truth = u >= U0 and t >= ins_on[u - U0]
            if truth:
                t_cont_u[u] = min(t_cont_u[u], t)
            else:
                false_sus[u] = True
            due_u[u] = (t + review, bool(truth))
        S |= new_s; St = stp & ~S
        # ---- forensic review: compromised entities stay contained, benign ones are restored and reset
        for i in [i for i, tt in due_d.items() if tt[0] <= t]:
            was = due_d.pop(i)[1]
            if was:
                perm_d[i] = True
            else:
                Q[i] = E[i] = False; method.reset(dev=np.array([i]))
        for u in [u for u, tt in due_u.items() if tt[0] <= t]:
            was = due_u.pop(u)[1]
            if was:
                perm_u[u] = True
            else:
                S[u] = False; method.reset(usr=np.array([u]))
        # ---- lateral movement over the live graph from infected devices that can still send
        src = comp & (kind != 'flood') & ~Q & ~E & (onset <= t)
        src &= (kind == 'worm')
        if src.any():
            Ed = W.topo[t]; uu = d['worm_u']; a_, b_ = np.r_[Ed[:, 0], Ed[:, 1]], np.r_[Ed[:, 1], Ed[:, 0]]
            p = W.sc.worm_p * np.where(Rr[a_], PHI_R, 1.0)
            hit = src[a_] & ~Q[b_] & np.isinf(onset[b_]) & (kind[b_] == 'benign') & (uu < p)
            for b in np.unique(b_[hit]):
                onset[b] = t + 1; kind[b] = 'worm'; mrel[b] = 0.25
    # ---- metrics
    never = np.isinf(W.onset) & (W.kind == 'benign') & np.isinf(onset)
    out = dict(method=method.name)
    out['dev_FIR'] = float(false_iso[never].mean())
    out['false_iso_events'] = int(false_iso_events)
    out['usr_FSR'] = float(false_sus[:U0].mean())
    out['avail'] = float(ben_del / max(ben_gen, 1))
    out['mal_delivered'] = float(mal_del); out['mal_generated'] = float(mal_gen)
    out['collateral'] = float(collat / max(ben_dev_windows, 1))
    out['restrict_benign'] = float(res_benign / max(ben_dev_windows, 1))
    out['lcc'] = float(np.mean(lcc))
    for nm in ('low', 'onoff', 'flood', 'worm', 'gw'):
        idx = np.where(kind == nm)[0]
        idx = idx[onset[idx] < T - 5]
        if len(idx):
            dt = t_cont_d[idx] - onset[idx]
            out[f'{nm}_n'] = int(len(idx)); out[f'{nm}_DR'] = float(np.isfinite(dt).mean())
            out[f'{nm}_delay'] = float(np.median(np.where(np.isfinite(dt), dt, T - onset[idx])))
    out['worm_infected'] = int((kind == 'worm').sum())
    iu = np.arange(U0, U); iu = iu[ins_on < T - 5]
    if len(iu):
        dt = t_cont_u[iu] - ins_on[iu - U0]
        out['hop_DR'] = float(np.isfinite(dt).mean()); out['hop_delay'] = float(np.median(np.where(np.isfinite(dt), dt, T - ins_on[iu - U0])))
    # collateral false isolation of the devices a credential-hopping insider visited
    out['hop_dev_FIR'] = float(false_iso[np.unique(np.concatenate(W.ins_reach))].mean()) if W.n_ins else np.nan
    yd = np.concatenate(lab_d); sd = np.concatenate(sc_d)
    out['auroc_dev'] = float(roc_auc_score(yd, sd)) if 0 < yd.mean() < 1 else np.nan
    yu = np.concatenate(lab_u); su = np.concatenate(sc_u)
    out['auroc_usr'] = float(roc_auc_score(yu, su)) if 0 < yu.mean() < 1 else np.nan
    if method.prob:
        p = np.clip(sd, 0, 1)
        out['brier_dev'] = float(np.mean((p - yd) ** 2))
        bins = np.minimum((p * 10).astype(int), 9)
        ece = 0.0
        for b in range(10):
            m = bins == b
            if m.any():
                ece += m.mean() * abs(p[m].mean() - yd[m].mean())
        out['ece_dev'] = float(ece)
        tier = np.digitize(p, [0.1, 0.5])
        out['tier'] = [[int(((tier == j) & (yd == c)).sum()) for c in (0, 1)] for j in range(3)]
    if record:
        out['recs'] = recs; out['onset_final'] = onset
    return out
