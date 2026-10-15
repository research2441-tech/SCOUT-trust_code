"""Entity-level trust methods. Every method receives the same per-window observation and returns a
risk score for each device and each user account; a policy turns risk into actions.

SCOUT (proposed): truncated beta-binomial window evidence against an enrolled entity-relative null,
one-sided negative-binomial volume evidence, responsibility-weighted user/device attribution, a
three-state (benign / active / dormant) compromise filter, and a contagion hazard that propagates
scrutiny over the live communication graph instead of propagating blame."""
import numpy as np
from scipy.special import betaln, betaincc

EPS = 1e-12


def bb_loglr(k, n, q0, a=1.0, b=4.0):
    """log of the truncated beta-binomial evidence ratio: theta ~ Beta(a, b) restricted to (q0, 1) under
    compromise against theta = q0 under the benign null."""
    k = np.asarray(k, float); n = np.asarray(n, float); q0 = np.broadcast_to(np.asarray(q0, float), k.shape)
    num = betaln(k + a, n - k + b) + np.log(np.maximum(betaincc(k + a, n - k + b, q0), 1e-300))
    den = betaln(a, b) + np.log(betaincc(a, b, q0))
    null = k * np.log(q0) + (n - k) * np.log1p(-q0)
    out = num - den - null
    return np.where(n > 0, out, 0.0)


def nb_loglr(n, mu, psi, r0):
    """One-sided negative-binomial volume evidence: mean psi*mu under compromise against mu, dispersion r0
    (scalar or per entity)."""
    mu = np.maximum(mu, 0.2)
    v = n * np.log(psi) + (n + r0) * (np.log(r0 + mu) - np.log(r0 + psi * mu))
    return np.maximum(v, 0.0)


# ------------------------------------------------------------------------------------- baselines
class Base:
    prob = True                      # risk is a probability-like quantity in [0, 1]
    name = 'base'

    def __init__(self, W, P):
        self.W, self.P = W, P
        self.N, self.U = W.N, W.U

    def reset(self, dev=None, usr=None):
        pass


class NoDefense(Base):
    name = 'None'

    def step(self, t, o):
        return np.zeros(self.N), np.zeros(self.U)


class EWMA(Base):
    """T_t = lambda T_{t-1} + (1 - lambda) Q_t with Q_t = 1 - k/n (the update in the original proposal)."""
    name = 'EWMA'

    def __init__(self, W, P, lam=0.9):
        super().__init__(W, P); self.lam = lam
        self.Td = np.ones(self.N); self.Tu = np.ones(self.U)

    def _up(self, T, n, k):
        Q = np.where(n > 0, 1 - k / np.maximum(n, 1), T)
        return self.lam * T + (1 - self.lam) * Q

    def step(self, t, o):
        self.Td = self._up(self.Td, o['n_dev'], o['k_dev']); self.Tu = self._up(self.Tu, o['n_usr'], o['k_usr'])
        return 1 - self.Td, 1 - self.Tu

    def reset(self, dev=None, usr=None):
        if dev is not None: self.Td[dev] = 1.0
        if usr is not None: self.Tu[usr] = 1.0


class BetaRep(Base):
    """Beta reputation with exponential forgetting."""
    name = 'Beta'

    def __init__(self, W, P, gamma=0.95):
        super().__init__(W, P); self.g = gamma
        self.ad, self.bd = np.ones(self.N), np.ones(self.N); self.au, self.bu = np.ones(self.U), np.ones(self.U)

    def step(self, t, o):
        self.ad = self.g * self.ad + (o['n_dev'] - o['k_dev']); self.bd = self.g * self.bd + o['k_dev']
        self.au = self.g * self.au + (o['n_usr'] - o['k_usr']); self.bu = self.g * self.bu + o['k_usr']
        return self.bd / (self.ad + self.bd), self.bu / (self.au + self.bu)

    def reset(self, dev=None, usr=None):
        if dev is not None: self.ad[dev] = 1; self.bd[dev] = 1
        if usr is not None: self.au[usr] = 1; self.bu[usr] = 1


class Asym(Base):
    """Asymmetric (slow-gain, fast-loss) trust update used against on-off attacks."""
    name = 'Asym'

    def __init__(self, W, P, up=0.02, down=0.4):
        super().__init__(W, P); self.up, self.down = up, down
        self.Td = np.ones(self.N); self.Tu = np.ones(self.U)

    def _up(self, T, n, k):
        Q = np.where(n > 0, 1 - k / np.maximum(n, 1), T)
        return T + np.where(Q >= T, self.up, self.down) * (Q - T)

    def step(self, t, o):
        self.Td = self._up(self.Td, o['n_dev'], o['k_dev']); self.Tu = self._up(self.Tu, o['n_usr'], o['k_usr'])
        return 1 - self.Td, 1 - self.Tu

    def reset(self, dev=None, usr=None):
        if dev is not None: self.Td[dev] = 1.0
        if usr is not None: self.Tu[usr] = 1.0


class CUSUM(Base):
    """Binomial CUSUM on the flag count with the global benign rate as null."""
    name = 'CUSUM'; prob = False

    def __init__(self, W, P, theta1=0.1):
        super().__init__(W, P)
        q0 = max(P['q0_global'], 1e-4); self.a = np.log(theta1 / q0); self.b = np.log((1 - theta1) / (1 - q0))
        self.Sd = np.zeros(self.N); self.Su = np.zeros(self.U)

    def step(self, t, o):
        self.Sd = np.maximum(0, self.Sd + o['k_dev'] * self.a + (o['n_dev'] - o['k_dev']) * self.b)
        self.Su = np.maximum(0, self.Su + o['k_usr'] * self.a + (o['n_usr'] - o['k_usr']) * self.b)
        return self.Sd, self.Su

    def reset(self, dev=None, usr=None):
        if dev is not None: self.Sd[dev] = 0
        if usr is not None: self.Su[usr] = 0


class BlameDiff(BetaRep):
    """Beta reputation followed by diffusion of risk over the communication graph (blame propagation)."""
    name = 'BetaDiff'

    def __init__(self, W, P, beta=0.3):
        super().__init__(W, P); self.beta = beta

    def step(self, t, o):
        rd, ru = super().step(t, o)
        A = o['A']; deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        rd = (1 - self.beta) * rd + self.beta * (A @ rd) / deg
        return rd, ru


class Learned(Base):
    """Supervised entity-risk model (gradient boosting on temporal and neighbourhood features),
    trained on validation worlds that contain known attack families only."""
    name = 'LearnedRisk'

    def __init__(self, W, P, model=None):
        super().__init__(W, P); self.model = model
        self.F = {k: np.zeros(self.N) for k in ('K1', 'N1', 'K5', 'N5', 'K20', 'N20', 'R')}
        self.G = {k: np.zeros(self.U) for k in ('K1', 'N1', 'K5', 'N5', 'K20', 'N20', 'R')}
        self.mu_d = np.ones(self.N); self.mu_u = np.ones(self.U); self.t = 0

    @staticmethod
    def _upd(S, n, k, mu):
        for lab, lam in (('1', 0.5), ('5', 0.8), ('20', 0.95)):
            S['K' + lab] = lam * S['K' + lab] + k; S['N' + lab] = lam * S['N' + lab] + n
        S['R'] = 0.5 * S['R'] + 0.5 * n / np.maximum(mu, 0.5)

    def feats(self, o):
        n, k = o['n_dev'], o['k_dev']
        if self.t < 20:
            self.mu_d = (self.mu_d * self.t + n) / (self.t + 1); self.mu_u = (self.mu_u * self.t + o['n_usr']) / (self.t + 1)
        self._upd(self.F, n, k, self.mu_d); self._upd(self.G, o['n_usr'], o['k_usr'], self.mu_u); self.t += 1
        def mk(S, nbr, deg, typ):
            r1, r5, r20 = [S['K' + l] / np.maximum(S['N' + l], 1) for l in ('1', '5', '20')]
            return np.c_[r1, r5, r20, S['R'], np.log1p(S['N5']), nbr, np.log1p(deg), np.full(len(r1), typ)]
        A = o['A']; deg = np.asarray(A.sum(1)).ravel()
        r5 = self.F['K5'] / np.maximum(self.F['N5'], 1)
        nbr = (A @ r5) / np.maximum(deg, 1)
        return mk(self.F, nbr, deg, 0.0), mk(self.G, np.zeros(self.U), np.zeros(self.U), 1.0)

    def step(self, t, o):
        Fd, Fu = self.feats(o)
        if self.model is None:
            return np.zeros(self.N), np.zeros(self.U)
        p = self.model.predict_proba(np.r_[Fd, Fu])[:, 1]
        return p[:self.N], p[self.N:]

    def reset(self, dev=None, usr=None):
        for S, idx in ((self.F, dev), (self.G, usr)):
            if idx is not None:
                for k in S:
                    S[k][idx] = 0


# ------------------------------------------------------------------------------------- proposed
class SCOUT(Base):
    name = 'SCOUT'

    def __init__(self, W, P, dormant=True, contagion='noisyor', attribution=True, volume=True, enrol=True,
                 clip=None, r=None, h0=None, omega=None, s=0.2, hmax=0.03, psi=4.0, q=None, a=1.0, b=4.0, beta_blame=0.3):
        super().__init__(W, P)
        self.dormant, self.contagion, self.attr, self.vol, self.enrol = dormant, contagion, attribution, volume, enrol
        self.c = clip if clip is not None else P.get('clip', 6.0)
        self.r = r if r is not None else P.get('r', 0.05)
        self.h0 = h0 if h0 is not None else P.get('h0', 2e-3)
        self.omega = omega if omega is not None else P.get('omega', 1.0)
        self.s = s if dormant else 0.0; self.hmax, self.psi, self.a, self.b = hmax, psi, a, b
        self.q = q if q is not None else W.sc.q; self.bb = beta_blame
        self.pd = np.zeros((self.N, 3)); self.pd[:, 0] = 1
        self.pu = np.zeros((self.U, 3)); self.pu[:, 0] = 1
        self.t = 0; self.En = {'nd': np.zeros(self.N), 'kd': np.zeros(self.N), 'nu': np.zeros(self.U), 'ku': np.zeros(self.U),
                               'n2': np.zeros(self.N)}
        self.q0d = np.full(self.N, self.q); self.q0u = np.full(self.U, self.q)
        self.mu = np.full(self.N, 6.0); self.r0 = 5.0; self.T_enrol = 20
        self.last_ell_d = np.zeros(self.N)

    # ---- filter (three-state forward recursion; s = 0 collapses it to the two-state Shiryaev posterior)
    def _filter(self, p, ell, h):
        pB, pA, pD = p[:, 0], p[:, 1], p[:, 2]
        prB = pB * (1 - h); prA = pB * h + pA * (1 - self.s) + pD * self.r; prD = pA * self.s + pD * (1 - self.r)
        e = np.exp(ell)
        Z = prB + prA * e + prD
        return np.c_[prB / Z, prA * e / Z, prD / Z]

    def risk(self):
        return 1 - self.pd[:, 0], 1 - self.pu[:, 0]

    def report(self):
        rd, ru = self.risk()
        return (self.rep_d if getattr(self, 'rep_d', None) is not None else rd), ru

    def step(self, t, o):
        nd, kd, nu, ku = o['n_dev'], o['k_dev'], o['n_usr'], o['k_usr']
        if self.t < self.T_enrol:                       # enrolment: entity-relative null and volume baseline
            E = self.En; E['nd'] += nd; E['kd'] += kd; E['nu'] += nu; E['ku'] += ku; E['n2'] += nd.astype(float) ** 2
            self.t += 1
            if self.t == self.T_enrol:
                from scipy.stats import beta as Bd
                m = 20.0
                if self.enrol:            # upper 90% posterior quantile of each entity's own benign flag rate
                    self.q0d = np.clip(Bd.ppf(0.9, E['kd'] + self.q * m, E['nd'] - E['kd'] + (1 - self.q) * m), self.q, 0.5)
                    self.q0u = np.clip(Bd.ppf(0.9, E['ku'] + self.q * m, E['nu'] - E['ku'] + (1 - self.q) * m), self.q, 0.5)
                self.mu = np.maximum(E['nd'] / self.T_enrol, 0.5)
                var = np.maximum(E['n2'] / self.T_enrol - self.mu ** 2, 0)
                phi_i = np.maximum(var / self.mu ** 2 - 1 / self.mu, 0)        # moment estimate of 1/r per entity
                phi_g = max(float(np.mean(phi_i)), 1e-3)
                phi = (self.T_enrol * phi_i + 20.0 * phi_g) / (self.T_enrol + 20.0)
                self.r0 = np.clip(1.0 / np.maximum(phi, 1e-3), 1.0, 50.0)
            return self.risk()
        self.t += 1
        pid, piu = self.risk()
        # ---- cell evidence and responsibility-weighted attribution
        cd, cu, cn, ck = o['cell_dev'], o['cell_usr'], o['cell_n'], o['cell_k']
        ell_c = bb_loglr(ck, cn, self.q0d[cd], self.a, self.b)
        if self.attr:
            ea = 0.01; mx = np.maximum(pid[cd], piu[cu]) + ea
            rd, ru = (pid[cd] + ea) / mx, (piu[cu] + ea) / mx
        else:
            rd = ru = np.ones(len(cd))
        ell_d = np.bincount(cd, rd * ell_c, self.N); ell_u = np.bincount(cu, ru * ell_c, self.U)
        if self.vol:
            ell_d = ell_d + nb_loglr(nd, self.mu, self.psi, self.r0)
        ell_d = self.omega * np.clip(ell_d, -self.c, self.c); ell_u = self.omega * np.clip(ell_u, -self.c, self.c)
        self.last_ell_d = ell_d
        # ---- hazard: contagion of scrutiny over the live communication graph
        h = np.full(self.N, self.h0)
        A = o['A']; deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        if self.contagion == 'noisyor':
            v = np.log1p(-np.clip(pid / deg, 0, 0.999))
            h = self.h0 + (self.hmax - self.h0) * (1 - np.exp(A @ v))
        self.pd = self._filter(self.pd, ell_d, h)
        self.pu = self._filter(self.pu, ell_u, np.full(self.U, self.h0))
        self.rep_d = None
        if self.contagion == 'blame_rec':             # ablation: recursive diffusion of the filter state itself
            pi = 1 - self.pd[:, 0]
            pi2 = (1 - self.bb) * pi + self.bb * (A @ pi) / deg
            scale = np.where(pi > 1e-12, pi2 / np.maximum(pi, 1e-12), 0.0)
            self.pd[:, 1:] = self.pd[:, 1:] * scale[:, None]
            self.pd[:, 1] += np.where(pi <= 1e-12, pi2, 0.0)
            self.pd[:, 0] = 1 - self.pd[:, 1] - self.pd[:, 2]
        if self.contagion == 'blame':                 # ablation: decisions use a one-step neighbour-smoothed posterior
            pi = 1 - self.pd[:, 0]
            self.rep_d = (1 - self.bb) * pi + self.bb * (A @ pi) / deg
        # volume baseline follows the entity only while it is trusted (prevents baseline poisoning)
        ok = (1 - self.pd[:, 0]) < 0.05
        self.mu = np.where(ok, 0.95 * self.mu + 0.05 * np.maximum(nd, 0.5), self.mu)
        return self.report()

    def reset(self, dev=None, usr=None):
        if dev is not None: self.pd[dev] = [1, 0, 0]
        if usr is not None: self.pu[usr] = [1, 0, 0]
