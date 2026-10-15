"""Flow-level open-set evidence (Section III-B) and the leave-one-family-out experiment E1.

For every (dataset, held-out family, seed) the script fits the known-attack classifier and the benign
novelty models on known data only, computes every competing flow score, converts each score into a
benign-conformity value u in (0, 1] with a disjoint benign calibration split, and stores the u-values of
the validation and test pools so the entity-level experiments replay identical evidence."""
import os
import sys, json, time, numpy as np, xgboost as xgb
from scipy.special import logsumexp
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from data import LOADERS, lofo_partition

RES = os.environ.get('SCOUT_RESULTS', 'results')


def rank_p(ref_sorted, s):
    """Conformal p-value of score s against a sorted benign reference: (1 + #{ref >= s}) / (n + 1)."""
    n = len(ref_sorted)
    return (1.0 + n - np.searchsorted(ref_sorted, s, side='left')) / (n + 1.0)


def cauchy_stat(P, wts):
    """Cauchy combination of dependent p-values: T = sum_j w_j tan((1/2 - p_j) pi)."""
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return (np.tan((0.5 - P) * np.pi) * wts).sum(1)


class Evidence:
    def __init__(self, seed=0, n_ref=8000, k=5):
        self.seed, self.n_ref, self.k = seed, n_ref, k

    def fit(self, X, y, benign, Xfit):
        rng = np.random.default_rng(self.seed)
        self.classes = np.array(sorted(set(y))); self.ib = int(np.where(self.classes == benign)[0][0])
        yi = np.searchsorted(self.classes, y)
        cw = len(yi) / (len(self.classes) * np.bincount(yi))
        self.clf = xgb.XGBClassifier(n_estimators=250, max_depth=8, learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, tree_method='hist', n_jobs=2, random_state=self.seed,
                                     objective='multi:softprob', eval_metric='mlogloss')
        self.clf.fit(X, yi, sample_weight=cw[yi])
        Xb = X[y == benign]
        self.sc = StandardScaler().fit(Xb)
        ref = self.sc.transform(Xb[rng.choice(len(Xb), min(self.n_ref, len(Xb)), replace=False)])
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(ref)
        self.iso = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=2).fit(ref)
        R = self.raw(Xfit)                    # rank references from a disjoint benign split
        self.ref = {c: np.sort(R[c]) for c in ('clf', 'knn', 'iso')}
        return self

    def raw(self, X):
        P = self.clf.predict_proba(X)
        Z = self.clf.predict(X, output_margin=True)
        Xs = self.sc.transform(X)
        d, _ = self.nn.kneighbors(Xs)
        return dict(clf=1 - P[:, self.ib], msp=1 - P.max(1), energy=-logsumexp(Z, axis=1),
                    knn=d.mean(1), iso=-self.iso.score_samples(Xs), P=P)

    def scores(self, X):
        """Every candidate flow score, oriented so that larger means less benign."""
        R = self.raw(X)
        rp = np.c_[[rank_p(self.ref[c], R[c]) for c in ('clf', 'knn', 'iso')]].T
        R['minp2'] = -np.minimum(rp[:, 0], rp[:, 1])                  # rank-max (Bonferroni-type) fusion
        R['cauchy2'] = cauchy_stat(rp[:, :2], np.array([0.5, 0.5]))
        R['cauchy3'] = cauchy_stat(rp, np.array([1 / 3, 1 / 3, 1 / 3]))  # proposed
        return R

    def fit_linear(self, Rv, yv):
        """Weighted-sum fusion S = a N + b A + c U with weights learned on benign vs KNOWN attacks."""
        F = self._lin_feats(Rv)
        self.lo, self.hi = np.percentile(F, 1, 0), np.percentile(F, 99, 0) + 1e-9
        self.lin = LogisticRegression(max_iter=2000).fit(np.clip((F - self.lo) / (self.hi - self.lo), 0, 1), yv)

    @staticmethod
    def _lin_feats(R):
        return np.c_[R['knn'], R['iso'], R['msp'], R['clf']]

    def linear(self, R):
        F = self._lin_feats(R)
        return self.lin.decision_function(np.clip((F - self.lo) / (self.hi - self.lo), 0, 1))


SCORES = ['clf', 'msp', 'energy', 'knn', 'iso', 'linear', 'minp2', 'cauchy2', 'cauchy3']


def run(ds, seeds=(0, 1, 2)):
    D = LOADERS[ds](); X, y, B = D['X'], D['y'], D['benign']
    out = []
    for seed in seeds:
        for held in D['families']:
            t0 = time.time()
            P = lofo_partition(D, held, seed)
            ev = Evidence(seed).fit(X[P['train']], y[P['train']], B, X[P['fit']])
            vidx = np.concatenate(list(P['val'].values())); yv = (y[vidx] != B).astype(int)
            Rv = ev.scores(X[vidx]); ev.fit_linear(Rv, yv); Rv['linear'] = ev.linear(Rv)
            Rc = ev.scores(X[P['cal']]); Rc['linear'] = ev.linear(Rc)
            cal = {s: np.sort(Rc[s]) for s in SCORES}
            pools = {}
            for split in ('val', 'test'):
                for grp, idx in P[split].items():
                    R = ev.scores(X[idx]); R['linear'] = ev.linear(R)
                    pools[(split, grp)] = (R, idx)
            # ---- metrics: unknown family vs benign test, known families vs benign test, known-class macro-F1
            Rb, _ = pools[('test', 'benign')]; Rh, _ = pools[('test', held)]
            kn = [g for g in P['test'] if g not in ('benign', held)]
            rec = dict(dataset=ds, seed=seed, held=held, n_unknown=len(P['test'][held]), n_benign_test=len(P['test']['benign']))
            for s in SCORES:
                ub, uh = rank_p(cal[s], Rb[s]), rank_p(cal[s], Rh[s])
                uk = np.concatenate([rank_p(cal[s], pools[('test', g)][0][s]) for g in kn])
                rec[f'{s}_auroc_unk'] = float(roc_auc_score(np.r_[np.zeros(len(ub)), np.ones(len(uh))], -np.r_[ub, uh]))
                for a in (0.001, 0.01, 0.05):
                    rec[f'{s}_fpr@{a}'] = float((ub <= a).mean())
                    rec[f'{s}_tpru@{a}'] = float((uh <= a).mean())
                    rec[f'{s}_tprk@{a}'] = float((uk <= a).mean())
            # known-class macro-F1 of the classifier on benign + known families
            kidx = np.concatenate([P['test']['benign']] + [P['test'][g] for g in kn])
            pred = ev.classes[ev.clf.predict_proba(X[kidx]).argmax(1)]
            rec['known_macroF1'] = float(f1_score(y[kidx], pred, average='macro'))
            # unknown flows: which known class does the classifier assign?
            ph = ev.classes[Rh['P'].argmax(1)]
            rec['unk_assigned_benign'] = float((ph == B).mean())
            rec['time_s'] = time.time() - t0
            out.append(rec)
            # ---- store conformity values for every pool and every score family used downstream
            store = {}
            for (split, grp), (R, idx) in pools.items():
                for s in ('cauchy3', 'minp2', 'linear', 'clf', 'knn'):
                    store[f'{split}|{grp}|{s}'] = rank_p(cal[s], R[s]).astype(np.float32)
            np.savez_compressed(f'{RES}/pool_{ds}_{held}_{seed}.npz', **store)
            print(ds, seed, held, 'AUROC unk cauchy3 %.3f lin %.3f clf %.3f | TPRu@1%% cauchy3 %.3f lin %.3f clf %.3f minp %.3f | FPR %.4f | F1k %.3f | %.0fs' % (
                rec['cauchy3_auroc_unk'], rec['linear_auroc_unk'], rec['clf_auroc_unk'], rec['cauchy3_tpru@0.01'],
                rec['linear_tpru@0.01'], rec['clf_tpru@0.01'], rec['minp2_tpru@0.01'], rec['cauchy3_fpr@0.01'],
                rec['known_macroF1'], rec['time_s']), flush=True)
            json.dump(out, open(f'{RES}/flow_{ds}.json', 'w'))


if __name__ == '__main__':
    run(sys.argv[1])
