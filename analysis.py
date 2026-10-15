"""Aggregates every result file into the numbers, tables and figures of Section VI.
Every value printed in the manuscript is produced here from the raw result files."""
import json, pickle, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.environ.get('SCOUT_RESULTS', 'results'); FIG = os.environ.get('SCOUT_FIGURES', 'figures')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.7, 'lines.linewidth': 1.2,
                     'axes.grid': True, 'grid.color': '#D9D9D9', 'grid.linewidth': 0.5, 'axes.axisbelow': True})
DS = ['TON_IoT', 'CICIoT2023']
BASE = ['EWMA', 'Beta', 'Asym', 'CUSUM', 'BetaDiff', 'Shiryaev', 'LearnedRisk']
MAIN = BASE + ['SCOUT']
ABL = ['SCOUT', 'SCOUT-noDormant', 'SCOUT-noContagion', 'SCOUT-blame', 'SCOUT-blameRec', 'SCOUT-noAttribution', 'SCOUT-noVolume',
       'SCOUT-noEnrol', 'SCOUT-noUtility', 'SCOUT-noTemper']
ABL_LAB = {'SCOUT': 'SCOUT (full)', 'SCOUT-noDormant': 'no dormant state ($s=0$)', 'SCOUT-noContagion': 'no contagion',
           'SCOUT-blame': 'one-step blame smoothing', 'SCOUT-blameRec': 'recursive blame diffusion', 'SCOUT-noAttribution': 'no attribution', 'SCOUT-noVolume': 'no volume term',
           'SCOUT-noEnrol': 'global null $q$', 'SCOUT-noUtility': 'threshold actions', 'SCOUT-noTemper': '$\\omega=1$'}
MK = {'EWMA': ('o', '#7F7F7F', '-'), 'Beta': ('s', '#A6A6A6', '--'), 'Asym': ('^', '#595959', ':'),
      'CUSUM': ('v', '#BFBFBF', '-.'), 'BetaDiff': ('D', '#8C8C8C', '--'), 'Shiryaev': ('P', '#404040', '-.'),
      'LearnedRisk': ('X', '#6E6E6E', ':'), 'SCOUT': ('o', '#000000', '-'), 'SCOUT-noDormant': ('s', '#5A5A5A', '--'),
      'SCOUT-noUtility': ('*', '#8C8C8C', '-.')}


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    r = np.random.default_rng(seed); m = r.choice(x, (B, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None]).sum() - (a[:, None] < b[None]).sum()) / (len(a) * len(b)))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj


def load_entity():
    R = pd.DataFrame([json.loads(l) for l in open(f'{RES}/entity_main.jsonl')])
    key = ['dataset', 'held', 'seed', 'kind']
    if os.path.exists(f'{RES}/entity_blame.jsonl'):          # blame ablation re-run (decision-level diffusion)
        Bl = pd.DataFrame([json.loads(l) for l in open(f'{RES}/entity_blame.jsonl')])
        rec = R[R.method == 'SCOUT-blame'].assign(method='SCOUT-blameRec')   # first pass: recursive diffusion of the filter state
        R = pd.concat([R[R.method != 'SCOUT-blame'], rec, Bl], ignore_index=True)
    if os.path.exists(f'{RES}/entity_ol.jsonl'):             # open-loop trust quality without lateral spread
        O = pd.DataFrame([json.loads(l) for l in open(f'{RES}/entity_ol.jsonl')]).drop_duplicates(key + ['method'])
        O = O.rename(columns={c: 'ol_' + c for c in ('auroc_dev', 'auroc_usr', 'brier_dev', 'ece_dev', 'tier')})
        R = R.drop(columns=[c for c in R.columns if c.startswith('ol_')]).merge(O, on=key + ['method'], how='left')
    none = R[R.method == 'None'].set_index(key)
    R = R.join(none[['mal_delivered', 'collateral', 'avail', 'lcc']].rename(columns=lambda c: c + '_none'), on=key)
    R['rat'] = R.mal_delivered / R.mal_delivered_none
    R['coll_x'] = R.collateral - R.collateral_none
    R['lcc_x'] = R.lcc - R.lcc_none
    R['avail_x'] = R.avail - R.avail_none
    return R


# ----------------------------------------------------------------------------------------------- E1 flow level
def flow():
    F = pd.concat([pd.DataFrame(json.load(open(f'{RES}/flow_{d}.json'))) for d in DS])
    S = ['msp', 'energy', 'knn', 'iso', 'linear', 'clf', 'minp2', 'cauchy3']
    rows = [['Dataset', 'Held family', 'TPR clf', 'TPR linear', 'TPR Cauchy', 'AUROC clf', 'AUROC Cauchy', 'Benign FPR', 'Known F1']]
    agg = F.groupby(['dataset', 'held']).mean(numeric_only=True)
    sd = F.groupby(['dataset', 'held']).std(numeric_only=True)
    for ds in DS:
        for h in sorted(F[F.dataset == ds].held.unique(), key=lambda s: s.lower()):
            a, s_ = agg.loc[(ds, h)], sd.loc[(ds, h)]
            rows.append([ds if h == sorted(F[F.dataset == ds].held.unique(), key=lambda s: s.lower())[0] else '', h,
                         '%.3f' % a['clf_tpru@0.01'], '%.3f' % a['linear_tpru@0.01'], '%.3f' % a['cauchy3_tpru@0.01'],
                         '%.3f' % a['clf_auroc_unk'], '%.3f' % a['cauchy3_auroc_unk'], '%.4f' % a['clf_fpr@0.01'],
                         '%.3f' % a['known_macroF1']])
    m = F.groupby('dataset').mean(numeric_only=True)
    for ds in DS:
        rows.append(['', 'Mean (%s)' % ('TON' if ds == 'TON_IoT' else 'CIC'), '%.3f' % m.loc[ds, 'clf_tpru@0.01'],
                     '%.3f' % m.loc[ds, 'linear_tpru@0.01'], '%.3f' % m.loc[ds, 'cauchy3_tpru@0.01'],
                     '%.3f' % m.loc[ds, 'clf_auroc_unk'], '%.3f' % m.loc[ds, 'cauchy3_auroc_unk'],
                     '%.4f' % m.loc[ds, 'clf_fpr@0.01'], '%.3f' % m.loc[ds, 'known_macroF1']])
    # paired test clf vs cauchy3 / linear at the operating point across folds
    st = {}
    for o in ('cauchy3', 'linear', 'minp2', 'knn', 'iso', 'msp', 'energy'):
        d = F['clf_tpru@0.01'] - F[f'{o}_tpru@0.01']
        st[o] = dict(mean=float(d.mean()), p=float(wilcoxon(F['clf_tpru@0.01'], F[f'{o}_tpru@0.01']).pvalue),
                     wins=int((d < 0).sum()), n=len(d))
        da = F['clf_auroc_unk'] - F[f'{o}_auroc_unk']
        st[o]['auroc_mean'] = float(da.mean()); st[o]['auroc_wins'] = int((da < 0).sum())
    # figure 3
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw=dict(width_ratios=[9, 7], wspace=0.08))
    lab = {'msp': 'MSP', 'energy': 'Energy', 'knn': '$k$-NN', 'iso': 'iForest', 'linear': 'Weighted sum',
           'clf': 'Classifier $u_f$', 'minp2': 'Min-$p$', 'cauchy3': 'Cauchy'}
    mk = {'msp': ('v', '#BFBFBF'), 'energy': ('^', '#A6A6A6'), 'knn': ('<', '#8C8C8C'), 'iso': ('>', '#7F7F7F'),
          'linear': ('s', '#595959'), 'clf': ('o', '#000000'), 'minp2': ('D', '#6E6E6E'), 'cauchy3': ('P', '#404040')}
    for j, ds in enumerate(DS):
        hs = sorted(F[F.dataset == ds].held.unique(), key=lambda s: s.lower())
        x = np.arange(len(hs))
        for k, s in enumerate(S):
            v = [agg.loc[(ds, h), f'{s}_tpru@0.01'] for h in hs]
            ax[j].plot(x + (k - 3.5) * 0.095, v, ls='none', marker=mk[s][0], ms=4.2 if s != 'clf' else 5,
                       mfc=mk[s][1] if s not in ('clf',) else 'black', mec='black', mew=0.5, label=lab[s])
        ax[j].set_xticks(x); ax[j].set_xticklabels(hs, rotation=35, ha='right')
        ax[j].set_ylim(-0.03, 1.05); ax[j].set_title('(%s) %s' % ('ab'[j], ds), loc='left')
        if j == 0: ax[j].set_ylabel('Unseen-family TPR at 1% benign FPR')
        else: ax[j].set_yticklabels([])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.17), columnspacing=1.0, handletextpad=0.3)
    fig.savefig(f'{FIG}/fig3_flow.png', dpi=600, bbox_inches='tight', pad_inches=0.03); plt.close(fig)
    return dict(tab=rows, st=st, mean=m, F=F)


# ----------------------------------------------------------------------------------------------- E2 main
def fmt_pct(x, d=1):
    return ('%.' + str(d) + 'f') % (100 * x)


def main_tables(R):
    rows = [['Method', 'FIR', 'FSR', 'RAT', 'Low', 'On-off', 'Flood', 'Gate', 'Lateral', 'Infected', 'Hop', 'Delay low']]
    G = R[R.method.isin(MAIN)].groupby(['dataset', 'method']).mean(numeric_only=True)
    for ds in [d for d in DS if d in set(R.dataset)]:
        rows.append(['<b>%s</b>' % ds] + [''] * 11)
        for m in MAIN:
            g = G.loc[(ds, m)]
            rows.append([m, fmt_pct(g.dev_FIR), fmt_pct(g.usr_FSR), fmt_pct(g.rat), fmt_pct(g.low_DR, 0), fmt_pct(g.onoff_DR, 0),
                         fmt_pct(g.flood_DR, 0), fmt_pct(g.gw_DR, 0), fmt_pct(g.worm_DR, 0), '%.0f' % g.worm_infected,
                         fmt_pct(g.hop_DR, 0), '%.1f' % g.low_delay])
    rows2 = [['Method', 'AUROC dev.', 'AUROC acc.', 'ECE', 'Brier', 'Avail. loss', 'Excess discon.', 'LCC change', 'Restricted']]
    G2 = R[R.method.isin(MAIN)].groupby('method').mean(numeric_only=True)
    for m in MAIN:
        g = G2.loc[m]
        ece = '%.3f' % g.ol_ece_dev if np.isfinite(g.get('ol_ece_dev', np.nan)) else '--'
        br = '%.4f' % g.ol_brier_dev if np.isfinite(g.get('ol_brier_dev', np.nan)) else '--'
        rows2.append([m, '%.3f' % g.ol_auroc_dev, '%.3f' % g.ol_auroc_usr, ece, br, fmt_pct(-g.avail_x, 2),
                      fmt_pct(g.coll_x, 2), fmt_pct(g.lcc_x, 2), fmt_pct(g.restrict_benign, 2)])
    return rows, rows2, G, G2


def abl_table(R):
    rows = [['Variant', 'FIR', 'FSR', 'RAT', 'Hop', 'Infected', 'Delay on-off', 'AUROC dev.', 'ECE', 'Excess discon.', 'Avail. loss']]
    G = R[R.method.isin(ABL)].groupby('method').mean(numeric_only=True)
    for m in ABL:
        g = G.loc[m]
        nf = lambda v, f: (f % v) if np.isfinite(v) else '--'
        rows.append([ABL_LAB[m], fmt_pct(g.dev_FIR), fmt_pct(g.usr_FSR), fmt_pct(g.rat), fmt_pct(g.hop_DR, 0),
                     '%.1f' % g.worm_infected, '%.1f' % g.onoff_delay, nf(g.ol_auroc_dev, '%.3f'), nf(g.ol_ece_dev, '%.3f'),
                     fmt_pct(g.coll_x, 2), fmt_pct(-g.avail_x, 2)])
    return rows, G


def stats(R):
    key = ['dataset', 'held', 'seed', 'kind']
    P = R.pivot_table(index=key, columns='method', values=['rat', 'dev_FIR', 'ol_ece_dev', 'ol_auroc_dev', 'onoff_delay', 'low_delay'])
    out = []
    for metric, better in (('rat', 'lower'), ('ol_ece_dev', 'lower'), ('ol_auroc_dev', 'higher'), ('low_delay', 'lower')):
        ps, rec = [], []
        for b in BASE + ['SCOUT-noDormant', 'SCOUT-noContagion', 'SCOUT-noUtility', 'SCOUT-noVolume']:
            if (metric, b) not in P.columns: continue
            a, c = P[(metric, 'SCOUT')].values, P[(metric, b)].values
            ok = np.isfinite(a) & np.isfinite(c)
            if ok.sum() < 5: continue
            try:
                p = wilcoxon(a[ok], c[ok]).pvalue
            except ValueError:
                p = 1.0
            ps.append(p); rec.append(dict(metric=metric, base=b, mean_s=float(a[ok].mean()), mean_b=float(c[ok].mean()),
                                          p=float(p), delta=cliff(a[ok], c[ok]), n=int(ok.sum()),
                                          better=float(((a[ok] < c[ok]) if better == 'lower' else (a[ok] > c[ok])).mean())))
        adj = holm(ps)
        for r, pa in zip(rec, adj):
            r['p_holm'] = float(pa); out.append(r)
    # Friedman over the main methods on RAT
    M = P['rat'][MAIN].dropna()
    fr = friedmanchisquare(*[M[m].values for m in MAIN])
    ranks = M.rank(axis=1).mean().sort_values()
    k, n = len(MAIN), len(M)
    cd = 3.031 * np.sqrt(k * (k + 1) / (6 * n))           # Nemenyi q_0.05 for k = 8
    return out, dict(chi2=float(fr.statistic), p=float(fr.pvalue), ranks=ranks.to_dict(), cd=float(cd), n=n)


def fig_profile(R):
    G = R[R.method.isin(MAIN)].groupby('method').mean(numeric_only=True)
    types = [('low_DR', 'Low-rate'), ('onoff_DR', 'On-off'), ('flood_DR', 'Flood'), ('gw_DR', 'Gateway'),
             ('worm_DR', 'Lateral'), ('hop_DR', 'Hopping\naccount')]
    meth = ['EWMA', 'Beta', 'Asym', 'Shiryaev', 'LearnedRisk', 'SCOUT']
    fills = ['#F2F2F2', '#D9D9D9', '#BFBFBF', '#8C8C8C', '#595959', '#000000']
    hatch = ['', '////', '', '\\\\\\\\', '', '']
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw=dict(width_ratios=[3.2, 1.25], wspace=0.28))
    x = np.arange(len(types)); w = 0.13
    for k, m in enumerate(meth):
        ax[0].bar(x + (k - 2.5) * w, [G.loc[m, c] for c, _ in types], w * 0.9, color=fills[k], edgecolor='black',
                  lw=0.5, hatch=hatch[k], label=m)
    ax[0].set_xticks(x); ax[0].set_xticklabels([t for _, t in types]); ax[0].set_ylim(0, 1.08)
    ax[0].set_ylabel('Containment rate'); ax[0].set_title('(a) Contained adversaries by type', loc='left')
    # (b) residual traffic with bootstrap CI
    for k, m in enumerate(MAIN):
        v = R[R.method == m].rat.values; mu, lo, hi = boot_ci(v)
        ax[1].errorbar(mu, k, xerr=[[mu - lo], [hi - mu]], fmt=MK[m][0], mfc=MK[m][1] if m != 'SCOUT' else 'black',
                       mec='black', ms=4.5, ecolor='black', elinewidth=0.8, capsize=2)
    ax[1].set_yticks(range(len(MAIN))); ax[1].set_yticklabels(MAIN); ax[1].set_xlabel('Residual attack traffic (RAT)')
    ax[1].set_title('(b) RAT, 95% CI', loc='left'); ax[1].set_xlim(0, None)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.36, -0.16), handletextpad=0.4)
    fig.savefig(f'{FIG}/fig4_profile.png', dpi=600, bbox_inches='tight', pad_inches=0.03); plt.close(fig)


def fig_reliability():
    f = f'{RES}/supp_relia.jsonl'
    if not os.path.exists(f): return None
    Q = pd.DataFrame([json.loads(l) for l in open(f)])
    meth = ['EWMA', 'Beta', 'Shiryaev', 'LearnedRisk', 'SCOUT']
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw=dict(wspace=0.3))
    out = {}
    for m in meth:
        q = Q[Q.method == m]; n = np.sum(q.n.tolist(), 0); pos = np.sum(q.pos.tolist(), 0); ps = np.sum(q.psum.tolist(), 0)
        ok = n > 50; conf = ps[ok] / n[ok]; acc = pos[ok] / n[ok]
        ax[0].plot(conf, acc, marker=MK[m][0], color='black' if m == 'SCOUT' else MK[m][1], mfc=MK[m][1] if m != 'SCOUT' else 'black',
                   mec='black', ls=MK[m][2], ms=4, lw=1.0, label=m)
        out[m] = dict(ece=float(np.sum(np.abs(ps - pos)) / n.sum()), share_hi=float(n[5:].sum() / n.sum()))
        frac = n / n.sum()
        nz = frac > 0
        ax[1].plot((np.arange(10) * 0.1 + 0.05)[nz], frac[nz], marker=MK[m][0], color='black' if m == 'SCOUT' else MK[m][1],
                   mfc=MK[m][1] if m != 'SCOUT' else 'black', mec='black', ls=MK[m][2], ms=4, lw=1.0, label=m)
    ax[0].plot([0, 1], [0, 1], color='#7F7F7F', lw=0.7, ls=(0, (1, 1)))
    ax[0].set_xlabel('Mean predicted compromise probability'); ax[0].set_ylabel('Observed compromise frequency')
    ax[0].set_title('(a) Reliability, pooled device-windows', loc='left'); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[1].set_yscale('log'); ax[1].set_xlabel('Predicted compromise probability (bin centre)'); ax[1].set_ylabel('Share of device-windows')
    ax[1].set_title('(b) Distribution of predictions', loc='left'); ax[1].set_xlim(0, 1)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(f'{FIG}/fig5_reliability.png', dpi=600, bbox_inches='tight', pad_inches=0.03); plt.close(fig)
    return out


def fig_duty():
    f = f'{RES}/supp_duty.jsonl'
    if not os.path.exists(f): return None
    Q = pd.DataFrame([json.loads(l) for l in open(f)])
    none = None
    meth = ['EWMA', 'Beta', 'Asym', 'Shiryaev', 'SCOUT-noDormant', 'SCOUT']
    G = Q.groupby(['method', 'gap']).mean(numeric_only=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.15), gridspec_kw=dict(wspace=0.28))
    gaps = sorted(Q.gap.unique())
    for m in meth:
        c = 'black' if m == 'SCOUT' else MK.get(m, ('o', '#7F7F7F', '-'))[1]
        kw = dict(marker=MK[m][0], color=c, mfc=c, mec='black', ls=MK[m][2], ms=4, lw=1.1, label=m.replace('SCOUT-noDormant', 'SCOUT, $s=0$'))
        ax[0].plot(gaps, [G.loc[(m, g), 'onoff_DR'] for g in gaps], **kw)
        ax[1].plot(gaps, [G.loc[(m, g), 'mal_delivered'] for g in gaps], **kw)
    ax[0].set_xscale('log'); ax[1].set_xscale('log')
    for a in ax: a.minorticks_off()
    for a in ax: a.set_xticks(gaps); a.set_xticklabels([str(g) for g in gaps]); a.set_xlabel('Dormant gap between 4-window bursts')
    ax[0].set_ylabel('On-off devices contained'); ax[0].set_ylim(0, 1.05); ax[0].set_title('(a) Containment', loc='left')
    ax[1].set_ylabel('Malicious flows delivered'); ax[1].set_title('(b) Delivered attack traffic', loc='left')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.21))
    fig.savefig(f'{FIG}/fig6_duty.png', dpi=600, bbox_inches='tight', pad_inches=0.03); plt.close(fig)
    return G


def fig_sens():
    f = f'{RES}/supp_sens.jsonl'
    if not os.path.exists(f): return None
    Q = pd.DataFrame([json.loads(l) for l in open(f)])
    G = Q.groupby(['param', 'value']).mean(numeric_only=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw=dict(wspace=0.32))
    hv = sorted(float(v) for v in Q[Q.param == 'hmax'].value.unique())
    nb = [G.loc[('hmax', v), 'nbr_hi'] for v in hv]; fa = [G.loc[('hmax', v), 'far_hi'] for v in hv]
    dl = [G.loc[('hmax', v), 'delay_med'] for v in hv]
    xs = np.arange(len(hv))
    ax[0].plot(xs, nb, marker='o', color='black', mfc='black', label='benign neighbours of compromised devices')
    ax[0].plot(xs, fa, marker='s', color='#7F7F7F', mfc='white', mec='black', ls='--', label='other benign devices')
    bl = G.loc[('contagion', 'blame')]
    ax[0].plot([len(hv) + 0.6], [bl.nbr_hi], marker='D', ms=6, color='black', mfc='#BFBFBF', ls='none', label='blame diffusion (neighbours)')
    ax[0].plot([len(hv) + 0.6], [bl.far_hi], marker='D', ms=6, color='black', mfc='white', ls='none', label='blame diffusion (others)')
    ax[0].set_xticks(list(xs) + [len(hv) + 0.6]); ax[0].set_xticklabels([('%g' % v) for v in hv] + ['blame'])
    ax[0].set_xlabel('$h_{\\max}$'); ax[0].set_ylabel('Share ever in high-risk tier'); ax[0].set_title('(a) Collateral suspicion (open loop)', loc='left')
    ax[0].legend(loc='upper left', frameon=False, fontsize=6.3)
    ax[0].set_ylim(0, max(nb + fa + [bl.nbr_hi]) * 1.65)
    ax[1].plot(xs, dl, marker='o', color='black', mfc='black', label='scrutiny contagion')
    ax[1].plot([len(hv) + 0.6], [bl.delay_med], marker='D', ms=6, color='black', mfc='#BFBFBF', ls='none', label='blame diffusion')
    ax[1].set_xticks(list(xs) + [len(hv) + 0.6]); ax[1].set_xticklabels([('%g' % v) for v in hv] + ['blame'])
    ax[1].set_xlabel('$h_{\\max}$'); ax[1].set_ylabel('Median windows to high-risk tier'); ax[1].set_title('(b) Detection delay of compromised devices', loc='left')
    ax[1].legend(loc='upper right', frameon=False)
    ax[1].set_ylim(0, max(dl + [bl.delay_med]) * 1.35)
    fig.savefig(f'{FIG}/fig7_scrutiny.png', dpi=600, bbox_inches='tight', pad_inches=0.03); plt.close(fig)
    return G


def fig_scale(R):
    f = f'{RES}/supp_scale.jsonl'
    if not os.path.exists(f): return None
    Q = pd.DataFrame([json.loads(l) for l in open(f)])
    sub = R[(R.seed == 0) & R.held.isin(Q.held.unique())].copy(); sub['sec_per_window'] = np.nan
    key = ['dataset', 'held', 'seed', 'kind']
    noneQ = Q[Q.method == 'Beta'][['N'] + key].copy()
    meth = ['Beta', 'Asym', 'Shiryaev', 'LearnedRisk', 'SCOUT', 'SCOUT-noUtility']
    Q2 = pd.concat([Q, sub[sub.method.isin(meth)].assign(N=500)], ignore_index=True)
    G = Q2.groupby(['method', 'N']).mean(numeric_only=True)
    Ns = sorted(Q2.N.unique())
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.1), gridspec_kw=dict(wspace=0.38))
    for m in meth:
        c = 'black' if m == 'SCOUT' else MK.get(m, ('o', '#7F7F7F', '-'))[1]
        kw = dict(marker=MK[m][0], color=c, mfc=c, mec='black', ls=MK[m][2], ms=4, lw=1.1, label=m.replace('SCOUT-noUtility', 'SCOUT, thr. actions'))
        ax[0].plot(Ns, [G.loc[(m, n), 'dev_FIR'] for n in Ns], **kw)
        ax[1].plot(Ns, [G.loc[(m, n), 'low_DR'] for n in Ns], **kw)
        t = [(n, G.loc[(m, n), 'sec_per_window'] * 1e3) for n in Ns if (m, n) in G.index and np.isfinite(G.loc[(m, n), 'sec_per_window'])]
        ax[2].plot([a for a, _ in t], [b for _, b in t], **kw)
    for a in ax: a.set_xscale('log'); a.minorticks_off(); a.set_xticks(Ns); a.set_xticklabels([str(n) for n in Ns]); a.set_xlabel('Devices $N$')
    ax[0].set_ylabel('FIR'); ax[0].axhline(0.02, color='#7F7F7F', lw=0.7, ls=(0, (2, 2))); ax[0].set_title('(a) False isolation', loc='left')
    ax[1].set_ylabel('Low-rate devices contained'); ax[1].set_title('(b) Detection', loc='left'); ax[1].set_ylim(0, 1.05)
    ax[2].set_ylabel('ms per window (closed loop)'); ax[2].set_yscale('log'); ax[2].set_title('(c) Runtime', loc='left')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.24))
    fig.savefig(f'{FIG}/fig8_scale.png', dpi=600, bbox_inches='tight', pad_inches=0.03); plt.close(fig)
    return G


def robust_table():
    f = f'{RES}/supp_robust.jsonl'
    if not os.path.exists(f): return None, None
    Q = pd.DataFrame([json.loads(l) for l in open(f)])
    G = Q.groupby(['cond', 'method']).mean(numeric_only=True)
    conds = [('rate0.5', 'benign rate ×0.5'), ('rate2', 'benign rate ×2'), ('churn0', 'no churn'), ('churn5', 'churn 5%'),
             ('hetero30', '30% atypical benign'), ('burst20', '20% bursty benign')]
    meth = ['Beta', 'Asym', 'Shiryaev', 'LearnedRisk', 'SCOUT']
    rows = [['Condition'] + ['%s' % m for m in meth]]
    for c, lab in conds:
        rows.append([lab] + ['%s / %s' % (fmt_pct(G.loc[(c, m), 'dev_FIR']), fmt_pct(G.loc[(c, m), 'low_DR'], 0)) for m in meth])
    return rows, G


def budget_fig(R):
    f = f'{RES}/supp_budget.jsonl'
    if not os.path.exists(f): return None
    Q = pd.DataFrame([json.loads(l) for l in open(f)])
    sub = R[(R.seed == 0) & R.held.isin(Q.held.unique()) & R.method.isin(Q.method.unique())].assign(eps=0.02)
    Q = pd.concat([Q, sub], ignore_index=True)
    key = ['dataset', 'held', 'seed', 'kind']
    none = R[R.method == 'None'].set_index(key).mal_delivered
    Q['rat'] = [r.mal_delivered / none.loc[(r.dataset, r.held, r.seed, r.kind)] for r in Q.itertuples()]
    return Q.groupby(['method', 'eps']).mean(numeric_only=True)


def failure(R):
    S = R[R.method == 'SCOUT'].groupby(['dataset', 'held']).mean(numeric_only=True)
    B = R[R.method == 'Shiryaev'].groupby(['dataset', 'held']).mean(numeric_only=True)
    rows = [['Dataset', 'Held family', '$p_a$', 'RAT SCOUT', 'RAT Shiryaev', 'Low', 'On-off', 'Hop', 'FIR']]
    for (ds, h), g in S.sort_values('pa').iterrows():
        rows.append([ds.replace('_IoT', '').replace('CICIoT2023', 'CIC'), h, '%.3f' % g.pa, fmt_pct(g.rat), fmt_pct(B.loc[(ds, h), 'rat']),
                     fmt_pct(g.low_DR, 0), fmt_pct(g.onoff_DR, 0), fmt_pct(g.hop_DR, 0), fmt_pct(g.dev_FIR)])
    return rows, S


def run():
    out = {}
    out['flow'] = flow()
    R = load_entity(); out['R'] = R
    out['main'] = main_tables(R); out['abl'] = abl_table(R); out['stats'] = stats(R)
    fig_profile(R)
    out['relia'] = fig_reliability(); out['duty'] = fig_duty(); out['sens'] = fig_sens(); out['scale'] = fig_scale(R)
    out['robust'] = robust_table(); out['budget'] = budget_fig(R); out['fail'] = failure(R)
    filt = [json.loads(l) for l in open(f'{RES}/filter_main.jsonl')]
    out['filter'] = filt
    pickle.dump(out, open(f'{RES}/analysis.pkl', 'wb'))
    return out


if __name__ == '__main__':
    o = run()
    for r in o['main'][0]: print(r)
    for r in o['abl'][0]: print(r)
