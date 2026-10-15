"""Fig. 1 (framework) and Fig. 2 (three-state trust filter with contagion hazard), grayscale."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

FIG = os.environ.get('SCOUT_FIGURES', 'figures')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman'], 'mathtext.fontset': 'stix'})


class Box:
    def __init__(self, ax, x, y, w, h, title, body='', fill='#FFFFFF', lw=1.0, ls='-', tsize=7.7, bsize=6.5, bold=True):
        self.x, self.y, self.w, self.h = x, y, w, h
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.0,rounding_size=0.05',
                                    fc=fill, ec='black', lw=lw, ls=ls, zorder=2))
        if body:
            ax.text(x, y + h / 2 - 0.14, title, ha='center', va='top', fontsize=tsize, weight='bold' if bold else 'normal', zorder=3)
            ax.text(x, y + h / 2 - 0.50, body, ha='center', va='top', fontsize=bsize, zorder=3, linespacing=1.25)
        else:
            ax.text(x, y, title, ha='center', va='center', fontsize=tsize, weight='bold' if bold else 'normal', zorder=3)

    def side(self, s, f=0.5):
        x, y, w, h = self.x, self.y, self.w, self.h
        return {'l': (x - w / 2, y - h / 2 + f * h), 'r': (x + w / 2, y - h / 2 + f * h),
                't': (x - w / 2 + f * w, y + h / 2), 'b': (x - w / 2 + f * w, y - h / 2)}[s]


def arrow(ax, p, q, ls='-', lw=1.1, rad=0.0, label=None, lpos=0.5, loff=(0, 0.12), lsize=7.0, head=9):
    a = FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=head, lw=lw, ls=ls, color='black',
                        connectionstyle=f'arc3,rad={rad}', zorder=4, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        x = p[0] + lpos * (q[0] - p[0]) + loff[0]; y = p[1] + lpos * (q[1] - p[1]) + loff[1]
        ax.text(x, y, label, ha='center', va='center', fontsize=lsize, zorder=5,
                bbox=dict(fc='white', ec='none', pad=0.6))


def poly(ax, pts, ls='-', lw=1.1, label=None, lxy=None, lsize=7.0):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color='black', lw=lw, ls=ls, zorder=4, solid_capstyle='butt')
    arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw)
    if label:
        ax.text(lxy[0], lxy[1], label, ha='center', va='center', fontsize=lsize, zorder=5, bbox=dict(fc='white', ec='none', pad=0.6))


def fig1():
    fig, ax = plt.subplots(figsize=(7.16, 4.75)); ax.set_xlim(0, 15.0); ax.set_ylim(0.1, 8.75); ax.axis('off')
    G1, G2, G3 = '#F2F2F2', '#DDDDDD', '#FFFFFF'
    c1, c2, c3, c4 = 1.6, 5.35, 9.1, 13.25; w = 3.0; y1, y2, y3 = 7.85, 4.85, 1.65; h1, h = 1.55, 1.65
    src = Box(ax, c1, y1, w, h1, 'Observation', 'flows of users, devices\nand gateways in window $t$;\nlive graph $\\mathcal{G}_t$', G1)
    scr = Box(ax, c2, y1, w, h1, 'Open-set flow screen', 'known-attack classifier;\nconformal benign value $u_f$;\nflag when $u_f\\leq q$', G3)
    evd = Box(ax, c3, y1, w, h1, 'Window evidence', 'beta-binomial ratio against\nenrolled null $q_{0,e}$; one-sided\nvolume ratio; clip, temper', G3)
    att = Box(ax, c4, y1, w, h1, 'Attribution', 'user and device share each\ncell by responsibility $\\rho$;\nentity evidence $\\tilde{\\ell}_{e,t}$', G3)
    enr = Box(ax, c2, y2, w, h, 'Enrolment', 'entity-relative null $q_{0,e}$;\nvolume baseline $\\mu_e$;\ndispersion $r_e$', G1, ls='--')
    hz = Box(ax, c3, y2, w, h, 'Scrutiny contagion', 'noisy-OR hazard $h_{i,t}$ from\nneighbours on $\\mathcal{G}_t$; raises\nthe prior, never the blame', G2, lw=1.5)
    flt = Box(ax, c4, y2, w, h, 'Three-state trust filter', 'benign, active, dormant;\nposterior $\\pi_{e,t}$ and\ntrust tier', G2, lw=1.5)
    rev = Box(ax, c2, y3, w, h, 'Forensic review', 'after $R$ windows: keep\ncontained, or restore and\nreset the posterior', G1)
    enf = Box(ax, c3, y3, w, h, 'Enforcement points', 'rate limits, step-up login,\negress filters, node or\naccount isolation', G1)
    pol = Box(ax, c4, y3, w, h, 'Cost-aware containment', 'allow, restrict, egress\nisolation or quarantine;\ntransit loss at cut vertices', G2, lw=1.5)
    # forward path
    arrow(ax, src.side('r'), scr.side('l'))
    arrow(ax, scr.side('r'), evd.side('l'))
    arrow(ax, evd.side('r'), att.side('l'))
    arrow(ax, att.side('b'), flt.side('t'), label='$\\tilde{\\ell}_{e,t}$', loff=(0.42, 0))
    arrow(ax, flt.side('b'), pol.side('t'), label='$\\pi_{e,t}$', loff=(0.42, 0))
    arrow(ax, pol.side('l'), enf.side('r'), label='$a_{e,t}$', loff=(0, 0.2))
    arrow(ax, enf.side('l'), rev.side('r'))
    # contagion loop between filter and hazard
    arrow(ax, flt.side('l', 0.74), hz.side('r', 0.74), label='$\\pi_{j,t-1}$', loff=(0, 0.2))
    arrow(ax, hz.side('r', 0.26), flt.side('l', 0.26), label='$h_{i,t}$', loff=(0, -0.2))
    # enrolment feeds the evidence layer
    yb = 6.55; xe = c3 - 0.75
    ax.plot([enr.x, enr.x], [enr.y + h / 2, yb], color='black', lw=1.0, zorder=4)
    ax.plot([enr.x, xe], [yb, yb], color='black', lw=1.0, zorder=4)
    arrow(ax, (xe, yb), (xe, evd.y - h1 / 2), lw=1.0)
    ax.text(0.5 * (enr.x + xe), yb + 0.08, '$q_{0,e},\\ \\mu_e,\\ r_e$', ha='center', va='bottom', fontsize=7.0)
    # live topology to contagion and containment
    yg = 3.45; xh = c3 - 0.75; xp = c4 - 0.75
    ax.plot([src.x, src.x], [src.y - h1 / 2, yg], color='black', lw=1.0, zorder=4)
    ax.plot([src.x, xp], [yg, yg], color='black', lw=1.0, zorder=4)
    arrow(ax, (xh, yg), (xh, hz.y - h / 2), lw=1.0)
    arrow(ax, (xp, yg), (xp, pol.y + h / 2), lw=1.0)
    ax.text(src.x + 0.9, yg + 0.06, '$\\mathcal{G}_t$', ha='left', va='bottom', fontsize=7.4)
    # dashed reverse paths
    yd = 2.95; xs = src.x + 0.6
    ax.plot([enf.x + 0.55, enf.x + 0.55], [enf.y + h / 2, yd], color='black', lw=1.1, ls='--', zorder=4)
    ax.plot([enf.x + 0.55, xs], [yd, yd], color='black', lw=1.1, ls='--', zorder=4)
    arrow(ax, (xs, yd), (xs, src.y - h1 / 2), ls='--', lw=1.1)
    ax.text(c2, yd - 0.08, 'isolation reshapes $\\mathcal{G}_{t+1}$', ha='center', va='top', fontsize=6.8)
    yr = 0.35; xr = 14.85
    ax.plot([rev.x, rev.x], [rev.y - h / 2, yr], color='black', lw=1.1, ls='--', zorder=4)
    ax.plot([rev.x, xr], [yr, yr], color='black', lw=1.1, ls='--', zorder=4)
    ax.plot([xr, xr], [yr, flt.y], color='black', lw=1.1, ls='--', zorder=4)
    arrow(ax, (xr, flt.y), (flt.x + w / 2, flt.y), ls='--', lw=1.1)
    ax.text(c3, yr + 0.08, 'restored entity: reset $\\pi_e$', ha='center', va='bottom', fontsize=6.8)
    fig.savefig(f'{FIG}/fig1_framework.png', dpi=600, bbox_inches='tight', pad_inches=0.03); plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(3.45, 2.75)); ax.set_xlim(0, 7.0); ax.set_ylim(0, 5.6); ax.axis('off')
    pos = {'B': (1.15, 3.6), 'A': (4.0, 3.6), 'D': (4.0, 0.95)}
    names = {'B': 'benign\n$z=\\mathrm{B}$', 'A': 'active\n$z=\\mathrm{A}$', 'D': 'dormant\n$z=\\mathrm{D}$'}
    fills = {'B': '#FFFFFF', 'A': '#D0D0D0', 'D': '#EDEDED'}
    R = 0.78
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), R, fc=fills[k], ec='black', lw=1.2, zorder=2))
        ax.text(x, y, names[k], ha='center', va='center', fontsize=7.6, zorder=3)
    arrow(ax, (pos['B'][0] + R, pos['B'][1]), (pos['A'][0] - R, pos['A'][1]), label='$h_{i,t}$', loff=(0, 0.2), lsize=7.6)
    arrow(ax, (pos['A'][0] + 0.25, pos['A'][1] - R + 0.04), (pos['D'][0] + 0.25, pos['D'][1] + R - 0.04), label='$s$', loff=(0.25, 0), lsize=7.6)
    arrow(ax, (pos['D'][0] - 0.25, pos['D'][1] + R - 0.04), (pos['A'][0] - 0.25, pos['A'][1] - R + 0.04), label='$r$', loff=(-0.25, 0), lsize=7.6)
    for k, lab in (('B', '$1-h_{i,t}$'), ('A', '$1-s$')):
        x, y = pos[k]
        a = FancyArrowPatch((x - 0.35, y + R - 0.08), (x + 0.35, y + R - 0.08), arrowstyle='-|>', mutation_scale=8, lw=1.0,
                            connectionstyle='arc3,rad=-1.6', color='black', zorder=4)
        ax.add_patch(a); ax.text(x, y + R + 0.62, lab, ha='center', va='bottom', fontsize=7.4)
    x, y = pos['D']
    a = FancyArrowPatch((x + R - 0.08, y + 0.35), (x + R - 0.08, y - 0.35), arrowstyle='-|>', mutation_scale=8, lw=1.0,
                        connectionstyle='arc3,rad=-1.6', color='black', zorder=4)
    ax.add_patch(a); ax.text(x + R + 0.62, y, '$1-r$', ha='left', va='center', fontsize=7.4)
    ax.text(5.65, 3.6, 'emission\n$\\exp(\\tilde{\\ell}_{i,t})$', ha='left', va='center', fontsize=7.2)
    ax.text(1.15, 1.65, 'emission 1\n(B and D emit\nbenign-like flows)', ha='center', va='center', fontsize=7.0)
    ax.text(1.15, 5.45, '', fontsize=1)
    fig.savefig(f'{FIG}/fig2_filter.png', dpi=600, bbox_inches='tight', pad_inches=0.03); plt.close(fig)


if __name__ == '__main__':
    fig1(); fig2()
