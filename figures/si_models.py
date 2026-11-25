"""SI Figures S2-S5 and Table S2: everyday-share allocation compared with the saved XGBoost predictions.

Inputs: data/grids/main_40x40_120min.parquet (observed arrivals, baseline and out-of-fold XGBoost predictions)
        and results/allocation/table_S2_model_metrics.csv (written by src/s01_allocation_similarity.py).
Outputs: figures/output/si/Figure_S2_scatter, Figure_S3_residuals, Figure_S4_ccdf, Figure_S5_groups (png, svg)
         and figures/output/si/Table_S2_metrics.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'figures' / 'output' / 'si'
OUTDIR.mkdir(parents=True, exist_ok=True)
SEED = 20261117

ink = '#252525'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
                     'axes.linewidth': .65, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'text.color': ink, 'axes.labelcolor': ink, 'axes.edgecolor': ink,
                     'savefig.facecolor': 'white', 'legend.frameon': False, 'svg.fonttype': 'none'})
WIDTH = 5.95                                             # same figure width as Figures S9 and S10
MODELS = {'alloc': ('Everyday-share allocation', '#2F4FD0', 'Blues'),
          'xgb': ('XGBoost', '#E07B39', 'Oranges')}
GROUPS = ['Small', 'Medium', 'Large', 'Mega']
GCOL = {'Small': '#5FACFC', 'Medium': '#FC6F68', 'Large': '#FFB64D', 'Mega': '#5BC39E'}   # as in Figure 4

# ---------------------------------------------------------------- data
g = pd.read_parquet(ROOT / 'data/grids/main_40x40_120min.parquet',
                    columns=['event_id', 'event_density', 'baseline_density', 'pred_xgb'])
grp = g.groupby('event_id')
g['alloc'] = grp.event_density.transform('sum') * g.baseline_density / grp.baseline_density.transform('sum')
g['xgb'] = g.pred_xgb
mb = grp.baseline_density.sum()
g['group'] = g.event_id.map(pd.qcut(mb, 4, labels=GROUPS))   # quartiles of baseline grid total, as in Figure 4
y = g.event_density.to_numpy()


def r2(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)


def ols_slope(obs, pred):
    return np.polyfit(obs, pred, 1)[0]


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUTDIR / f'{name}.{ext}', dpi=450, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def letters(fig, items, dy=.01):
    """Place bold panel letters at the upper-left corner of each axis's tight box (figure coordinates)."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for lab, ax, x in items:
        bb = inv.transform(ax.get_tightbbox(fig.canvas.get_renderer()))
        fig.text(bb[0, 0] if x is None else x, bb[1, 1] + dy, lab, weight='bold', fontsize=10, ha='left', va='bottom')


def left_edge(fig, axes):
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    return min(inv.transform(a.get_tightbbox(fig.canvas.get_renderer()))[0, 0] for a in axes)


# ---------------------------------------------------------------- Figure S2: predicted versus observed arrivals
lim = float(np.percentile(np.r_[y, g.alloc, g.xgb], 99.5))
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.75), gridspec_kw={'wspace': .45})
for ax, (key, (name, col, cmap)) in zip(axs, MODELS.items()):
    p = g[key].to_numpy()
    hb = ax.hexbin(y, p, gridsize=60, cmap=cmap, mincnt=1, norm=mcolors.LogNorm(), extent=(0, lim, 0, lim))
    ax.plot([0, lim], [0, lim], 'k--', lw=.8, alpha=.6)
    ax.set(xlim=(0, lim), ylim=(0, lim), aspect='equal', xlabel='Observed arrivals per cell')
    ax.set_title(f'{name}  ($R^2$ = {r2(y, p):.3f})', fontsize=8)
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(hb, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Grid cells', fontsize=7)
axs[0].set_ylabel('Predicted arrivals per cell')
x0 = left_edge(fig, [axs[0]])
letters(fig, [('a', axs[0], x0), ('b', axs[1], None)])
save(fig, 'Figure_S2_scatter')

# ---------------------------------------------------------------- Figure S3: residual density and absolute error by group
rng = np.random.default_rng(SEED)
sample = rng.choice(len(g), 60000, replace=False)
fig = plt.figure(figsize=(WIDTH, 2.6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.35], wspace=.3)
ax = fig.add_subplot(gs[0])
xs = np.linspace(-12, 12, 481)
for key, (name, col, _) in MODELS.items():
    res = (g[key].to_numpy() - y)[sample]
    ax.plot(xs, gaussian_kde(res, bw_method=.05)(xs), color=col, lw=1.1, label=name)
ax.axvline(0, color='#858585', ls='--', lw=.8)
ax.set(xlim=(-12, 12), xlabel='Residual (predicted − observed)', ylabel='Density')
ax.set_ylim(bottom=0)
ax_a = ax
ax = fig.add_subplot(gs[1])
pos, data, cols = [], [], []
for i, grp_name in enumerate(GROUPS):
    m = (g.group == grp_name).to_numpy()
    for j, (key, (name, col, _)) in enumerate(MODELS.items()):
        pos.append(i + (j - .5) * .36)
        data.append(np.abs(g[key].to_numpy()[m] - y[m]))
        cols.append(col)
bp = ax.boxplot(data, positions=pos, widths=.3, showfliers=False, patch_artist=True,
                medianprops=dict(color=ink, lw=.9), whiskerprops=dict(lw=.7), capprops=dict(lw=.7),
                boxprops=dict(lw=.7))
for patch, col in zip(bp['boxes'], cols):
    patch.set_facecolor(col)
    patch.set_alpha(.75)
ax.set(xticks=range(4), xticklabels=GROUPS, xlabel='Baseline traffic magnitude group',
       ylabel='Absolute error (arrivals per cell)')
ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c, alpha=.75, ec=ink, lw=.7) for _, c, _ in MODELS.values()],
          [n for n, _, _ in MODELS.values()], fontsize=6.5, ncol=2, loc='upper center')
ax_b = ax
x0 = left_edge(fig, [ax_a])
letters(fig, [('a', ax_a, x0), ('b', ax_b, None)])
save(fig, 'Figure_S3_residuals')

# ---------------------------------------------------------------- Figure S4: CCDFs of observed and predicted arrivals
xgrid = np.logspace(-2, np.log10(max(y.max(), g.alloc.max(), g.xgb.max()) * 1.05), 300)


def ccdf(v):
    v = np.sort(v)
    return 1 - np.searchsorted(v, xgrid, side='right') / len(v)


fig = plt.figure(figsize=(WIDTH, 3.0))
outer = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.4], wspace=.28)
ax = fig.add_subplot(outer[0])
ax.plot(xgrid, ccdf(y), color=ink, lw=1.2, label='Observed')
for key, (name, col, _) in MODELS.items():
    ax.plot(xgrid, ccdf(g[key].to_numpy()), color=col, lw=1, ls='--', label=name)
ax.set(xscale='log', yscale='log', xlabel='Arrivals per cell', ylabel='P(X > x)', title='All records')
ax.set_title('All records', fontsize=8)
ax.legend(fontsize=6.3, loc='lower left')
ax_a = ax
inner = outer[1].subgridspec(2, 2, hspace=.55, wspace=.3)
ax_b = None
for k, grp_name in enumerate(GROUPS):
    ax = fig.add_subplot(inner[k // 2, k % 2], sharex=ax_a, sharey=ax_a)
    m = (g.group == grp_name).to_numpy()
    ax.plot(xgrid, ccdf(y[m]), color=ink, lw=1.1)
    for key, (name, col, _) in MODELS.items():
        ax.plot(xgrid, ccdf(g[key].to_numpy()[m]), color=col, lw=.9, ls='--')
    ax.set_title(grp_name, fontsize=8, color=ink)
    ax.tick_params(labelsize=6)
    if k // 2 == 1:
        ax.set_xlabel('Arrivals per cell', fontsize=7)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    if k % 2 == 0:
        ax.set_ylabel('P(X > x)', fontsize=7)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    if k == 0:
        ax_b = ax
ax_a.set_ylim(1e-6, 1.5)
x0 = left_edge(fig, [ax_a])
letters(fig, [('a', ax_a, x0), ('b', ax_b, None)])
save(fig, 'Figure_S4_ccdf')

# ---------------------------------------------------------------- Figure S5: group-level diagnostics
fig = plt.figure(figsize=(WIDTH, 5.0))
gs = gridspec.GridSpec(3, 4, hspace=.55, wspace=.42)
row_axes = []
for r, (key, (name, col, cmap)) in enumerate(MODELS.items()):
    first = None
    for k, grp_name in enumerate(GROUPS):
        ax = fig.add_subplot(gs[r, k])
        m = (g.group == grp_name).to_numpy()
        yo, p = y[m], g[key].to_numpy()[m]
        top = float(np.percentile(np.r_[yo, p], 99.9))
        ax.hexbin(yo, p, gridsize=40, cmap=cmap, mincnt=1, norm=mcolors.LogNorm(), extent=(0, top, 0, top))
        ax.plot([0, top], [0, top], 'k--', lw=.7, alpha=.6)
        b = ols_slope(yo, p)
        ax.plot([0, top], [np.mean(p) - b * np.mean(yo), np.mean(p) + b * (top - np.mean(yo))], color=ink, lw=.8)
        ax.set(xlim=(0, top), ylim=(0, top), aspect='equal')
        if r == 0:
            ax.set_title(grp_name, fontsize=8)
        ax.text(.04, .96, f'$R^2$ = {r2(yo, p):.3f}\nslope = {b:.2f}', transform=ax.transAxes, va='top',
                ha='left', fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Observed', fontsize=7)
        if k == 0:
            ax.set_ylabel(f'Predicted ({"allocation" if key == "alloc" else "XGBoost"})', fontsize=7)
            first = ax
    row_axes.append(first)
first = None
for k, grp_name in enumerate(GROUPS):
    ax = fig.add_subplot(gs[2, k])
    m = (g.group == grp_name).to_numpy()
    for key, (name, col, _) in MODELS.items():
        p = g[key].to_numpy()[m]
        res = p - y[m]
        order = np.argsort(p)
        bins = np.array_split(order, 40)                    # 40 equal-count bins of the predicted value
        ax.plot([p[i].mean() for i in bins], [res[i].mean() for i in bins], color=col, lw=1, marker='o', ms=1.6)
    ax.axhline(0, color='#858585', ls='--', lw=.7)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=-1)
    ticks = [0, 1, 10, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(labelsize=6)
    ax.set_xlabel('Predicted', fontsize=7)
    if k == 0:
        ax.set_ylabel('Mean residual', fontsize=7)
        first = ax
row_axes.append(first)
handles = [plt.Line2D([], [], color=c, lw=1, marker='o', ms=2) for _, c, _ in MODELS.values()]
fig.legend(handles, [n for n, _, _ in MODELS.values()], fontsize=6.5, ncol=2, loc='lower center',
           bbox_to_anchor=(.5, -.02))
x0 = left_edge(fig, row_axes)
letters(fig, [(lab, ax, x0) for lab, ax in zip('abc', row_axes)])
save(fig, 'Figure_S5_groups')

# ---------------------------------------------------------------- Table S2
t = pd.read_csv(ROOT / 'results/allocation/table_S2_model_metrics.csv').set_index('model')
names = {'allocation': 'Everyday-share allocation', 'xgb': 'XGBoost'}
cols = ['R2', 'ExpVar', 'Pearson', 'Spearman', 'RMSE', 'MAE', 'MedAE', 'MAPE@5', 'Bias']
head = [r'\(R^{2}\)', 'Exp/Var', r'Pearson \(r\)', r'Spearman \(\rho\)', 'RMSE', 'MAE', 'MedAE', 'MAPE@5', 'Bias']
lines = [r'\begin{tabular}{@{}l' + 'c' * len(cols) + r'@{}}\toprule',
         ' & ' + ' & '.join(rf'\textbf{{{h}}}' for h in head) + r' \\ \midrule']
for k, name in names.items():
    v = t.loc[k, cols].copy()
    v['Bias'] = 0.0 if abs(v['Bias']) < 5e-3 else v['Bias']
    lines.append(name + ' & ' + ' & '.join(f'{x:.2f}' for x in v) + r' \\')
lines.append(r'\bottomrule\end{tabular}')
(OUTDIR / 'Table_S2_metrics.tex').write_text('\n'.join(lines) + '\n')

print(t.loc[list(names), cols].round(3).to_string())
for grp_name in GROUPS:
    m = (g.group == grp_name).to_numpy()
    print(grp_name, int(g.loc[m, 'event_id'].nunique()),
          {k: (round(r2(y[m], g[k].to_numpy()[m]), 3), round(ols_slope(y[m], g[k].to_numpy()[m]), 2)) for k in MODELS})
print('S2-S5 and Table S2 written to', OUTDIR)
