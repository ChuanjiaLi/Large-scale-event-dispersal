"""SI items tied to Figure 4: Tables S3, S4 and S7, Figures S7, S8 and S14.

Inputs: results/figure_data/{df_fig4_r1.csv, grid_allocation_r1.parquet} and results/venue_context/*.csv (src/s03).
Outputs: figures/output/si/, results/venue_context/Table_S3_rows.tex, Table_S4_rows.tex, Table_S7_associations.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'figures' / 'output' / 'si'; ASSETS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'results' / 'venue_context'; OUT.mkdir(parents=True, exist_ok=True)
FD = ROOT / 'results' / 'figure_data'
d = pd.read_csv(FD / 'df_fig4_r1.csv')
order = ['Small', 'Medium', 'Large', 'Mega']

def rows(cols, labels, scale, fmt):
    out = []
    for c, lab in zip(cols, labels):
        for k, g in enumerate(order):
            x = d.loc[d.magnitude_group == g, c].dropna() * scale
            vals = [x.mean(), x.std(), x.median(), x.quantile(.25), x.quantile(.75), x.min(), x.max()]
            out.append(f"{lab if k == 0 else ''} & {g} & {len(x)} & " + ' & '.join(fmt.format(v) for v in vals) + r' \\')
        out.append(r'\addlinespace[4pt]')
    return '\n'.join(out[:-1])

s3 = rows(['SUBWAY', 'BUS', 'DRV', 'BYC', 'WLK'], ['SUBWAY', 'BUS', 'DRV', 'BYC', 'WLK'], 100, '{:.2f}')
s4 = rows(['radius_of_gyration', 'concentration_ratio', 'directionality', 'centroid_displacement'],
          ['Radius of gyration', 'Spatial Gini coefficient', 'Directionality', 'Centroid displacement'], 1, '{:.2f}')
(OUT / 'Table_S3_rows.tex').write_text(s3 + '\n'); (OUT / 'Table_S4_rows.tex').write_text(s4 + '\n')

# ---- Figure S7 (same design as the original Sup-fig-7)
modes = ['SUBWAY', 'BUS', 'DRV', 'BYC', 'WLK']
summ = d.groupby('magnitude_group')[modes].mean().reindex(order)
plt.rcParams['font.sans-serif'] = ['Arial']
fig, axes = plt.subplots(1, 2, figsize=(22, 9), dpi=300)
pal = sns.color_palette('viridis', len(modes))
bottom = np.zeros(4)
for i, m in enumerate(modes):
    v = summ[m].values
    axes[0].bar(order, v, bottom=bottom, label=m, color=pal[i], edgecolor='white', linewidth=0.5)
    for j, (val, b) in enumerate(zip(v, bottom)):
        if val > 0.03:
            axes[0].text(j, b + val / 2, f'{val * 100:.1f}%', ha='center', va='center', fontsize=22, color='white', fontweight='bold')
    bottom += v
axes[0].set_ylabel('Cumulative Mode Share', fontsize=24); axes[0].set_xlabel('Magnitude Group', fontsize=24)
axes[0].tick_params(labelsize=22); axes[0].set_ylim(0, 1.05)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
for i, m in enumerate(modes):
    v = summ[m].values
    axes[1].plot(range(4), v, marker='o', linewidth=3, markersize=10, label=m, color=pal[i])
# Value labels to the right of each point, pushed apart vertically where lines are close so they do not overlap.
gap = 0.0165
for j in range(4):
    vals = summ.iloc[j][modes].to_numpy()
    idx = np.argsort(vals)
    pos = vals[idx].copy()
    for _ in range(50):
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < gap:
                mid = (pos[k] + pos[k - 1]) / 2
                pos[k - 1], pos[k] = mid - gap / 2, mid + gap / 2
    for k, i in enumerate(idx):
        axes[1].text(j + 0.1, pos[k], f'{vals[i] * 100:.1f}%', ha='left', va='center', fontsize=20, color=pal[i], zorder=5,
                     bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none', alpha=0.85))
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(order); axes[1].set_xlim(-0.25, 3.55)
axes[1].set_ylabel('Mean Mode Share', fontsize=24); axes[1].set_xlabel('Magnitude Group', fontsize=24)
axes[1].tick_params(labelsize=22); axes[1].grid(True, alpha=0.3)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
axes[1].legend(loc='upper right', fontsize=20, ncol=5, frameon=False, bbox_to_anchor=(1, 1.12))
plt.tight_layout(); fig.savefig(ASSETS / 'Figure_S7_modal_share.png', dpi=300); plt.close(fig)

# ---- Figure S8: corner cases in (radius, Gini) space
n = d.copy()
for c, k in [('radius_of_gyration', 'r'), ('concentration_ratio', 'g')]:
    n[k] = (n[c] - n[c].min()) / (n[c].max() - n[c].min())
corners = {'A: High Gini, low radius (concentrated, compact)': (0, 1), 'B: High Gini, high radius (concentrated, spread)': (1, 1),
           'C: Low Gini, low radius (diffuse, compact)': (0, 0), 'D: Low Gini, high radius (diffuse, spread)': (1, 0)}
cases = {name: n.loc[np.hypot(n.r - rx, n.g - gy).idxmin(), 'event_id'] for name, (rx, gy) in corners.items()}
grid = pd.read_parquet(FD / 'grid_allocation_r1.parquet')
base = mcolors.LinearSegmentedColormap.from_list('seq', [(1, 1, 1, .2), (1, 0, 0, .5), (0, 0, 0, .8)], 256)
neg = mcolors.LinearSegmentedColormap.from_list('neg', [(0.1, 0.2, 0.6, 0.8), (0.4, 0.6, 1, 0.5), (1, 1, 1, 0.2)], 128)
div = mcolors.ListedColormap(np.vstack((neg(np.linspace(0, 1, 128)), base(np.linspace(0, 1, 256))[:216:2][:128])))
fig, axes = plt.subplots(4, 3, figsize=(12, 15.5), dpi=300)
for r, (name, eid) in enumerate(cases.items()):
    g = grid[grid.event_id == eid].sort_values(['y_km', 'x_km'])
    vals = [g.baseline_density.values, g.event_density.values, (g.event_density - g.baseline_density).values]
    vmax = max(vals[0].max(), vals[1].max())
    row = d[d.event_id == eid].iloc[0]
    for c, (v, title) in enumerate(zip(vals, ['Baseline', 'Event day', 'Change from baseline'])):
        ax = axes[r, c]
        im = ax.imshow(v.reshape(40, 40), origin='lower', cmap=div, vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(title, fontsize=14)
        if c == 0: ax.set_ylabel(name.replace(' (', '\n('), fontsize=11)
    axes[r, 2].text(1.03, 0.5, f"{row.city}\nRg = {row.radius_of_gyration:.1f} km\nGini = {row.concentration_ratio:.2f}",
                    transform=axes[r, 2].transAxes, fontsize=10, va='center')
    cb = fig.colorbar(im, ax=axes[r, :], fraction=0.015, pad=0.12); cb.set_label('OD trips', fontsize=10)
fig.savefig(ASSETS / 'Figure_S8_morphology_cases.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('cases', cases)
print(d.groupby('magnitude_group')[['radius_of_gyration', 'concentration_ratio', 'directionality', 'centroid_displacement']].mean().reindex(order).round(2))
print((d.groupby('magnitude_group')[modes].mean().reindex(order) * 100).round(2))

# ---- inputs for Table S7 and Figure S14 (src/s03)
A = pd.read_csv(OUT / 'associations.csv')
OUTCOMES = list(A[['outcome', 'label', 'panel']].drop_duplicates().itertuples(index=False, name=None))
t = d.dropna(subset=['T50']).copy()
# ---- Table S7
def fmt(x): return f'{x:.2f}'.replace('-', '\\(-\\)')
lines = [r'\begin{addedblock}',
         r'\noindent\textbf{Table S7. Associations of venue and dispersal characteristics with baseline traffic magnitude.}',
         r'\par\smallskip', r'{\small',
         r'\begin{tabular}{@{}llcccc@{}}\toprule',
         r'Panel & Outcome & Unadjusted & + \(\log q\) & + \(\log q\), \(\log d_{\mathrm{CBD}}\) & 95\% CI \\ \midrule']
for col, label, panel in OUTCOMES:
    r = A[A.outcome == col].set_index('spec')
    ci = f"[{fmt(r.loc['q_cbd', 'lo'])}, {fmt(r.loc['q_cbd', 'hi'])}]"
    lines.append(f"4{panel} & {label} & {fmt(r.loc['raw', 'b'])} & {fmt(r.loc['q', 'b'])} & {fmt(r.loc['q_cbd', 'b'])} & {ci} \\\\")
lines += [r'\bottomrule\end{tabular}\par}', r'\end{addedblock}']
(OUT / 'Table_S7_associations.tex').write_text('\n'.join(lines) + '\n')

# ---- Figure S14: T50 and T80 against extra trips, coloured by baseline group (style of Figure S12)
colors = {'Small': '#5FACFC', 'Medium': '#FC6F68', 'Large': '#FFB64D', 'Mega': '#5BC39E'}; ink = '#252525'
styles = {'Small': ':', 'Medium': '--', 'Large': '-.', 'Mega': '-'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': .65, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'xtick.direction': 'out', 'ytick.direction': 'out', 'text.color': ink,
                     'axes.labelcolor': ink, 'axes.edgecolor': ink, 'savefig.facecolor': 'white',
                     'legend.frameon': False, 'svg.fonttype': 'none'})
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.6)); fig.subplots_adjust(left=.08, right=.98, bottom=.2, top=.9, wspace=.3)
for ax, y, lab, letter in zip(axs, ['T50', 'T80'], [r'$T_{50}$ (h)', r'$T_{80}$ (h)'], 'ab'):
    for g, c in colors.items():
        z = t[t.magnitude_group == g]
        ax.scatter(z.extra_trips, z[y], s=6, color=c, alpha=.5, linewidths=0, label=g)
        b1, a1 = np.polyfit(np.log10(z.extra_trips), np.log10(z[y]), 1)   # power-law fit per group, as in Figure 4e
        xr = np.logspace(np.log10(z.extra_trips.min()), np.log10(z.extra_trips.max()), 50)
        ax.plot(xr, 10 ** a1 * xr ** b1, color=c, lw=1.4, ls=styles[g])
    ax.set(xscale='log', yscale='log', ylim=(0.05, 5), xlabel='Extra dispersal trips', ylabel=lab)
    ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(['0.1', '0.25', '0.5', '1', '2', '4']); ax.minorticks_off()
    ax.text(-.14, 1.06, letter, transform=ax.transAxes, weight='bold', fontsize=10)
axs[0].legend(loc='upper left', ncol=4, fontsize=6.5, handletextpad=.2, columnspacing=.8, markerscale=1.5)
for ext in ('png', 'svg'):
    fig.savefig(ASSETS / f'Figure_S14_dispersal_time.{ext}', dpi=450, bbox_inches='tight', pad_inches=.04)


