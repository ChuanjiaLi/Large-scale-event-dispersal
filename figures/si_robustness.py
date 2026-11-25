"""SI Figures S9 and S10: everyday-share allocation under other grid resolutions and post-event windows.

Inputs: data/grids/main_40x40_120min.parquet and data/grids/robustness_*.parquet.
Outputs: figures/output/si/Figure_S9*, Figure_S10*, results/robustness/robustness_summary.csv
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[1]
GRIDS = ROOT / 'data' / 'grids'
OUTDIR = ROOT / 'figures' / 'output' / 'si'
OUTDIR.mkdir(parents=True, exist_ok=True)

# Venue identifiers for the cluster bootstrap.
venues = pd.read_csv(ROOT / 'data' / 'events.csv').set_index('event_id')['venue_id']

ink = '#252525'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
                     'axes.linewidth': .65, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'text.color': ink, 'axes.labelcolor': ink, 'axes.edgecolor': ink,
                     'savefig.facecolor': 'white', 'legend.frameon': False, 'svg.fonttype': 'none'})
# Same per-resolution colours and colormaps as the original Appendix_4 figures.
gran_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
gran_cmaps = ['Reds', 'Blues', 'Greens', 'Oranges']


def everyday_share(df):
    """Add the everyday-share allocation column 'pred' to a grid table."""
    df = df[['event_id', 'event_density', 'baseline_density']].copy()
    grp = df.groupby('event_id')
    df['pred'] = grp.event_density.transform('sum') * df.baseline_density / grp.baseline_density.transform('sum')
    return df


def event_table(df):
    """Event-level FVU, R2 and baseline grid total."""
    rows = []
    for eid, g in df.groupby('event_id'):
        y, p = g.event_density.values, g.pred.values
        sse, sst = ((y - p) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        rows.append((eid, sse / sst, 1 - sse / sst, g.baseline_density.sum()))
    t = pd.DataFrame(rows, columns=['event_id', 'fvu', 'r2', 'Mb']).set_index('event_id')
    t['venue'] = venues.reindex(t.index).values
    return t


def slope_with_ci(t, n_boot=1000, seed=20261117):
    """OLS slope of log FVU on log Mb and a venue-bootstrap 95% interval."""
    x, y = np.log10(t.Mb.values), np.log10(t.fvu.clip(lower=1e-6).values)
    b = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(seed)
    groups = {v: np.flatnonzero(t.venue.values == v) for v in t.venue.unique()}
    keys = list(groups)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        if np.ptp(x[idx]) > 0:
            boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    return b, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def draw(configs, fig_name, xlabel):
    """configs: list of (label, parquet path). Layout follows the original Appendix_4 figure."""
    data, summary = [], []
    for label, path in configs:
        df = everyday_share(pd.read_parquet(path))
        t = event_table(df)
        y, p = df.event_density.values, df.pred.values
        pooled = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        b, lo, hi = slope_with_ci(t)
        summary.append(dict(label=label, n=len(t), pooled_r2=pooled, median_r2=t.r2.median(),
                            r2_q1=t.r2.quantile(.25), r2_q3=t.r2.quantile(.75), slope=b, ci025=lo, ci975=hi))
        data.append(df)
    s = pd.DataFrame(summary)
    lim = max(max(np.percentile(d.event_density, 99.5), np.percentile(d.pred, 99.5)) for d in data)

    n = len(configs)
    # Same figure width for S9 (n=3) and S10 (n=4), so both print with the same font size;
    # the upper row is shortened when it holds four square panels.
    top_h = 1.0 if n == 3 else .72
    fig = plt.figure(figsize=(5.95, 2.55 * top_h + 2.05))
    outer = gridspec.GridSpec(2, 1, height_ratios=[top_h, .85], hspace=.48 if n == 3 else .62)
    top = outer[0].subgridspec(1, n, wspace=.42 if n == 3 else .55)
    bottom = outer[1].subgridspec(1, 2, wspace=.32)
    # Upper row: everyday-share allocation versus observed arrivals, shared axes.
    for k, (d, row) in enumerate(zip(data, s.itertuples())):
        ax = fig.add_subplot(top[0, k])
        hb = ax.hexbin(d.event_density, d.pred, gridsize=60, cmap=gran_cmaps[k], mincnt=1,
                       norm=mcolors.LogNorm(), extent=(0, lim, 0, lim))
        ax.plot([0, lim], [0, lim], 'k--', lw=.8, alpha=.6)
        ax.set(xlim=(0, lim), ylim=(0, lim), aspect='equal')
        ax.set_title(f'{row.label}  ($R^2$ = {row.pooled_r2:.3f})', fontsize=8)
        ax.set_xlabel('Observed arrivals')
        if k == 0:
            # Two lines when the four panels are short, so the label stays within the axes height.
            ax.set_ylabel('Everyday-share allocation' if n == 3 else 'Everyday-share\nallocation')
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=.05)
        cb = plt.colorbar(hb, cax=cax)
        cb.ax.tick_params(labelsize=6)
        if k == n - 1:
            cb.set_label('Grid cells', fontsize=7)
            cax_last = cax
        if k == 0:
            ax_a = ax
    # Lower left: pooled and median event-level R2.
    ax = fig.add_subplot(bottom[0, 0])
    x = np.arange(n)
    ax.bar(x - .18, s.pooled_r2, .34, color=gran_colors[:n], edgecolor='white', label='Pooled $R^2$')
    ax.bar(x + .18, s.median_r2, .34, color=gran_colors[:n], alpha=.5, edgecolor='white', hatch='///',
           label='Median event-level $R^2$')
    ax.errorbar(x + .18, s.median_r2, yerr=[s.median_r2 - s.r2_q1, s.r2_q3 - s.median_r2],
                fmt='none', ecolor=ink, elinewidth=.8, capsize=2)
    for i, r in s.iterrows():
        ax.text(i - .18, r.pooled_r2 + .02, f'{r.pooled_r2:.2f}', ha='center', fontsize=5.8)
        ax.text(i + .18, r.r2_q3 + .02, f'{r.median_r2:.2f}', ha='center', fontsize=5.8)
    ax.set(xticks=x, xticklabels=s.label, ylim=(0, 1.12), ylabel='$R^2$', xlabel=xlabel)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#8c8c8c', label='Pooled $R^2$'),
                       Patch(facecolor='#8c8c8c', alpha=.5, hatch='///', edgecolor='white', label='Median event-level $R^2$ (IQR)')],
              fontsize=6.2, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False, columnspacing=1.2)
    ax_b = ax
    # Lower right: slope of log FVU on log baseline grid total.
    ax = fig.add_subplot(bottom[0, 1])
    ax.errorbar(x, s.slope, yerr=[s.slope - s.ci025, s.ci975 - s.slope], fmt='none',
                ecolor=ink, elinewidth=.9, capsize=3)
    ax.scatter(x, s.slope, c=gran_colors[:n], s=30, zorder=3)
    for i, r in s.iterrows():
        ax.text(i + .12, r.slope, f'{r.slope:.2f}', va='center', fontsize=6.5)
    ax.axhline(0, color='#858585', ls='--', lw=.8)
    ax.set(xticks=x, xticklabels=s.label, xlim=(-.5, n - .5 + (.2 if n == 4 else 0)), ylim=(min(-1.0, s.ci025.min() - .1), .1),
           ylabel='Slope of log FVU vs log $M_b$', xlabel=xlabel)
    ax_c = ax
    # Stretch the lower row so that the right spine of c meets the right edge of the upper row
    # (the colour-bar label); b and c each take half of the extra width.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    box = lambda a: inv.transform(a.get_tightbbox(fig.canvas.get_renderer()))
    right = box(cax_last)[1, 0]
    pb, pc = ax_b.get_position(), ax_c.get_position()
    extra = right - pc.x1
    ax_b.set_position([pb.x0, pb.y0, pb.width + extra / 2, pb.height])
    ax_c.set_position([pc.x0 + extra / 2, pc.y0, pc.width + extra / 2, pc.height])
    fig.canvas.draw()
    # Panel letters: a and b share the left edge of the figure content; b and c share a height.
    (ax0, _), (_, ay1) = box(ax_a)
    (bx0, _), (_, by1) = box(ax_b)
    (cx0, _), (_, cy1) = box(ax_c)
    x_left = min(ax0, bx0)
    y_bc = max(by1, cy1)
    for lab, x, y in (('a', x_left, ay1), ('b', x_left, y_bc), ('c', cx0, y_bc)):
        fig.text(x, y + .01, lab, weight='bold', fontsize=10, ha='left', va='bottom')
    for ext in ('png', 'svg'):
        fig.savefig(OUTDIR / f'{fig_name}.{ext}', dpi=450, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    return s


spatial = draw([('20×20', GRIDS / 'robustness_grid20x20_120min.parquet'),
                ('40×40', GRIDS / 'main_40x40_120min.parquet'),
                ('60×60', GRIDS / 'robustness_grid60x60_120min.parquet')],
               'Figure_S9_spatial_everyday_share', 'Grid resolution')
temporal = draw([('30 min', GRIDS / 'robustness_grid40x40_30min.parquet'),
                 ('60 min', GRIDS / 'robustness_grid40x40_60min.parquet'),
                 ('90 min', GRIDS / 'robustness_grid40x40_90min.parquet'),
                 ('120 min', GRIDS / 'main_40x40_120min.parquet')],
                'Figure_S10_temporal_everyday_share', 'Post-event window')
out = pd.concat([spatial.assign(test='spatial'), temporal.assign(test='temporal')])
(ROOT / 'results' / 'robustness').mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / 'results' / 'robustness' / 'robustness_summary.csv', index=False)
print(out.round(3).to_string())
