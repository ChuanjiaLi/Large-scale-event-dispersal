"""SI Figure S13: reference-day variability and event-day changes for the three Figure S1 cases.

Each reference day is compared with the mean of the other two; the event day with the mean of all three.
Input: data/grids/reference_day_cases.parquet (daily 40 x 40 grids, aggregated and smoothed as the main fields).
Output: figures/output/si/Figure_S13_daily_variability.{png,svg}
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'output' / 'si'
OUT.mkdir(parents=True, exist_ok=True)
CASES = [(1705, 'Hangzhou'), (4263, 'Xi’an'), (4340, 'Harbin')]

cases = pd.read_parquet(ROOT / 'data' / 'grids' / 'reference_day_cases.parquet')
main = pd.read_parquet(ROOT / 'data' / 'grids' / 'main_40x40_120min.parquet',
                       columns=['event_id', 'row', 'col', 'baseline_density', 'event_density'])
rows = []
for eid, _ in CASES:
    c = cases[cases.event_id == eid]
    ref = c[c.role == 'reference']
    dates = sorted(ref.date.unique())
    ordinary = np.stack([ref[ref.date == dt].sort_values('cell').flow.to_numpy().reshape(40, 40) for dt in dates])
    observed = c[c.role == 'event'].sort_values('cell').flow.to_numpy().reshape(40, 40)
    saved = main[main.event_id == eid].sort_values(['row', 'col'])
    np.testing.assert_allclose(ordinary.mean(axis=0).ravel(), saved.baseline_density, atol=1e-8)   # same fields as the main data
    np.testing.assert_allclose(observed.ravel(), saved.event_density, atol=1e-8)
    event_date = c[c.role == 'event'].date.iloc[0]
    comparisons = [np.delete(ordinary, i, axis=0).mean(axis=0) for i in range(3)] + [ordinary.mean(axis=0)]
    panels = []
    for dt, field, comp in zip(dates + [event_date], [*ordinary, observed], comparisons):
        diff = field - comp
        panels.append((pd.Timestamp(dt).strftime('%d %b'), diff, np.abs(diff).sum() / comp.sum()))
    rows.append(panels)

maximum = max(float(np.abs(p[1]).max()) for r in rows for p in r)
norm = colors.AsinhNorm(linear_width=1., vmin=-maximum, vmax=maximum)
palette = colors.LinearSegmentedColormap.from_list('difference', ['#1b5589', '#79a9ca', '#ffffff', '#e49775', '#a33823'], N=513)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': .5, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(3, 4, figsize=(7.25, 5.95))
fig.subplots_adjust(left=.105, right=.991, top=.89, bottom=.145, hspace=.39, wspace=.06)
for ri, r in enumerate(rows):
    for ci, (date, diff, dev) in enumerate(r):
        ax = axes[ri, ci]
        im = ax.imshow(diff, origin='lower', norm=norm, cmap=palette, interpolation='nearest')
        ax.set(xticks=[], yticks=[])
        for sp in ax.spines.values():
            sp.set_color('#969696')
        if ri == 0:
            ax.text(.5, 1.27, 'Event day' if ci == 3 else f'Reference day {ci + 1}', transform=ax.transAxes,
                    ha='center', weight='bold', fontsize=9)
        if ci == 0:
            ax.set_ylabel(CASES[ri][1], weight='bold', fontsize=9, labelpad=7)
        ax.set_title(f'{date}   ·   D = {dev:.2f}', fontsize=7.6, pad=3)
bar_ax = fig.add_axes([.215, .075, .61, .021])
ticks = [v for v in [-1000, -100, -10, -1, 0, 1, 10, 100, 1000] if abs(v) <= maximum]
bar = fig.colorbar(im, cax=bar_ax, orientation='horizontal', ticks=ticks)
bar.ax.set_xticklabels([str(v).replace('-', '−') for v in ticks])
bar.ax.tick_params(labelsize=7, length=2, width=.5)
bar.set_label('Difference in grid-cell flow from the comparison mean', fontsize=8, labelpad=3)
for suffix in ('png', 'svg'):
    fig.savefig(OUT / f'Figure_S13_daily_variability.{suffix}', dpi=420, bbox_inches='tight', pad_inches=.06)
for (_, city), r in zip(CASES, rows):
    print(city, [round(p[2], 3) for p in r])
