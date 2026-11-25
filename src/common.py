"""Shared paths and loaders for the large-scale event dispersal analysis."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
FIGDATA = RESULTS / 'figure_data'      # inputs for figures/main_fig.py
FIGURES = ROOT / 'figures' / 'output'
for _p in (RESULTS, FIGDATA, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20261117


def out_dir(name):
    """results/<name>/, created on first use."""
    p = RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_events():
    """One row per record (433): anonymized venue, dates, flags, venue features and mode shares."""
    return pd.read_csv(DATA / 'events.csv')


def load_grid(name='main_40x40_120min', columns=None):
    """Destination grids, one row per record and cell (row-major), sorted by event_id and cell."""
    g = pd.read_parquet(DATA / 'grids' / f'{name}.parquet', columns=columns)
    return g


def everyday_share(grid):
    """Everyday-share allocation: D_hat = M_e * D_b / sum(D_b), per record."""
    grp = grid.groupby('event_id')
    return grp.event_density.transform('sum') * grid.baseline_density / grp.baseline_density.transform('sum')


def unique_input_mask(events):
    """True for one record per group of identical input fields (and for all records outside a group)."""
    grp = events.identical_input_group.fillna('')
    return (grp == '') | ~grp.duplicated()


def venue_bootstrap(frame, stat, n=1000, seed=SEED, venue_col='venue_id'):
    """Percentile interval of stat(frame) when venues are resampled with all their records."""
    rng = np.random.default_rng(seed)
    groups = [idx for _, idx in frame.groupby(venue_col).indices.items()]
    vals = []
    for _ in range(n):
        rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        vals.append(stat(frame.iloc[rows]))
    return np.nanquantile(np.array(vals), [0.025, 0.975], axis=0)
