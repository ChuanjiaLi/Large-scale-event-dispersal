"""Step 4: venue groups and modal-shift scenario responses (Figure 5; Table S8; Note S5).
"""
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import adjusted_rand_score

from common import DATA, FIGDATA, load_events, out_dir

O = out_dir('scenarios')
ev = load_events().set_index('event_id')
d = pd.read_csv(FIGDATA / 'df_fig4_r1.csv')                       # written by s03
d['min_dis_to_trunk'] = d.event_id.map(ev.dist_to_trunk_road_m)
d['car_taxi'] = d.DRV + d.TAXI
M, D = d.baseline_magnitude.to_numpy(), d.dist_to_cbd.to_numpy()
LABELS = ['H', 'LC', 'LP']
NAMES = {'H': 'CBD core area', 'LC': 'Sub-center area', 'LP': 'Remote suburban area'}


def assign(m_cut, d_cut):
    """Ties at a cut go to the low / central side."""
    return np.where(M > m_cut, 'H', np.where(D > d_cut, 'LP', 'LC'))


m50, d50 = np.quantile(M, .5), np.quantile(D, .5)
main = assign(m50, d50)
d['group'] = main

# ---------------------------------------------------------------- threshold grid (40th-60th percentiles)
qs = np.round(np.arange(0.40, 0.601, 0.05), 2)
schemes = [(qm, qd, assign(np.quantile(M, qm), np.quantile(D, qd))) for qm in qs for qd in qs]
grid = pd.DataFrame([{'q_Mb': qm, 'q_dist': qd, 'share_same_label': (lab == main).mean(),
                      'ARI_vs_main': adjusted_rand_score(main, lab), **{f'n_{k}': int((lab == k).sum()) for k in LABELS}}
                     for qm, qd, lab in schemes])
grid.to_csv(O / 'threshold_grid.csv', index=False)
d['grid_stability'] = np.mean([lab == main for _, _, lab in schemes], axis=0)

# ---------------------------------------------------------------- venue-cluster bootstrap of the two medians
rng = np.random.default_rng(20261122)
venues = d.venue.unique()
idx = {v: np.flatnonzero(d.venue.to_numpy() == v) for v in venues}
keep = np.zeros(len(d))
boot = []
for _ in range(2000):
    ii = np.concatenate([idx[v] for v in rng.choice(venues, size=len(venues), replace=True)])
    lab = assign(np.median(M[ii]), np.median(D[ii]))
    keep += lab == main
    boot.append({'Mb_cut': np.median(M[ii]), 'dist_cut_km': np.median(D[ii]), 'share_same_label': (lab == main).mean()})
d['bootstrap_stability'] = keep / 2000
pd.DataFrame(boot).to_csv(O / 'bootstrap_thresholds.csv', index=False)
d[['event_id', 'venue', 'city', 'baseline_magnitude', 'dist_to_cbd', 'group', 'grid_stability', 'bootstrap_stability']] \
    .to_csv(O / 'event_groups.csv', index=False)

# ---------------------------------------------------------------- Table S8: group means, ranges across the 25 schemes, p-values
FEATS = [('dist_to_cbd', 'Distance to city centre (km)', '{:.1f}'),
         ('metro_centrality', 'Metro stations within 2 km', '{:.1f}'),
         ('density', 'Road network density (km km\\(^{-2}\\))', '{:.2f}'),
         ('SUBWAY', 'Metro share (\\%)', 'pct'), ('car_taxi', 'Car and taxi share (\\%)', 'pct')]
vt = d.groupby('venue').agg(group=('group', lambda s: s.value_counts().index[0]), **{c: (c, 'mean') for c, _, _ in FEATS})
fmt = lambda v, f: f'{100 * v:.1f}' if f == 'pct' else f.format(v)
comp = []
for c, lab, f in FEATS:
    r = {'feature': lab}
    for k in LABELS:
        vals = [d.loc[s == k, c].mean() for _, _, s in schemes]
        r[k], r[k + '_range'] = fmt(d.loc[main == k, c].mean(), f), f'{fmt(min(vals), f)}--{fmt(max(vals), f)}'
    r['p_events'] = kruskal(*[d.loc[main == k, c] for k in LABELS]).pvalue
    r['p_venues'] = kruskal(*[vt.loc[vt.group == k, c] for k in LABELS]).pvalue
    comp.append(r)
comp = pd.DataFrame(comp)
comp.to_csv(O / 'group_comparison.csv', index=False)
n_ev = {k: [(s == k).sum() for _, _, s in schemes] for k in LABELS}
n_vn = {k: [d.loc[s == k, 'venue'].nunique() for _, _, s in schemes] for k in LABELS}
p_str = lambda p: '\\(<\\)0.001' if p < 0.001 else f'{p:.3f}'
two = lambda a, b: f'\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{a}\\\\{b}\\end{{tabular}}'
lines = [r'\begin{addedblock}', r'\noindent\textbf{Table S8. Venue groups used in the modal-shift scenarios.}',
         r'\par\smallskip', r'{\small', r'\begin{tabular}{@{}lccccc@{}}\toprule',
         ' & ' + ' & '.join([two('CBD core', 'area (H)'), two('Sub-center', 'area (LC)'), two('Remote suburban', 'area (LP)'),
                             two('\\(p\\)', '(events)'), two('\\(p\\)', '(venues)')]) + r' \\ \midrule',
         'Number of events & ' + ' & '.join(f'{(main == k).sum()} ({min(n_ev[k])}--{max(n_ev[k])})' for k in LABELS) + ' & & \\\\',
         'Number of venues & ' + ' & '.join(f'{d.loc[main == k, "venue"].nunique()} ({min(n_vn[k])}--{max(n_vn[k])})' for k in LABELS) + ' & & \\\\']
for _, r in comp.iterrows():
    lines.append(f'{r.feature} & ' + ' & '.join(f'{r[k]} ({r[k + "_range"]})' for k in LABELS)
                 + f' & {p_str(r.p_events)} & {p_str(r.p_venues)} \\\\')
lines += [r'\bottomrule\end{tabular}\par}', r'\end{addedblock}']
(O / 'Table_S8_groups.tex').write_text('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- scenario responses
sc = pd.read_csv(DATA / 'scenarios/modal_shift_metrics.csv').sort_values(['event_id', 'tau'])
tau = np.sort(sc.tau.unique())
P = sc.pivot(index='event_id', columns='tau', values='peak_intensity').loc[d.event_id].to_numpy()
R = sc.pivot(index='event_id', columns='tau', values='peak_weighted_radius_km').loc[d.event_id].to_numpy()
S = np.stack([P, R], axis=1)                                    # (433, 2, n_tau)
mid = np.argsort(np.abs(tau))[:2]                               # the two central transfer shares
REL = S / S[:, :, mid].mean(axis=2, keepdims=True) - 1
curves = {k: REL[main == k].mean(axis=0) for k in LABELS}
scheme_curves = {k: np.stack([REL[s == k].mean(axis=0) for _, _, s in schemes]) for k in LABELS}
i_neg, i_pos = int(np.argmin(np.abs(tau + 0.184))), int(np.argmin(np.abs(tau - 0.184)))
rows = []
for k in LABELS:
    for i, lab in [(i_neg, 'to private'), (i_pos, 'to public')]:
        rows.append({'group': k, 'shift': lab, 'tau': tau[i], 'peak_mean': curves[k][0, i],
                     'peak_lo': scheme_curves[k][:, 0, i].min(), 'peak_hi': scheme_curves[k][:, 0, i].max(),
                     'radius_mean': curves[k][1, i], 'radius_lo': scheme_curves[k][:, 1, i].min(),
                     'radius_hi': scheme_curves[k][:, 1, i].max(), 'share_peak_up': (REL[main == k, 0, i] > 0).mean(),
                     'share_radius_up': (REL[main == k, 1, i] > 0).mean()})
pd.DataFrame(rows).to_csv(O / 'scenario_summary.csv', index=False)
diffs = []
for a, b in [('LP', 'H'), ('LP', 'LC'), ('LC', 'H')]:
    v = [REL[s == a, 0, i_pos].mean() - REL[s == b, 0, i_pos].mean() for _, _, s in schemes]
    diffs.append({'pair': f'{a}-{b}', 'main': curves[a][0, i_pos] - curves[b][0, i_pos], 'min': min(v), 'max': max(v)})
pd.DataFrame(diffs).to_csv(O / 'group_differences_at_plus_18pct.csv', index=False)
np.savez(FIGDATA / 'fig5_curves.npz', tau=tau, labels=np.array(LABELS), **{f'{k}_mean': curves[k] for k in LABELS},
         **{f'{k}_schemes': scheme_curves[k] for k in LABELS})

pd.set_option('display.width', 200)
print('cuts: M_b %.0f, distance %.2f km; sizes %s' % (m50, d50, {k: int((main == k).sum()) for k in LABELS}))
print('share keeping label across 25 schemes: median %.3f, min %.3f' % (grid.share_same_label.median(), grid.share_same_label.min()))
print(comp.to_string(index=False))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(pd.DataFrame(diffs).round(3).to_string(index=False))
