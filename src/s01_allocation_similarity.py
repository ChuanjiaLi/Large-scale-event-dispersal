"""Step 1: spatial similarity, everyday-share allocation and saved model predictions.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import explained_variance_score, median_absolute_error, r2_score

from common import FIGDATA, DATA, SEED, everyday_share, load_events, load_grid, out_dir, unique_input_mask

OUT = out_dir('allocation')
ev = load_events().set_index('event_id')
g = load_grid().sort_values(['event_id', 'row', 'col']).reset_index(drop=True)
assert g.event_id.nunique() == 433 and len(g) == 433 * 1600
g['Y_allocated'] = everyday_share(g)
grp = g.groupby('event_id')

# ---------------------------------------------------------------- figure inputs (Figures 1c, 1d, 2, 3)
cities = pd.read_csv(DATA / 'cities.csv')
pd.DataFrame({'city name': cities.city, 'count': cities.n_events, 'city_tier_ordered': cities.city_tier}) \
    .set_index('city name').to_csv(FIGDATA / 'city_tier_features_r1.csv')
tot = grp.agg(baseline_magnitude=('baseline_density', 'sum'), event_magnitude=('event_density', 'sum'))
tot.reset_index().to_csv(FIGDATA / 'event_mag_info_r1.csv', index=False)
g[['event_id', 'x_km', 'y_km', 'baseline_density', 'event_density', 'Y_allocated']] \
    .to_parquet(FIGDATA / 'grid_allocation_r1.parquet', index=False)


def cor(x, y, rank=False):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float((spearmanr if rank else pearsonr)(x, y)[0])


def scores(y, p):
    r = y - p
    sse, sst = float(r @ r), float(np.square(y - y.mean()).sum())
    return {'r2': 1 - sse / sst, 'fvu': sse / sst, 'mae': float(np.abs(r).mean()),
            'rmse': float(np.sqrt(np.square(r).mean())), 'sse': sse, 'sst': sst}


PRED = {'allocation': 'Y_allocated', 'xgb': 'pred_xgb'}
rows = []
for eid, d in grp:
    y, b = d.event_density.to_numpy(), d.baseline_density.to_numpy()
    act = y > 1
    row = {'event_id': eid, 'venue_id': ev.loc[eid, 'venue_id'], 'city': ev.loc[eid, 'city'],
           'baseline_total': b.sum(), 'event_total': y.sum(), 'q': y.sum() / b.sum(),
           # Figure 2a: active cells (arrivals > 1); Figure S11: all 1,600 cells
           'sss': cor(y[act], b[act], True), 'pearson': cor(y[act], b[act]),
           'total_pearson': cor(y, b), 'change_pearson': cor(y - b, b),
           'TV': 0.5 * np.abs(y / y.sum() - b / b.sum()).sum()}
    for name, col in PRED.items():
        p = d[col].to_numpy()
        for target, yy, pp in [('total', y, p), ('change', y - b, p - b)]:
            for k, v in scores(yy, pp).items():
                row[f'{name}_{target}_{k}'] = v
    rows.append(row)
em = pd.DataFrame(rows)
em.to_csv(OUT / 'event_metrics.csv', index=False)

sim = em[['event_id', 'sss', 'pearson', 'total_pearson', 'change_pearson']].rename(
    columns={'total_pearson': 'pearson_total_all', 'change_pearson': 'pearson_extra_all'})
sim.to_csv(FIGDATA / 'df_similarity_r1.csv', index=False)
fvu = em[['event_id', 'allocation_total_fvu', 'baseline_total', 'event_total', 'city', 'q']].rename(
    columns={'allocation_total_fvu': 'fvu', 'baseline_total': 'baseline_magnitude', 'event_total': 'event_magnitude'})
fvu['q_band'] = pd.cut(fvu.q, [0, 1, 1.5, 2, 3, np.inf], labels=['<=1', '1-1.5', '1.5-2', '2-3', '>3'])
fvu.to_csv(FIGDATA / 'df_fvu_r1.csv', index=False)

# ---------------------------------------------------------------- pooled scores (Table S6 rows, Table S2)
y, b = g.event_density.to_numpy(), g.baseline_density.to_numpy()
pooled = []
for name, col in PRED.items():
    p = g[col].to_numpy()
    for target, yy, pp in [('total', y, p), ('extra_trips', y - b, p - b)]:
        pooled.append({'predictor': name, 'target': target, **scores(yy, pp)})
pd.DataFrame(pooled).to_csv(OUT / 'pooled_metrics.csv', index=False)

table_s2 = []
for name in ['allocation', 'xgb']:
    p = g[PRED[name]].to_numpy()
    m5 = y > 5
    table_s2.append({'model': name, 'R2': r2_score(y, p), 'ExpVar': explained_variance_score(y, p),
                     'Pearson': pearsonr(y, p)[0], 'Spearman': spearmanr(y, p)[0],
                     'RMSE': float(np.sqrt(np.mean((y - p) ** 2))), 'MAE': float(np.mean(np.abs(y - p))),
                     'MedAE': median_absolute_error(y, p),
                     'MAPE@5': float(np.mean(np.abs((y[m5] - p[m5]) / y[m5])) * 100),
                     'Bias': float(np.mean(p - y))})
pd.DataFrame(table_s2).to_csv(OUT / 'table_S2_model_metrics.csv', index=False)

# ---------------------------------------------------------------- similarity medians with venue-bootstrap intervals
ev2 = ev.reindex(em.event_id)
em['unique_input'] = unique_input_mask(ev2.reset_index()).to_numpy()
em['no_shared_reference'] = (ev2.n_reference_days_shared_with_listed_event == 0).to_numpy()
subsets = {'all': em, 'q_gt_3': em[em.q > 3], 'unique_input': em[em.unique_input],
           'no_shared_reference': em[em.no_shared_reference]}
metrics = ['total_pearson', 'change_pearson', 'sss', 'pearson', 'allocation_total_r2', 'allocation_change_r2']
rng = np.random.default_rng(SEED)
summ = []
for label, s in subsets.items():
    groups = [idx for _, idx in s.reset_index(drop=True).groupby('venue_id').indices.items()]
    vals = s[metrics].reset_index(drop=True)
    boots = np.array([vals.iloc[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])]
                      .median().to_numpy() for _ in range(1000)])
    for j, m in enumerate(metrics):
        summ.append({'subset': label, 'n_records': len(s), 'n_venues': s.venue_id.nunique(), 'metric': m,
                     'median': s[m].median(), 'q25': s[m].quantile(.25), 'q75': s[m].quantile(.75),
                     'ci025': np.nanquantile(boots[:, j], .025), 'ci975': np.nanquantile(boots[:, j], .975)})
summ = pd.DataFrame(summ)
summ.to_csv(OUT / 'similarity_summary.csv', index=False)

pd.set_option('display.width', 200)
print(pd.DataFrame(pooled).query("predictor == 'allocation'")[['target', 'r2', 'mae']].round(3).to_string(index=False))
print('median event R2 (allocation):', round(em.allocation_total_r2.median(), 3))
print(pd.DataFrame(table_s2).round(2).to_string(index=False))
print(summ.query("metric in ['total_pearson','change_pearson','sss','pearson']")
      .pivot(index='subset', columns='metric', values='median').round(3))
print(summ.query("metric == 'change_pearson'")[['subset', 'n_records', 'n_venues', 'median', 'ci025', 'ci975']].round(3).to_string(index=False))
