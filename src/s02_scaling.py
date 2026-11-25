"""Step 2: deviation from everyday shares (FVU) and its relation to baseline magnitude and relative surge.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

from common import DATA, SEED, everyday_share, load_events, load_grid, out_dir

O = out_dir('scaling')
rng = np.random.default_rng(SEED)
ev = load_events().set_index('event_id')
g = load_grid().sort_values(['event_id', 'row', 'col']).reset_index(drop=True)
assert len(g) == 433 * 1600

# signature for "one record per identical input field": the group label, or the record itself
sig = np.where(ev.identical_input_group.fillna('') == '', 'E' + ev.index.astype(str), ev.identical_input_group)
meta = pd.DataFrame({'venue': ev.venue_id, 'city': ev.city, 'field_signature': sig}, index=ev.index)

names = {'xgb': 'pred_xgb', 'proportional': None}
y, b = g.event_density.to_numpy(), g.baseline_density.to_numpy()
rows, pooled = [], []
for name, col in names.items():
    pp = g[col].to_numpy() if col else everyday_share(g).to_numpy()
    for target, truth, pred in [('total', y, pp), ('signed_change', y - b, pp - b)]:
        resid = truth - pred
        sse, sst = float(resid @ resid), float(np.square(truth - truth.mean()).sum())
        pooled.append(dict(model=name, target=target, n_events=433, n_cells=len(y), r2=1 - sse / sst,
                           mae=float(np.abs(resid).mean()), rmse=float(np.sqrt(np.square(resid).mean())), sse=sse, sst=sst))
        for eid, idx in g.groupby('event_id').indices.items():
            yy, rr = truth[idx], resid[idx]
            ss, tt = float(rr @ rr), float(np.square(yy - yy.mean()).sum())
            rows.append(dict(event_id=eid, model=name, target=target, n_cells=len(idx), mae=float(np.abs(rr).mean()),
                             rmse=float(np.sqrt(np.square(rr).mean())), sse=ss, sst=tt, fvu=ss / tt, r2=1 - ss / tt))
d = pd.DataFrame(rows)
tot = g.groupby('event_id').agg(baseline_total=('baseline_density', 'sum'), event_total=('event_density', 'sum'))
tot['total_ratio'] = tot.event_total / tot.baseline_total
d = d.merge(meta, left_on='event_id', right_index=True).merge(tot, left_on='event_id', right_index=True)
d = d[['event_id', 'model', 'target', 'n_cells', 'mae', 'rmse', 'sse', 'sst', 'fvu', 'r2', 'venue', 'city',
       'baseline_total', 'event_total', 'total_ratio', 'field_signature']]
d.to_csv(O / 'event_metrics.csv', index=False)
pd.DataFrame(pooled).to_csv(O / 'pooled_metrics.csv', index=False)

# ---------------------------------------------------------------- regressions of log error measures (Table S5B, Note S3.1)
reg = []
for model, z in d.query("target == 'total'").groupby('model'):
    for subset, zz in [('all433', z), ('unique_fields', z.drop_duplicates('field_signature'))]:
        for spec in ['baseline_only', 'relative_impact_city', 'event_size_city']:
            X = pd.DataFrame({'log_baseline': np.log(zz.baseline_total)}, index=zz.index)
            if spec != 'baseline_only':
                key = 'total_ratio' if spec == 'relative_impact_city' else 'event_total'
                X['log_' + key] = np.log(zz[key])
                X = pd.concat([X, pd.get_dummies(zz.city, prefix='city', drop_first=True, dtype=float)], axis=1)
            X = sm.add_constant(X)
            for target in ['fvu', 'sse', 'sst', 'mae', 'rmse']:
                fit = sm.OLS(np.log(zz[target]), X).fit(cov_type='cluster', cov_kwds={'groups': zz.venue})
                ci = fit.conf_int()
                for term in [v for v in X.columns if v.startswith('log_')]:
                    reg.append(dict(model=model, subset=subset, specification=spec, dependent='log_' + target, term=term,
                                    coefficient=fit.params[term], ci025=ci.loc[term, 0], ci975=ci.loc[term, 1],
                                    pvalue=fit.pvalues[term], n_events=len(zz), n_venues=zz.venue.nunique(), r2=fit.rsquared))
reg = pd.DataFrame(reg)
reg.to_csv(O / 'scale_error_regressions.csv', index=False)


def slope(x, y):
    return float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1])


def boot_slope(z, n=1000, col='fvu'):
    x, yv = np.log(z.baseline_total.to_numpy()), np.log(z[col].to_numpy())
    groups = [np.flatnonzero(z.venue.to_numpy() == v) for v in z.venue.unique()]
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if np.ptp(x[idx]) > 1e-10:
            vals.append(slope(x[idx], yv[idx]))
    return slope(x, yv), np.quantile(vals, [.025, .975]).tolist(), len(vals)


def curve_fit(kind, mb, y):
    center = float(np.exp(np.mean(np.log(mb))))
    x = mb / center
    lx = np.log(x)
    if kind in ['power', 'exponential']:
        xx = lx if kind == 'power' else x
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(x)), xx]), y, rcond=None)[0]
        return {'kind': kind, 'center': center, 'params': coef.tolist()}
    start = np.linalg.lstsq(np.column_stack([np.ones(len(x)), lx]), y, rcond=None)[0]

    def pred(p):
        return np.logaddexp(p[0], p[1] + p[2] * lx)
    fits = []
    for floor in [-20, float(np.quantile(y, .1)) - 2]:
        p0 = [max(-29, min(29, floor)), float(np.clip(start[0], -29, 29)), float(np.clip(start[1], -4.9, -.001))]
        fits.append(least_squares(lambda p: pred(p) - y, p0, bounds=([-30, -30, -5], [30, 30, 0]), max_nfev=3000))
    f = min(fits, key=lambda z: np.square(z.fun).sum())
    return {'kind': kind, 'center': center, 'params': f.x.tolist(), 'converged': bool(f.success)}


def curve_predict(f, mb):
    x, p = mb / f['center'], f['params']
    if f['kind'] == 'power':
        return p[0] + p[1] * np.log(x)
    if f['kind'] == 'exponential':
        return p[0] + p[1] * x
    return np.logaddexp(p[0], p[1] + p[2] * np.log(x))


def skip_draws(z):
    """Advance the random stream exactly as one model block below does (slope and city bootstraps).

    The original analysis also bootstrapped a linear comparison model, which came first in this loop.
    Skipping its draws keeps the published intervals of the allocation and XGBoost unchanged."""
    n_venues = z.venue.nunique()
    for _ in range(1000):
        rng.integers(0, n_venues, n_venues)
    for _, cz in z.groupby('city'):
        if len(cz) >= 10 and cz.venue.nunique() >= 3:
            for _ in range(500):
                rng.integers(0, cz.venue.nunique(), cz.venue.nunique())


fits, slopes, cityrows, strata = [], [], [], []
skip_draws(d.query("target == 'total' and model == 'proportional'").reset_index(drop=True))
for model, z in d.query("target == 'total'").groupby('model'):
    z = z.reset_index(drop=True)
    mb, yv = z.baseline_total.to_numpy(), np.log(z.fvu.to_numpy())
    est, ci, nboot = boot_slope(z)
    slopes.append(dict(model=model, beta=est, ci025=ci[0], ci975=ci[1], bootstrap_valid=nboot, n_events=len(z),
                       n_venues=z.venue.nunique()))
    for kind in ['power', 'exponential', 'power_with_floor']:
        f = curve_fit(kind, mb, yv)
        err = curve_predict(f, mb) - yv
        cvpred = np.full(len(z), np.nan)
        for tr, te in GroupKFold(5).split(mb, yv, z.venue):
            cvpred[te] = curve_predict(curve_fit(kind, mb[tr], yv[tr]), mb[te])
        mse = float(np.mean(err ** 2))
        k = 3 if kind == 'power_with_floor' else 2
        fits.append(dict(model=model, curve=kind, n_events=len(z), log_mse=mse, log_rmse=np.sqrt(mse),
                         aic=len(z) * np.log(mse) + 2 * k, venue_cv_log_rmse=np.sqrt(np.mean((cvpred - yv) ** 2)),
                         parameters=json.dumps(f)))
    for city, cz in z.groupby('city'):
        enough = len(cz) >= 10 and cz.venue.nunique() >= 3
        span = cz.baseline_total.max() / cz.baseline_total.min()
        row = dict(model=model, city=city, n_events=len(cz), n_venues=cz.venue.nunique(), baseline_span=span,
                   eligible=enough and span >= 10)       # a power-law fit needs at least one order of magnitude
        if enough:
            bb, cc, nb = boot_slope(cz, 500)             # drawn for every city with enough records (fixed random stream)
            row.update(beta=bb, ci025=cc[0], ci975=cc[1], valid_bootstraps=nb)
        cityrows.append(row)
    z['impact_bin'] = pd.cut(z.total_ratio, [0, 1, 1.5, 2, 3, np.inf], labels=['<=1', '1-1.5', '1.5-2', '2-3', '>3'])
    for label, zz in z.groupby('impact_bin', observed=True):
        strata.append(dict(model=model, impact_bin=str(label), n_events=len(zz), n_venues=zz.venue.nunique(),
                           median_fvu=zz.fvu.median(), median_mae=zz.mae.median(),
                           spearman_baseline_fvu=spearmanr(zz.baseline_total, zz.fvu)[0] if len(zz) > 3 else np.nan,
                           loglog_slope=slope(np.log(zz.baseline_total), np.log(zz.fvu)) if len(zz) > 3 else np.nan))
pd.DataFrame(fits).to_csv(O / 'curve_comparison.csv', index=False)
pd.DataFrame(slopes).to_csv(O / 'power_slope_bootstrap.csv', index=False)
pd.DataFrame(cityrows).to_csv(O / 'city_slopes.csv', index=False)
pd.DataFrame(strata).to_csv(O / 'impact_strata.csv', index=False)

# ---------------------------------------------------------------- extra-trip FVU (Note S3.1; response to Reviewer 2, M3)
zc = d.query("model == 'proportional' and target == 'signed_change'").reset_index(drop=True)
est, ci, _ = boot_slope(zc)
X = sm.add_constant(pd.concat([pd.DataFrame({'log_baseline': np.log(zc.baseline_total), 'log_total_ratio': np.log(zc.total_ratio)}),
                               pd.get_dummies(zc.city, prefix='city', drop_first=True, dtype=float)], axis=1))
fit = sm.OLS(np.log(zc.fvu), X).fit(cov_type='cluster', cov_kwds={'groups': zc.venue})
pd.DataFrame([{'quantity': 'extra_trip_fvu_slope', 'estimate': est, 'ci025': ci[0], 'ci975': ci[1]},
              {'quantity': 'extra_trip_fvu_slope_adjusted_q_city', 'estimate': fit.params['log_baseline'],
               'ci025': fit.conf_int().loc['log_baseline', 0], 'ci975': fit.conf_int().loc['log_baseline', 1]}]) \
    .to_csv(O / 'extra_trip_fvu.csv', index=False)

# ---------------------------------------------------------------- reference-day variability at three venues (Figure S13)
cases = pd.read_parquet(DATA / 'grids/reference_day_cases.parquet')
crow = []
for eid, c in cases.groupby('event_id'):
    fields = {dt: q.sort_values('cell').flow.to_numpy() for dt, q in c[c.role == 'reference'].groupby('date')}
    avg = np.mean(list(fields.values()), axis=0)
    evf = c[c.role == 'event'].sort_values('cell').flow.to_numpy()
    for label, field, base, role in [('event', evf, avg, 'event_vs_reference_mean')] + \
            [(dt, f, np.mean([v for k2, v in fields.items() if k2 != dt], axis=0), 'selected_reference_vs_other_two')
             for dt, f in fields.items()]:
        diff = field - base
        crow.append(dict(event_id=eid, date=label, role=role, total_ratio=field.sum() / base.sum(),
                         pearson_total=pearsonr(field, base)[0], pearson_signed_change=pearsonr(diff, base)[0],
                         normalized_l1=np.abs(diff).sum() / base.sum()))
pd.DataFrame(crow).to_csv(O / 'three_case_reference_variability.csv', index=False)

pd.set_option('display.width', 200)
print(pd.DataFrame(slopes).round(3).to_string(index=False))
print(pd.DataFrame(fits)[['model', 'curve', 'venue_cv_log_rmse']].round(3).to_string(index=False))
print(reg.query("subset == 'all433' and model == 'proportional' and term == 'log_baseline'")
      [['specification', 'dependent', 'coefficient', 'ci025', 'ci975']].round(3).to_string(index=False))
print(pd.DataFrame(strata).query("model == 'proportional'")[['impact_bin', 'n_events', 'median_fvu', 'loglog_slope']].round(3).to_string(index=False))
cs = pd.DataFrame(cityrows).query("model == 'proportional' and eligible")
print('eligible cities', len(cs), 'negative', int((cs.beta < 0).sum()), 'CI excludes 0', int((cs.ci975 < 0).sum()))
print(pd.read_csv(O / 'extra_trip_fvu.csv').round(3).to_string(index=False))
