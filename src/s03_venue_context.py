"""Step 3: venue context, dispersal morphology and timing (Figure 4; Tables S3, S4, S7; Figure S14).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from common import DATA, FIGDATA, load_events, load_grid, out_dir

OUT = out_dir('venue_context')
ev = load_events().set_index('event_id')
fvu = pd.read_csv(FIGDATA / 'df_fvu_r1.csv')          # written by s01
g = load_grid(columns=['event_id', 'row', 'col', 'x_km', 'y_km', 'event_density']).sort_values(['event_id', 'row', 'col'])

fig4 = fvu[['event_id', 'city', 'baseline_magnitude', 'event_magnitude', 'q', 'fvu']].copy()
fig4['magnitude_group'] = pd.qcut(fig4.baseline_magnitude, 4, labels=['Small', 'Medium', 'Large', 'Mega'])
fig4['venue'] = fig4.event_id.map(ev.venue_id)
for src, dst in [('share_car', 'DRV'), ('share_metro', 'SUBWAY'), ('share_bus', 'BUS'), ('share_bicycle', 'BYC'),
                 ('share_walk', 'WLK'), ('share_taxi', 'TAXI'), ('isochrone_area_15min', 'iso_area'),
                 ('isochrone_compactness', 'iso_compactness'), ('metro_stations_2km', 'metro_centrality'),
                 ('road_density_km_per_km2', 'density'), ('dist_to_city_centre_km', 'dist_to_cbd')]:
    fig4[dst] = fig4.event_id.map(ev[src])


def morphology(X, Y, D):
    """Methods Eqs. 3-12 on positive-flow cells; X, Y are km offsets from the venue (111 km per degree)."""
    m = D > 0
    X, Y, D = X[m], Y[m], D[m]
    M = D.sum()
    cx, cy = (D * X).sum() / M, (D * Y).sum() / M
    rg = np.sqrt((D * ((X - cx) ** 2 + (Y - cy) ** 2)).sum() / M)
    s = np.sort(D)
    n = len(s)
    gini = ((2 * np.arange(1, n + 1) - n - 1) * s).sum() / (n ** 2 * s.mean())
    w = np.linalg.eigvalsh(np.cov(np.vstack([X, Y]), aweights=D, bias=True))
    return rg, gini, np.sqrt(w[1] / w[0]), np.hypot(cx, cy)


rows, rows_lat = [], []
for eid, d in g.groupby('event_id'):
    X, Y, D = d.x_km.to_numpy(float), d.y_km.to_numpy(float), d.event_density.to_numpy()
    rows.append((eid, *morphology(X, Y, D)))
    c = ev.loc[eid, 'cos_lat_venue']           # latitude-corrected distances (m3 sensitivity)
    rows_lat.append((eid, *morphology(X * 111.32 * c / 111, Y * 110.57 / 111, D)))
cols = ['event_id', 'radius_of_gyration', 'concentration_ratio', 'directionality', 'centroid_displacement']
fig4 = fig4.merge(pd.DataFrame(rows, columns=cols), on='event_id')

series = pd.read_csv(DATA / 'timeseries/extra_departures_5min.csv').sort_values(['event_id', 'step'])
trows = []
for eid, s in series.groupby('event_id'):
    f = s.extra_trips.clip(lower=0).to_numpy()
    if f.sum() <= 0:
        continue
    cum = np.cumsum(f) / f.sum()
    t = s.minutes_after_end.to_numpy() / 60
    trows.append((eid, t[np.argmax(cum >= 0.5)], t[np.argmax(cum >= 0.8)], f.max() / f.sum(), f.sum()))
fig4 = fig4.merge(pd.DataFrame(trows, columns=['event_id', 'T50', 'T80', 'peak_share', 'extra_trips']), on='event_id', how='left')
fig4.to_csv(FIGDATA / 'df_fig4_r1.csv', index=False)
ser = series.rename(columns={'extra_trips': 'extra_flow'}).assign(pos=lambda x: x.step - 1, index_col=lambda x: x.event_id)
ser['extra_magnitude'] = ser.groupby('event_id').extra_flow.transform(lambda v: v.clip(lower=0).sum())
ser.merge(fig4[['event_id', 'magnitude_group']], on='event_id').to_csv(FIGDATA / 'df_series_r1.csv', index=False)

# ---------------------------------------------------------------- group cut points and descriptive tables (Tables S3, S4)
cuts = fig4.baseline_magnitude.quantile([.25, .5, .75])
pd.DataFrame({'quantile': [.25, .5, .75], 'baseline_magnitude': cuts.values}).to_csv(OUT / 'group_cut_points.csv', index=False)
desc = []
for col in ['DRV', 'SUBWAY', 'BUS', 'BYC', 'WLK', 'radius_of_gyration', 'concentration_ratio', 'directionality',
            'centroid_displacement', 'iso_area', 'iso_compactness', 'metro_centrality', 'density']:
    for grp_name, z in fig4.groupby('magnitude_group', observed=True):
        x = z[col].dropna()
        desc.append(dict(indicator=col, group=grp_name, n=len(x), mean=x.mean(), sd=x.std(), median=x.median(),
                         q25=x.quantile(.25), q75=x.quantile(.75), min=x.min(), max=x.max()))
pd.DataFrame(desc).to_csv(OUT / 'group_descriptives.csv', index=False)

# ---------------------------------------------------------------- associations (Table S7)
d = fig4.copy()
d['lMb'], d['lq'] = np.log10(d.baseline_magnitude), np.log10(d.q)
d['ldcbd'] = np.log10(d.dist_to_cbd.clip(lower=0.1))
OUTCOMES = [('SUBWAY', 'Metro share', 'a'), ('BUS', 'Bus share', 'a'), ('DRV', 'Car share', 'a'),
            ('BYC', 'Bicycle share', 'a'), ('WLK', 'Walking share', 'a'),
            ('iso_area', '15-min isochrone area', 'b'), ('iso_compactness', 'Isochrone compactness', 'b'),
            ('metro_centrality', 'Metro stations within 2 km', 'b'), ('density', 'Road network density', 'b'),
            ('radius_of_gyration', 'Radius of gyration', 'c'), ('concentration_ratio', 'Spatial Gini coefficient', 'c'),
            ('directionality', 'Directionality', 'c'), ('centroid_displacement', 'Centroid displacement', 'c')]
SPECS = [('raw', 'z ~ lMb'), ('q', 'z ~ lMb + lq'), ('q_cbd', 'z ~ lMb + lq + ldcbd')]
rows = []
for col, label, panel in OUTCOMES:
    dd = d.dropna(subset=[col, 'ldcbd']).copy()
    dd['z'] = (dd[col] - dd[col].mean()) / dd[col].std()
    gv, gc = pd.factorize(dd.venue)[0], pd.factorize(dd.city)[0]
    for name, f in SPECS:
        m = smf.ols(f, dd).fit(cov_type='cluster', cov_kwds={'groups': gv})
        mc = smf.ols(f, dd).fit(cov_type='cluster', cov_kwds={'groups': gc})
        rows.append(dict(outcome=col, label=label, panel=panel, spec=name, n=int(m.nobs), b=m.params.lMb,
                         lo=m.conf_int().loc['lMb', 0], hi=m.conf_int().loc['lMb', 1], p=m.pvalues.lMb,
                         lo_city=mc.conf_int().loc['lMb', 0], hi_city=mc.conf_int().loc['lMb', 1], p_city=mc.pvalues.lMb))
A = pd.DataFrame(rows)
A.to_csv(OUT / 'associations.csv', index=False)

# ---------------------------------------------------------------- timing at equal extra demand (Figure 4e, Figure S14)
t = d.dropna(subset=['T50']).copy()
t['lV'] = np.log10(t.extra_trips)
gv = pd.factorize(t.venue)[0]
srows = []
for yv in ['T50', 'T80', 'peak_share']:
    for name, f in [('V', f'np.log10({yv}) ~ lV'), ('V_Mb', f'np.log10({yv}) ~ lV + lMb'),
                    ('V_Mb_cbd', f'np.log10({yv}) ~ lV + lMb + ldcbd')]:
        m = smf.ols(f, t).fit(cov_type='cluster', cov_kwds={'groups': gv})
        r = dict(outcome=yv, spec=name, n=int(m.nobs), b_V=m.params.lV)
        if 'lMb' in m.params:
            lo, hi = m.conf_int().loc['lMb']
            r.update(b_Mb=m.params.lMb, lo=lo, hi=hi, p=m.pvalues.lMb, pct=100 * (10 ** m.params.lMb - 1),
                     pct_lo=100 * (10 ** lo - 1), pct_hi=100 * (10 ** hi - 1))
        srows.append(r)
S = pd.DataFrame(srows)
S.to_csv(OUT / 'size_matched.csv', index=False)

# ---------------------------------------------------------------- latitude-corrected morphology (m3 sensitivity)
lat = pd.DataFrame(rows_lat, columns=cols).set_index('event_id')
dl = d.set_index('event_id')
lrows = []
for col, label in [('radius_of_gyration', 'Radius of gyration'), ('directionality', 'Directionality'),
                   ('centroid_displacement', 'Centroid displacement')]:
    r = {'indicator': label, 'median_ratio': (lat[col] / dl[col]).median(), 'spearman': spearmanr(lat[col], dl[col])[0]}
    for tag, values in [('reported', dl[col]), ('latitude_corrected', lat[col].reindex(dl.index))]:
        dd = dl.assign(z=(values - values.mean()) / values.std())
        m = smf.ols('z ~ lMb + lq + ldcbd', dd).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.venue)[0]})
        r.update({f'b_{tag}': m.params.lMb, f'lo_{tag}': m.conf_int().loc['lMb', 0],
                  f'hi_{tag}': m.conf_int().loc['lMb', 1], f'p_{tag}': m.pvalues.lMb})
    lrows.append(r)
pd.DataFrame(lrows).to_csv(OUT / 'latitude_sensitivity.csv', index=False)

pd.set_option('display.width', 200)
print('group cut points:', cuts.round(0).tolist())
print(A[A.spec == 'q_cbd'][['label', 'b', 'lo', 'hi', 'p', 'p_city']].round(3).to_string(index=False))
print(S.dropna(subset=['pct'])[['outcome', 'spec', 'pct', 'pct_lo', 'pct_hi']].round(1).to_string(index=False))
print(pd.DataFrame(lrows).round(3).to_string(index=False))
