"""Step 1b: matched five-fold XGBoost refits (Table S6; response to Reviewer 2, M1).

"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import load_events, load_grid, out_dir

FEATURES = ['baseline_density', 'baseline_gradient_mag', 'baseline_laplacian', 'dist_to_venue', 'angle_to_venue',
            'baseline_magnitude', 'event_magnitude', 'metro_station_density']
NO_TOTAL = [f for f in FEATURES if f != 'event_magnitude']
FRACTIONS = (0.02, 0.1, 0.6, 1.0)
EXPECTED_SAMPLED_ROWS = {1: 110721, 2: 108733, 3: 107662, 4: 109032, 5: 112515}   # original training samples
PARAMS = dict(n_estimators=1000, max_depth=7, learning_rate=0.05, subsample=0.7, colsample_bytree=0.8,
              random_state=42, verbosity=0)


def main(threads):
    out = out_dir('xgboost_refit')
    frame = load_grid().reset_index(drop=True)          # keep the stored row order: sampling depends on it
    fold = frame.event_id.map(load_events().set_index('event_id').cv_fold)
    groups = frame.groupby('event_id')
    event_total = groups.event_density.transform('sum')
    baseline_total = groups.baseline_density.transform('sum')
    target, background = frame.event_density, frame.baseline_density
    original = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    designs = {'refit_original': original,
               'xgb_grid_total': original.assign(event_magnitude=event_total),
               'xgb_no_event_total': original[NO_TOTAL]}
    pred = pd.DataFrame({'event_id': frame.event_id, 'fold': fold, 'saved_xgb': frame.pred_xgb,
                         'allocation': event_total * background / baseline_total})
    strata = pd.cut(target, [-np.inf, 2, 5, 10, np.inf], right=False)
    for k in sorted(fold.unique()):
        train, test = fold.ne(k), fold.eq(k)
        sampled = np.concatenate([target.loc[train & strata.eq(level)].sample(frac=frac, random_state=42).index
                                  for level, frac in zip(strata.cat.categories, FRACTIONS)])
        assert len(sampled) == EXPECTED_SAMPLED_ROWS[k]
        weights = 1.0 + 0.5 * target.loc[sampled]
        for name, X in designs.items():
            scaler = StandardScaler().fit(X.loc[train])
            model = XGBRegressor(**PARAMS, n_jobs=threads)
            model.fit(scaler.transform(X.loc[sampled]), target.loc[sampled], sample_weight=weights)
            pred.loc[test, name] = model.predict(scaler.transform(X.loc[test]))
            print(f'fold {k}: {name}', flush=True)
    clipped = pred.xgb_grid_total.clip(lower=0)
    pred['xgb_grid_total_rescaled'] = event_total * clipped / clipped.groupby(frame.event_id).transform('sum')
    rows = []
    for name in ['saved_xgb', *designs, 'xgb_grid_total_rescaled', 'allocation']:
        p = pred[name]
        rows.append({'model': name, 'total_r2': r2_score(target, p), 'mae': mean_absolute_error(target, p),
                     'extra_trip_r2': r2_score(target - background, p - background)})
    res = pd.DataFrame(rows)
    res.to_csv(out / 'pooled_metrics.csv', index=False)
    pred.to_parquet(out / 'held_out_predictions.parquet', index=False)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--threads', type=int, default=4)
    main(ap.parse_args().threads)
