# Data

Aggregated, anonymized inputs for all analyses in the paper (433 planned-special-event records, 32 Chinese cities,
130 venues, 2024). Raw trip records, event and venue names, and venue coordinates are not included. Grid coordinates
are given as kilometre offsets from the venue (`x_km`, `y_km`; 111 km per degree on both axes, as in the paper).

## `events.csv` (one row per record)

| Column | Description |
|---|---|
| `event_id` | Record identifier used in all files |
| `city` | City |
| `venue_id` | Anonymized venue identifier (V001–V130); repeated events at a venue share it |
| `event_type` | concert, sports or exhibition |
| `event_date`, `end_time` | Event date and end time; the post-event window starts at `end_time` |
| `reference_dates` | The three reference days averaged for the baseline, separated by a vertical bar |
| `n_reference_days_after_event` | Reference days that fall after the event date (retrospective baseline) |
| `n_reference_days_shared_with_listed_event` | Reference days that coincide with another listed event at the same venue |
| `n_reference_days_weekday_mismatch` | Reference days on a different weekday from the event (0 for all records) |
| `identical_input_group` | Records whose event and baseline fields are identical share a group label (blank otherwise) |
| `cv_fold` | Event-record fold (1–5) of the five-fold validation of the learned models |
| `cos_lat_venue` | Cosine of the venue latitude (3 decimals), for latitude-corrected distances |
| `dist_to_city_centre_km` | Distance from the venue to the city centre |
| `isochrone_area_15min`, `isochrone_compactness` | 15-min driving isochrone area and Polsby–Popper compactness |
| `metro_stations_2km` | Metro stations within 2 km of the venue |
| `road_density_km_per_km2` | Length of major roads per unit area within 25 km |
| `dist_to_trunk_road_m` | Distance to the nearest trunk road |
| `share_metro`, `share_bus`, `share_car`, `share_bicycle`, `share_walk`, `share_taxi` | Mode shares of departing trips (provider mode labels) |

## `cities.csv`

City, city tier used in Figure 1, and number of records.

## `grids/`

All grids hold one row per record and cell, in row-major order (`row`, `col`), with `x_km`, `y_km` the cell centre
relative to the venue. Destination counts are trips departing within 1 km of the venue, 2–20 km long, binned on a grid
spanning the 0.5th–99.5th percentiles of event- and reference-day destinations and smoothed with a Gaussian filter
(sigma = 0.5 cell). The baseline is the mean of the three reference days.

| File | Content |
|---|---|
| `main_40x40_120min.parquet` | Main analysis: `event_density` (event day), `baseline_density` (baseline), the eight model features (`baseline_density`, `baseline_gradient_mag`, `baseline_laplacian`, `dist_to_venue`, `angle_to_venue`, `baseline_magnitude`, `event_magnitude`, `metro_station_density`; the two magnitudes are pre-grid trip counts) and the saved out-of-fold predictions of XGBoost (`pred_xgb`) |
| `robustness_grid20x20_120min.parquet`, `robustness_grid60x60_120min.parquet` | Same window on coarser and finer grids (Figure S9) |
| `robustness_grid40x40_{30,60,90}min.parquet` | 40 × 40 grid with shorter post-event windows (Figure S10) |
| `reference_day_cases.parquet` | Separate daily grids (three reference days and the event day) for the three Figure S1 records (Figure S13) |

## `timeseries/extra_departures_5min.csv`

Extra departures (event day minus baseline mean) in 48 consecutive 5-min intervals after the event ends
(Figure 4d–e, Figure S14).

## `scenarios/modal_shift_metrics.csv`

Peak intensity and peak-weighted radius of the scenario field for each record and transfer share `tau`
(Figure 5; generated with `src/modal_shift_model.py` from mode-specific destination grids, which are not distributed).
