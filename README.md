# Large-scale event dispersal

Data and code for **“Revealing urban mobility inertia of crowd dispersal after large-scale events”** (iScience, revised
version). Starting from aggregated, anonymized inputs for 433 planned special event records in 32 Chinese cities, the
repository reproduces the tables and figure panels of the paper and its Supplemental Information listed below; the few
items that need confidential inputs are named under "What is not included".

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh            # tables and figure panels listed below, about 1-2 min
bash run_all.sh --refit    # also refit XGBoost on the original folds (Table S6), about 15-30 min
```

Tables and intermediate results are written to `results/`, figure panels to `figures/output/`.
The published figures were assembled from these panels; figure fonts use Arial.

## Repository structure

```text
data/        aggregated inputs (see data/README.md)
src/         analysis steps, run in order
  common.py                      paths, loaders, everyday-share allocation
  s01_allocation_similarity.py   similarity, everyday-share allocation, saved model predictions
  s01b_xgboost_refit.py          matched XGBoost refits on the original five folds (optional, slow)
  s02_scaling.py                 FVU, power-law fits, surge bands, error decomposition, city fits
  s03_venue_context.py           venue groups, morphology, timing, multivariable associations
  s04_scenarios.py               venue groups and modal-shift scenario responses
  modal_shift_model.py           reference implementation of the modal-shift scenarios
figures/     plotting scripts (run after src/)
  main_fig.py                    main-text figure panels
  si_models.py                   Figures S2-S5; Table S2
  si_scaling.py                  Figures S6, S11, S12; Table S5; response-letter Table R2
  si_robustness.py               Figures S9, S10
  si_venue_context.py            Figures S7, S8, S14; Tables S3, S4, S7
  si_reference_days.py           Figure S13
run_all.sh   runs everything in order
```

## Where each result comes from

| Paper item | Script | Output |
|---|---|---|
| Figure 1c–d | `src/s01`, `figures/main_fig.py` | `MainFig-1-3`, `MainFig-1-4` |
| Figure 2a–c; Figure S11 | `src/s01`, `figures/main_fig.py`, `figures/si_scaling.py` | `MainFig-2-*`, `si/Figure_S11_background` |
| Table S2; Figures S2–S5 (everyday-share allocation and XGBoost) | `src/s01`, `figures/si_models.py` | `results/allocation/table_S2_model_metrics.csv`, `si/Table_S2_metrics.tex`, `si/Figure_S2*`–`si/Figure_S5*` |
| Table S6 (allocation and model comparison) | `src/s01`, `src/s01b` | `results/allocation/pooled_metrics.csv`, `results/xgboost_refit/pooled_metrics.csv` |
| Figure 3; Table S5; Figures S6, S12 | `src/s02`, `figures/main_fig.py`, `figures/si_scaling.py` | `MainFig-3-*`, `results/scaling/`, `si/Table_S5_scaling.tex` |
| Figures S9, S10 | `figures/si_robustness.py` | `si/Figure_S9*`, `si/Figure_S10*`, `results/robustness/` |
| Figure S13 | `src/s02`, `figures/si_reference_days.py` | `si/Figure_S13_daily_variability` |
| Figure 4; Tables S3, S4, S7; Figures S7, S8, S14 | `src/s03`, `figures/main_fig.py`, `figures/si_venue_context.py` | `MainFig-4-*`, `results/venue_context/` |
| Figure 5 (bottom row); Table S8 | `src/s04`, `figures/main_fig.py` | `MainFig-5-1-*`, `results/scenarios/Table_S8_groups.tex` |

Uncertainty intervals resample venues with all their records (fixed random seeds), so repeated events at a venue are
not treated as independent.

## What is not included

The raw trip records and the event inventory with event names, venue names and coordinates are provided by Amap and
are confidential, as are the queries used to extract them. The repository therefore starts from the aggregated
destination grids. Five items cannot be rebuilt from these inputs: the maps in Figure 1a–b (licensed base maps and raw
trip records), the schematics in the top row of Figure 5, the departure time series of the three named events in
Figure S1 (raw trip records), the sample of named events in Table S1 (event names), and the mode-specific destination
grids used to generate `data/scenarios/modal_shift_metrics.csv` (the generating functions are in
`src/modal_shift_model.py`).
Requests for additional information can be directed to the lead contact of the paper.

## Citation

Please cite the paper when using these data or code.
