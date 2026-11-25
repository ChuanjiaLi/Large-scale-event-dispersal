#!/usr/bin/env bash
# Reproduce the tables and figure panels listed in README.md from data/. Run from the repository root:
#   bash run_all.sh            # analysis + figures (about 1-2 min)
#   bash run_all.sh --refit    # also refit XGBoost on the original folds (Table S6; about 15-30 min)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
cd src
$PY s01_allocation_similarity.py
[[ "${1:-}" == "--refit" ]] && $PY s01b_xgboost_refit.py
$PY s02_scaling.py
$PY s03_venue_context.py
$PY s04_scenarios.py
cd ../figures
$PY main_fig.py
$PY si_models.py
$PY si_scaling.py
$PY si_robustness.py
$PY si_venue_context.py
$PY si_reference_days.py
echo "Done: tables in results/, figure panels in figures/output/"
