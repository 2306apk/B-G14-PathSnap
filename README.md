# G14 PathSnap - Phase 1

Independent, trip-local map-matching pipeline for the PathSnap ML Hackathon. Phase 1 includes strict raw-file validation, EDA, per-trip metric projection, graph audits, ordered reference labelling, candidate generation, causal features, five initial classifiers, and a separate HMM/Viterbi decoder.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATHSNAP_DATA_ROOT="../Student folder"
```

Raw data is never copied or modified. `--data-root` overrides the environment variable.

## Reproduce Phase 1

```bash
.venv/bin/python scripts/run_eda.py --data-root "$PATHSNAP_DATA_ROOT"
.venv/bin/python scripts/run_preprocessing.py --data-root "$PATHSNAP_DATA_ROOT"
.venv/bin/python scripts/build_features.py --data-root "$PATHSNAP_DATA_ROOT"
.venv/bin/python scripts/run_experiments.py --max-train-rows 150000 --cv-folds 3 --seed 42
.venv/bin/python scripts/run_hmm_baseline.py --data-root "$PATHSNAP_DATA_ROOT" --model data/processed/models/hist_gradient_boosting.joblib
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/pytest -q
```

Bayesian optimization is implemented but deliberately not run in the Phase 1 baseline:

```bash
.venv/bin/python scripts/optimize_model.py --trials 30 --seed 42
```

## Data contract and leakage

- `.track`: longitude, latitude, timestamp seconds.
- `.nodes`: longitude, latitude; row index is trip-local node ID.
- `.arcs`: from-node, to-node; row index is trip-local arc ID.
- `.route`: ordered reference arc IDs, never assumed one-per-GPS-observation.
- Training fits and radius calibration use only the 80 training trips.
- Student-test trips are evaluated only. Their labels never enter training transforms.
- Prediction/HMM functions consume candidates and model probabilities, not reference routes.

## Repository layout

```text
src/pathsnap/       reusable pipeline modules
scripts/            executable Phase 1 stages
tests/              unit and invalid-input tests
notebooks/          thin, reproducible views over script outputs
data/processed/     generated labels, features, models (ignored)
outputs/            generated CSVs, plots, maps, logs (ignored)
reports/audits/     evidence and readiness reports
```

See `reports/audits/PASTE_BACK_HERE.md` for executed results and limitations.
