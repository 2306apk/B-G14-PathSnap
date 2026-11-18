# Phase 1 Status

## Implemented and tested

- Strict four-file loader and validation.
- Complete 90-trip EDA and graph audit.
- Per-trip UTM selection, projection, endpoint arc geometry, and STRtree.
- Monotonic dynamic-programming reference association with uncertainty fields.
- Training-derived robust 70 m radius and explicit candidate/no-candidate misses.
- Fourteen causal candidate features and Parquet model interface.
- Required static plots and interactive Folium map.
- Five probability-producing initial classifiers with trip-group CV.
- Separate classifier-emission Viterbi decoder with gap/no-candidate resets.
- Unit tests and whole-repository syntax compilation.

## Partially implemented

- Model experimentation is an initial 150,000-row balanced baseline, not final full-data selection.
- HMM transition parameters are defensible baselines, not tuned.
- Notebooks are reproducible views over script outputs; scripts are the authoritative execution path.

## Implemented but not executed

- Optuna Bayesian optimization framework (`scripts/optimize_model.py`).

## Planned / out of Phase 1

- Bayesian trials and tuned-vs-untuned comparison.
- Final model selection, persisted production configuration, API, UI, and two-laptop deployment.
- Predicted-route Folium overlay after final model selection.

## Readiness

Phase 1 is ready for independent review. It is not ready to claim a final competition model because optimization and final selection remain unexecuted.
