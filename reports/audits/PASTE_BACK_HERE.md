# PathSnap Phase 1 - Consolidated Independent Audit

Generated from executed work in `/Users/apoorva_kogta/Documents/ML-Hackathon-Codex/G14-PathSnap` on 2026-11-18. Numbers below come from saved logs/CSVs, not estimates.

## 1. Source and integrity audit - completed

- Read all 22 pages across the five official PDFs and visually checked their rendered pages.
- Student archive SHA-256: `4804b0399d1d56f4844b9c74e2f998aae0a281fb5947a1f47ad1bd013214d248`.
- Materials archive SHA-256: `db368793f68c76baa8eaa30d783d376df50841287c29c43f6911feb98cab4367`.
- `unzip -t` reported no errors. All 360 extracted data files and all five extracted PDFs matched their archive members byte-for-byte.
- Repository cloned cleanly at commit `feba3f0`; no force-push, merge, PR, release, reset, clean, stash, or destructive history operation was performed during implementation.
- No evaluator-test partition was present or accessed. Original ZIPs and extracted raw files were not modified.

## 2. Dataset audit - completed and executed

| Item | Training | Student test |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Nodes | 1,933,485 | 90,689 |
| Arcs | 3,876,904 | 185,041 |
| Reference-route arcs | 31,336 | 3,477 |

- All 90 trips had all four files. Parsing/validation failures, NaNs, invalid arc-to-node references, invalid route IDs, exact duplicate observations, and non-increasing timestamps: zero.
- Repeated coordinate observations (timestamps may differ): 578.
- Exact 1-second gaps: 219,934 / 227,487 = 96.679810%. Gaps >10/30/60/120 s: 617/135/70/42. Maximum: 153,064 s in training trip `00000090`.
- Observation range: 262 (`student_test/00000052`) to 19,313 (`train/00000054`).
- GPS extent: longitude -100.418538 to 137.237029; latitude -34.640291 to 61.699590. This disproves a Netherlands-only/global-CRS assumption.
- Ninety unique node+arc SHA-256 fingerprints establish independent trip-local networks.
- No isolated nodes. Weak component counts ranged 1-421. Across reference routes, 14 consecutive transitions were disconnected; weighted connectivity was 99.959681%.
- Full per-trip inventory: `outputs/csv/dataset_inventory.csv`; CRS table: `outputs/csv/trip_crs.csv`.

## 3. Preprocessing and CRS - completed and executed

- WGS84 ranges and numeric types are validated before projection.
- UTM zone is selected per trip from the median of combined GPS and network coordinates: `zone=floor((longitude+180)/6)+1`; EPSG is `32600+zone` north or `32700+zone` south.
- Twenty-five EPSG codes were selected: 32614, 32615, 32616, 32617, 32622, 32629-32641 (present zones), 32643, 32648, 32651, 32653, 32721, 32722, 32735. Exact trip mapping is in `outputs/csv/trip_crs.csv`.
- `pyproj` uses `always_xy=True`. No distance is computed in longitude/latitude degrees.
- Arc geometry is a straight projected line from the documented from-node to to-node. This is explicitly an endpoint approximation because intermediate road geometry is unavailable.
- Each trip builds its own geometry array, graph, and Shapely STRtree; trip-local arc IDs never cross contexts.

## 4. Reference-route labelling - completed and executed

- `.route` is treated as an ordered route, never as one label per observation.
- A dynamic program assigns a nondecreasing route position to every GPS observation. Objective: sum of point-to-arc metres + `5 m * skipped route positions` + `2 m * omitted start/end positions`.
- Each label retains assigned arc ID, route position, assigned distance, nearest-route distance, distance margin, confidence, and uncertainty flag.
- Labels are explicitly reference-derived and not represented as independently verified physical truth.
- Two training trips have gross route/GPS mismatch: `00000033` median 122.413 m and `00000054` median 140.088 m.

## 5. Candidate radius and misses - completed and executed

- All-training distance p95/p99: 142.096 m/205.823 m, distorted by the two mismatched trips.
- A training-only extreme-outlier rule excludes trip medians above `Q3 + 5*IQR = 11.8613 m` from radius calibration only. It identified exactly `00000033` and `00000054`.
- On the other 78 trips (183,260 observations), p95/p99 were 18.842 m/69.829 m. Selected radius: 70 m (p99 rounded upward to 5 m).
- Robust calibration recall/miss: 99.002510%/0.997490%.
- All 80 training trips were still processed. Actual reference miss: 21,684/208,417 = 10.404142%; no-candidate: 15,286/208,417 = 7.334335%.
- Student-test reference miss: 21/19,160 = 0.109603%; no-candidate: 0.
- Generated rows: 2,817,917 training and 324,291 student-test. No nearest-arc fallback hides misses.
- Radius comparison: `outputs/csv/candidate_radius_recall.csv`; coverage: `outputs/csv/candidate_coverage.csv`.

## 6. Features - completed and executed

One Parquet row is one trip-local observation/candidate pair. Target is 1 only when the candidate equals the monotonic reference assignment.

| Feature | Formula | Unit |
|---|---|---|
| Point-arc distance | Shapely point-to-line distance | m |
| From/to-node distance | projected Euclidean norm | m |
| Projection position | located distance / arc length, clipped [0,1] | ratio |
| Vehicle heading | `atan2(delta_easting, delta_northing)` from previous point | degrees |
| Arc bearing | same formula on arc endpoints | degrees |
| Heading difference | `abs((vehicle-arc+180) mod 360 - 180)` | degrees |
| Reverse indicator | heading difference >90 | binary |
| Speed | previous-to-current projected displacement / positive time gap | m/s |
| Time gap | current timestamp - previous timestamp | s |
| Arc length | projected endpoint-line length | m |
| Rank/count | distance rank and candidates in observation | ordinal/count |
| Nearest alternative | second distance for rank 1; nearest otherwise | m |

First-point heading/speed/gap and unavailable alternative distances remain NaN; model-fold median imputers handle them. All dynamics are backward-looking; future observations are not features.

## 7. Initial models - completed and executed baseline

Training used a deterministic balanced 150,000-row sample from generated training candidates, seed 42, with 3-fold `GroupKFold` by trip. Student-test rows were never fitted or used for imputers/scalers.

| Model | CV candidate Macro F1 | Student candidate Macro F1 | Log loss | Student Top-1 |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0.936107 | 0.844368 | 0.116687 | 0.822338 |
| Random Forest | 0.938742 | 0.843527 | 0.120731 | 0.822129 |
| Extra Trees | 0.937657 | 0.831693 | 0.133511 | 0.816962 |
| MLP | 0.933995 | 0.820820 | 0.136118 | 0.809969 |
| Logistic Regression | 0.924291 | 0.751118 | 0.183448 | 0.796921 |

All five expose probabilities compatible with emission scoring. These are initial results, not a final model-selection claim. Full configurations/timings: `outputs/csv/initial_model_results.csv`; artifacts: `data/processed/models/`.

Optuna/TPE Bayesian optimization is implemented in `scripts/optimize_model.py` but **not executed**. No tuned-model result is claimed.

## 8. HMM/Viterbi - completed and executed baseline

- Hidden states are candidate arcs at each observation.
- Emissions are log classifier probabilities from the initial HistGradientBoosting artifact.
- Transition cost: 0 for the same arc, 0.25 for directed adjacency, 1 for a shared node, otherwise 6 plus endpoint distance/50 m.
- Missing candidate spans and timestamp gaps >60 s reset the chain rather than invent states.
- Weighted student-test classifier Top-1: 0.822338; Viterbi Top-1: 0.889666.
- Weighted route continuity: 0.902715 classifier-only; 0.995300 Viterbi.
- Viterbi Top-1 improved on all 10 student-test trips. Parameters are not yet tuned.
- Per-trip evidence: `outputs/csv/hmm_baseline_results.csv`.

## 9. Tests and output inspection

- Unit tests: **17 passed, 0 failed** (`pytest -q`).
- Whole `src`, `scripts`, and `tests` compile check: passed.
- Required plot montage was visually inspected: no clipping or unreadable labels found.
- Folium HTML is 3,208,596 bytes and contains blue GPS, green reference route, neutral road network, named layers, fit bounds, and layer control.

Visualisations:

- `outputs/plots/gps_trajectory_00000001.png`
- `outputs/plots/timestamp_gap_distribution.png`
- `outputs/plots/node_arc_distribution.png`
- `outputs/plots/network_connectivity.png`
- `outputs/plots/candidate_distance_distribution.png`
- `outputs/plots/candidate_count_distribution.png`
- `outputs/plots/feature_distributions.png`
- `outputs/plots/model_cv_macro_f1.png`
- `outputs/plots/model_student_top1.png`
- `outputs/maps/trip_00000001_reference_map.html`

## 10. Files created or modified

- Root: `.gitignore`, `README.md`, `pyproject.toml`, `requirements.txt`.
- Package: `src/pathsnap/{__init__,io,geo,graph,candidates,labels,features,visualize,metrics,models,hmm}.py`.
- Scripts: `scripts/{run_eda,run_preprocessing,build_features,run_experiments,run_hmm_baseline,optimize_model}.py`.
- Tests: `tests/conftest.py`, `tests/test_io.py`, `tests/test_geo_graph.py`, `tests/test_candidates_labels_features.py`, `tests/test_hmm.py`.
- Notebooks: `notebooks/01_eda.ipynb` through `04_feature_engineering.ipynb`.
- Audits: `DATASET_AUDIT.md`, `PREPROCESSING_AUDIT.md`, `FEATURE_ENGINEERING_AUDIT.md`, `PHASE1_STATUS.md`, this file.
- Generated/cache outputs: 90 label Parquets, 90 feature Parquets, five model joblibs, one ranking JSON, CSVs/plots/map/logs listed above. Generated data is Git-ignored.

## 11. Commands actually executed

```bash
git clone https://github.com/2306apk/G14-PathSnap.git G14-PathSnap
brew install poppler
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy pandas pyproj shapely networkx matplotlib seaborn folium scikit-learn scipy joblib jupyter pytest optuna pyarrow
.venv/bin/pip install -e .
.venv/bin/python scripts/run_eda.py --data-root '../Student folder'
.venv/bin/python scripts/run_preprocessing.py --data-root '../Student folder'
.venv/bin/python scripts/build_features.py --data-root '../Student folder'
.venv/bin/python scripts/run_experiments.py --max-train-rows 150000 --cv-folds 3 --seed 42
.venv/bin/python scripts/run_hmm_baseline.py --data-root '../Student folder' --model data/processed/models/hist_gradient_boosting.joblib
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/pytest -q
```

One first preprocessing invocation failed immediately with an unmatched parenthesis before data work. It was fixed; compile check and the successful full run followed. No failed experiment result is presented as valid.

## 12. Remaining bugs, blockers, and readiness

- **Known data limitation:** two training reference routes are grossly inconsistent with their GPS. Their labels/misses remain reported; they are excluded only from radius calibration.
- **Partial:** initial models use a balanced 150k-row training sample, not all 2.82m rows.
- **Pending:** Bayesian optimization, final model selection, tuned decoder parameters, final predicted-route map, API/UI/deployment.
- **No blocker for Phase 1 review.** Data pipeline, EDA, preprocessing, candidate generation, labels, features, visualisations, five initial models, HMM/Viterbi baseline, and tests are implemented and executed.
- **Not ready for final competition submission** until pending model optimization/selection and later deployment phases are executed.
