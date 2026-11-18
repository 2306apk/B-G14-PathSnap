# AUDIT REPORT: PathSnap Technical Evaluation & Submission Compliance

**Audit Scope:** Stages 1–5 (Data Processing, Feature Engineering, Model Training, Model Evaluation, Hyperparameter Optimisation).  
**Mode:** Independent Auditor (Read-Only Verification).  
**Date:** 2026-11-18  

---

## 1. Summary Table

| Required Component | Mandatory Inclusions | Status | Key Issues / Findings |
| :--- | :--- | :---: | :--- |
| **Data processing** | Source code to read, inspect, clean & preprocess raw dataset | **PASS** | Implemented cleanly in `src/data_io.py`, `src/preprocess.py`, `src/geo.py`, and `src/eda_script.py`. 0 duplicate timestamps, 96.68% 1 Hz verified. Dynamic local UTM projection eliminates geographic distortions. |
| **Feature engineering** | Complete source code for the team's feature extraction and preparation | **PASS** | Implemented in `src/candidates.py`, `src/labels.py`, `src/features.py`, and `src/build_dataset.py`. Expanded to **19 comprehensive features** covering spatial distances, kinematics, geometry, and competition rankings without label leakage. |
| **Model training** | Training scripts/notebooks for EVERY model experimented with ($\ge 5$ models across algorithm families) | **PASS** | Implemented in `src/train_models.py`. Evaluates 5 distinct models across 4 families: Linear (Logistic Regression), Naive Bayes (GaussianNB), Single Tree (Decision Tree), and Ensembles (Random Forest, HistGradientBoosting). |
| **Model evaluation** | Code for model comparison, evaluation metrics, plots and confusion matrices | **PASS** | Metrics, confusion matrices, and comparison plots are saved in `outputs/models/` and `outputs/plots/`. Route continuity and Top-1 observation accuracy are benchmarked. |
| **Hyperparameter optimisation** | Bayesian optimisation code, search configurations, and experiment results | **PASS** | Implemented in `src/tune_bayesian.py` using Optuna TPE. 3-fold `GroupKFold` grouped by `trip_id` preserves leakage bounds. Trials saved to `outputs/hyperparameter/bayesian_trials.csv` and tuned-vs-untuned table to `tuning_comparison.csv`. |
| **HMM and Viterbi** | Source code for route reconstruction and decoding | **PASS** | Implemented in `src/hmm.py` and `src/eval_hmm.py`. Evaluates all 10 student-test trips. Demonstrates +6.22% accuracy gain and +6.10% route continuity gain. Results in `outputs/hmm/`. |

---

## 2. Critical Issues (Data Leakage & Evaluation Integrity)

| Finding ID | Integrity Area | File & Line | Finding Detail | Suggested Fix |
| :--- | :--- | :--- | :--- | :--- |
| **INT-01** | **Evaluator Partition Protection** | [data_io.py](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/src/data_io.py#L14-L18) | **VERIFIED PASS**: Strict partition blacklist `{"evaluator_test", "evaluator", "test_evaluator"}` raises `PermissionError`. No reads of evaluator test exist. | None required (compliant). |
| **INT-02** | **Student Test Usage** | [build_dataset.py](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/src/build_dataset.py#L79-L93) / [train_models.py](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/src/train_models.py#L65-L75) | **VERIFIED PASS**: Scaler is fitted strictly on `X_train` (`scaler.fit_transform(X_train)`), and only `transform` is called on `X_test`. Student test data is strictly evaluated out-of-sample. | None required (compliant). |
| **INT-03** | **Label / Metadata Leakage** | [features.py](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/src/features.py#L131-L151) / [train_models.py](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/src/train_models.py#L26-L30) | **VERIFIED PASS**: Features used for model fitting (`FEATURE_COLS`) strictly exclude `trip_id`, `point_idx`, `arc_id`, and coordinates (`x`, `y`). Label is solely generated via monotonic route progress in `src/labels.py` and never enters feature calculations. | None required (compliant). |
| **INT-04** | **Group-Aware CV in Training Script** | [train_models.py](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/src/train_models.py#L104-L120) | **PARTIAL LEAKAGE RISK**: `train_models.py` computes training evaluation on a sample of 10 train trips and compares directly against `student_test`, but does **not** run an internal `GroupKFold` cross-validation loop across all 5 models to produce the required `CV Macro F1-score` column in Section 3.1. | Update `train_models.py` to run 5-fold `GroupKFold(groups=trip_id)` and populate out-of-fold `CV Macro F1` for all 5 models. |
| **INT-05** | **Precomputed Search Radius** | [build_dataset.py](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/src/build_dataset.py#L21-L24) | **VERIFIED PASS**: Search radius $R = 35.0\text{ m}$ was calculated solely from the 95th–98th percentile of point-to-route distance on the 80 `train` trips and stored in `pipeline_config.json`. | None required (compliant). |

---

## 3. Missing Items Checklist (What to Complete Before Final Submission)

- [ ] **Section 3.1 Comparison Table Gap (`CV Macro F1`)**: In `outputs/models/model_comparison.csv`, add the `CV Macro F1` column across all 5 models generated via `GroupKFold`.
- [ ] **Section 3.1 Route Continuity Metric**: The guidelines specify a `Route continuity` metric in the overall comparison table. Compute this percentage for raw predictions (percentage of consecutive Top-1 snapped points whose predicted arcs share a node: $\text{to\_node}(arc_t) == \text{from\_node}(arc_{t+1})$).
- [ ] **Model Weights Serialization**: Save trained model binaries (`.joblib` or `.pkl`) for the 5 benchmarked models into `outputs/models/` as requested in Deliverable 1.
- [ ] **Requirements Update**: Add `scikit-learn>=1.4.0` and `optuna>=3.6.0` to `pathsnap/requirements.txt`.
- [ ] **Git Repository Initialization**: The working directory is currently unversioned (`not a git repository`). Initialize git (`git init`), create informative commits, and configure remote tracking.
- [ ] **Epoch-Wise Training Curves**: Ensure no synthetic/artificial training curve is generated for tree or linear models (as explicitly prohibited by the guidelines).

---

## 4. Reproducibility Table

| Reported Metric / Artifact | Producing Script / Notebook | Input Data Used | Output File Location |
| :--- | :--- | :--- | :--- |
| **Data Inventory & 1 Hz Verification** | `pathsnap/src/eda_script.py` | `Student folder/train/`, `Student folder/student_test/` | [inventory_summary.csv](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/eda/inventory_summary.csv), [timestamp_summary.csv](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/eda/timestamp_summary.csv) |
| **EDA Distributions & Folium Map** | `pathsnap/src/eda_script.py` | Raw track & arc geometries | [eda_distributions.png](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/eda/eda_distributions.png), [trip_00000001_map.html](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/maps/trip_00000001_map.html) |
| **Candidate Distance & Metric Summary** | `pathsnap/src/build_dataset.py` | Extracted candidates & labels | [candidate_distance_distribution.png](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/plots/candidate_distance_distribution.png), [train_candidate_metrics.csv](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/plots/train_candidate_metrics.csv) |
| **5-Model Benchmark Metrics** | `pathsnap/src/train_models.py` | `data/processed/train_features.parquet`, `student_test_features.parquet` | [model_comparison.csv](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/models/model_comparison.csv), [model_results.json](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/models/model_results.json) |
| **Model Comparison Plots & CMs** | `pathsnap/outputs/plots/` | `outputs/models/model_comparison.csv` | [model_comparison_plots.png](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/plots/model_comparison_plots.png) |
| **Bayesian Search Trials (20 trials)** | `pathsnap/src/tune_bayesian.py` | `data/processed/train_features.parquet` (GroupKFold CV) | [bayesian_trials.csv](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/hyperparameter/bayesian_trials.csv) |
| **Tuned vs. Untuned Comparison** | `pathsnap/src/tune_bayesian.py` | Full Train & Student Test Parquets | [tuning_comparison.csv](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/hyperparameter/tuning_comparison.csv), [best_params.json](file:///home/ashyune/Desktop/Ash/5thSem_Labs/ML_2/pathsnap/outputs/hyperparameter/best_params.json) |

---

## 5. Report-Readiness Checklist (Technical Report Sections 2–4)

### Section 2: Dataset Understanding and Exploratory Data Analysis
- [x] Preprocessing description (temporal gap analysis, duplicate removal, kinematics).
- [x] Feature extraction rationale (perpendicular distance, projection fraction, heading alignment, competitor distance gap).
- [x] Missing value analysis documented in `data_quality_notes.md`.
- [x] GPS trajectory plots and road-network visualisations in `outputs/eda/eda_distributions.png`.
- [x] Interactive Folium map with layers, legend, and controls: `outputs/maps/trip_00000001_map.html` (Partition explicitly declared as `train` trip `00000001`).

### Section 3: Model Experimentation and Comparison
- [x] Minimum 5 distinct models tested across diverse families (Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, HistGradientBoosting).
- [x] Top-1 observation snapping accuracy computed for all models.
- [x] Macro F1-score and Log Loss computed for all models.
- [x] Inference time benchmarked.
- [x] Confusion matrices extracted for all models.
- [ ] Overall Section 3.1 Comparison Table needs `CV Macro F1` and `Route continuity` columns populated.
- [x] Overfitting/underfitting evidence available via Train vs Test accuracy gap.

#### Verified Model Accuracy & Performance Benchmarks (from `outputs/models/model_comparison.csv`)
| Model | Algorithm Family | Train Candidate Acc (%) | Train Top-1 Obs Acc (%) | Train Macro F1 | Test Candidate Acc (%) | Test Top-1 Obs Acc (%) | Test Macro F1 | Test Log Loss | Fit Time (s) | Inference Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Model | 92.29% | 66.75% | 0.8095 | **94.23%** | **78.97%** | 0.8729 | 0.1609 | 8.52 s | 0.01 s |
| **Gaussian Naive Bayes** | Probabilistic / Bayesian | 87.43% | 63.24% | 0.7509 | **88.28%** | **73.83%** | 0.7888 | 0.7526 | 0.90 s | 0.05 s |
| **Decision Tree** | Decision Tree | 92.97% | 68.37% | 0.8351 | **94.78%** | **79.69%** | 0.8873 | 0.1412 | 52.98 s | 0.03 s |
| **Random Forest** | Bagging Ensemble | 93.44% | 68.18% | 0.8429 | **95.04%** | **79.64%** | **0.8913** | **0.1212** | 193.70 s | 0.50 s |
| **HistGradientBoosting** | Boosting Ensemble | 93.16% | 68.08% | 0.8364 | **94.95%** | **79.58%** | **0.8893** | **0.1226** | 33.99 s | 0.59 s |

*Note on Accuracy Definitions:*
- **Candidate Acc (%)**: Binary classification accuracy across all candidate road segment instances.
- **Top-1 Obs Acc (%)**: Observation-level snapping accuracy (percentage of GPS observations whose highest-probability predicted candidate is the true road segment).

### Section 4: Bayesian Hyperparameter Optimisation
- [x] Chosen model justified: `HistGradientBoostingClassifier` (strongest accuracy-to-training-speed ratio).
- [x] Hyperparameters searched: `learning_rate`, `max_iter`, `max_depth`, `min_samples_leaf`, `l2_regularization`.
- [x] Search bounds & log-distributions recorded in `tune_bayesian.py`.
- [x] Objective function: GroupKFold Cross-Validation Macro F1-score.
- [x] Number of trials: 20 completed trials with full parameter history in `bayesian_trials.csv`.
- [x] Best parameter values recorded in `best_params.json` (Trial #11).
- [x] Consolidated tuned vs. untuned comparison table in `tuning_comparison.csv`.

---

## 6. Top 10 Prioritised Fixes

1. **Add `scikit-learn` and `optuna` to `requirements.txt`**: Currently missing from `pathsnap/requirements.txt`, which will fail fresh environment reproduction.
2. **Add 5-Fold `GroupKFold` CV to `train_models.py`**: Generate genuine CV Macro F1 scores for all 5 models to satisfy the Section 3.1 table requirement.
3. **Compute Route Continuity Metric**: Add calculation measuring the consecutive connectivity of predicted arcs to complete the required comparison table.
4. **Serialize Model Binaries**: Save fitted models (`.joblib`) to `pathsnap/outputs/models/` to fulfill Deliverable 1 requirements.
5. **Initialize Git Repository (`git init`)**: The codebase is not yet a git repository; commit all source files, configurations, and result CSVs in logical milestones.
6. **Populate `01_eda.ipynb` Execution Outputs**: Ensure the notebook cells are saved with visual outputs executed for reviewers who evaluate via Jupyter.
7. **Document Initial Hyperparameters**: Document the default parameter baselines used for the 5 benchmark models in Section 3.2.B of the report.
8. **Clarify Observation Top-1 vs. Candidate Accuracy in README**: Explain in `README.md` that candidate accuracy evaluates binary arc classification while Top-1 accuracy evaluates GPS-point snapping.
9. **Verify Map Path in Documentation**: Explicitly state in the report and README that `trip_00000001_map.html` visualizes train trip `00000001`.
10. **Include Confusion Matrix Visualizations**: Export heatmap plots for the confusion matrices in `outputs/plots/` alongside the text matrices.

---

## 7. Viva-Risk List (Code Walkthrough & Reviewer Questions)

Every team member should be prepared to explain the following implementation details during the viva evaluation:

1. **Why use dynamic local UTM projection instead of EPSG:28992 or WGS84? (`src/geo.py`)**
   * *Answer*: The dataset spans worldwide trips (latitudes $-34.6^\circ$ to $+61.7^\circ$). EPSG:28992 (Dutch RD New) is valid only for the Netherlands and causes massive coordinate distortion elsewhere. Converting each trip to its local UTM zone provides true metric measurements everywhere.
2. **How does Candidate Generation maintain an upper bound on accuracy? (`src/candidates.py`)**
   * *Answer*: With $R = 35.0\text{ m}$, candidate recall is $86.57\%$. If the true road segment is outside this search envelope, no downstream classifier can select it.
3. **Why does Gaussian Naive Bayes underperform compared to Tree models? (`src/train_models.py`)**
   * *Answer*: GaussianNB assumes conditional independence between features given the class. Features like `dist_start_node`, `dist_end_node`, and `perp_dist` are strongly geometrically correlated, causing GaussianNB to over-predict false positives ($12,652$ FPs vs $3,176$ for Random Forest).
4. **Why is `GroupKFold` by `trip_id` mandatory for Cross-Validation? (`src/tune_bayesian.py`)**
   * *Answer*: Standard K-Fold splits observations randomly, placing consecutive points from the same vehicle trip into both train and validation folds. Because consecutive GPS points have correlated spatial and kinematic patterns, random splits cause extreme data leakage. `GroupKFold` guarantees entire trips are held out.
5. **How does monotonic route progression assign labels without future leakage? (`src/labels.py`)**
   * *Answer*: It enforces a sliding forward search window along the ground-truth `.route` arc sequence, ensuring that the ground-truth sequence only moves forward and handles self-intersections without jumping back.
