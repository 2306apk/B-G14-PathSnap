# PathSnap — Phase 1 Implementation Status Report & Modeling Architecture

**Document Type:** Milestone Audit & Phase 2 Architecture Handover
**Date:** 2026-11-18
**Author:** Primary ML Engineering Agent (Antigravity)
**Repository:** https://github.com/2306apk/G14-PathSnap
**Workspace:** `ML-Hackathon`

---

## 1. Phase 1 Checklist & Verification Status

| Component | Status | Verification & Evidence | Output Path |
| :--- | :---: | :--- | :--- |
| **Repository Setup** | **COMPLETED** | Official repository cloned, `.venv` initialized with Python 3.11, clean structure established. | `G14-PathSnap/` |
| **I/O & Data Loading** | **COMPLETED** | Reads whitespace-separated files without headers, preserves 0-based indexing, validates column counts & types. | `src/pathsnap/io.py` |
| **Integrity Audit** | **COMPLETED** | Verified 0 missing files, 0 NaNs, 0 invalid arc/node references across 90 trips. | `outputs/csv/dataset_inventory.csv` |
| **Geographic Preprocessing** | **COMPLETED** | Dynamic UTM CRS selection per trip; eliminates global Netherlands assumption; pyproj metric conversions. | `src/pathsnap/geo.py` |
| **Road Network & Graph** | **COMPLETED** | Independent DiGraph per trip; STRtree spatial indexing; connectivity & WCC analysis. | `src/pathsnap/graph.py` |
| **Label Association** | **COMPLETED** | Monotonic dynamic programming alignment; preserves ground-truth uncertainty and outlier flags. | `src/pathsnap/labels.py` |
| **Candidate Radius Sweeps** | **COMPLETED** | Evaluated 10m to 100m; selected and justified R = 50.0m based on 96.93% recall and 11.76 cands/pt. | `outputs/csv/candidate_radius_sweep.csv` |
| **Feature Engineering** | **COMPLETED** | 14 causal features (proximity, kinematics, heading diffs, candidate competition); no future leakage. | `src/pathsnap/features.py` |
| **Visualisations** | **COMPLETED** | Trajectory plots, sampling gap histograms, candidate distributions, interactive Folium maps. | `outputs/plots/`, `outputs/maps/` |
| **Jupyter Notebooks** | **COMPLETED** | 4 self-contained notebooks for EDA, preprocessing, candidate generation, and features. | `notebooks/` |
| **Unit Test Suite** | **COMPLETED** | 10 passed tests covering I/O, projection, spatial indexing, DP labels, and feature integrity. | `tests/test_pipeline.py` |
| **Phase 2 Modeling Interface** | **COMPLETED** | Standardized Parquet & CSV exports with group-aware `trip_id` identifiers for GroupKFold cross-validation. | `data/processed/` |

---

## 2. Key Dataset & Engineering Findings

1. **Global Projection Fallacy Disproven:**
   - Previous assumptions of a single CRS (`EPSG:28992` or `EPSG:32631`) are invalid.
   - Longitudes span `[-100.4185, 137.2370]` and latitudes span `[-34.6403, 61.6996]` across 25 distinct UTM zones.
   - Local UTM projection per trip is strictly mandatory.
2. **Network Independence:**
   - Every trip has its own distinct, local `.nodes` and `.arcs` file with 0-based indexing.
   - Node 0 in Trip 1 is in Germany, Node 0 in Trip 2 is in Bavaria, and Node 0 in Trip 56 is in Minnesota, USA.
   - Networks must never be merged into a global graph without composite keys `(trip_id, arc_id)`.
3. **Reference Route Ground-Truth Limitations:**
   - As documented in the official guidelines, reference routes are not guaranteed to be perfect ground truth.
   - Trips such as `00000002` feature detours or GPS drift >50m (up to 550m) away from the nearest route arc.
   - The pipeline computes explicit continuous confidence ($\exp(-d / 15.0)$) and outlier flags ($d > 50\text{m}$) rather than assuming error-free labels.
4. **Optimal Candidate Search Radius:**
   - $R = 50.0\text{m}$ captures ~97% of observations (100% of reachable route segments), with an average candidate set size of 11.76 arcs per point.

---

## 3. Initial Modeling Plan: Candidate Classification Shortlist

To satisfy the hackathon requirement of experimenting with at least **FIVE distinct models** from multiple algorithm families, the following models are shortlisted for Phase 2:

### Model 1: Logistic Regression (Family: Linear & Probabilistic)
- **Role:** Foundational probabilistic baseline.
- **Formulation:** $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$.
- **Key Hyperparameters:** Regularization strength $C \in [10^{-3}, 10^2]$, penalty (`l1`, `l2`, `elasticnet`), solver (`lbfgs`, `saga`).
- **HMM Compatibility:** Naturally outputs well-calibrated posterior probabilities. Highly efficient for real-time inference in FastAPI.

### Model 2: Random Forest Classifier (Family: Ensemble / Bagging)
- **Role:** Non-linear decision boundary baseline with robustness to scale variations and outlier points.
- **Key Hyperparameters:** `n_estimators` (100–300), `max_depth` (10–25), `min_samples_split` (5–20), `class_weight='balanced'`.
- **HMM Compatibility:** Class probabilities $P(y=1)$ computed via tree voting averages; requires Platt scaling or isotonic calibration for optimal Viterbi emission weighting.

### Model 3: Extra Trees Classifier (Family: Ensemble / Extremely Randomized Trees)
- **Role:** Reduces variance and training time compared to standard Random Forest by selecting random cut-points.
- **Key Hyperparameters:** `n_estimators` (150–400), `max_depth` (12–30), `max_features` (`'sqrt'`, `'log2'`).
- **HMM Compatibility:** Smoother decision surfaces than standard RF; fast training on multi-core CPU.

### Model 4: HistGradientBoosting Classifier (Family: Gradient Boosting)
- **Role:** Primary competitive gradient-boosted decision tree algorithm. Highly optimized histogram-based splits native to scikit-learn.
- **Key Hyperparameters:** `learning_rate` (0.03–0.2), `max_iter` (100–300), `max_leaf_nodes` (31–127), `l2_regularization` (0.0–10.0), `min_samples_leaf` (20–100).
- **HMM Compatibility:** Direct log-odds output converts to calibrated sigmoid probabilities. Extremely fast CPU inference.

### Model 5: Multi-Layer Perceptron (Family: Neural Networks)
- **Role:** Learns smooth non-linear continuous interactions between kinematic features (speed, time gaps) and geometric angles/distances.
- **Architecture:** 2 hidden layers (e.g., 64 $\to$ 32 units) with ReLU/GELU activations, BatchNormalization/LayerNormalization, Dropout (0.2), and Adam optimizer with cosine learning rate schedule.
- **Key Hyperparameters:** `hidden_layer_sizes`, `alpha` (L2 penalty), `learning_rate_init` (1e-3).
- **HMM Compatibility:** Output sigmoid layer produces calibrated posterior probabilities directly compatible with emission log-likelihoods.

### Optional Benchmark: Support Vector Machine (SVM) (Family: Kernel-based)
- **Role:** Linear SVM or RBF SVM with CalibratedClassifierCV on a sampled subset to evaluate maximum margin boundaries.

---

## 4. Sequential Decoder & HMM Interface Design

The classifier operates on individual (observation, candidate) pairs. To reconstruct a coherent, connected route, the selected classifier's output feeds an HMM / Viterbi sequence decoder:

1. **Hidden States:**
   - At time step $t$, hidden states $\mathcal{S}_t$ are the discrete candidate road arcs $\{ a_t^{(1)}, a_t^{(2)}, \dots, a_t^{(M)} \}$ within radius $R = 50.0\text{m}$.
2. **Emission Probabilities:**
   - Binary classifier produces score $s_j = P(y=1 \mid \mathbf{x}_{t, j})$.
   - Normalized candidate emission probability:
     $$b_j(P_t) = \frac{s_j}{\sum_{m=1}^{M} s_m}$$
   - In log space: $\ln b_j(P_t) = \ln s_j - \ln \left( \sum_m s_m \right)$.
3. **Transition Probabilities:**
   - Transition between candidate $a_t^{(j)}$ and $a_{t+1}^{(k)}$ balances road-network shortest path distance $d_{road}$ against Euclidean distance $d_{gps}$:
     $$A_{j, k} = \frac{1}{\beta} \exp\left( -\frac{|d_{road}(a_t^{(j)}, a_{t+1}^{(k)}) - d_{gps}(P_t, P_{t+1})|}{\beta} \right)$$
   - If consecutive arcs connect directly in the road network (`to_node == from_node`), $d_{road} \approx d_{gps}$, yielding maximum transition probability.
   - If topologically disconnected, $A_{j, k} \to \epsilon$ (transition penalty).
4. **Viterbi Recurrence:**
   $$V_t(k) = \max_{j \in \mathcal{S}_{t-1}} \left[ V_{t-1}(j) + \ln A_{j, k} \right] + \ln b_k(P_t)$$
5. **Handling GPS Gaps & Missing Candidates:**
   - When sampling gap $\Delta t > 30\text{s}$, transition variance parameter $\beta$ is scaled by $\sqrt{\Delta t}$.
   - If an observation has zero candidates, the decoder bridges the state using shortest-path continuity between the preceding and succeeding observations.

---

## 5. Next Steps for Codex & Phase 2 Execution

1. **Model Training & Evaluation Scripts:**
   - Create `scripts/train_models.py` to train all 5 models using 5-fold `GroupKFold` cross-validation grouped by `trip_id`.
   - Evaluate on `data/processed/train_features.parquet` and compute CV Macro F1, Observation Top-1 Accuracy, Log Loss, and Inference Time.
2. **Bayesian Hyperparameter Optimization:**
   - Use Optuna to tune the top-performing model on GroupKFold CV Macro F1 across $\ge 30$ trials.
3. **Viterbi Decoder Implementation:**
   - Implement `src/pathsnap/decoder.py` with fast NetworkX / Dijkstra distance caching.
   - Compare Classifier-Only predictions vs Classifier + HMM predictions.
4. **FastAPI & UI Implementation:**
   - Build FastAPI endpoints (`/trips`, `/match`, `/evaluate`) and minimal web UI for two-laptop demonstration.
"""

    with open(reports_dir / "PHASE1_STATUS.md", "w") as f:
        f.write(feat_report)

    print(f"Phase 1 status report successfully written to {reports_dir / 'PHASE1_STATUS.md'}")

