# PathSnap Engineering Report: Methodological Decisions, Justifications, and Pipeline Execution

This document provides a comprehensive breakdown of **what was done, why each decision was made, and how it was implemented** across the entire PathSnap GPS map-matching pipeline. Use this document as your primary reference for the technical report (Sections 1–5) and reviewer evaluations.

---

## 1. Data Integrity & Partitioning

### What Was Done
* Built an academic-integrity loader in `src/data_io.py` that strictly manages raw dataset files (`.track`, `.nodes`, `.arcs`, `.route`).
* Enforced an immediate exception trigger for forbidden partitions:
  ```python
  FORBIDDEN_PARTITIONS = {"evaluator_test", "evaluator", "test_evaluator"}
  ```
* Enforced strict partition boundaries: all processing, candidate search, and feature extraction operate trip-by-trip (`trip_id`).

### Why It Was Done (Justification)
* **Prevent Evaluation Disqualification**: Evaluation rules strictly forbid loading, inspecting, or training on evaluator test sets. Raising a `PermissionError` guarantees that evaluator data is never accessed accidentally.
* **Prevent Information Leakage**: Map-matching data is sequential. If points from the same trip are mixed across training and validation sets, consecutive GPS pings will leak trajectory information. Grouping strictly by `trip_id` preserves legitimate validation bounds.

---

## 2. Dynamic Metric Coordinate Reference System (CRS)

### What Was Done
* Implemented dynamic local UTM zone projection in `src/geo.py` using `pyproj`.
* For each individual trip, the system determines the optimal UTM zone from the mean longitude and latitude of the trajectory and projects all GPS points and road network nodes from WGS84 degrees into projected metres $(x, y)$.

### Why It Was Done (Justification)
* **Initial Assumption Disproven**: The task materials initially hinted that data looks like the Netherlands (which typically uses Dutch RD New / EPSG:28992).
* **Empirical Discovery**: During our Exploratory Data Analysis (EDA), we analyzed the bounding coordinates and found that trips span worldwide coordinates:
  * Longitude: $[-100.42^\circ, +137.24^\circ]$
  * Latitude: $[-34.64^\circ, +61.70^\circ]$
* **Consequence of Flat Euclidean Degrees**: Calculating distances directly on latitude/longitude degrees ($\sqrt{\Delta \text{lat}^2 + \Delta \text{lon}^2}$) creates severe spatial distortion because $1^\circ$ of longitude shrinks rapidly as latitude moves away from the equator ($\cos(\text{lat})$ scaling).
* **Consequence of a Single Regional CRS**: EPSG:28992 would completely distort trips located in North America or Asia. Dynamic local UTM guarantees that 1 unit equals exactly 1 metre everywhere on Earth without regional distortion.

---

## 3. Spatial STRtree Indexing & Candidate Generation

### What Was Done
* Implemented 2D spatial candidate generation using Shapely's `STRtree` (Sort-Tile-Recursive R-tree) in `src/candidates.py`.
* Fitted a fixed candidate search radius of $R = 35.0\text{ metres}$ determined from the training partition.
* Extracted candidate road segments (`arc_id`) within $R = 35.0\text{ m}$ of each GPS point.

### Why It Was Done (Justification)
* **Algorithmic Complexity ($O(N \cdot M)$ vs $O(N \log M)$)**: Each trip contains thousands of GPS points and road networks with up to $88,809$ arcs. A naive distance check compares every point against every arc, requiring billions of operations and taking hours per trip. The `STRtree` spatial index evaluates bounding box overlaps in logarithmic time ($O(\log M)$), completing candidate queries in seconds.
* **Why $R = 35.0\text{ metres}$?**:
  * If $R$ is too small (e.g. $15\text{ m}$), GPS multipath noise causes the true road segment to be omitted entirely (candidate miss). Once missed, no downstream machine learning model can ever select it.
  * If $R$ is too large (e.g. $70\text{ m}$), the number of distractor candidates explodes, drastically increasing false positives and computational overhead.
  * Analysis on training trips showed that $R = 35\text{ m}$ captures **$86.57\%$ candidate recall / hit rate** while keeping candidate ambiguity manageable (average of only $7.96$ candidates per point).

---

## 4. Ground-Truth Reference Labeling via Monotonic Progression

### What Was Done
* Implemented monotonic ground-truth reference arc assignment in `src/labels.py` (`assign_reference_arcs_monotonic`).
* For each GPS point, the algorithm restricts candidate reference matching to a forward-sliding window $[curr\_pos - 1, curr\_pos + 20]$ along the true `.route` sequence.

### Why It Was Done (Justification)
* **Handling Complex Intersections and Loops**: Real vehicle trajectories frequently pass near highway cloverleafs, overpasses, or return along the same avenue later in the trip.
* A naive nearest-road assignment causes the labeler to snap onto an overlapping future or past segment of the route.
* Enforcing monotonic forward progress along the sequence guarantees that each GPS point is mapped to the physically traversed segment corresponding to that specific point in time.

---

## 5. Feature Engineering: Multi-Aspect Representation

### What Was Done
* Expanded feature engineering in `src/features.py` to **19 comprehensive discriminative features** (satisfying the 18–20 features requirement) for every `(trip_id, point_idx, candidate_arc_id)` tuple across 4 structural dimensions:

#### A. Geometric & Spatial Distance (6 features)
1. `perp_dist`: Perpendicular distance to the candidate road segment (m).
2. `dist_start_node`: Distance to the start node of the road arc (m).
3. `dist_end_node`: Distance to the end node of the road arc (m).
4. `proj_frac`: Fractional projection $[0.0 \dots 1.0]$ along the arc length.
5. `norm_perp_dist`: Normalized perpendicular distance relative to the search radius (`perp_dist / 35.0`).
6. `dist_to_segment_midpoint`: Euclidean distance from GPS point to the midpoint of the road arc.

#### B. Kinematic & Heading Alignment (6 features)
7. `bearing_diff`: Angular divergence $[0^\circ \dots 180^\circ]$ between vehicle velocity vector and road orientation.
8. `is_reverse_direction`: Binary flag ($1$ if `bearing_diff` $> 90^\circ$).
9. `cos_bearing_diff`: Continuous cosine alignment metric $\cos(\Delta \theta) \in [-1.0, 1.0]$.
10. `speed`: Instantaneous vehicle speed in m/s ($v = \Delta d / \Delta t$).
11. `speed_ratio_arc`: Ratio of expected distance traveled to candidate arc length (`speed * dt / arc_length`).
12. `dt`: Time elapsed since the previous ping (s).

#### C. Road Network & Topology Geometry (3 features)
13. `arc_length`: Physical length of the candidate road arc in metres.
14. `log_arc_length`: Log-transformed arc length $\log(1 + \text{arc\_length})$ to prevent skew from extreme highways.
15. `arc_straightness`: Ratio of Euclidean chord length to curvilinear arc length (curvature metric $\in (0, 1]$).

#### D. Candidate Competition & Ranking Context (4 features)
16. `candidate_rank`: Distance rank among candidates ($1 = \text{nearest}$).
17. `num_candidates`: Number of candidates within $R = 35\text{ m}$ (local intersection complexity).
18. `dist_gap_to_next`: Distance margin between this candidate and the nearest competitor.
19. `dist_ratio_to_best`: Ratio of this candidate's distance relative to the closest candidate (`dist / min_dist`).

### Why It Was Done (Justification)
* **Resolving Parallel One-Way Roads**: Distance alone (`perp_dist`) cannot distinguish between two parallel one-way streets 10 metres apart. `bearing_diff` and `is_reverse_direction` instantly eliminate oncoming lanes.
* **Resolving Ambiguity at Intersections**: `dist_gap_to_next` provides an explicit margin of certainty. When `dist_gap_to_next` is large, the nearest candidate is unambiguous; when it is near zero, the model relies more heavily on kinematic heading and connectivity.
* **Strict Leakage Prevention**: Metadata columns (`trip_id`, `point_idx`, `arc_id`) and raw coordinates (`x`, `y`) are strictly excluded from the training matrix.

---

## 6. Multi-Model Benchmarking

### What Was Done
* Implemented a comprehensive benchmarking pipeline in `src/train_models.py` training 5 distinct algorithms across 4 diverse families on 1,294,644 training instances and evaluating on 137,380 test instances:
  1. **Logistic Regression** (Linear baseline)
  2. **Gaussian Naive Bayes** (Probabilistic baseline)
  3. **Decision Tree** (Non-linear rule-based)
  4. **Random Forest** (Bagging ensemble)
  5. **HistGradientBoosting** (Gradient boosting ensemble)

### Why It Was Done (Justification)
* **Submission Requirement**: The rubric strictly mandates testing at least 5 distinct models from multiple algorithm families.
* **Algorithmic Findings**:
  * **Gaussian Naive Bayes failed** (**73.83%** Top-1, **12,652** false positives) because spatial distance features (`perp_dist`, `dist_start_node`, `dist_end_node`) are highly correlated, violating the naive independence assumption.
  * **Tree Ensembles dominated**: **Random Forest** achieved **95.04%** candidate accuracy and **0.8913** Macro F1; **HistGradientBoosting** achieved **94.95%** candidate accuracy and **0.8893** Macro F1 while training in only **33.99 seconds** (5.7x faster than Random Forest's 193.70 s).

---

## 7. Bayesian Hyperparameter Optimization

### What Was Done
* Implemented Bayesian optimization in `src/tune_bayesian.py` using **Optuna (Tree-structured Parzen Estimator - TPE)** on `HistGradientBoostingClassifier`.
* Executed 20 trials optimizing out-of-fold validation Macro F1 under a 3-fold `GroupKFold` split (grouped by `trip_id`).
* Optimal hyperparameters discovered (Trial #11):
  * `learning_rate`: `0.0864`
  * `max_iter`: `200`
  * `max_depth`: `9`
  * `min_samples_leaf`: `50`
  * `l2_regularization`: `0.0052`

### Why It Was Done (Justification)
* **GroupKFold Integrity**: Standard K-Fold splits observations randomly, placing consecutive points from the same vehicle trip into both train and validation folds (severe leakage). Grouping by `trip_id` ensures that out-of-fold evaluation tests on unseen physical trips.
* **Why Optuna TPE over Grid Search**: Grid search evaluates combinations blindly without memory. Optuna's Tree-structured Parzen Estimator builds a probability model of the objective function, focusing trials on promising parameter regions.
* **Impact of Tuning**:
  * Validation Macro F1 improved from **0.8312 to 0.8482** (+1.70%).
  * Test Log Loss dropped from **0.1245 to 0.1223**. Lower log loss yields better-calibrated probabilities, which directly empowers the subsequent HMM stage.

---

## 8. HMM & Viterbi Sequence Decoding

### What Was Done
* Implemented a Hidden Markov Model with dynamic programming Viterbi decoding in `src/hmm.py` and benchmarked all 10 test trips in `src/eval_hmm.py`.
* **Hidden States**: Candidate road segments within radius $R$.
* **Emission Probabilities**: $P(z_t \mid s_t) = \text{Tuned Classifier Predicted Probability } \hat{p}_t$.
* **Transition Probabilities**: Modeled using exponential path distance discrepancies:
  $$\log P(s_t \mid s_{t-1}) = -\frac{|d_{\text{network}}(s_{t-1}, s_t) - d_{\text{sensor}}(z_{t-1}, z_t)|}{\beta}$$
* Network routing computed dynamically using **NetworkX Dijkstra shortest-path algorithms**.

### Why It Was Done (Justification)
* **The Fundamental Limitation of Classifiers**: Even the best machine learning classifier evaluates each GPS point in isolation. If a momentary GPS multipath spike shifts a single point 12 metres sideways onto a parallel alley, the classifier will incorrectly snap to that alley.
* **Why HMM Solves This**: An HMM incorporates temporal sequence constraints. Transitioning to a disconnected parallel alley incurs an impossible network detour penalty. The Viterbi algorithm identifies the globally optimal trajectory across the entire trip.
* **Results Achieved**:
  * Observation Top-1 Accuracy: **$81.85\% \rightarrow 88.07\%$ (+6.22% gain)**.
  * Route Continuity: **$89.82\% \rightarrow 95.92\%$ (+6.10% gain)**.
  * Completely eliminated erratic zig-zag jumps across road dividers.
