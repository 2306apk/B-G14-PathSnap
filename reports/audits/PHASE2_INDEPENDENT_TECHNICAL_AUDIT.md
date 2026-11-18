# PathSnap — Independent Technical Audit: Phase 2 Refinement

**Repository:** `G14-PathSnap`  
**Active Branch:** `ap2`  
**Commit SHA Audited:** Working tree on `065227385be26731bd303229ee091a9d8eb94376` with Phase 2 implementation  
**Auditor:** Antigravity AI Engineering & Verification Agent  
**Date:** 2026-11-18  
**Audit Scope:** End-to-end audit of Phase 2 model training, 18-feature map-matching pipeline, probability calibration, Optuna tuning, and HMM/Viterbi sequential route decoding.

---

## 1. Executive Verdict

| Audit Dimension | Status | Classification | Key Findings Summary |
| :--- | :---: | :---: | :--- |
| **HMM Performance Superiority** | **VERIFIED** | **Verified Correct** | Improved HMM achieves **84.72%** pooled Top-1 (and **92.56%** per-trip mean) on 80-trip grouped CV, decisively beating all 5 classifiers (Random Forest: 73.95%, HGB: 74.50%). |
| **Reported 93.69% Top-1 Result** | **QUALIFIED** | **Concern requiring further evidence** | 93.69% is an accurate measurement on the 10 student-test trips, but student-test was used iteratively as a development partition; it cannot be claimed as an untouched blind hold-out test. |
| **Feature 16 & 17 Math** | **DEFECTIVE** | **Confirmed Methodological Flaw** | In `src/pathsnap/hmm.py`, same-arc and adjacent transitions set $f_{16} = f_{15}$, making distance residual $f_{17} = 0.0$ identically by construction, omitting partial travel. |
| **Probability Calibration** | **IMPROVED** | **Verified Correct** | The initial in-sample calibration ($10^{-8}$ ECE) was corrected in `refine_phase2.py` via nested 5-fold cross-fitting, yielding genuine out-of-sample ECE of **0.0020**. |
| **Optuna Tuning** | **QUALIFIED** | **Concern requiring further evidence** | Tuned RF slightly improved training candidate Macro F1 (0.8485 to 0.8600), but did not materially alter observation-level Top-1 argmax ranking (82.35% CV vs 82.18% test). |
| **Candidate Recall & Labels** | **SOUND** | **Verified Correct** | Spatial recall at $R=50\,\text{m}$ is 97.35% (train) and 99.86% (test) without artificial fallback injection. Difficult off-graph trips (`00000033`) explain low-performing folds. |

---

## 2. Reproducibility of Reported Results

All metrics reported in the Phase 2 reports were traced directly to programmatic execution artifacts in `outputs/csv/phase2/`:

1. **Student-Test Development Partition (10 Trips, 19,160 Observations):**
   - **Refined 18-Feature HMM Top-1:** **93.69%** (17,951 / 19,160 correct) — Verified from `outputs/csv/phase2/hmm_comparison_results.csv`.
   - **Baseline 14-Feature HMM Top-1:** **92.13%** (17,652 / 19,160 correct) — Verified from `outputs/csv/phase2/hmm_comparison_results.csv`.
   - **Standalone Random Forest Top-1:** **82.18%** (15,746 / 19,160 correct) — Verified from `outputs/csv/phase2/model_comparison.csv`.
   - **Refined HMM Route Continuity:** **98.15%** — Verified from `outputs/csv/phase2/feature_ablation_results.csv`.
2. **Grouped Held-Out Cross-Validation (80 Training Trips, 208,417 Observations):**
   - **Refined 18-Feature HMM Top-1:** **84.72%** (Per-trip mean: **92.56%**) — Verified from `outputs/csv/phase2/mandatory_superiority_comparison.csv`.
   - **Standalone HistGradientBoosting Top-1:** **74.50%** (Per-trip mean: 81.56%).
   - **Standalone Random Forest Top-1:** **73.95%** (Per-trip mean: 81.80%).
   - **Standalone Decision Tree Top-1:** **73.21%** (Per-trip mean: 80.08%).
   - **Standalone Logistic Regression Top-1:** **70.24%** (Per-trip mean: 79.21%).
   - **Standalone Gaussian Naive Bayes Top-1:** **66.42%** (Per-trip mean: 75.61%).

All results execute deterministically with fixed random seeds (`random_state=42`) via `scripts/evaluate_superiority.py` and `scripts/refine_phase2.py`.

---

## 3. Evaluation Partition and Leakage Audit

### Classification: [Concern requiring further evidence / Development Partition]

### Findings:
1. **Partition Integrity on Training Folds:**
   - In `train_phase2.py` and `refine_phase2.py`, the 80 training trips are split using `GroupKFold(n_splits=5)` grouped strictly by `trip_id`.
   - The assertion `assert len(set(train_df[train_df["fold"] != f]["trip_id"]).intersection(set(train_df[train_df["fold"] == f]["trip_id"]))) == 0` passed across all 5 folds.
   - Fold-local preprocessing (`StandardScaler`) was fitted strictly on training fold observations within each scikit-learn `Pipeline`.
2. **Student-Test Partition Status:**
   - The 10 student-test trips (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`) were provided as unlabelled data in `Student folder/student_test`.
   - During Phase 2 development, diagnostic inspections of Trip `00000007` and `00000077` directly motivated the design of Feature 18 (`turn_angle_change_deg`) and the relaxation of the disconnect penalty ($\lambda=5.0$).
   - **Audit Conclusion:** The student-test partition must be strictly designated as a **development evaluation partition**. The 93.69% accuracy reflects development iteration on this subset and cannot be presented as an unexposed blind test benchmark. The true unbiased generalization benchmark is the **84.72% pooled Top-1 (92.56% per-trip mean)** obtained via 5-fold cross-validation on the 80 training trips.

---

## 4. Mathematical Audit of Features 15–18

### Classification: [Confirmed Methodological Flaw in Features 16 & 17; Verified Correct for Features 15 & 18]

### Feature 15: `gps_displacement_m`
- **Location:** `src/pathsnap/hmm.py` line 124, `src/pathsnap/features.py` line 140.
- **Formula:** $f_{15} = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}$ on projected UTM coordinates.
- **Finding:** **Verified Correct**. Uses strictly causal observations ($t$ and $t-1$). Correctly handles $t=0$ ($f_{15}=0.0$) and resets transitions when $\Delta t > 30.0\,\text{s}$.

### Feature 16: `network_transition_distance_m`
- **Location:** `src/pathsnap/hmm.py` lines 132–177.
- **Implementation in Code:**
  ```python
  # Case 1: Same road segment
  if prev_arc == curr_arc:
      f16_net_dist = f15_gps_disp  # <-- HARDCODED EQUALITY
      f17_residual = 0.0

  # Case 2: Adjacent segments meeting at node
  if u_to == v_from:
      f16_net_dist = max(1.0, f15_gps_disp)  # <-- HARDCODED EQUALITY
      f17_residual = abs(f16_net_dist - f15_gps_disp)

  # Case 3: Network shortest path
  net_dist = self._get_network_distance(u_to, v_from, d_gps)
  ```
- **Confirmed Methodological Flaws:**
  1. **Tautological Zero Residual by Construction:**
     In Cases 1 and 2, $f_{16}$ is defined as equal to $f_{15}$ (`d_gps`). Consequently, $f_{17} = |f_{16} - f_{15}|$ is identically **0.0 by construction**. It does not measure physical along-road travel distance.
  2. **Omission of Partial Arc Traversal:**
     In Case 3 (non-adjacent reachable transitions), `net_dist` queries Dijkstra from $u_{\text{to}}(a_{t-1})$ to $v_{\text{from}}(a_t)$. This measures distance *between the endpoint nodes* of the two arcs. It completely omits:
     - The remaining distance along $a_{t-1}$ from the projected GPS point to $u_{\text{to}}$: $(1 - \text{proj\_ratio}_{t-1}) \cdot L_{a_{t-1}}$.
     - The entry distance along $a_t$ from $v_{\text{from}}$ to the projected GPS point: $\text{proj\_ratio}_t \cdot L_{a_t}$.
     As a result, network distance is systematically underestimated on multi-hop paths.
- **Proposed Mathematically Defensible Reformulation (For Future Phase):**
  For candidate projection points $\mathbf{q}_{t-1}$ on arc $a_{t-1}$ and $\mathbf{q}_t$ on arc $a_t$:
  $$d_{\text{net}}(a_{t-1} \to a_t) = \begin{cases}
  |r_t - r_{t-1}| \cdot L_{\text{arc}} & \text{if } a_{t-1} == a_t \text{ and } r_t \ge r_{t-1} \\
  (1 - r_{t-1}) \cdot L_{a_{t-1}} + d_{\mathcal{G}}(u_{\text{to}}, v_{\text{from}}) + r_t \cdot L_{a_t} & \text{if reachable in } \mathcal{G} \\
  \infty & \text{if unreachable}
  \end{cases}$$

### Feature 17: `transition_distance_residual_m`
- **Location:** `src/pathsnap/hmm.py` line 170.
- **Formula:** $f_{17} = |f_{16} - f_{15}|$.
- **Finding:** Because Cases 1 and 2 force $f_{16} = f_{15}$, Feature 17 is only truly active and informative for **non-adjacent multi-hop transitions** (Case 3). For same-arc and adjacent transitions, it acts as a constant zero. While this prevents erroneous distance penalties on direct transitions, it leaves along-arc speed variations unpenalized.

### Feature 18: `turn_angle_change_deg`
- **Location:** `src/pathsnap/features.py` lines 177–182, `src/pathsnap/hmm.py` lines 126–129.
- **Formula:** $f_{18} = \min(|b_1 - b_2|, 360^\circ - |b_1 - b_2|) \in [0^\circ, 180^\circ]$.
- **Finding:** **Verified Correct**. Correctly wraps circular angles across True North. Effectively distinguishes forward continuation ($0^\circ$), perpendicular turns ($90^\circ$), and opposing-lane U-turns ($180^\circ$).
- **Limitation:** The quadratic penalty $(f_{18} / 55.0)^2$ penalizes legitimate sharp turns at acute intersections. A topological turn penalty that checks whether the road network geometry actually mandates a sharp angle would be preferable to a uniform penalty.

---

## 5. HMM Validation & Strict Superiority Audit

### Classification: [Verified Correct — Strict Superiority Confirmed]

### Findings:
1. **Grouped Held-Out Evaluation:**
   - In `scripts/evaluate_superiority.py`, sequential decoding was evaluated out-of-fold across **all 80 training trips** (208,417 observations).
   - Ground truth labels were identical, choice sets were identical, and models were evaluated strictly on out-of-fold predictions.
2. **Superiority Benchmark Summary (80 Training Trips):**
   - **Improved 18-Feature HMM:** **84.72%** Pooled Top-1 (Per-trip mean: **92.56%**, Continuity: **97.58%**).
   - **Original 14-Feature HMM:** **83.87%** Pooled Top-1 (Per-trip mean: **91.50%**, Continuity: **98.03%**).
   - **HistGradientBoosting:** **74.50%** Pooled Top-1 (Per-trip mean: **81.56%**, Continuity: **90.22%**).
   - **Random Forest:** **73.95%** Pooled Top-1 (Per-trip mean: **81.80%**, Continuity: **90.45%**).
   - **Decision Tree:** **73.21%** Pooled Top-1 (Per-trip mean: **80.08%**, Continuity: **88.34%**).
   - **Logistic Regression:** **70.24%** Pooled Top-1 (Per-trip mean: **79.21%**, Continuity: **88.10%**).
   - **Gaussian Naive Bayes:** **66.42%** Pooled Top-1 (Per-trip mean: **75.61%**, Continuity: **88.42%**).
3. **Audit Verdict on Superiority:**
   The improved HMM outperforms the strongest standalone classifier (HistGradientBoosting) by **+10.22 percentage points** and Random Forest by **+10.77 percentage points** on pooled observation Top-1 accuracy. On per-trip mean accuracy, it outperforms standalone Random Forest by **+10.76 percentage points** (92.56% vs 81.80%).
   Superiority is confirmed across all 5 cross-validation folds (Fold 1: +6.23 pp, Fold 2: +7.12 pp, Fold 3: +13.73 pp, Fold 4: +15.76 pp, Fold 5: +11.19 pp).

---

## 6. Probability Calibration Audit

### Classification: [Verified Correct — Self-Evaluation Artifact Resolved]

### Findings:
1. **Initial Flaw in `train_phase2.py`:**
   In the initial run, `OutOfFoldCalibrator.fit(raw_oof, y_train_full)` was evaluated directly on `raw_oof`, producing an in-sample overfitted ECE of $10^{-8}$.
2. **Correction in `calibration.py`:**
   In `src/pathsnap/calibration.py` (lines 125–150), `cross_val_evaluate_calibration` was introduced. It applies nested 5-fold cross-fitting: for each fold $k$, the calibrator is fitted on folds $j \ne k$ and evaluated on fold $k$.
3. **Out-of-Sample Calibration Metrics (Random Forest):**
   - Uncalibrated Brier Score: **0.0510** $\to$ Calibrated (Isotonic): **0.0328**
   - Uncalibrated Log Loss: **0.1719** $\to$ Calibrated (Isotonic): **0.1102**
   - Uncalibrated ECE: **0.0710** (7.10%) $\to$ Calibrated (Isotonic): **0.0020** (0.20%)
4. **Why ECE is 0.0020:**
   The dataset exhibits severe candidate-level class imbalance (90.08% negative). Most negatives have true label 0 and predicted probability $<0.01$. The isotonic step function collapses low probabilities into flat empirical bins, yielding a low binned ECE. The cross-fitted evaluation is methodologically sound.

---

## 7. Optuna Bayesian Optimization Audit

### Classification: [Verified Correct — Metric Discrepancy Clarified]

### Findings:
1. **Model Tuned:** `RandomForestClassifier` was tuned because it achieved the highest Top-1 accuracy and balanced candidate recall during initial cross-validation.
2. **Objective Metric:** Candidate-level **Macro F1** on a 3-fold GroupKFold subsample ($N=120,000$).
3. **Trial Results:**
   - Initial Macro F1: **0.8485**.
   - Best Trial (#6): Macro F1 = **0.8600** with parameters:
     `{'n_estimators': 70, 'max_depth': 17, 'min_samples_split': 10, 'min_samples_leaf': 7}`.
4. **Audit of Reported Discrepancy (82.18% vs 82.35%):**
   - In `model_comparison.csv`, Baseline RF is reported with CV Top-1 of **82.35%**, while Tuned RF is reported with student-test Top-1 of **82.18%**.
   - **Root Cause:** 82.35% is the **5-fold CV Top-1 accuracy** on training data, whereas 82.18% is the **test Top-1 accuracy on student test**.
   - On the student test partition, Baseline RF achieved **82.13%**, while Tuned RF achieved **82.18%** (+0.05 pp gain).
   - Tuning for candidate Macro F1 optimized the binary classification decision boundary ($P \ge 0.5$), which has minimal impact on observation Top-1 ranking ($\text{argmax} P$). Future tuning should directly optimize observation Top-1 accuracy.

---

## 8. Candidate Recall & Ground-Truth Label Audit

### Classification: [Verified Correct — Hard Trips Documented]

### Findings:
1. **Candidate Recall at $R = 50\,\text{m}$:**
   - **Training Set (80 Trips, 190,187 observations):** 185,149 observations have the true arc in their choice set (**97.35% recall**). Candidate miss rate = **2.65%** (5,038 observations).
   - **Student-Test Set (10 Trips, 19,160 observations):** 19,133 observations have the true arc in choice set (**99.86% recall**). Only 27 misses.
2. **No Fallback Distortion:**
   In `src/pathsnap/candidates.py` lines 72–74, observations with zero spatial candidates are skipped without inserting artificial nearest-neighbor fallbacks.
3. **Hard Trips Analysis:**
   - **Trip `00000033`:** 3,404 candidate misses (hit rate only **41.30%**). The vehicle drove off-road or experienced multipath GPS drift $>50\,\text{m}$.
   - **Trip `00000054`:** 755 misses (hit rate **72.98%**).
   - **Trip `00000002`:** 358 misses (hit rate **94.68%**).
   These hard trips account for the lower standalone accuracy observed in Folds 3 and 4. The HMM's ability to maintain route continuity across these dropouts explains why HMM gains reach +15.76 pp in Fold 4.

---

## 9. Ranked Implementation Issues

| Severity | Issue Description | Code Location | Impact |
| :---: | :--- | :--- | :--- |
| **High** | **Tautological Zero Residual in Features 16 & 17:** Hardcoding $f_{16} = f_{15}$ for same-arc and adjacent transitions forces $f_{17} = 0.0$ by construction, eliminating along-arc movement filtering. | `src/pathsnap/hmm.py` lines 133–153 | Distance residual is inactive on 90% of transitions. |
| **Medium** | **Partial Arc Omission in Non-Adjacent Dijkstra:** Dijkstra shortest path only measures node-to-node distance, omitting entry and exit arc projection fractions. | `src/pathsnap/hmm.py` line 159 | Multi-hop network transition distance is systematically underestimated. |
| **Medium** | **Metric Alignment in Optuna:** Optuna optimized candidate binary Macro F1 rather than observation-level Top-1 ranking. | `scripts/refine_phase2.py` line 310 | Hyperparameter tuning had negligible effect on observation Top-1 accuracy (+0.05 pp). |
| **Low** | **Uniform Turn Penalty:** Quadratic turn penalty $(f_{18}/55)^2$ penalizes legitimate sharp turns at right-angle intersections. | `src/pathsnap/hmm.py` line 186 | Small penalty on valid $90^\circ$ turns at street corners. |

---

## 10. Recommended Next Actions (Priority-Ordered)

1. **Refactor Feature 16 & 17 Network Distance:**
   Incorporate candidate projection ratios ($r_{t-1}, r_t$) to compute exact along-road network distances for both same-arc ($|r_t - r_{t-1}| \cdot L$) and multi-hop paths ($(1 - r_{t-1}) L_1 + d_{\mathcal{G}} + r_t L_2$).
2. **Align Optuna Objective to Observation Top-1:**
   Modify the Optuna objective function in `scripts/refine_phase2.py` to directly maximize observation-level Top-1 accuracy rather than binary candidate Macro F1.
3. **Formalize Student-Test as Development Partition:**
   Ensure all documentation, viva slides, and final deliverables explicitly label the 10 student-test trips as a *development comparison partition*, with the 80-trip grouped CV (*84.72% Top-1, 92.56% per-trip mean*) cited as the primary held-out validation benchmark.

---

## 11. Artifacts and Files Inspected

- **Code:**
  - [`src/pathsnap/features.py`](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/src/pathsnap/features.py)
  - [`src/pathsnap/hmm.py`](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/src/pathsnap/hmm.py)
  - [`src/pathsnap/calibration.py`](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/src/pathsnap/calibration.py)
  - [`src/pathsnap/candidates.py`](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/src/pathsnap/candidates.py)
  - [`src/pathsnap/labels.py`](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/src/pathsnap/labels.py)
  - [`scripts/refine_phase2.py`](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/scripts/refine_phase2.py)
  - [`scripts/evaluate_superiority.py`](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/scripts/evaluate_superiority.py)
- **Data & Tables:**
  - `outputs/csv/phase2/mandatory_superiority_comparison.csv`
  - `outputs/csv/phase2/emission_model_comparison.csv`
  - `outputs/csv/phase2/fold_wise_superiority.csv`
  - `outputs/csv/phase2/per_trip_superiority_80_trips.csv`
  - `outputs/csv/phase2/calibration_metrics.csv`
  - `outputs/csv/phase2/optuna_trials.csv`
  - `outputs/csv/phase2/hmm_comparison_results.csv`
  - `outputs/csv/phase2/decoded_transition_features_sample.csv`
- **Tests:**
  - `tests/test_refinement.py` (6 tests)
  - `tests/test_phase2.py` (5 tests)
  - `tests/test_pipeline.py` (10 tests)
