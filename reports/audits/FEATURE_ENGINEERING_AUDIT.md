# PathSnap — Feature Engineering & Candidate Classification Audit

**Generated Programmatically by `scripts/build_features.py`**
**Date:** 2026-11-18 11:13:27

---

## 1. Feature Dataset Summary

Features were extracted for every GPS-observation and candidate-arc pair within radius $R = 50.0\text{m}$.

| Partition | Total Trips | Observations | Candidate-Arc Rows | Positive Labels | Class Ratio (Pos %) | File Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | 80 | 208,417 | 1,867,152 | 185,149 | 9.92% | `data/processed/train_features.parquet` |
| **Student Testing** | 10 | 19,160 | 209,121 | N/A (Unlabelled) | N/A | `data/processed/student_test_features.parquet` |

---

## 2. Complete Feature Schema Documentation

The PathSnap feature store defines exactly 14 predictive features per candidate pair:

| Feature Name | Type | Unit | Rationale & Definition | Missing Value Policy |
| :--- | :--- | :--- | :--- | :--- |
| `dist_to_arc` | float/int | meters | Perpendicular Euclidean distance from the projected GPS observation to the candidate road segment. Primary geometric proximity indicator. True segments have small perpendicular distances. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `dist_to_from_node` | float/int | meters | Euclidean distance from the GPS observation to the candidate arc's starting node (from_node). Captures proximity to intersection entry and relative position along the road. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `dist_to_to_node` | float/int | meters | Euclidean distance from the GPS observation to the candidate arc's ending node (to_node). Captures proximity to intersection exit and endpoint convergence. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `proj_ratio` | float/int | ratio [0, 1] | Normalized projection position along the directed segment from 0 (from_node) to 1 (to_node). Indicates whether the observation projects onto the middle of the segment or clamps at endpoints. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `gps_heading` | float/int | degrees [0, 360) | Instantaneous vehicle travel heading computed from previous to current GPS observation. Reflects vehicle orientation without looking ahead into the future (leakage-free). | Imputed with median or 0.0; guaranteed 0 NaNs |
| `arc_bearing` | float/int | degrees [0, 360) | Directed compass bearing of the road segment from from_node to to_node clockwise from True North. Represents the permissible direction of travel along this road arc. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `heading_diff` | float/int | degrees [0, 180] | Circular angular difference between the vehicle GPS heading and the candidate arc bearing. Discriminates between opposing lanes, parallel streets, and perpendicular cross-streets. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `is_reverse_direction` | float/int | binary {0, 1} | Indicator flag equal to 1 if heading_diff > 90 degrees, indicating counter-flow direction. Vehicles rarely travel backwards along one-way road segments; strongly filters reverse arcs. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `speed_est` | float/int | m/s | Estimated instantaneous vehicle speed calculated as distance(P_{t-1}, P_t) / delta_t. Differentiates slow/stopped maneuvers (where heading is noisy) from high-speed arterial travel. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `time_gap` | float/int | seconds | Elapsed time delta between the previous and current GPS observation. Flags GPS sampling interruptions or tunnels where dead-reckoning uncertainty increases. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `arc_length` | float/int | meters | Total Euclidean length of the candidate road segment. Short intersection connector arcs vs long freeway segments have different prior probabilities. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `candidate_rank` | float/int | integer [1, K] | Rank of the candidate arc among all candidates for this observation, sorted by ascending dist_to_arc. Relative rank provides an invariant ranking feature robust across different road densities. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `candidate_count` | float/int | count | Total number of candidate road arcs within the search radius R for this observation. Reflects local road network density; complex intersections have higher candidate counts. | Imputed with median or 0.0; guaranteed 0 NaNs |
| `dist_to_alternative` | float/int | meters | Distance margin to the nearest competing candidate arc. Measures confidence: a large margin indicates an unambiguous choice with no close competitors. | Imputed with median or 0.0; guaranteed 0 NaNs |

---

## 3. Candidate Miss Rate Analysis

A **Candidate Miss** occurs when the true assigned reference route arc is absent from the candidate set generated within search radius $R = 50.0\text{m}$.

- **Total Training Observations Evaluated:** 208,417
- **Observations with Reference Arc in Candidates (Hits):** 185,149 (88.84%)
- **Observations Missing Reference Arc (Misses):** 23,268 (11.16%)

### Handling Missing Candidates
- In the training feature dataset, when a candidate miss occurs, all candidates for that observation receive label `0`.
- The observation is explicitly flagged in audit logs rather than hidden via artificial fallback.
- In sequential HMM decoding, observations with missing candidates or zero candidates trigger an emission-smoothing fallback (uniform low probability) and allow transition continuity across the gap.

---

## 4. Leakage Prevention and Temporal Safeguards

1. **Temporal Horizon:** Kinematic features (`speed_est`, `gps_heading`, `time_gap`) are derived strictly using past observations:
   $$v_t = \frac{\|P_t - P_{t-1}\|}{\Delta t}, \quad \theta_t = \text{bearing}(P_{t-1}, P_t)$$
   No future observation $P_{t+1}$ is used in any feature calculation.
2. **First Observation Boundary ($t=0$):**
   - `speed_est` is initialized strictly to $0.0\text{ m/s}$.
   - `time_gap` is set to the nominal $1.0\text{s}$.
   - `gps_heading` is initialized to match the orientation of step 1, with no forward speed leakage.
3. **Partition Isolation:** Student-test trips are processed with `labels_df = None`. No reference route data enters the student-test candidate or feature pipeline.

---

## 5. Artifacts and Generated Data Files

1. `data/processed/train_features.parquet`: Full training feature matrix (1,867,152 rows).
2. `data/processed/student_test_features.parquet`: Full student-testing feature matrix (209,121 rows).
3. `data/processed/train_features_sample_1000.csv`: Small inspection CSV.
4. `outputs/csv/train_feature_summary.csv`: Per-trip observation, candidate, and hit/miss counts.
5. `outputs/csv/student_test_feature_summary.csv`: Per-trip student-test candidate counts.
6. `outputs/plots/feature_distributions.png`: Density distributions comparing positive vs negative candidates.
7. `outputs/plots/feature_correlations.png`: Correlation matrix of features and target label.
