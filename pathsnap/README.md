# PathSnap: Preprocessing, Candidate Generation, and Feature Extraction Pipeline

Map-matching pipeline built for the university ML hackathon (PathSnap).
Matches noisy 1 Hz GPS trajectory points to road-network arcs using spatial geometry indexing, candidate generation, and feature extraction.

---

## 1. Directory Structure

```text
pathsnap/
├── data/
│   └── processed/
│       ├── pipeline_config.json             # Fitted config (radius, thresholds, CRS strategy)
│       ├── train_features.parquet           # 1,294,644 rows (Train candidate-point pairs)
│       ├── train_features_sample.csv        # First 2,000 rows sample
│       ├── student_test_features.parquet    # 137,380 rows (Student test pairs)
│       └── student_test_features_sample.csv # First 2,000 rows sample
├── notebooks/
│   └── 01_eda.ipynb                         # Jupyter notebook for Step 1 exploratory data analysis
├── outputs/
│   ├── eda/                                 # Summary CSVs and EDA distribution plot
│   │   ├── inventory_summary.csv
│   │   ├── timestamp_summary.csv
│   │   ├── extent_summary.csv
│   │   ├── network_summary.csv
│   │   ├── route_summary.csv
│   │   └── eda_distributions.png
│   ├── maps/
│   │   └── trip_00000001_map.html           # Interactive Folium map with layers & legend
│   └── plots/
│       ├── candidate_distance_distribution.png
│       ├── feature_correlations.png
│       └── train_candidate_metrics.csv
├── src/
│   ├── data_io.py                           # Loader with academic integrity enforcement
│   ├── geo.py                               # Local metric UTM projection & STRtree index
│   ├── preprocess.py                        # Cleaning & kinematics calculation (dt, speed, heading)
│   ├── labels.py                            # Monotonic route ground-truth reference-arc assignment
│   ├── candidates.py                        # STRtree candidate generation (R = 35m)
│   ├── features.py                          # Feature extraction per (point, candidate) pair
│   ├── eda_script.py                        # Script executing Step 1 EDA
│   └── build_dataset.py                     # End-to-end pipeline processing train & test partitions
├── data_quality_notes.md                    # Data anomalies, findings, and decisions
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## 2. Academic Integrity Rules Enforced

1. **Evaluator Partition Protection**: `src/data_io.py` strictly prevents reading or opening `evaluator_test`. Requesting any forbidden partition raises an immediate `PermissionError`.
2. **Trip Partitioning**: Splits are grouped exclusively by trip ID. Points from a trip are never mixed across partitions.
3. **No Hardcoded Predictions**: Reference labels from `.route` are used solely to establish reference training targets, never as input features.
4. **Metric CRS**: All spatial queries, distances, and bearings are computed in projected metres using each trip's optimal local UTM projection, never Euclidean degrees.
5. **Fitted Transforms**: Radius ($R = 35.0 \text{ m}$) and pipeline thresholds are determined strictly from `train` data and saved to `data/processed/pipeline_config.json`.

---

## 3. How to Run Each Step

### Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Run Data Validation & EDA
Generates inventory summaries, timestamp gap distributions, network metrics, and the Folium interactive map:
```bash
python pathsnap/src/eda_script.py
```
Outputs saved to `pathsnap/outputs/eda/` and `pathsnap/outputs/maps/trip_00000001_map.html`.

### Step 2, 3, 4, & 5: Run Full Preprocessing, Labelling, Candidates & Features
Runs the end-to-end pipeline over all 80 training trips and 10 student-test trips:
```bash
python pathsnap/src/build_dataset.py
```
This produces:
- `pathsnap/data/processed/train_features.parquet` (1,294,644 rows, 16 columns)
- `pathsnap/data/processed/student_test_features.parquet` (137,380 rows, 16 columns)
- `pathsnap/outputs/plots/candidate_distance_distribution.png`
- `pathsnap/outputs/plots/feature_correlations.png`

---

## 4. Feature Definitions

Each row represents a `(trip_id, point_idx, candidate_arc_id)` tuple. Raw IDs and coordinates are excluded from model training features:

| Feature Name | Type | Description |
|---|---|---|
| `perp_dist` | `float` | Perpendicular Euclidean distance from the GPS point to the candidate arc line (metres). |
| `dist_start_node` | `float` | Distance from the GPS point to the arc's start node (metres). |
| `dist_end_node` | `float` | Distance from the GPS point to the arc's end node (metres). |
| `proj_frac` | `float` | Fractional projection along the candidate arc $[0.0 \dots 1.0]$. |
| `bearing_diff` | `float` | Angular difference between vehicle motion heading and arc orientation $[0^\circ \dots 180^\circ]$. |
| `is_reverse_direction` | `int (0/1)` | Binary flag: $1$ if `bearing_diff` $> 90^\circ$ (vehicle travelling against arc flow). |
| `speed` | `float` | Vehicle instantaneous speed at this point ($\text{m/s}$). |
| `dt` | `float` | Time elapsed since the previous GPS observation (seconds). |
| `arc_length` | `float` | Length of the candidate road arc in metres. |
| `candidate_rank` | `int` | Distance rank among all candidate arcs for this GPS point ($1 = \text{nearest}$). |
| `num_candidates` | `int` | Total count of candidates within radius $R$ for this point. |
| `dist_gap_to_next` | `float` | Distance margin between this candidate and the nearest competitor (detects parallel street ambiguity). |
| `label` | `int (0/1)` | Target label ($1$ if candidate is the reference arc, $0$ otherwise). |

**Dataset Stats**:
- **Train Class Imbalance**: $13.94\%$ positive (`label = 1`), $86.06\%$ negative (`label = 0`) (Ratio $1 : 6.2$).
- **Mean Candidates per Point**: $7.96$ arcs.
- **Candidate Hit Rate**: $86.57\%$ with $R = 35\text{ m}$.
