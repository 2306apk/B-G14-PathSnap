# PathSnap: Distributed Map-Matching & Route Reconstruction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Academic-informational.svg)]()

**Group 14 (PathSnap)** — University Machine Learning Hackathon Project.

PathSnap is an end-to-end distributed machine learning system that matches noisy vehicle GPS trajectory observations to road networks and reconstructs coherent, topologically valid routes.

---

## 1. Project Architecture & Repository Layout

```text
G14-PathSnap/
├── data/
│   ├── raw/                      # Symlinked or configured path to raw Student folder
│   │   ├── train/                # 80 training trips (.track, .nodes, .arcs, .route)
│   │   └── student_test/         # 10 student-testing trips
│   └── processed/                # Extracted feature datasets (Parquet / CSV)
│       ├── train_features.parquet
│       └── student_test_features.parquet
├── src/
│   └── pathsnap/
│       ├── __init__.py
│       ├── io.py                 # Robust whitespace file parsing & data integrity validation
│       ├── geo.py                # Dynamic per-trip UTM CRS selection & metric conversions
│       ├── graph.py              # Independent road networks, topology & WCC analysis
│       ├── candidates.py         # STRtree spatial queries & candidate radius optimization
│       ├── labels.py             # Order-aware monotonic DP point-to-route alignment
│       ├── features.py           # 14-feature schema with past-only temporal safeguards
│       ├── visualize.py          # Folium interactive maps [lat, lon] & scientific plots
│       └── metrics.py            # Observation Top-1 accuracy, miss rate, continuity
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis & quality audit
│   ├── 02_preprocessing.ipynb    # Coordinate projections & spatial indexing
│   ├── 03_candidate_generation.ipynb # Candidate radius sweeps & recall curves
│   └── 04_feature_engineering.ipynb  # Feature schemas, distributions & correlations
├── scripts/
│   ├── run_eda.py                # Generates dataset audit, inventory & quality metrics
│   ├── run_preprocessing.py      # Computes per-trip CRS & candidate radius sweeps
│   └── build_features.py         # Builds processed Parquet datasets & feature audits
├── outputs/
│   ├── csv/                      # Generated audit tables and parameter sweeps
│   ├── plots/                    # Static distribution and trajectory visualizations
│   └── maps/                     # Interactive Folium HTML trajectory maps
├── reports/
│   └── audits/
│       ├── DATASET_AUDIT.md      # Full programmatic dataset quality audit
│       ├── PREPROCESSING_AUDIT.md# Coordinate, projection & radius justifications
│       ├── FEATURE_ENGINEERING_AUDIT.md # Schema, miss rates & leakage safeguards
│       └── PHASE1_STATUS.md      # Milestone summary & Phase 2 modeling roadmap
├── tests/
│   └── test_pipeline.py          # Unit tests covering all core modules
├── requirements.txt              # Pinned Python dependencies
└── README.md
```

---

## 2. Installation & Quickstart

### Prerequisites
- Python 3.11 on Apple Silicon / macOS or Linux
- Virtual environment (`venv`)

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/2306apk/G14-PathSnap.git
cd G14-PathSnap

# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Dataset Path
By default, PathSnap automatically resolves raw data from `data/raw/` or `Student folder` in parent directories. You can also set a custom location via environment variable:
```bash
export PATHSNAP_DATA_DIR="/path/to/Student folder"
```

---

## 3. Execution Pipeline

### Step 1: Run Exploratory Data Analysis & Audit
```bash
python scripts/run_eda.py
```
Outputs:
- Audit CSVs in `outputs/csv/` (`dataset_inventory.csv`, `gps_quality_audit.csv`, `network_quality_audit.csv`)
- Statistical distribution plots in `outputs/plots/`
- Interactive Folium maps in `outputs/maps/`
- Report: `reports/audits/DATASET_AUDIT.md`

### Step 2: Run Preprocessing & Candidate Radius Optimization
```bash
python scripts/run_preprocessing.py
```
Outputs:
- Per-trip UTM CRS mapping in `outputs/csv/trip_crs_mapping.csv`
- Candidate radius sweep metrics in `outputs/csv/candidate_radius_sweep.csv`
- Candidate distance and recall plots in `outputs/plots/`
- Report: `reports/audits/PREPROCESSING_AUDIT.md`

### Step 3: Build Candidate Features & Target Labels
```bash
python scripts/build_features.py
```
Outputs:
- Processed feature matrices: `data/processed/train_features.parquet` and `data/processed/student_test_features.parquet`
- Feature distribution and correlation plots in `outputs/plots/`
- Report: `reports/audits/FEATURE_ENGINEERING_AUDIT.md`

### Step 4: Run Unit Tests
```bash
pytest -v tests/
```

---

## 4. Key Engineering Decisions

1. **Independent Per-Trip Networks:** No single global graph is constructed. Node 0 and Arc 0 in trip 1 are completely distinct from trip 2. Graphs and spatial indices are built strictly per trip.
2. **Local Metric Projections:** Eliminates arbitrary assumptions (such as EPSG:28992). The pipeline automatically derives the true UTM zone from each trip's geographic extent across 25 distinct zones.
3. **Monotonic Dynamic Programming Alignment:** Associates reference routes to GPS points while honoring chronological vehicle progression and preserving ground-truth uncertainty flags.
4. **Justified Candidate Radius ($R = 50.0\text{m}$):** Balances high candidate recall (96.93% overall, ~100% of reachable route arcs) with manageable candidate density (11.76 candidates/point).
5. **Zero Lookahead Leakage:** Speed, heading, and time gaps are derived strictly from past observations ($t$ and $t-1$). Student-testing reference routes are strictly isolated.
