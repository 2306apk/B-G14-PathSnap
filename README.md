# PathSnap: Map-Matching & Route Reconstruction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Academic-informational.svg)]()

**Group 14 (PathSnap)** — University Machine Learning Hackathon Project.

PathSnap is an end-to-end map-matching system that matches raw, noisy vehicle GPS trajectories to directed road networks and reconstructs continuous, topologically valid driving routes.

---

## 1. Project Overview

Vehicle GPS traces collected in real-world environments often suffer from sensor noise, clock drift, and multipath reflections in urban canyons. As a result, raw coordinates drift off the actual road, jump across multi-lane highways, or suggest impossible driving maneuvers (such as sudden U-turns or traveling against traffic on one-way streets).

PathSnap solves this problem in two stages:
1. **Candidate Classification:** Generates nearby candidate road segments within a 50m radius and scores each candidate using a machine learning classifier trained on 14 kinematic and geometric observation features.
2. **Sequential Route Decoding:** Feeds calibrated classifier probabilities and 4 road-network transition features into a Hidden Markov Model (HMM) solved via the Viterbi algorithm. The HMM enforces physical road network connectivity and directional turn constraints to reconstruct the most likely continuous route.

---

## 2. How It Works

```
Raw GPS Input (timestamp, lat, lon)
        │
        ▼
Local UTM Projection (metric coordinate conversion per trip)
        │
        ▼
Spatial Candidate Generation (STRtree spatial query, R = 50m)
        │
        ▼
14 Observation Features (distance to arc, heading diff, reverse direction, speed, etc.)
        │
        ▼
Calibrated Classifier Probabilities (Random Forest + Platt Sigmoid Calibration)
        │
        ▼
4 Transition Features (GPS displacement, road distance, residual, turn-angle change)
        │
        ▼
HMM / Viterbi Sequence Decoder (directed graph shortest paths, gap resets, partial arc progress)
        │
        ▼
Reconstructed Route (topologically continuous sequence of road arcs)
```

### The 18-Feature Architecture
The system uses a total of 18 features divided into two operational levels:

- **14 Observation-Level Features (Classifier Emissions):**
  1. `dist_to_arc`: Perpendicular distance to candidate road arc (meters).
  2. `dist_to_from_node`: Distance to segment start node (meters).
  3. `dist_to_to_node`: Distance to segment end node (meters).
  4. `proj_ratio`: Longitudinal projection ratio along segment $[0.0, 1.0]$.
  5. `gps_heading`: Vehicle bearing between consecutive GPS points $[0, 360^\circ)$.
  6. `arc_bearing`: Permissible travel bearing along the directed road arc $[0, 360^\circ)$.
  7. `heading_diff`: Angular alignment between vehicle heading and road bearing $[0, 180^\circ]$.
  8. `is_reverse_direction`: Flag for counter-flow travel ($1$ if heading diff $> 90^\circ$).
  9. `speed_est`: Estimated vehicle speed between $t-1$ and $t$ (m/s).
  10. `time_gap`: Time interval $\Delta t$ between consecutive observations (seconds).
  11. `arc_length`: Total length of the candidate road arc (meters).
  12. `candidate_rank`: Distance rank of candidate within the observation choice set.
  13. `candidate_count`: Total number of candidates within 50m radius.
  14. `dist_to_alternative`: Distance margin between nearest and second-nearest candidates.

- **4 Transition-Level Features (HMM Transitions):**
  15. `gps_displacement_m`: Euclidean displacement between consecutive GPS points on the UTM plane.
  16. `network_transition_distance_m`: Directed shortest path distance along the road network between consecutive candidate arcs, including continuous partial traversal progress calculated via `proj_ratio`.
  17. `transition_distance_residual_m`: Absolute difference $|d_{\text{net}} - d_{\text{gps}}|$ measuring movement plausibility.
  18. `turn_angle_change_deg`: Circular angular deviation between consecutive road arc bearings $[0, 180^\circ]$, penalizing unphysical sharp turns and U-turns.

---

## 3. Models and Evaluation

### Standalone Classifiers vs. HMM Decoder
We trained and evaluated five standalone classification algorithms using 5-fold `GroupKFold` cross-validation across 80 training trips (208,417 observations). Observations from the same trip were never split across training and validation folds.

Two HMM versions were evaluated:
- **14-Feature HMM (Baseline):** Uses classifier emission probabilities with discrete endpoint graph transitions.
- **18-Feature HMM (Improved):** Integrates all 18 features, partial arc traversal modeling, log-space numerical stabilization, backward-movement penalties, and 30-second timestamp gap resets.

### Benchmark Results (80 Training Trips, Out-of-Fold)

| Method / Architecture | Mean Per-Trip Top-1 | Median Per-Trip Top-1 | Pooled Observation Top-1 | Route Continuity | Candidate Recall Ceiling | Coverage-Adjusted Top-1 | Avg Latency / Trip |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Improved HMM (18-feat)** | **92.56%** | **96.23%** | **84.72%** | **97.58%** | **97.51%** | **94.63%** | **52.6 ms** |
| Original HMM (14-feat) | 91.50% | 95.47% | 83.87% | 98.03% | 97.51% | 93.55% | 48.2 ms |
| **Random Forest** | 81.80% | 84.75% | 73.95% | 90.45% | 97.51% | 83.68% | 22.4 ms |
| **HistGradientBoosting** | 81.56% | 84.30% | 74.50% | 90.22% | 97.51% | 83.46% | 14.8 ms |
| **Decision Tree** | 80.08% | 82.53% | 73.21% | 88.34% | 97.51% | 82.01% | 1.1 ms |
| **Logistic Regression** | 79.21% | 81.85% | 70.24% | 88.10% | 97.51% | 80.78% | 1.5 ms |
| **Gaussian Naive Bayes** | 75.61% | 77.98% | 66.42% | 88.42% | 97.51% | 77.09% | 0.9 ms |

### Understanding the Metrics: Mean Per-Trip vs. Pooled Top-1
- **Unweighted Mean Per-Trip Top-1 (92.56%):** Evaluates Top-1 accuracy per trajectory and averages across all 80 trips equally ($1/80$ each). This measures general performance across varied routes without letting long trips dominate the score.
- **Pooled Observation Top-1 (84.72%):** Computes total correct predictions over all 208,417 observations. This score is lower because a single outlier trajectory, **Trip 00000054**, contains **19,313 points** (~9.3% of the entire dataset) and suffered from severe hardware GPS drift (>50m off-road), which truncated candidate recall to 10.56% and accuracy to 8.56%. In pooled evaluation, this one trip contributes nearly 10% of all evaluation points.
- **Win Rate:** The 18-feature HMM outperforms Random Forest on **78 out of 80 trips (97.5% win rate)** with 0 ties and only 2 minor regressions ($-0.57\%$ on Trip 5 and $-0.15\%$ on Trip 33). Across all observations, the HMM fixes **27,191 classifier errors** while introducing **4,744 regressions**, yielding a **5.73 : 1 net correction ratio**.

---

## 4. Repository Structure

```text
G14-PathSnap/
├── src/pathsnap/                 # Core Python package
│   ├── __init__.py
│   ├── io.py                     # Whitespace file parsers and dataset loaders
│   ├── geo.py                    # Dynamic UTM projection selection per trip
│   ├── graph.py                  # Directed road network construction and Dijkstra routing
│   ├── candidates.py             # STRtree spatial indexing and 50m candidate queries
│   ├── labels.py                 # Monotonic DP ground-truth route association
│   ├── features.py               # 18-feature schema and transition feature calculation
│   ├── calibration.py            # Platt sigmoid probability calibrator and ECE metrics
│   ├── hmm.py                    # HMMRouteDecoder, Viterbi recurrence and path backtracking
│   ├── visualize.py              # Interactive Folium map generation and plotting utilities
│   ├── metrics.py                # Top-1 accuracy, route continuity, and recall calculations
│   └── api.py                    # FastAPI backend with Stage 5 and modern REST endpoints
├── frontend/                     # Modern Leaflet/Vite web application
│   ├── index.html                # Main web interface
│   ├── package.json              # Frontend dependencies
│   ├── vite.config.js            # Vite bundler configuration
│   └── src/
│       ├── main.js               # Application logic, API polling, and Leaflet map rendering
│       └── style.css              # Custom styling
├── ui/
│   └── index.html                # Stage 5 offline HTML5 Canvas Evaluator interface
├── notebooks/                    # Educational and exploratory notebooks
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Projections and network inspection
│   ├── 03_candidate_generation.ipynb # Candidate radius sweeps and recall analysis
│   └── 04_feature_engineering.ipynb  # Feature schemas and distributions
├── scripts/                      # Execution and pipeline scripts
│   ├── start_backend.py          # Two-laptop backend launcher with host discovery
│   ├── train_phase2.py           # Classifier training, Bayesian optimization, and HMM tuning
│   ├── refine_phase2.py          # 18-feature refinement and ablation analysis
│   ├── evaluate_superiority.py   # Master benchmark evaluation across all 7 models
│   ├── generate_final_hmm_evaluation.py # Artifact generator for reports and plots
│   ├── build_features.py         # Feature matrix extractor
│   ├── run_eda.py                # Exploratory analysis runner
│   ├── run_preprocessing.py      # Preprocessing and candidate radius runner
│   ├── train_models.py           # Baseline classifier training
│   ├── smoke_test_models.py      # Quick sanity checks on models
│   └── generate_notebooks.py     # Notebook synchronization
├── outputs/                      # Saved experimental outputs
│   ├── csv/                      # Dataset audits and Phase 2 fold results
│   ├── hmm_final/                # Final benchmark CSVs and comparative tables
│   ├── maps/                     # Generated interactive Folium HTML maps
│   └── plots/                    # Diagnostic PNG figures, calibration curves, and deltas
├── reports/                      # Project documentation and specifications
│   ├── HMM_FINAL_PERFORMANCE.md  # Detailed validation report and two-laptop guide
│   ├── model_comparison.md       # Consolidated five-model comparison report
│   ├── model_training_report.md  # Training configurations and hyperparameters
│   ├── hmm_evaluation_report.md  # HMM routing and Viterbi evaluation
│   ├── feature_engineering/
│   │   └── 18_FEATURES.md        # Mathematical definitions of all 18 features
│   └── reliability/
│       ├── calibration_report.md # Expected Calibration Error (ECE) and Brier scores
│       └── overfitting_analysis.md # Generalization gap analysis across folds
├── tests/                        # Automated unit and regression test suite (45 tests)
│   ├── test_api.py               # API endpoint contracts and Stage 5 compatibility
│   ├── test_final_hmm.py         # Final HMM superiority and benchmark regressions
│   ├── test_hmm_behavior.py      # HMM transitions, normalization, and gap resets
│   ├── test_phase2.py            # GroupKFold isolation and calibration tests
│   ├── test_pipeline.py          # Coordinate projection, spatial index, and geometry
│   └── test_refinement.py        # 18-feature extraction and transition penalties
├── requirements.txt              # Pinned Python package dependencies
├── .gitignore                    # Git exclusion rules
└── README.md                     # Project documentation
```

---

## 5. Setup and Installation

### Prerequisites
- Python 3.11 (tested on macOS Apple Silicon and Linux)
- Node.js 18+ (optional, only required if running the Vite frontend)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/2306apk/G14-PathSnap.git
cd G14-PathSnap

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Dataset Configuration
By default, the pipeline automatically searches for raw data in `data/raw/` or in a `Student folder` in parent directories. You can also specify an explicit directory via an environment variable:
```bash
export PATHSNAP_DATA_DIR="/absolute/path/to/Student folder"
```

---

## 6. Running the Backend API

Launch the backend with host discovery enabled:
```bash
python scripts/start_backend.py --host 0.0.0.0 --port 8000
```

The script prints the local and local area network (LAN) URLs:
```text
============================================================================
🚀  PathSnap Map-Matching Backend API Starting...
============================================================================
• Local Machine URL : http://127.0.0.1:8000
• LAN URL           : http://10.158.203.111:8000
• Stage 5 UI        : http://10.158.203.111:8000/ui
• Swagger API Docs  : http://10.158.203.111:8000/docs
============================================================================
```

---

## 7. Two-Laptop Demonstration Setup

PathSnap is designed to run live across two separate laptops connected to the same Wi-Fi network or mobile hotspot:

```text
┌────────────────────────────────┐         Local Wi-Fi / Hotspot         ┌────────────────────────────────┐
│      Laptop 1: macOS (M2)      │ <───────────────────────────────────> │       Laptop 2: Windows        │
│                                │                                       │                                │
│  - Backend Host: 0.0.0.0:8000  │        HTTP / REST API Requests       │  - Client: Web Browser         │
│  - LAN IP: 10.158.203.111      │   (CORS: Allow-Origin: * on all IPs)  │  - URL: http://<LAN_IP>:8000/ui│
│  - Models & HMM Decoder Loaded │                                       │  - Offline HTML5 Canvas UI     │
└────────────────────────────────┘                                       └────────────────────────────────┘
```

### Demonstration Steps:
1. **Laptop 1 (Mac Backend):**
   - Connect to local Wi-Fi or phone hotspot.
   - Start the server: `python scripts/start_backend.py --host 0.0.0.0 --port 8000`.
   - Note the reported LAN IP (e.g., `10.158.203.111`).
2. **Laptop 2 (Windows Frontend):**
   - Connect to the same Wi-Fi or phone hotspot.
   - Open a browser (Chrome or Edge) and visit:
     ```
     http://10.158.203.111:8000/ui
     ```
   - Click **Connect & load** to fetch available trips from the backend.
   - Select a trip (e.g., `00000052`) and click **Reconstruct trip**.
   - The map and evaluation metrics render immediately on Laptop 2.

> **Important:** The client on Laptop 2 must navigate to Laptop 1's **LAN IP** (`http://10.158.203.111:8000/ui`), not `localhost`. The backend is configured with permissive CORS (`allow_origins=["*"]`) so that browser requests across the local network are accepted without origin errors.

---

## 8. Frontend Applications

PathSnap provides two frontend interfaces depending on deployment needs:

1. **Stage 5 Canvas Evaluator UI (`/ui`):**
   - Served directly by FastAPI at `/ui` (from `ui/index.html`).
   - Built with plain HTML5 Canvas and vanilla JavaScript.
   - Zero external npm dependencies and zero external tile server calls—runs completely offline.
   - Renders raw GPS observations (blue dots), nearby candidate road network (gray lines), and the reconstructed route (coral red line) alongside observation accuracy and route continuity KPIs.

2. **Modern Leaflet Frontend (`frontend/`):**
   - Located in `frontend/` (built with Vite and Leaflet).
   - Provides an interactive web map with pan/zoom, layer toggles, and model switching (comparing HMM predictions against Random Forest, Decision Tree, Logistic Regression, etc.).
   - To run locally:
     ```bash
     cd frontend
     npm install
     npm run dev
     ```

---

## 9. Deliverables, Evaluation Reports & Maps

All experimental outputs, comparison tables, and visual deliverables are stored in the repository:

- **Technical Reports:**
  - [reports/HMM_FINAL_PERFORMANCE.md](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/reports/HMM_FINAL_PERFORMANCE.md): Complete performance audit, model superiority verification, error analysis, and viva defense guide.
  - [reports/feature_engineering/18_FEATURES.md](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/reports/feature_engineering/18_FEATURES.md): Formal mathematical definitions for all 14 observation and 4 transition features.
  - [reports/model_comparison.md](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/reports/model_comparison.md): Detailed comparison across all five ML classifiers.
  - [reports/model_training_report.md](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/reports/model_training_report.md): Training configurations, hyperparameter search spaces, and loss functions.
- **Evaluation CSV Tables:**
  - `outputs/hmm_final/final_models_comparison.csv`: Consolidated comparison across all seven architectures.
  - `outputs/hmm_final/final_per_trip_comparison.csv`: Trip-by-trip accuracy and continuity records for all 80 training trips.
  - `outputs/hmm_final/student_test_isolated_evaluation.csv`: Independent evaluation on the student-test development partition.
  - `outputs/hmm_final/classifier_to_hmm_transitions.csv`: Observation-level transition matrix (corrections vs. regressions).
- **Interactive Folium HTML Maps:**
  - [outputs/maps/trip_00000000.html](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/outputs/maps/trip_00000000.html): Interactive map with Folium LayerControl showing raw GPS, road network, reference route, and matched route.
  - [outputs/maps/trip_00000001.html](file:///Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap/outputs/maps/trip_00000001.html): Interactive map for Trip 00000001.
- **Diagnostic Plots:**
  - `outputs/hmm_final/hmm_vs_rf_per_trip_delta.png`: Per-trip accuracy delta plot (78 positive wins, 2 regressions).
  - `outputs/hmm_final/per_trip_distribution_boxplot.png`: Distribution of Top-1 accuracy across all models.
  - `outputs/hmm_final/raw_vs_coverage_adjusted_bars.png`: Raw vs. coverage-adjusted accuracy comparison.

---

## 10. Running Tests

The test suite covers API endpoint contracts, HMM behavior, numerical stability, coordinate geometry, and performance regressions:
```bash
PYTHONPATH=src pytest tests/ -v
```
**Current Status:** All **45 unit and integration tests** pass cleanly with zero failures and zero warnings.

---

## 11. Known Limitations and Discussion

1. **Approximate Route-Aligned Ground Truth:**
   Reference routes in the dataset are provided as ordered sequences of arc IDs rather than pointwise GPS-to-arc timestamps. Target training labels were generated using monotonic dynamic programming projection, which can have slight ambiguity near dense overpasses or multi-level interchanges.
2. **Candidate Recall Limits:**
   The 50m spatial search radius yields a 97.51% global average candidate recall. However, if a vehicle travels through areas where GPS drift exceeds 50m (such as severe satellite multipath or unmapped road segments), the true road arc is absent from the candidate set. A sequence decoder cannot select an arc that was never generated.
3. **Severe GPS Drift Trajectories:**
   On **Trip 00000054**, severe hardware sensor drift caused GPS points to deviate >50m from the road network, restricting candidate recall to **10.56%** and raw Top-1 accuracy to **8.56%**. When evaluated only on observations where candidate arcs were present, the coverage-adjusted accuracy reached **81.07%**.
4. **Student-Testing Partition Label:**
   All student-test evaluations reported in `outputs/hmm_final/student_test_isolated_evaluation.csv` (e.g., 98.09% Top-1 on Trip `00000052`) reflect development evaluation on the provided student-testing sample trips. They are strictly separate from any blind evaluator test sets used during hackathon judging.
