# PathSnap: Map-Matching & Route Reconstruction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-Academic-informational.svg)](LICENSE)

> **Group 14 (PathSnap)** — University Machine Learning Hackathon Project.  
> An end-to-end pipeline that takes noisy, real-world vehicle GPS trajectories, snaps them to directed road networks, and reconstructs continuous, physically drivable routes.

---

## 1. What is PathSnap?

Anyone who has used navigation apps in dense downtown areas has seen their GPS dot wander across city blocks, jump across highway dividers, or claim they are driving the wrong way down a one-way street. 

In the real world, consumer GPS chips face constant interference:
- **Urban canyons:** Signals bounce off skyscrapers (multipath reflection).
- **Sensor drift & jitter:** Low-cost hardware, satellite geometry, and clock drift degrade precision.
- **Topological blind spots:** GPS has no built-in knowledge of one-way streets, median barriers, or turn restrictions.

**PathSnap** tackles this challenge by combining modern machine learning with temporal sequence decoding:

1. **Spatial Candidate Scoring (Stage 1):** For every GPS point, we search a 50-meter radius for nearby road segments. A trained classifier evaluates 14 geometric and kinematic features (such as road distance, heading difference, and vehicle speed) to estimate how likely each road candidate is.
2. **Sequential Route Decoding (Stage 2):** Individual points do not tell the whole story—driving is sequential. We feed calibrated classifier probabilities and 4 transition features into a Hidden Markov Model (HMM) solved with the Viterbi algorithm. The HMM enforces physical road connectivity, one-way constraints, and turn plausibility to reconstruct the single most likely driving path from start to finish.

---

## 2. System Architecture & How It Works

Here is how a raw trajectory flows through the PathSnap pipeline:

```
Raw GPS Pings (timestamp, latitude, longitude)
         │
         ▼
Local UTM Projection (converts coordinates to a metric plane for accurate distance math)
         │
         ▼
Spatial Candidate Search (STRtree spatial query finds all road segments within 50m)
         │
         ▼
14 Observation Features (road proximity, heading alignment, speed, candidate rankings)
         │
         ▼
Calibrated Classifier Probabilities (Random Forest + Platt Sigmoid Calibration)
         │
         ▼
4 Road-Network Transition Features (shortest-path road distance vs. GPS displacement, turn angles)
         │
         ▼
HMM / Viterbi Sequence Decoder (evaluates connected road paths, handles time gaps, tracks arc progress)
         │
         ▼
Reconstructed Route (clean, unbroken sequence of directed road arcs)
```

---

### The 18-Feature Breakdown

Rather than relying purely on simple distance thresholds, PathSnap evaluates each candidate using 18 distinct features across two operational levels:

#### Observation-Level Features (14 Features — Classifier Emissions)
These features tell us how well an individual GPS ping matches a candidate road segment at that exact moment:

1. **`dist_to_arc`**: Perpendicular distance from the GPS point to the road centerline (meters).
2. **`dist_to_from_node`**: Distance to the road segment's start node (meters).
3. **`dist_to_to_node`**: Distance to the road segment's end node (meters).
4. **`proj_ratio`**: How far along the segment the vehicle projects, scaled from $0.0$ (start) to $1.0$ (end).
5. **`gps_heading`**: Direction of travel between consecutive GPS pings ($0^\circ$ to $360^\circ$).
6. **`arc_bearing`**: The legal travel direction of the road segment ($0^\circ$ to $360^\circ$).
7. **`heading_diff`**: Angular difference between vehicle travel heading and road bearing ($0^\circ$ to $180^\circ$).
8. **`is_reverse_direction`**: Flag ($1$ or $0$) indicating if the vehicle appears to travel against the legal flow (heading difference $> 90^\circ$).
9. **`speed_est`**: Estimated instantaneous speed between successive pings (m/s).
10. **`time_gap`**: Elapsed time $\Delta t$ between consecutive pings (seconds).
11. **`arc_length`**: Total physical length of the candidate road segment (meters).
12. **`candidate_rank`**: Proximity rank of this candidate among all segments within the 50m search radius ($1$ = closest).
13. **`candidate_count`**: Total number of candidate segments competing in the 50m neighborhood.
14. **`dist_to_alternative`**: Proximity margin between the closest candidate and the runner-up candidate (helps gauge ambiguity).

#### Transition-Level Features (4 Features — HMM Sequential Graph Transitions)
These features check whether moving from candidate road segment $A$ at time $t-1$ to candidate road segment $B$ at time $t$ makes physical sense:

15. **`gps_displacement_m`**: Straight-line distance between successive GPS pings on the metric UTM plane.
16. **`network_transition_distance_m`**: True directed shortest-path distance along the road network from candidate $A$ to candidate $B$, including partial arc traversal progress tracked via `proj_ratio`.
17. **`transition_distance_residual_m`**: The absolute difference $|d_{\text{net}} - d_{\text{gps}}|$. If this value is large, the vehicle would have had to teleport or speed impossibly to make that transition.
18. **`turn_angle_change_deg`**: Heading change between consecutive road arcs ($0^\circ$ to $180^\circ$). Strongly penalizes unnatural hairpin maneuvers and impossible instant U-turns.

---

## 3. Benchmark Results & Key Findings

We evaluated five standalone classification algorithms and two HMM sequence decoders across **80 training trips** (208,417 total observations) using strict 5-fold `GroupKFold` cross-validation. Trips were kept completely intact within their respective folds to prevent data leakage.

### Performance Summary

| Architecture / Model | Mean Per-Trip Top-1 | Median Per-Trip Top-1 | Pooled Observation Top-1 | Route Continuity | Candidate Recall Ceiling | Coverage-Adjusted Top-1 | Avg Latency / Trip |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Improved HMM (18-feat)** | **92.56%** | **96.23%** | **84.72%** | **97.58%** | **97.51%** | **94.63%** | **52.6 ms** |
| Original HMM (14-feat) | 91.50% | 95.47% | 83.87% | 98.03% | 97.51% | 93.55% | 48.2 ms |
| **Random Forest** | 81.80% | 84.75% | 73.95% | 90.45% | 97.51% | 83.68% | 22.4 ms |
| **HistGradientBoosting** | 81.56% | 84.30% | 74.50% | 90.22% | 97.51% | 83.46% | 14.8 ms |
| **Decision Tree** | 80.08% | 82.53% | 73.21% | 88.34% | 97.51% | 82.01% | 1.1 ms |
| **Logistic Regression** | 79.21% | 81.85% | 70.24% | 88.10% | 97.51% | 80.78% | 1.5 ms |
| **Gaussian Naive Bayes** | 75.61% | 77.98% | 66.42% | 88.42% | 97.51% | 77.09% | 0.9 ms |

---

### Reading the Numbers: Why Two Accuracy Metrics?

- **Unweighted Mean Per-Trip Top-1 (92.56%):**  
  Each trip is scored individually, and all 80 trips are weighted equally. This reflects how reliably the system reconstructs normal everyday trips without letting a single very long trip skew the overall picture.
- **Pooled Observation Top-1 (84.72%):**  
  All 208,417 observations are lumped together into one calculation. The score drops because of **Trip 00000054**, a notorious outlier containing **19,313 points (~9.3% of the entire dataset)**. On this trip, the hardware GPS drifted more than 50 meters into surrounding terrain, dropping candidate recall to 10.56% and raw accuracy to 8.56%. Because Trip 54 accounts for nearly a tenth of all datapoints, pooled metrics are pulled down heavily by this single hardware failure.
- **Head-to-Head Win Rate:**  
  The 18-feature HMM beat the best standalone classifier (Random Forest) on **78 out of 80 trips (a 97.5% win rate)**, tied on 0 trips, and had negligible minor regressions on only two trips ($-0.57\%$ on Trip 5 and $-0.15\%$ on Trip 33).
- **Error Correction Ratio:**  
  Across the full dataset, the HMM corrected **27,191 raw classifier mistakes** while introducing only **4,744 regressions**, achieving an impressive **5.73 to 1 net correction ratio**.

---

## 4. Repository Structure

```text
G14-PathSnap/
├── src/pathsnap/                 # Core Python package
│   ├── __init__.py
│   ├── io.py                     # Dataset loaders and whitespace-delimited file parsers
│   ├── geo.py                    # Automatic UTM projection selection per trip
│   ├── graph.py                  # Directed road network builder and Dijkstra routing
│   ├── candidates.py             # Spatial indexing (STRtree) and 50m radius search
│   ├── labels.py                 # Ground-truth alignment using dynamic programming
│   ├── features.py               # 18-feature calculation (observation + transition)
│   ├── calibration.py            # Platt scaling (sigmoid) probability calibrator & ECE metrics
│   ├── hmm.py                    # HMMRouteDecoder, log-domain Viterbi solver, backtracking
│   ├── visualize.py              # Interactive Folium map generation and plotting helpers
│   ├── metrics.py                # Top-1 accuracy, route continuity, and recall metrics
│   └── api.py                    # FastAPI server supporting Stage 5 and REST endpoints
├── frontend/                     # Modern interactive web interface (Vite + Leaflet)
│   ├── index.html                # Single-page application entrypoint
│   ├── package.json              # Web app dependencies
│   ├── vite.config.js            # Vite development server configuration
│   └── src/
│       ├── main.js               # Frontend application logic, Leaflet map renderer, API client
│       └── style.css              # Custom dashboard styling
├── ui/
│   └── index.html                # Lightweight, self-contained HTML5 Canvas Evaluator (runs offline)
├── notebooks/                    # Jupyter notebooks for data analysis and step-by-step exploration
│   ├── 01_eda.ipynb              # Exploratory data analysis of trips and networks
│   ├── 02_preprocessing.ipynb    # Projections and coordinate sanity checks
│   ├── 03_candidate_generation.ipynb # Candidate radius sweeps and recall analysis
│   └── 04_feature_engineering.ipynb  # Feature distributions and correlation analysis
├── scripts/                      # Standalone CLI tools and pipeline runners
│   ├── start_backend.py          # Starts FastAPI with automatic local IP/LAN discovery
│   ├── train_phase2.py           # Classifier training, hyperparameter tuning, and HMM fitting
│   ├── refine_phase2.py          # 18-feature extraction and ablation experiments
│   ├── evaluate_superiority.py   # Benchmark runner across all 7 models
│   ├── generate_final_hmm_evaluation.py # Generates final comparison CSVs and diagnostic plots
│   ├── build_features.py         # Builds feature datasets from raw trips
│   ├── run_eda.py                # CLI runner for exploratory data analysis
│   ├── run_preprocessing.py      # CLI runner for coordinate projections
│   ├── train_models.py           # Quick training for baseline models
│   ├── smoke_test_models.py      # Sanity check script for saved model artifacts
│   └── generate_notebooks.py     # Re-synchronizes notebook contents
├── outputs/                      # Experimental artifacts, charts, and generated maps
│   ├── csv/                      # Dataset audits, radius sweeps, and fold outputs
│   ├── hmm_final/                # Final benchmark CSVs and comparative tables
│   ├── maps/                     # Rendered interactive Folium HTML maps
│   └── plots/                    # Diagnostic PNG plots (histograms, boxplots, ROC curves)
├── reports/                      # Detailed technical documentation and experiment writeups
│   ├── HMM_FINAL_PERFORMANCE.md  # Complete performance audit & viva defense guide
│   ├── model_comparison.md       # Comparative study of all 5 classifiers
│   ├── model_training_report.md  # Training configurations and hyperparameters
│   ├── hmm_evaluation_report.md  # Detailed HMM sequence decoding analysis
│   ├── feature_engineering/
│   │   └── 18_FEATURES.md        # Mathematical formulation of all 18 features
│   └── reliability/
│       ├── calibration_report.md # Probability calibration (ECE and Brier scores)
│       └── overfitting_analysis.md # Generalization gap and cross-fold stability
├── tests/                        # Comprehensive automated test suite (45 unit tests)
│   ├── test_api.py               # API endpoint tests and Stage 5 contract validation
│   ├── test_final_hmm.py         # HMM benchmark regressions and superiority checks
│   ├── test_hmm_behavior.py      # Transition penalties, gap resets, and Viterbi behavior
│   ├── test_phase2.py            # GroupKFold fold isolation and calibration checks
│   ├── test_pipeline.py          # Projection, STRtree spatial search, and geometry tests
│   └── test_refinement.py        # 18-feature extraction and transition math tests
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # You are here!
```

---

## 5. Quick Start & Installation

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** *(optional, only needed if using the Vite/Leaflet frontend)*

### Step 1: Clone and Set Up Environment
```bash
# Clone the repository
git clone https://github.com/2306apk/G14-PathSnap.git
cd G14-PathSnap

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

### Step 2: Configure Dataset Path
The pipeline automatically looks for data in `data/raw/` or a folder named `Student folder` located in the project or its parent directories. 

If your dataset is located elsewhere, point to it using an environment variable:
```bash
# On Linux/macOS:
export PATHSNAP_DATA_DIR="/path/to/Student folder"

# On Windows (PowerShell):
$env:PATHSNAP_DATA_DIR="C:\path\to\Student folder"
```

---

## 6. Running the Application

### Option A: Start the Backend API Server
Launch the FastAPI backend with network discovery enabled:
```bash
python scripts/start_backend.py --host 0.0.0.0 --port 8000
```

When started, it outputs both local and network URLs:
```text
============================================================================
🚀  PathSnap Map-Matching Backend API Starting...
============================================================================
• Local URL         : http://127.0.0.1:8000
• LAN URL           : http://192.168.1.100:8000
• Stage 5 UI        : http://192.168.1.100:8000/ui
• Interactive Docs  : http://192.168.1.100:8000/docs
============================================================================
```

You can view the interactive Swagger API documentation directly at `http://localhost:8000/docs`.

---

### Option B: Interactive Frontends

PathSnap includes two frontends to suit different demonstration needs:

1. **Lightweight Canvas UI (`/ui`):**  
   - Built directly into FastAPI; simply open `http://localhost:8000/ui` in your browser.
   - Built with pure HTML5 Canvas and zero external dependencies—works completely offline without needing internet tile servers.
   - Displays raw GPS points (blue), candidate roads (gray), and the reconstructed route (coral red), with live accuracy metrics.

2. **Modern Leaflet Frontend (`frontend/`):**  
   - An interactive web application built with Vite and Leaflet.
   - Features rich zoom/pan, street map layers, and interactive model switching to compare the HMM output against individual classifiers.
   - To launch:
     ```bash
     cd frontend
     npm install
     npm run dev
     ```

---

## 7. Two-Laptop Live Demo Setup

For hackathon judging and live demonstrations, PathSnap is pre-configured to run seamlessly across two machines on the same local Wi-Fi or mobile hotspot:

```text
┌────────────────────────────────┐         Local Wi-Fi / Hotspot         ┌────────────────────────────────┐
│       Laptop 1 (Backend)       │ <───────────────────────────────────> │       Laptop 2 (Client)        │
│                                │                                       │                                │
│  - Host: 0.0.0.0:8000          │        HTTP / REST API Requests       │  - Browser (Chrome, Edge, etc) │
│  - Example IP: 192.168.1.100   │   (CORS enabled for cross-origin LAN) │  - URL: http://<LAN_IP>:8000/ui│
│  - ML Models & HMM Engine      │                                       │  - Interactive Evaluation UI   │
└────────────────────────────────┘                                       └────────────────────────────────┘
```

### Steps to Demo:
1. **On Laptop 1 (Server):**
   - Connect to Wi-Fi or phone hotspot.
   - Run: `python scripts/start_backend.py --host 0.0.0.0 --port 8000`.
   - Note the printed LAN IP (for example, `192.168.1.100`).
2. **On Laptop 2 (Client):**
   - Connect to the same Wi-Fi or hotspot.
   - Open a browser and visit: `http://192.168.1.100:8000/ui`.
   - Click **Connect & load** to fetch available trips from the server.
   - Pick a trip (such as `00000052`) and click **Reconstruct trip**.
   - The matched route, road network, and performance metrics will render live on Laptop 2!

*(Note: Permissive CORS `allow_origins=["*"]` is enabled by default, ensuring smooth communication across local devices without browser security blocks).*

---

## 8. Reports, Documentation & Interactive Maps

Detailed experimental reports and visual outputs are available in the repository:

- **Technical Reports:**
  - [reports/HMM_FINAL_PERFORMANCE.md](reports/HMM_FINAL_PERFORMANCE.md): Comprehensive validation audit, error analysis, and viva defense guide.
  - [reports/feature_engineering/18_FEATURES.md](reports/feature_engineering/18_FEATURES.md): Formal definitions and formulas for all 18 features.
  - [reports/model_comparison.md](reports/model_comparison.md): In-depth comparison across all five ML models.
  - [reports/model_training_report.md](reports/model_training_report.md): Training methodology, loss functions, and hyperparameters.
- **Evaluation Tables:**
  - `outputs/hmm_final/final_models_comparison.csv`: Summary metrics across all seven models.
  - `outputs/hmm_final/final_per_trip_comparison.csv`: Trip-by-trip accuracy and continuity for all 80 training trips.
  - `outputs/hmm_final/student_test_isolated_evaluation.csv`: Independent evaluation on student-test development trips.
- **Diagnostic Visualizations:**
  - `outputs/hmm_final/hmm_vs_rf_per_trip_delta.png`: Per-trip accuracy improvements (78 wins, 2 minor regressions).
  - `outputs/hmm_final/per_trip_distribution_boxplot.png`: Top-1 accuracy distributions across models.
  - `outputs/hmm_final/raw_vs_coverage_adjusted_bars.png`: Raw vs. coverage-adjusted accuracy comparison.
- **Interactive Maps:**
  - [outputs/maps/trip_00000000.html](outputs/maps/trip_00000000.html): Interactive Folium map with layer controls showing raw GPS, road network, and reconstructed path.
  - [outputs/maps/trip_00000001.html](outputs/maps/trip_00000001.html): Interactive map for Trip 00000001.

---

## 9. Running Tests

The test suite validates API contracts, HMM numerical stability, spatial indexing, coordinate conversions, and regression thresholds:
```bash
python -m pytest tests/ -v
```

---

## 10. Real-World Limitations & Discussion

1. **Approximate Route Labels:** Reference routes are given as ordered sequences of road arc IDs rather than exact timestamped point-to-arc matches. Ground-truth labels were generated using monotonic dynamic programming, which can introduce subtle ambiguities near complex multi-level highway interchanges.
2. **The 50-Meter Candidate Search Radius:** A 50-meter search radius achieves a strong 97.51% candidate recall across the dataset while keeping computation fast. However, if GPS drifts farther than 50 meters off-road, the true segment will not be included in the candidate pool. A sequence decoder cannot pick an arc that does not exist in its candidate set.
3. **Extreme Sensor Drift (Trip 54):** On Trip `00000054`, hardware sensor failure led to GPS points drifting far into unpaved terrain (>50m off-road), limiting candidate recall to 10.56% and raw accuracy to 8.56%. When evaluated solely on the observations where the true road was within the 50m search radius, the coverage-adjusted accuracy reached **81.07%**.
4. **Student Testing Partition:** The student-test numbers reported in `outputs/hmm_final/student_test_isolated_evaluation.csv` (e.g. 98.09% Top-1 on Trip `00000052`) reflect evaluation on the sample student testing split provided for development, kept strictly separate from the training folds.
