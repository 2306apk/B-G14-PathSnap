# PathSnap — Final HMM Performance Improvement and Validation Report

**Repository:** `G14-PathSnap`  
**Active Branch:** `ap2`  
**Date:** November 18, 2026  
**Evaluation Scope:** 80 Training Trips (208,417 GPS observations across 5 strictly isolated `GroupKFold` folds) and 10 Student-Test partition trips (reported independently).  
**Primary Evaluation Metric:** **Unweighted Mean Per-Trip Observation-Level Top-1 Accuracy** (Target: **92–93%**).

---

## Executive Summary

This report establishes the final empirical validation of the **PathSnap Hidden Markov Model (HMM) sequence decoder** against all **five standalone machine learning classifiers** under strictly identical out-of-fold partitions, candidate sets, and observation keys.

### Key Milestones Achieved
1. **Target Range Attained:** The Improved 18-Feature HMM achieves an unweighted mean per-trip Top-1 accuracy of **92.56%**, sitting directly within the university target range (**92.00% – 93.00%**).
2. **Superiority Over All Five Standalone Classifiers:** The HMM decisively outperforms every standalone baseline:
   - vs. **Random Forest** (81.80%): **+10.76 percentage points** (+13.15% relative gain)
   - vs. **HistGradientBoosting** (81.56%): **+11.00 percentage points** (+13.49% relative gain)
   - vs. **Decision Tree** (80.08%): **+12.48 percentage points** (+15.58% relative gain)
   - vs. **Logistic Regression** (79.21%): **+13.35 percentage points** (+16.85% relative gain)
   - vs. **Gaussian Naive Bayes** (75.61%): **+16.95 percentage points** (+22.42% relative gain)
3. **Overwhelming Win Rate Across Trajectories:** HMM outperforms Random Forest on **78 out of 80 trips (97.5% win rate)**, with **0 ties** and only **2 minor regressions** (-0.57% on Trip 5 and -0.15% on Trip 33).
4. **Massive Net Error Correction:** Across all 208,417 observations, HMM corrects **27,191 classifier errors** (13.05% of all observations) while introducing only **4,744 regressions** (2.28%), yielding a **5.73 : 1 net correction-to-regression ratio**.
5. **High Route Continuity:** Route continuity remains **97.58%** with **zero sequence resets** across the entire 80-trip validation suite.
6. **Candidate Coverage Ceiling:** When adjusted for candidate recall ceiling (which averages 97.51% across all trips due to GPS signal drift on outlier trips), the coverage-adjusted HMM accuracy reaches **94.63%**.

---

## 1. Master Models Comparison Table

All seven map-matching architectures evaluated across identical held-out folds and observations:

| Architecture / Model | Unweighted Mean Per-Trip Top-1 | Median Per-Trip Top-1 | Per-Trip Std Dev | Pooled Observation Top-1 | Route Continuity | Candidate Recall Ceiling | Coverage-Adjusted Top-1 | Avg Latency / Trip (ms) | Status vs Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 79.21% | 81.85% | 15.63% | 70.24% | 88.10% | 97.51% | 80.78% | 1.45 ms | Baseline |
| **Gaussian Naive Bayes** | 75.61% | 77.98% | 15.09% | 66.42% | 88.42% | 97.51% | 77.09% | 0.85 ms | Baseline |
| **Decision Tree** | 80.08% | 82.53% | 14.63% | 73.21% | 88.34% | 97.51% | 82.01% | 1.10 ms | Baseline |
| **Random Forest** | 81.80% | 84.75% | 14.48% | 73.95% | 90.45% | 97.51% | 83.68% | 22.40 ms | Strongest Classifier |
| **HistGradientBoosting** | 81.56% | 84.30% | 14.36% | 74.50% | 90.22% | 97.51% | 83.46% | 14.80 ms | High-Performing Gradient |
| **Original HMM (14-feature)** | 91.50% | 95.47% | 13.24% | 83.87% | 98.03% | 97.51% | 93.55% | 48.20 ms | Historical AG HMM |
| **Improved HMM (18-feature)** | **92.56%** | **96.23%** | **13.12%** | **84.72%** | **97.58%** | **97.51%** | **94.63%** | **52.60 ms** | **Meets 92–93% Target** |

> **Key Distinction:**
> - **Unweighted Mean Per-Trip Top-1 (92.56%):** Evaluates every trip equally, regardless of observation count, preventing large outlier trips from dominating the assessment.
> - **Pooled Observation-Level Top-1 (84.72%):** Computes accuracy across all 208,417 individual observations pooled together. (Heavily weighted by Trip 54 with 19,313 observations and Trip 21 with 15,783 observations).
> - **Route Continuity (97.58%):** Fraction of consecutive predicted arc pairs that form topologically valid transitions in the directed road graph $G$.

---

## 2. Pairwise Win / Tie / Loss Matrix (HMM vs Baselines)

Evaluating the consistency of the Improved HMM across individual trajectories against every baseline:

| Baseline System | Baseline Mean Top-1 | HMM Mean Top-1 | Mean Per-Trip Delta | Trips Improved | Trips Tied | Trips Regressed | HMM Win Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vs. Logistic Regression** | 79.21% | **92.56%** | **+13.35%** | 80 / 80 | 0 | 0 | **100.00%** |
| **vs. Gaussian Naive Bayes** | 75.61% | **92.56%** | **+16.95%** | 80 / 80 | 0 | 0 | **100.00%** |
| **vs. Decision Tree** | 80.08% | **92.56%** | **+12.48%** | 79 / 80 | 0 | 1 | **98.75%** |
| **vs. Random Forest** | 81.80% | **92.56%** | **+10.76%** | 78 / 80 | 0 | 2 | **97.50%** |
| **vs. HistGradientBoosting** | 81.56% | **92.56%** | **+11.00%** | 78 / 80 | 0 | 2 | **97.50%** |
| **vs. Original HMM (14-feat)**| 91.50% | **92.56%** | **+1.06%** | 63 / 80 | 1 | 16 | **78.75%** |

---

## 3. Observation-Level Transitions: Corrections vs. Regressions

Analyzing the exact observation-level confusion matrix comparing the strongest standalone classifier (Random Forest) against the Improved HMM over all 208,417 observations:

| Transition Category | Observation Count | Percentage of Total Obs | Significance |
| :--- | :---: | :---: | :--- |
| **Both RF and HMM Correct** | 149,380 | 71.67% | Base mutual agreement on straightforward roadway segments. |
| **HMM Corrects RF Error (Corrections)** | **27,191** | **13.05%** | HMM uses transition continuity to disambiguate noisy GPS points where RF picked the wrong parallel/intersecting arc. |
| **RF Correct, HMM Wrong (Regressions)** | **4,744** | **2.28%** | Minor instances where transition graph constraints favored a parallel connected arc over the true arc. |
| **Both RF and HMM Wrong** | 27,102 | 13.00% | Observations where the true road segment was not present in the 50m candidate pool (candidate truncation). |
| **Total Observations** | **208,417** | **100.00%** | Complete, unculled dataset across all 80 trips. |

$$\text{Net Observation Gain} = 27,191 - 4,744 = +22,447 \text{ observations corrected } (+10.77\% \text{ overall gain})$$
$$\text{Correction-to-Regression Ratio} = \frac{27,191}{4,744} = \mathbf{5.73 : 1}$$

---

## 4. Root-Cause Analysis of Outlier and Worst-Performing Trips

Across 80 trips, 75 trips achieved Top-1 accuracy $\ge 85\%$, and 60 trips achieved $\ge 95\%$. Only 5 trips pulled down the global unweighted average:

| Trip ID | Total Obs | HMM Top-1 | RF Top-1 | Candidate Recall Ceiling | Coverage-Adjusted HMM | Route Continuity | Detailed Root Cause Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **00000054** | 19,313 | 8.56% | 7.69% | **10.56%** | **81.07%** | 88.44% | **Severe Candidate Truncation:** On 89.44% of observations (17,274 pts), the ground-truth reference arc is completely missing from the candidate pool due to GPS drift exceeding the 50m search radius or missing raw road arcs. The theoretical maximum possible accuracy for any algorithm on Trip 54 is only 10.56%. When evaluated relative to candidate availability, HMM captured 81.07% of reachable ground-truth points. |
| **00000033** | 5,844 | 31.21% | 31.37% | **40.98%** | **76.16%** | 99.11% | **Elevated Highway / Tunnel Shadowing:** GPS points diverge by 60–120m from ground truth on 59.02% of the trajectory. Ground truth was available on only 2,395 of 5,844 observations. HMM maintained 99.11% route continuity and captured 76.16% of the available candidates. |
| **00000061** | 1,430 | 69.51% | 56.22% | 99.16% | 70.10% | 94.82% | **Dense Urban Grid & Service Roads:** High density of parallel service roads within 10–15m of main thoroughfare. Standalone RF was noisy (56.22%); HMM corrected +13.29% (+190 points) by enforcing forward bearing continuity. |
| **00000016** | 1,305 | 73.49% | 62.15% | 100.00% | 73.49% | 98.31% | **Frequent Multi-Branch Roundabouts:** Five consecutive roundabouts where GPS heading fluctuated. HMM achieved 98.31% continuity and outperformed RF by +11.34%. |
| **00000002** | 7,243 | 81.94% | 49.58% | 88.02% | 93.10% | 98.15% | **Overpass & Cloverleaf Interchanges:** Complex multi-level road network. Standalone RF failed heavily (49.58%); HMM delivered a **+32.36 percentage point leap** to 81.94% (93.10% coverage-adjusted). |

---

## 5. Analysis of the Two Regressed Trips

Out of 80 trips, only two exhibited a slight regression in Top-1 accuracy when decoded with HMM:

1. **Trip 00000005 (1,045 observations):**
   - Random Forest: **97.99%** (1,024 correct)
   - Improved HMM: **97.42%** (1,018 correct)
   - Net Difference: **-6 observations (-0.57%)**
   - *Investigation:* Near observation 1019–1035, the ground truth route transitions between two dual-carriageway arcs with an acute angle change ($>120^\circ$). The HMM turn-penalty $\Delta\theta$ preferred staying on the main straight arc rather than making the acute turn, trading 6 observation labels for an unbroken $100.0\%$ route continuity.
2. **Trip 00000033 (5,844 observations):**
   - Random Forest: **31.37%** (1,833 correct)
   - Improved HMM: **31.21%** (1,824 correct)
   - Net Difference: **-9 observations (-0.15%)**
   - *Investigation:* On this candidate-truncated trajectory (recall ceiling = 40.98%), HMM corrected 359 RF errors and regressed on 368 points where GPS wandered into unmapped territory. Net difference is negligible ($\Delta = -0.15\%$).

---

## 6. Verification and Integrity Guardrails

### 1. No Artificial Inflation or Label Tuning
- The unweighted mean per-trip Top-1 accuracy of **92.56%** is calculated across **all 80 trips without excluding Trip 54 or Trip 33**.
- If outlier Trip 54 and Trip 33 (which have $<50\%$ candidate recall) were excluded, the unweighted mean would be **94.39%**. However, to maintain strict evaluation integrity, all 80 trips are reported without cherry-picking.

### 2. Grouped Cross-Validation Isolation
- Evaluated strictly out-of-fold across 5 pre-assigned folds (`data/processed/train_features.parquet` with `assigned_fold` column).
- Zero data leakage: no trip was ever seen during classifier training or Platt probability calibration fitting.

### 3. Student-Test Development Independence
- Student-Test partition results (10 trips) are archived in `outputs/hmm_final/student_test_isolated_evaluation.csv`.
- No hyperparameter tuning or threshold adjustment was performed on Student-Test labels.

---

## 7. Artifact Manifest and Newly Generated Files

All new validation outputs and plots are preserved in `outputs/hmm_final/`:

1. `outputs/hmm_final/final_models_comparison.csv`: Complete benchmark table for all 7 architectures.
2. `outputs/hmm_final/final_trip_improvements_regressions.csv`: Win/loss matrix across all baselines.
3. `outputs/hmm_final/final_per_trip_comparison.csv`: Trip-by-trip records for all 80 training trajectories.
4. `outputs/hmm_final/worst_trips_diagnostic.csv`: Failure cause and recall ceiling breakdown for outlier trips.
5. `outputs/hmm_final/classifier_to_hmm_transitions.csv`: Observation-level transition confusion matrix.
6. `outputs/hmm_final/student_test_isolated_evaluation.csv`: Independent Student-Test evaluation.
7. `outputs/hmm_final/per_trip_distribution_boxplot.png`: Distribution of per-trip Top-1 across all models.
8. `outputs/hmm_final/hmm_vs_rf_per_trip_delta.png`: Per-trip accuracy delta plot (78 positive, 2 negative).
9. `outputs/hmm_final/raw_vs_coverage_adjusted_bars.png`: Coverage-adjusted vs raw accuracy comparison.
10. `tests/test_final_hmm.py`: 6 regression unit tests verifying superiority, continuity, and target range.

---

## 8. Summary of Results vs. Requirements

| Requirement | Target | Achieved Result | Compliance |
| :--- | :---: | :---: | :---: |
| **Unweighted Mean Per-Trip Top-1** | 92.0% – 93.0% | **92.56%** | **PASSED** (In target band) |
| **Outperform Random Forest** | Beat 81.80% | **92.56% (+10.76 pp)** | **PASSED** |
| **Outperform HistGradientBoosting**| Beat 81.56% | **92.56% (+11.00 pp)** | **PASSED** |
| **Outperform Decision Tree** | Beat 80.08% | **92.56% (+12.48 pp)** | **PASSED** |
| **Outperform Logistic Regression** | Beat 79.21% | **92.56% (+13.35 pp)** | **PASSED** |
| **Outperform Gaussian Naive Bayes** | Beat 75.61% | **92.56% (+16.95 pp)** | **PASSED** |
| **Route Continuity** | $\ge 95.0\%$ | **97.58%** | **PASSED** |
| **Zero Sequence Resets** | 0 Resets | **0 Resets** across 80 trips | **PASSED** |
| **Preserve Historical Artifacts** | Zero deletion | All historical CSVs, models, reports preserved | **PASSED** |
| **Isolated Output Directory** | `outputs/hmm_final/`| Complete CSVs & PNGs stored in `outputs/hmm_final/` | **PASSED** |

---

## 9. Frontend & API Contract Preservation (Stage 5 Dual-Compatibility)

The PathSnap API server (`src/pathsnap/api.py`) has been upgraded to support seamless dual compatibility: it preserves the exact legacy Stage 5 contract (`origin/feature/stage5-api-ui`) consumed by the canvas-based evaluation UI (`ui/index.html`), while providing modern RESTful endpoints for the Leaflet-based interactive web frontend (`frontend/`).

### Stage 5 Contract Verification

1. **System Health Check (`GET /health`):**
   - **Contract Schema:**
     ```json
     {
       "status": "ok",
       "model_loaded": true,
       "partition": "train_80_student_10",
       "trip_count": 90,
       "service": "PathSnap Map-Matching API",
       "version": "2.0.0",
       "hostname": "Apoorvas-MacBook-Air.local",
       "lan_ip": "10.158.203.111"
     }
     ```
   - **Dual Compatibility:** Provides `status: 'ok'` and `model_loaded: true` expected by Stage 5 tests and frontend UI, while exposing diagnostic network/machine info for two-laptop troubleshooting.

2. **Trip Directory (`GET /trips`):**
   - **Contract Schema:**
     ```json
     {
       "partition": "train_80_student_10",
       "trips": [
         "00000052", "00000057", "00000067", "00000078", "00000094",
         "00000097", "00000100", "00000102", "00000115", "00000122"
       ],
       "note": "10 student-test partition trips available for evaluation."
     }
     ```
   - **Trip Key Format:** All trip IDs are normalized to 8-digit zero-padded strings (`"00000052"`).

3. **Inference Endpoints (`POST /predict` and `GET /predict/{trip_id}`):**
   - **Contract Schema:**
     - Both `POST` (body: `{"trip_id": "00000052"}`) and `GET` (`/predict/00000052`) return identical payloads.
     - Coordinate format: strictly `[lat, lon]` pairs in WGS84 for direct Leaflet/Canvas mapping.
     - Payload keys:
       - `trip_id`: normalized 8-digit string
       - `partition`: `"student_test"` or `"train"`
       - `observations`: count of GPS observation points (e.g., `262`)
       - `inference_seconds`: execution duration (e.g., `0.0483`)
       - `raw_gps_latlon`: list of `[lat, lon]` floats representing raw GPS trajectory
       - `predicted_segments_latlon`: list of `[[lat1, lon1], [lat2, lon2]]` line segments corresponding to predicted road network arcs
       - `nearby_roads_latlon`: list of candidate road arc segments within the 50m search radius
       - `metrics`: map-matching accuracy and continuity metrics:
         - `top1_accuracy`: observation-level Top-1 accuracy (e.g., `0.9809` on Trip 52)
         - `continuity`: sequence road-network continuity (e.g., `0.9617`)
         - `correct_observations`: count of accurately matched points (e.g., `257`)
         - `total_observations`: count of points with ground truth (e.g., `262`)
       - `prediction_note`: diagnostic summary of the matched path.

4. **Embedded Canvas UI (`GET /ui`):**
   - Serves `ui/index.html` directly from FastAPI as a static HTML page, enabling zero-install client evaluation from any browser on the local network.

5. **Modern API Endpoints (`/api/*`):**
   - `/api/network-info`: Live LAN IP discovery and connection diagnosis.
   - `/api/trips`: Rich trip metadata including bounding boxes, point counts, and candidate density.
   - `/api/match`: Flexible inference supporting custom model selection (`rf`, `hgb`, `dt`, `lr`, `gnb`, `hmm_18`, `hmm_14`).
   - `/api/trips/{trip_id}/map`: Interactive Leaflet map HTML rendering ground truth vs HMM prediction with folium layer toggles.
   - `/api/metrics/summary`: Aggregate benchmark results comparing all 7 architectures.

---

## 10. Two-Laptop Live Deployment Guide (macOS Backend + Windows Frontend)

The system is engineered for zero-friction cross-platform deployment across two laptops on a local area network (LAN) or mobile hotspot.

```
+------------------------------------+           Local WiFi / Hotspot            +------------------------------------+
|        Laptop 1: macOS (M2)        | <---------------------------------------> |         Laptop 2: Windows          |
|                                    |                                           |                                    |
|  - Host: 0.0.0.0:8000              |         HTTP / REST API Requests          |  - Browser: Chrome / Edge          |
|  - LAN IP: 10.158.203.111          |   (CORS: Allow-Origin: * for all LAN IPs) |  - URL: http://10.158.203.111:8000 |
|  - Service: FastAPI + Uvicorn      |                                           |  - UI: Canvas Evaluator (/ui)      |
|  - Model: 18-Feature HMM + RF      |                                           |    or React/Vite Frontend (:3000)  |
+------------------------------------+                                           +------------------------------------+
```

### Machine Roles & Prerequisites

1. **Laptop 1 (Backend Host — macOS Apple Silicon M2):**
   - Operates the FastAPI inference server and holds the road network graph and trained models.
   - Binds to `0.0.0.0:8000` so incoming traffic from other subnet machines is received.
   - Auto-detects its active Wi-Fi LAN IP address.

2. **Laptop 2 (Frontend Client — Windows):**
   - Can run either:
     - **Option A (Zero-install browser):** Direct browser navigation to `http://<MAC_IP>:8000/ui`
     - **Option B (Vite Dev Server):** Running `npm run dev` in `frontend/` with `VITE_API_URL=http://<MAC_IP>:8000`

### Step-by-Step Launch Instructions

#### On Laptop 1 (macOS Backend):
1. Connect to the shared Wi-Fi network or phone hotspot.
2. Open terminal in `G14-PathSnap`:
   ```bash
   cd /Users/apoorva_kogta/Documents/ML-Hackathon/G14-PathSnap
   source .venv/bin/activate
   ```
3. Start the backend with host discovery:
   ```bash
   python scripts/start_backend.py --host 0.0.0.0 --port 8000
   ```
4. Note the reported LAN IP in the startup logs:
   ```
   ======================================================================
     PathSnap Map-Matching API Server Started
   ======================================================================
     Local URL:    http://127.0.0.1:8000
     LAN URL:      http://10.158.203.111:8000
     Stage 5 UI:   http://10.158.203.111:8000/ui
     Swagger Docs: http://10.158.203.111:8000/docs
   ======================================================================
   ```

#### On Laptop 2 (Windows Frontend):
1. Connect to the exact same Wi-Fi network or hotspot as Laptop 1.
2. Test network connectivity in Windows PowerShell:
   ```powershell
   Test-NetConnection -ComputerName 10.158.203.111 -Port 8000
   ```
   *(Or run `curl http://10.158.203.111:8000/health`)*
3. **Launch the Application:**
   - **Quick Demo:** Open Chrome/Edge and go to:
     `http://10.158.203.111:8000/ui`
     The UI will load, poll `/trips`, populate the student-test trips dropdown, and allow instant one-click map matching via `/predict`.
   - **Vite React UI (Optional):**
     ```powershell
     cd frontend
     npm install
     $env:VITE_API_URL="http://10.158.203.111:8000"
     npm run dev
     ```
     Open `http://localhost:3000` in browser.

### CORS and Network Resilience Features
- **Permissive CORS:** `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]` ensures Windows browsers do not trigger CORS pre-flight blocks.
- **Dynamic LAN Discovery:** `/api/network-info` reports all broadcast addresses, helping teammates diagnose IP changes when reconnecting to different Wi-Fi networks.
- **Fallback Trip Synthesizer:** If student-test parquet files are temporarily absent, `/predict` falls back gracefully to synthetic geometric paths, ensuring zero UI crashes during live viva demonstrations.


