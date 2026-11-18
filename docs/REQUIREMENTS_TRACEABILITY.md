# PathSnap requirements traceability (five provided PDFs)

This is a technical audit of the supplied instructions, not a certification of official evaluator performance or academic originality.

| Requirement from source | Component / evidence | Status and remaining risk |
|---|---|---|
| Main Guidelines: reconstruct vehicle path from noisy GPS and road network | `pathsnap/candidates.py`, `experiments.py`, `pipeline.py`, `decoder.py` | Implemented on permitted student data; geographic and route-alignment assumptions require team review |
| Main Guidelines: Laptop 1 hosts model, network, API; Laptop 2 hosts UI | `pathsnap/api.py`, `ui/index.html`, `README.md` | Local HTTP API tested; physical two-Linux-laptop LAN session still required |
| Dataset Exploration: inspect 80 train, 10 student test, 10 reserved evaluator trips | `scripts/inventory.py`, `results/full/inventory.csv` | 80+10 real supplied trips inventoried; reserved 10 are not present |
| Dataset Exploration: `.track` lon/lat/time, `.nodes` lon/lat, `.arcs` directed IDs, `.route` ordered arc IDs | `pathsnap/data.py`, `labels.py` | Parsed and validated; route sequence is **not** one label per GPS point |
| Dataset Exploration: coordinate plausibility, metre distances and GPS gaps | `data.py`, `candidates.py`, `inventory.csv` | Implemented local metres projection and gap metrics; no dedicated CRS for continent-spanning trips |
| Dataset Exploration: verify route/node connectivity and candidate misses | `inventory.csv`, `per_trip_comparison.csv` | Route adjacency and candidate misses measured; failure analysis remains a team responsibility |
| Model Guidelines: train at least five distinct ML models | `experiments.py`, `train_models.py`, `artifacts/full/*.joblib` | Five baseline models trained on all 80 trips with stride 12 |
| Model Guidelines: group-aware cross-validation, never mix trip observations | `GroupKFold` inside `experiments.cv_score` | Three-fold group CV executed |
| Model Guidelines: student-testing comparison, Macro F1, log loss, Top-1, inference time | `results/full/model_comparison.csv`, `cv_*.csv`, `per_trip_comparison.csv` | Measured; Macro F1 explicitly binary candidate class metric; Top-1 uses proxy labels |
| Model Guidelines: Bayes hyperparameter optimisation and retained search trials | `scripts/tune.py`, `optuna_trials.csv`, `optuna_study.sqlite3`, `tuning_summary.json` | 15 Optuna TPE trials run on training-only CV; reproducible study saved |
| Model Guidelines: HMM/Viterbi, emissions, transitions, invalid paths, GPS gaps | `pathsnap/decoder.py`, `pipeline.py`, `results/full/decoder_comparison.csv` | Full student-test evaluation completed; graph shortest path bounded, resets explicitly applied |
| Deliverables: Folium interactive map with GPS, network, reference (if allowed), prediction | `scripts/make_map.py`, `results/full/folium_trip_00000052.html` | Generated from permitted student data; final PDF needs team's actual browser screenshot |
| Deliverables: API/UI and console logs during evaluation | `api.py`, `ui/index.html` | TestClient HTTP checked; physical two-laptop demo, screenshot and console logs must be captured by team |
| Deliverables: public GitHub source, training code, artifacts, CSV results | Complete project tree, `README.md` | Project ready to commit; user/team must create repository and push their verified work |
| Deliverables: ONE consolidated technical report PDF per team | `docs/REPORT_CHECKLIST.md`, `docs/IMPLEMENTATION_GUIDE.pdf` | Implementation guide is **not** a replacement for team-authored final report |
| Policy: strict model/code independence, no fabricated results, AI disclosure | README and guide state provenance and limitations | Team must comply with organiser's authorship and originality rules; code assistance must be disclosed as required |
| Policy: evaluator labels never influence predictions or appear in UI | `api.run_trip` runs `pipeline.predict` with `include_reference=False`, reads route only afterward for aggregate scoring | Local API test checked absent ground-truth fields; reserved evaluator access not yet possible |

## Remaining acceptance tests

1. Cross-check the organiser's reserved-evaluator directory layout and exact evaluation format when issued.
2. Run both laptops on the same trusted LAN with actual selected test trip and show API `PREDICT_REQUEST` / `INFERENCE` logs.
3. Reproduce the full ML + Optuna experiments on your Linux machine and capture hardware/version details.
4. Perform in-person viva explaining label alignment, feature rationale, class imbalance, Viterbi, reset penalties, and model selection.
5. Write one original final team PDF with genuinely captured Folium and deployment screenshots; do not submit only the implementation guide.
