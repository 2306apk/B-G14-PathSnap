# One consolidated PDF — technical report checklist (NOT a fabricated report)

The event requires two final deliverables: a public GitHub repository and **one** PDF report per team. The content below is a fill-in checklist; replace it with original, verified team discussion.

## Cover
- Project title PathSnap, course and team names/roll numbers, GitHub URL, date, environment and Python versions.
- Explicit note of permitted AI assistance according to the event's instructions.

## 2.2 Dataset understanding and EDA
- Prove 80 training / 10 student-testing trips, inventory CSV and missing-value statistics.
- Explain coordinate order `lon,lat`, timestamps, node IDs, directed arc IDs, route arc IDs.
- Describe missing / irregular GPS gaps, basic network connectivity and local metre projection.
- Show a real GPS plot, network plot, and at least one **Folium map** of a real trip.
- State limitations of approximate GPS-to-route label alignment and why candidate misses occur.

## 3. Model experimentation
For **each of the five trained models**:
- Suitability for PathSnap; measured strengths and weaknesses, not textbook-only prose.
- Important initial hyperparameters, training objective, class balancing, feature preprocessing.
- Mean and standard deviation across 3 *trip-grouped* CV folds.
- Student-testing Top-1 accuracy, binary-candidate Macro F1, log loss and route continuity.
- Training/inference time, overfitting/underfitting evidence from train CV vs student testing.
- Actual metric comparison plots. Do **not** invent epoch loss curves for models that do not train in epochs.

**Evidence:** `results/full/model_comparison.csv`, `cv_*.csv`, plots, `run_manifest.json`.

## 4. Bayesian optimisation
- Selected model and rationale based on training CV criterion.
- TPE search distributions; number of trials from CSV, best params, objective and fold selection.
- Untuned vs tuned CV Top-1, student-test Top-1, binary macro F1, log loss, inference time.
- Discuss whether tuning genuinely helped, on what measure, and what it cost.

**Evidence:** `results/full/optuna_trials.csv`, `tuning_summary.json`, SQLite study DB.

## 5. HMM and Viterbi
- Define hidden state (candidate directed arc ID), emission (normalised ML probability), transition (bounded graph path/travel plausibility), initial score, Viterbi max-product in log space.
- Explain stationary points, missing candidates, invalid transitions, gaps and graph disconnects.
- Compare classifier-only and HMM observation-level accuracy/continuity **per trip** and weighted total.
- Include at least one failure case, explain what happened instead of claiming universal improvement.

**Evidence:** `results/full/decoder_comparison.csv`.

## 6. Geographical visualisation
- Actual generated Folium HTML, with GPS blue, predicted route red, reference green (student data only), roads grey.
- Save a **real browser screenshot** of that Folium map and place it inside the PDF.
- Do not show the evaluator reference routes on the client/UI.

## 7. API / UI / deployment
- Architecture and exact commands on Laptop 1 / Laptop 2, their LAN IPs (no credentials).
- Real screenshots of working UI and Laptop 1 terminal logs showing API requests and processing.
- Demonstrate at least all 10 available student-testing trips before evaluator test; validate the 10 reserved trips only when organisers provide them.
- Show API responses and explicitly note they contain aggregated scores but no reference arc IDs.

## Final audit
- Every table/plot/screenshot can be regenerated using committed source and files.
- Include the source limitations: proxy aligned labels, candidate misses, feature choices, route continuity definition, possible HMM failure, model training stride, known runtime on real hardware.
- Each member can defend the code, results, and trade-offs.
- GitHub repo public with source code, model config/weights if size permits, all experiment CSVs and map HTML.
- Submit one final PDF (not separate reports for each model).
