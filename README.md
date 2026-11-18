# PathSnap — Distributed Map Matching and Route Reconstruction

> **Course:** UE24CS352A Machine Learning | PES University, RR Campus
>
> **Status:** Working student implementation scaffold with five trained ML baselines, group-aware CV, Bayesian TPE search, road-network HMM/Viterbi, Folium, FastAPI and a separate offline-capable browser UI. Teams must independently validate every assumption, reproduce experiments, and meet the event's originality rules. **Do not represent the provided code or analysis as independently authored if it was generated with AI assistance.**

## 1. What the files contain

The uploaded `Student folder.zip` contains exactly **80** `train` and **10** `student_test` trip directories, each with `.track`, `.nodes`, `.arcs`, `.route`. The five supporting PDFs in `PathSnap_student_materials(1).zip` require a trained five-model comparison, Bayesian optimisation, HMM/Viterbi, two-laptop FastAPI/UI deployment, Folium maps, a public GitHub repository, and one consolidated team PDF report.

**Reserved evaluation trips are NOT provided.** Do not invent their IDs or train/tune on them when they become available. During the demo, the API automatically switches from 10 student-testing trips to `evaluator_test` when that folder exists.

| File | Format | Meaning |
|---|---|---|
| `.track` | `lon lat timestamp_seconds` (3 whitespace-separated columns) | GPS trajectory |
| `.nodes` | `lon lat` (2 columns) | Node coordinates; row = node ID |
| `.arcs` | `from_node to_node` (2 integer columns) | Directed road arcs; row = arc ID |
| `.route` | `arc_id` (1 integer column) | Chronological route arcs, **not GPS labels** |

## 2. Architecture

```text
Raw trip files, 80 train + 10 student-test
      |
      +--> EDA / validation / coordinates-to-metres
      |
      +--> Ordered reference-route alignment => approximate per-point labels
      |          (TRAIN/TEST evaluation only; NEVER inference)
      |
      +--> Spatial index => nearest candidate directed arcs => 14 features
      |                                          |
      |                            Five ML classifiers + grouped 3-fold CV
      |                                          |
      |                          Optuna TPE Bayesian hyperparameter search
      |                                          |
      |                                joblib model artifact (Laptop 1)
      |                                          |
GPS + road network (no labels) -----------------+--> candidate probabilities
                                                     |
                                               HMM + Viterbi
                                                     |
                                      predicted road arc per GPS observation
                                                     |
                              Laptop 1 FastAPI + internal post-inference scoring
                                                     |
                                       aggregate metrics + predicted geometry
                                                     |
                                          Laptop 2 offline-capable UI
```

**Important evaluation caveat:** There are no per-GPS true arc IDs in the source. `pathsnap/labels.py` projects GPS onto the supplied *ordered* route geometry with a monotone path alignment. Model accuracy against these labels is a **proxy metric**, not verified GPS pointwise truth. Report it accurately and investigate alignment errors.

## 3. Install on Linux (Laptop 1, training and API)

Use Python 3.11 or newer. The included trained model artifacts were generated with scikit-learn 1.8.0, NumPy 2.3.5 and Python 3.13.5. If your environment cannot load the provided artifacts, rerun training with the requirements from your own Python virtual environment.

Download `PathSnap_Implementation.zip` and the original `Student folder.zip` to your Linux `~/Downloads` folder.

```bash
cd "$HOME"
unzip -q "$HOME/Downloads/PathSnap_Implementation.zip"
cd "$HOME/PathSnap_Implementation"

mkdir -p data
tmpdir="$(mktemp -d)"
unzip -q "$HOME/Downloads/Student folder.zip" -d "$tmpdir"
cp -a "$tmpdir/Student folder/train" data/
cp -a "$tmpdir/Student folder/student_test" data/
rm -rf "$tmpdir"

sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.inventory --data-root data --out results/inventory.csv
```

**Verify:** `find data/train -mindepth 1 -maxdepth 1 -type d | wc -l` should print `80`; `find data/student_test -mindepth 1 -maxdepth 1 -type d | wc -l` should print `10`.

## 4. Run all required model training (not the smoke test)

```bash
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
python -m scripts.train_models \
  --data-root data --candidate-k 8 \
  --train-stride 12 --test-stride 1 \
  --cv-folds 3 --out-dir results/full --artifact-dir artifacts/full
```

This fits **five distinct algorithms** on the 80 training trips with group-aware CV (no observation from a trip enters both training and validation folds), evaluates them on all 10 student-testing trips, saves each model to `artifacts/full`, and creates:

- `results/full/model_comparison.csv`
- `results/full/per_trip_comparison.csv`
- `results/full/cv_*.csv`
- Measured comparison plots and `results/full/run_manifest.json`
- `results/full/selected_model.txt` based on **training CV Top-1**

Model names: `logistic`, `gaussian_nb`, `decision_tree`, `random_forest`, `hist_gradient_boosting`.

### Optuna Bayesian tuning

```bash
python -m scripts.tune \
  --data-root data --candidate-k 8 \
  --train-stride 12 --test-stride 1 \
  --trials 15 --out-dir results/full --artifact-dir artifacts/full
```

The selected model is read from `results/full/selected_model.txt`. The study is **resumable** in `results/full/optuna_study.sqlite3`; the `--trials` flag adds that many new trials. Training-only grouped CV chooses tuning parameters. The 10 student-testing trips are used for post-tuning comparison, not optimisation. The tuned model is written to `artifacts/full/best_tuned.joblib` and trial records to `results/full/optuna_trials.csv`.

### HMM / Viterbi comparison

```bash
python -m scripts.evaluate \
  --data-root data \
  --model artifacts/full/best_tuned.joblib \
  --partition student_test --stride 1 \
  --out results/full/decoder_comparison.csv
```

Measures classifier-only versus classifier+HMM observation Top-1 (approximate labels), candidate miss rate, direct arc continuity, and inference timing on every student-testing observation.

### Folium map

```bash
python -m scripts.make_map \
  --data-root data --model artifacts/full/best_tuned.joblib \
  --partition student_test --trip-id 00000052 \
  --out results/full/folium_trip_00000052.html
xdg-open results/full/folium_trip_00000052.html
```

Map layers show blue raw GPS, red predicted road arcs, green reference route (student report only), and grey nearby network. **Never show the green reference layer for reserved evaluator data.** The HTML Folium map can use OSM background tiles when internet is available, but the road and GPS overlays are sourced from the real dataset.

## 5. Two-laptop final deployment

**Laptop 1** runs the model, the road network, the API, and internal post-inference aggregate scoring:

```bash
cd "$HOME/PathSnap_Implementation"
source .venv/bin/activate
export PATHSNAP_DATA_ROOT="$(pwd)/data"
export PATHSNAP_MODEL_PATH="$(pwd)/artifacts/full/best_tuned.joblib"
uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000 --access-log
```

In a second Laptop 1 terminal:

```bash
hostname -I
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/trips
```

**Laptop 2** hosts only the UI. Download/extract `PathSnap_Implementation.zip` there (no Python dependencies except the OS standard library required):

```bash
cd "$HOME/PathSnap_Implementation"
python3 -m http.server 5500 --directory ui --bind 0.0.0.0
```

Open `http://127.0.0.1:5500` in Laptop 2's browser. In the **Laptop 1 API address** field, type `http://LAPTOP_1_LAN_IP:8000` using the IP displayed by `hostname -I`. Click **Connect & load**, choose any available trip, and click **Reconstruct trip**. The UI draws the network and matched route without requiring external tile services. On Laptop 1, show the incoming `PREDICT_REQUEST` and `INFERENCE` logs to the evaluator.

If the firewall blocks the private LAN connection on Laptop 1, allow access only on the trusted lab network:

```bash
sudo ufw status
sudo ufw allow from LAPTOP_2_LAN_IP to any port 8000 proto tcp
```

Do not expose this development API to the public Internet. If both laptops are temporarily unavailable, run the API and UI on one laptop with `http://127.0.0.1:8000`.

## 6. Reserved evaluator testing — strict isolation

The evaluator data is not included in either uploaded ZIP. Only when provided by organisers, create `data/evaluator_test/000000XX/000000XX.track`, `.nodes`, `.arcs` and, if the organiser allows post-inference scoring, `.route` in the **server-only** data directory.

- Never use reserved evaluator samples for training, feature selection or tuning.
- Never enter any reference route file into `pathsnap.pipeline.predict` or the browser UI.
- The API loads references **after** inference to calculate aggregate metrics; the UI receives no reference sequence or geometry.
- API requests are generated by the evaluator's selected trip ID; predictions are computed, never hardcoded.

## 7. Team responsibilities and Git

See `docs/TEAM_PLAN.md` for a four-person ownership and integration plan. To initiate your own GitHub repository:

```bash
git init
git branch -M main
git add pathsnap scripts tests ui docs README.md requirements.txt pyproject.toml .gitignore
git commit -m "Implement PathSnap experimental pipeline and two-laptop demo"
gh auth login
gh repo create YOUR_GITHUB_USERNAME/PathSnap --public --source=. --remote=origin --push
```

After reproducing the complete run, selectively commit the actual `results/full/` CSVs, plots, train/test configuration, and required trained artifacts; exclude raw datasets, evaluator data, credentials, temporary cache, and unsubstantiated smoke-test results. Inspect `git status` and `git diff --cached --stat` before each push.

## 8. What remains for the team to own

The repository and run commands implement the required components, but the team still must:

1. Review or improve the reference-route alignment and document its bias, especially for loops / gaps.
2. Verify any selected hyperparameters and replicate the full experiments on the team laptops.
3. Decide whether GPS stride / candidate count, metrics and model family decisions are justified for the challenge.
4. Perform two-laptop live testing against all ten reserved evaluator trips when available.
5. Capture genuine Folium, UI and terminal API screenshots; verify full report figures and conclusions.
6. Prepare **one consolidated PDF**, with each model's reason for selection, important configuration, train/CV versus student-test results, HMM comparison, optimisation trials and evidence (see `docs/REPORT_CHECKLIST.md`).
7. Disclose AI assistance according to the organisers' policy; every member must be able to explain and defend the implementation.

### Known limitations

- GPS-point supervision is **approximate** because the raw route is only arc-sequential ground truth.
- The road graph uses directed segments and bounded Dijkstra transitions, with explicit gaps/disconnections resets.
- Map projection is a per-trip local metres approximation; very large or polar graphs should use a dedicated CRS.
- Closest-candidate generation uses segment midpoints and incident-node search; some true route arcs can be absent.
- The supplied challenge PDFs do **not** define the final evaluator-test file path or exact scoring protocol. The organiser may provide additional instructions that supersede the current adapter.
- HMM decoding is CPU-based; record runtime on the real demo laptop.
- The current default run has `--train-stride 12`; report that sampling rather than claiming all training observations were fitted.

## 9. Rebuild the measured implementation guide PDF (optional)

```bash
python -m scripts.build_implementation_guide --results results/full --out docs/IMPLEMENTATION_GUIDE.pdf
```

This PDF is a **measured engineering handoff guide**, not the final team-authored technical report demanded by the challenge. Add your own screenshots, reasoning and explanations in your final PDF.

## One-command reproducible workflow

After extracting the implementation ZIP into your home directory, the Linux scripts can perform the installation and all experiments:

```bash
cd "$HOME/PathSnap_Implementation"
bash scripts/setup_linux.sh "$HOME/Downloads/Student folder.zip"
bash scripts/run_all.sh
```

The second command retrains all five models, runs group CV, adds **15 new Optuna trials**, compares classifier/HMM predictions on all ten student-testing trips, generates Folium and EDA plots, and rebuilds the engineering guide. To add fewer new trials on a constrained laptop: `PATHSNAP_TRIALS=5 bash scripts/run_all.sh`. For a clean 15-trial study, remove the old `results/full/optuna_study.sqlite3` before running. Any final report must be based on the results you actually reproduced.
