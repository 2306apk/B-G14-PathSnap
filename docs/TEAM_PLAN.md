# Four-person implementation and integration plan

## Ownership table

| Member | Primary responsibility | Files / output | Verification |
|---|---|---|---|
| **1 — Dataset / geographic preprocessing** | Dataset audit, coordinate conversion, raw file validation, route-sequence analysis, nearest-road candidates, reference alignment; Folium EDA | `pathsnap/data.py`, `candidates.py`, `labels.py`; `scripts/inventory.py`, `make_map.py`; `results/inventory.csv`; EDA maps | Explain why route arc count != GPS count; measure label-alignment error and candidate-miss rate on each partition; GPS gaps and long arc handling |
| **2 — ML experimentation** | Feature selection, five independent model experiments, grouped CV, model comparison, plots, confusion matrices, class imbalance | `pathsnap/experiments.py`, `scripts/train_models.py`, `results/full/model_comparison.csv`, `artifacts/full/*.joblib` | Run all 5 models on 80 training + all 10 student-test trips; explain each model and its actual scores |
| **3 — Bayesian tuning / sequential decoding** | Optuna TPE and trial record, HMM emissions, network transitions, bounded shortest paths, Viterbi, gaps and disconnected transitions, classifier vs HMM metrics | `scripts/tune.py`, `pathsnap/decoder.py`, `metrics.py`, `pipeline.py`, `scripts/evaluate.py` | Inspect `optuna_trials.csv`, tuning summary; full `decoder_comparison.csv`; quantify improvement/worsening and limits |
| **4 — API / UI / deployment and final report integration** | FastAPI, route serialization, no-label-leakage verification, Laptop 2 offline UI, LAN connectivity, final PDF report, GitHub coordination | `pathsnap/api.py`, `ui/index.html`, `README.md`, report PDF | Demonstrate requests on Laptop 1 logs, choose all 10 available trips on Laptop 2, record real screenshots and report evidence |

Cross-review: Member 1 checks Member 2's candidate truth mapping, Member 2 reviews Member 3's metric definitions, Member 3 checks Member 4's prediction/reference isolation, and Member 4 checks data/reports for leaked reserved data.

## Git collaboration (run individually on Linux)

Team lead, from an empty GitHub repository:

```bash
cd ~/PathSnap_Implementation
git init
git branch -M main
git add .gitignore README.md pyproject.toml requirements.txt pathsnap scripts ui tests docs
git commit -m "Initialize PathSnap architecture"
gh repo create YOUR_USERNAME/PathSnap --public --source=. --remote=origin --push
```

Each contributor clones the repository and creates their **own** branch:

```bash
git clone https://github.com/YOUR_USERNAME/PathSnap.git
cd PathSnap
git switch -c feature/YOUR_COMPONENT
# Implement, execute tests, and explain changes in this branch.
python3 -m pytest -q
git add <ONLY_THE_FILES_YOU_OWN>
git commit -m "Explain contribution and verified behavior"
git push -u origin feature/YOUR_COMPONENT
gh pr create --base main --head feature/YOUR_COMPONENT --title "PathSnap: verified component" --body "Changes, executed commands, measured results, and limitations"
```

Suggested branch names: `feature/data-eda`, `feature/models`, `feature/tuning-decoder`, `feature/api-ui`.

The team lead reviews and merges PRs to `main` only after tests pass. Avoid the four members editing the same files at the same time. Git history should accurately reflect which member made and verified each change.

## Suggested work stages (do not treat these as fabricated completion evidence)

**Stage A — Data contract and EDA:** Member 1 publishes a data inventory + alignment explanation. Others review route mapping and all missing-candidate cases.

**Stage B — Model experiments:** Member 2 publishes five model experiments, train CV folds, student-test metrics, and plots. Member 3 starts HMM and writes regression tests.

**Stage C — Model improvement:** Member 3 publishes Optuna trial history and full decoder comparison. Member 4 runs API and UI integration checks with demo data.

**Stage D — Final validation and submission:** Member 4 connects two laptops. Each member tests their own component, signs off on results and screenshots. Merge to GitHub main. Prepare and verify a *single* PDF report with actual evidence.

## Viva: each member must answer

1. Which file contains ground-truth **arc sequence**, and why isn't it per-observation truth?
2. Why are geographic degrees not valid metric distances?
3. Why must CV group by `trip_id`?
4. How are candidate misses and candidate-binary Macro F1 defined?
5. Why are classification probabilities used as HMM emissions, and why must they be normalised per observation?
6. What is the Viterbi recurrence, and how does the graph transition penalise disconnected arcs?
7. Why must the evaluator's `.route` remain inaccessible to the prediction stage?
8. What changes between running one laptop and the required two-laptop API/UI system?
