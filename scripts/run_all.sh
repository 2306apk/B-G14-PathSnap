#!/usr/bin/env bash
# Runs all PathSnap development experiments and regenerates evidence.
# NEVER insert evaluator_test into this script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f .venv/bin/activate ]]; then
  echo 'Please run: bash scripts/setup_linux.sh "/path/to/Student folder.zip"' >&2
  exit 2
fi
source .venv/bin/activate
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
TRIALS="${PATHSNAP_TRIALS:-15}"
mkdir -p results/full artifacts/full
python -m pytest -q
python -m scripts.inventory --data-root data --out results/full/inventory.csv
python -m scripts.eda_plots --data-root data --inventory results/full/inventory.csv \
  --partition train --trip-id 00000005 --out-dir results/full/eda
python -m scripts.train_models --data-root data --candidate-k 8 \
  --train-stride 12 --test-stride 1 --cv-folds 3 \
  --out-dir results/full --artifact-dir artifacts/full
python -m scripts.tune --data-root data --candidate-k 8 \
  --train-stride 12 --test-stride 1 --trials "$TRIALS" \
  --out-dir results/full --artifact-dir artifacts/full
python -m scripts.evaluate --data-root data \
  --model artifacts/full/best_tuned.joblib --partition student_test \
  --out results/full/decoder_comparison.csv
python -m scripts.make_map --data-root data \
  --model artifacts/full/best_tuned.joblib \
  --partition student_test --trip-id 00000052 \
  --out results/full/folium_trip_00000052.html
python -m scripts.build_implementation_guide \
  --results results/full --out docs/IMPLEMENTATION_GUIDE.pdf
echo 'All permitted development stages completed. See results/full and docs/IMPLEMENTATION_GUIDE.pdf.'
