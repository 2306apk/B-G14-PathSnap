#!/usr/bin/env bash
# Reproducible Linux environment; expects raw Student folder.zip from the organisers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ZIP_PATH="${1:-$HOME/Downloads/Student folder.zip}"
if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Missing dataset ZIP: $ZIP_PATH" >&2
  echo "Usage: bash scripts/setup_linux.sh '/path/to/Student folder.zip'" >&2
  exit 2
fi
mkdir -p data
TMPDIR_EXTRACT="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_EXTRACT"' EXIT
unzip -q "$ZIP_PATH" -d "$TMPDIR_EXTRACT"
for split in train student_test; do
  if [[ ! -d "$TMPDIR_EXTRACT/Student folder/$split" ]]; then
     echo "Expected split not found in dataset: $split" >&2; exit 2
  fi
  if [[ -e "data/$split" ]]; then
    echo "data/$split already exists. Leaving it unchanged."
  else
    cp -a "$TMPDIR_EXTRACT/Student folder/$split" "data/$split"
  fi
done
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
printf '\nDataset ready: '
find data/train -mindepth 1 -maxdepth 1 -type d | wc -l
printf ' training trips, '
find data/student_test -mindepth 1 -maxdepth 1 -type d | wc -l
printf ' student-testing trips.\n'
