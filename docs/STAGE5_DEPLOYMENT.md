# PathSnap — Stage 5 Two-Laptop Deployment

## Architecture

Laptop 1:
- Trained ML model
- GPS and road network files
- HMM/Viterbi route reconstruction
- FastAPI backend on port 8000

Laptop 2:
- Browser-based evaluator UI
- Local HTTP server on port 5500
- Sends requests to Laptop 1 over the LAN

## Clone the project

git clone --branch feature/stage5-api-ui \
  --single-branch \
  https://github.com/2306apk/G14-PathSnap.git

## Laptop 1 — Backend

Install dependencies in a Python virtual environment.

Keep the original dataset under data/.
The dataset is intentionally not committed to GitHub.

Set the model path to the existing Stage 3 tuned artifact.

Commands:

    source .venv/bin/activate
    export PATHSNAP_DATA_ROOT="$(pwd)/data"
    export PATHSNAP_MODEL_PATH="$(pwd)/artifacts/stage3/best_tuned.joblib"
    uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000

Check:

    curl http://127.0.0.1:8000/health

## Laptop 2 — Frontend

No ML dependencies or dataset required.

Commands:

    python3 -m http.server 5500 --directory ui --bind 127.0.0.1

Open:

    http://127.0.0.1:5500

Enter Laptop 1's current LAN address in the API field,
for example:

    http://LAPTOP1_IP:8000

Click Connect & load, choose a trip, then Reconstruct trip.

## API endpoints

GET  /health
GET  /trips
POST /predict
GET  /predict/{trip_id}

POST example:

    curl -X POST http://LAPTOP1_IP:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"trip_id":"00000052"}'

## Evaluation integrity

- Predictions use GPS and road-network inputs.
- Reference-route labels are loaded after inference for scoring.
- Reference geometry and pointwise truth are not sent to the UI.
- Accuracy and continuity are calculated from model results.
- Scores use approximate GPS-to-reference-route alignment.
- Do not report one trip's accuracy as the overall result.
- Reserved evaluator data must not be used to tune the model.

## Current evaluation

On 10 student-testing trips:
- 19,160 observations
- Classifier Top-1: 76.25%
- HMM Top-1: 83.87%

On 80 training trips using grouped out-of-fold evaluation:
- 208,417 observations
- HMM Top-1: 72.48%

These are development proxy metrics, not official hidden-test scores.

## Security

Run the API only on a trusted LAN.
Do not expose port 8000 to the public Internet.
Do not commit raw data, passwords, tokens or private evaluator files.
