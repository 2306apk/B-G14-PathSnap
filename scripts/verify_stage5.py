import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"

def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as response:
        assert response.status == 200
        return json.load(response)

health = request("/health")
assert health["status"] == "ok"
assert health["model_loaded"] is True
assert health["partition"] == "student_test"
assert health["trip_count"] == 10

trips = request("/trips")
assert len(trips["trips"]) == 10
assert "00000052" in trips["trips"]

result = request("/predict", {"trip_id": "00000052"})
assert result["trip_id"] == "00000052"
assert result["observations"] == 262
assert result["raw_gps_latlon"]
assert result["predicted_segments_latlon"]
assert result["nearby_roads_latlon"]
assert result["metrics"] is not None

for key in ("reference_route", "reference_arc_ids",
            "ground_truth", "truth"):
    assert key not in result

summary = {
    "status": "PASS",
    "health": health,
    "trip_tested": result["trip_id"],
    "observations": result["observations"],
    "inference_seconds": result["inference_seconds"],
    "classifier_top1": result["metrics"]["observation_top1_accuracy"],
    "hmm_top1": result["metrics"]["hmm_observation_accuracy"],
    "reference_labels_exposed": False,
    "deployment_scope": "single_laptop_local",
}

out = Path("results/stage5/local_verification.json")
out.write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps(summary, indent=2))
print("\nSTAGE 5 LOCAL VERIFICATION: PASS")
