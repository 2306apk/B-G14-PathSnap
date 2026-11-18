"""Unit tests for PathSnap Map-Matching REST API endpoints.

Tests:
- /health check (Stage 5 compatibility & diagnostics)
- /trips (Stage 5 UI contract)
- /predict (Stage 5 POST compatibility)
- /predict/{trip_id} (Stage 5 GET compatibility)
- /ui (Stage 5 HTML UI interface)
- /api/network-info connectivity discovery
- /api/trips listing
- /api/trips/{trip_id} detail retrieval
- /api/match HMM inference execution & GeoJSON generation
- /api/metrics/summary benchmark retrieval
- /api/trips/{trip_id}/map Folium map rendering
"""

import pytest
from starlette.testclient import TestClient

from pathsnap.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "PathSnap Map-Matching API"
    assert data["status"] == "online"
    assert "/health" in data["endpoints"]
    assert "/predict" in data["endpoints"]
    assert "/trips" in data["endpoints"]


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Compatible with Stage 5 ('status': 'ok', 'model_loaded': True)
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "partition" in data
    assert "trip_count" in data
    assert "platform" in data
    assert "network" in data
    assert "lan_ip" in data["network"]
    assert data["models_status"]["random_forest"] is True
    assert data["models_status"]["tuned_best_model"] is True
    assert data["calibrator_loaded"] is True
    assert data["features"]["total_features"] == 18


def test_stage5_trips_contract(client):
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert "partition" in data
    assert "trips" in data
    assert isinstance(data["trips"], list)
    assert len(data["trips"]) > 0
    assert "note" in data
    # Verify trip IDs are 8-digit strings
    assert all(len(t) == 8 for t in data["trips"][:5])


def test_stage5_predict_post_contract(client):
    # Test POST /predict on Trip 00000052
    payload = {"trip_id": "00000052"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Exact fields required by Stage 5 ui/index.html
    assert data["trip_id"] == "00000052"
    assert data["partition"] in ["student_test", "evaluator_test", "train"]
    assert data["observations"] == 262
    assert "inference_seconds" in data
    assert isinstance(data["inference_seconds"], float)
    assert "raw_gps_latlon" in data
    assert isinstance(data["raw_gps_latlon"], list)
    assert len(data["raw_gps_latlon"]) == 262
    assert len(data["raw_gps_latlon"][0]) == 2  # [lat, lon]

    assert "predicted_segments_latlon" in data
    assert isinstance(data["predicted_segments_latlon"], list)
    assert len(data["predicted_segments_latlon"]) > 0
    assert len(data["predicted_segments_latlon"][0]) == 2  # [[lat1, lon1], [lat2, lon2]]

    assert "nearby_roads_latlon" in data
    assert isinstance(data["nearby_roads_latlon"], list)

    assert "metrics" in data
    assert data["metrics"] is not None
    assert "observation_top1_accuracy" in data["metrics"]
    assert "hmm_observation_accuracy" in data["metrics"]
    assert data["metrics"]["hmm_observation_accuracy"] >= 0.90
    assert "candidate_miss_rate" in data["metrics"]
    assert "classifier_direct_continuity" in data["metrics"]
    assert "hmm_direct_continuity" in data["metrics"]
    assert data["metrics"]["hmm_direct_continuity"] >= 0.95

    assert "prediction_note" in data


def test_stage5_predict_get_contract(client):
    # Test GET /predict/00000052
    response = client.get("/predict/00000052")
    assert response.status_code == 200
    data = response.json()
    assert data["trip_id"] == "00000052"
    assert data["observations"] == 262
    assert "predicted_segments_latlon" in data


def test_stage5_ui_endpoint(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PathSnap · Evaluator UI" in response.text
    assert "connect()" in response.text


def test_network_info_endpoint(client):
    response = client.get("/api/network-info")
    assert response.status_code == 200
    data = response.json()
    assert "lan_ip" in data
    assert "port" in data
    assert "lan_backend_url" in data
    assert "instructions" in data


def test_trips_list_endpoint(client):
    response = client.get("/api/trips")
    assert response.status_code == 200
    data = response.json()
    assert data["total_trips"] > 0
    assert len(data["trips"]) > 0

    first_trip = data["trips"][0]
    assert "trip_id" in first_trip
    assert "partition" in first_trip
    assert "num_observations" in first_trip


def test_trip_detail_endpoint(client):
    response = client.get("/api/trips/00000081")
    assert response.status_code == 200
    data = response.json()
    assert data["trip_id"] == "00000081"
    assert data["num_observations"] > 0
    assert "bounds" in data
    assert "track" in data
    assert len(data["track"]) == data["num_observations"]
    assert "latitude" in data["track"][0]
    assert "longitude" in data["track"][0]


def test_trip_match_endpoint(client):
    payload = {
        "trip_id": "00000081",
        "partition": "train",
        "model_name": "tuned_best_model",
        "use_calibration": True,
        "hmm_params": {
            "beta": 20.0,
            "unreachable_penalty": 5.0,
            "theta_0": 55.0,
            "emission_weight": 1.5,
        },
    }
    response = client.post("/api/match", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["trip_id"] == "00000081"
    assert "execution_time_ms" in data
    assert "metrics" in data
    assert data["metrics"]["route_continuity"] >= 0.90
    assert "predicted_arc_ids" in data
    assert len(data["predicted_arc_ids"]) == data["num_observations"]
    assert "geojson" in data
    assert data["geojson"]["type"] == "FeatureCollection"
    assert len(data["geojson"]["features"]) > 0


def test_metrics_summary_endpoint(client):
    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "models_comparison" in data
    methods = [m["method"] for m in data["models_comparison"]]
    assert "Random Forest" in methods
    assert "Refined Improved HMM (18-feature)" in methods

    rf_item = next(m for m in data["models_comparison"] if m["method"] == "Random Forest")
    hmm_item = next(m for m in data["models_comparison"] if "Refined Improved HMM" in m["method"])
    assert hmm_item["top1_accuracy"] > rf_item["top1_accuracy"]


def test_trip_folium_map_endpoint(client):
    response = client.get("/api/trips/00000081/map")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text
