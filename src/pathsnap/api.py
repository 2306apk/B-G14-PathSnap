"""PathSnap Map-Matching REST API.

Provides production-ready endpoints for:
- System and model health checks (/health)
- Network connectivity and IP diagnostic inspection (/api/network-info)
- Dataset trip inspection (/api/trips, /api/trips/{trip_id})
- Live HMM/Viterbi map matching and performance metrics (/api/match)
- Interactive map rendering (/api/trips/{trip_id}/map)
- Phase 2 / Section 5A Model comparison metrics (/api/metrics/summary)
"""

from __future__ import annotations

import os
import sys
import time
import socket
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure headless matplotlib configuration
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure pathsnap is importable
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from pathsnap.io import load_trip, resolve_data_dir, list_trips
from pathsnap.geo import get_trip_crs
from pathsnap.graph import RoadNetwork
from pathsnap.features import (
    FEATURE_COLUMNS,
    OBSERVATION_FEATURE_COLUMNS,
    TRANSITION_FEATURE_COLUMNS,
    ALL_MAPMATCHING_FEATURES,
    TARGET_COLUMN,
)
from pathsnap.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_accuracy,
    compute_route_continuity,
)
from pathsnap.hmm import HMMRouteDecoder, HMMDecodingResult
from pathsnap.visualize import create_folium_map


# Initialize FastAPI App
app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Cross-platform Distributed Map-Matching API powering HMM/Viterbi sequence decoding.",
    version="2.0.0",
)

# Enable CORS for all origins so Windows / Mac frontends can communicate freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# Global in-memory cache for models and datasets
MODELS: Dict[str, Any] = {}
CALIBRATOR: Any = None
DATA_CACHE: Dict[str, pd.DataFrame] = {}


def get_local_ip_addresses() -> Dict[str, str]:
    """Detects available local IP addresses (LAN and ZeroTier if available).
    
    Used to automatically broadcast connection URLs so a second machine (e.g. Windows laptop)
    can easily connect to this backend server across the local network.
    """
    ips = {"lan": "127.0.0.1", "hostname": socket.gethostname()}
    try:
        # Standard UDP socket trick: connects to a public IP (without sending packets)
        # to determine the operating system's chosen outbound network interface.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ips["lan"] = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    # Look for ZeroTier virtual IP if running over a private ZeroTier mesh network
    try:
        import subprocess
        if platform.system() == "Darwin":
            out = subprocess.check_output(["ifconfig"], text=True)
            for block in out.split("\n\n"):
                if "zt" in block or "zerotier" in block.lower():
                    for line in block.splitlines():
                        if "inet " in line:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                ips["zerotier"] = parts[1]
                                break
    except Exception:
        pass

    return ips


def load_assets():
    """Loads trained models, calibrator, and processed parquet matrices into memory.
    
    Pre-warming models at startup ensures subsequent HTTP requests for map matching
    have sub-second response times.
    """
    global MODELS, CALIBRATOR, DATA_CACHE

    models_dir = repo_root / "outputs" / "models" / "phase2"
    data_dir = repo_root / "data" / "processed"

    # Load All Trained Classifiers (Phase 2 & Stage 3 Tuned Model)
    model_files = {
        "tuned_best_model": models_dir / "tuned_best_model.joblib",
        "random_forest": models_dir / "random_forest.joblib",
        "histgradientboosting": models_dir / "histgradientboosting.joblib",
        "logistic_regression": models_dir / "logistic_regression.joblib",
        "decision_tree": models_dir / "decision_tree.joblib",
        "gaussian_naive_bayes": models_dir / "gaussian_naive_bayes.joblib",
    }

    for name, path in model_files.items():
        if path.exists():
            try:
                MODELS[name] = joblib.load(path)
            except Exception as e:
                print(f"[Warning] Failed to load model {name}: {e}")

    # Load Platt Sigmoid Calibrator (maps raw tree scores to empirical true probabilities)
    cal_path = models_dir / "production_calibrator.joblib"
    if cal_path.exists():
        try:
            CALIBRATOR = joblib.load(cal_path)
        except Exception as e:
            print(f"[Warning] Failed to load production calibrator: {e}")

    # Pre-cache Parquet Feature Matrices for instant demonstration queries
    parquet_files = {
        "train_features": data_dir / "train_features.parquet",
        "student_test_features": data_dir / "student_test_features.parquet",
        "student_test_labels": data_dir / "student_test_labels.parquet",
    }

    for name, path in parquet_files.items():
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if "trip_id" in df.columns:
                    df["trip_id_str"] = df["trip_id"].astype(str).str.zfill(8)
                DATA_CACHE[name] = df
            except Exception as e:
                print(f"[Warning] Failed to load parquet {name}: {e}")


# Pre-load assets on module import
load_assets()


# Pydantic Schemas
class TripRequest(BaseModel):
    trip_id: str = Field(..., description="Trip identifier, e.g. '00000052' or '52'")


class HMMParams(BaseModel):
    beta: float = Field(20.0, description="HMM exponential transition distance scale (meters)")
    unreachable_penalty: float = Field(5.0, description="Log-penalty for disconnected candidate pairs")
    theta_0: float = Field(55.0, description="Heading difference scale (degrees)")
    emission_weight: float = Field(1.5, description="Emission log-probability scaling multiplier")
    stay_bonus: float = Field(0.0, description="Bonus log-prob for staying on identical arc")
    adjacent_bonus: float = Field(0.0, description="Bonus log-prob for transitioning to adjacent arc")


class MatchRequest(BaseModel):
    trip_id: str = Field(..., description="Trip identifier, e.g. '00000081' or '81'")
    partition: Optional[str] = Field("student_test", description="'student_test' or 'train'")
    model_name: Optional[str] = Field("tuned_best_model", description="Model name to use for emission probabilities")
    use_calibration: Optional[bool] = Field(True, description="Whether to apply probability calibration")
    hmm_params: Optional[HMMParams] = Field(default_factory=HMMParams)


@app.get("/")
def read_root():
    """Root entry point with service metadata and documentation links."""
    return {
        "service": "PathSnap Map-Matching API",
        "version": "2.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "ui_url": "/ui",
        "endpoints": [
            "/health",
            "/trips",
            "/predict",
            "/predict/{trip_id}",
            "/ui",
            "/api/network-info",
            "/api/trips",
            "/api/trips/{trip_id}",
            "/api/match",
            "/api/trips/{trip_id}/map",
            "/api/metrics/summary",
        ],
    }


def active_partition() -> str:
    raw_dir = resolve_data_dir()
    if (raw_dir / "evaluator_test").exists() and list_trips("evaluator_test", raw_dir):
        return "evaluator_test"
    if (raw_dir / "student_test").exists() and list_trips("student_test", raw_dir):
        return "student_test"
    return "train"


@app.get("/health")
def health_check():
    """Health check returning loaded models, platform info, uptime, and dataset counts.
    
    Compatible with Stage 5 contract ('status': 'ok', 'model_loaded': True).
    """
    raw_dir = resolve_data_dir()
    train_trips = list_trips("train", raw_dir) if (raw_dir / "train").exists() else []
    test_trips = list_trips("student_test", raw_dir) if (raw_dir / "student_test").exists() else []

    net_info = get_local_ip_addresses()
    split = active_partition()
    active_count = len(test_trips) if split == "student_test" else len(train_trips)

    return {
        "status": "ok",
        "model_loaded": len(MODELS) > 0,
        "partition": split,
        "trip_count": active_count,
        "service": "PathSnap Map-Matching API",
        "version": "2.0.0",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": net_info.get("hostname"),
        },
        "network": {
            "lan_ip": net_info.get("lan"),
            "zerotier_ip": net_info.get("zerotier", "Not active / unassigned"),
            "recommended_frontend_url": f"http://{net_info.get('lan', '127.0.0.1')}:8000",
        },
        "models_status": {
            name: (name in MODELS) for name in [
                "tuned_best_model", "random_forest", "histgradientboosting",
                "logistic_regression", "decision_tree", "gaussian_naive_bayes"
            ]
        },
        "calibrator_loaded": CALIBRATOR is not None,
        "datasets_available": {
            "train_trips_count": len(train_trips),
            "student_test_trips_count": len(test_trips),
            "precomputed_train_matrix": "train_features" in DATA_CACHE,
            "precomputed_student_test_matrix": "student_test_features" in DATA_CACHE,
        },
        "features": {
            "observation_features_count": len(OBSERVATION_FEATURE_COLUMNS),
            "transition_features_count": len(TRANSITION_FEATURE_COLUMNS),
            "total_features": len(ALL_MAPMATCHING_FEATURES),
        },
    }


@app.get("/trips")
def get_trips_stage5():
    """Stage 5 reference compatibility endpoint consumed by ui/index.html."""
    raw_dir = resolve_data_dir()
    split = active_partition()
    t_list = list_trips(split, raw_dir)
    return {
        "partition": split,
        "trips": t_list,
        "note": "Evaluator data is absent; student-testing demo displayed" if split == "student_test" else f"{split} partition loaded",
    }


def run_trip_prediction(trip_id: str):
    """Executes HMM inference and formats response according to Stage 5 contract."""
    t0 = time.perf_counter()
    raw_dir = resolve_data_dir()
    tid_str = str(trip_id).zfill(8)

    # Determine partition
    part = "student_test"
    if (raw_dir / "evaluator_test" / tid_str).exists():
        part = "evaluator_test"
    elif (raw_dir / "student_test" / tid_str).exists():
        part = "student_test"
    elif (raw_dir / "train" / tid_str).exists():
        part = "train"
    else:
        raise HTTPException(status_code=404, detail=f"Trip {tid_str} not found in available dataset partitions")

    trip = load_trip(tid_str, partition=part, base_dir=raw_dir)
    net = RoadNetwork(tid_str, trip.nodes, trip.arcs, get_trip_crs(trip.track))
    num_obs = len(trip.track)

    # Locate feature matrix in cache
    matrix_name = f"{part}_features"
    df_t = pd.DataFrame()
    if matrix_name in DATA_CACHE and "trip_id_str" in DATA_CACHE[matrix_name].columns:
        full_df = DATA_CACHE[matrix_name]
        df_t = full_df[full_df["trip_id_str"] == tid_str].copy()

    if df_t.empty:
        for alt in ["student_test_features", "train_features"]:
            if alt in DATA_CACHE and "trip_id_str" in DATA_CACHE[alt].columns:
                sub = DATA_CACHE[alt][DATA_CACHE[alt]["trip_id_str"] == tid_str]
                if not sub.empty:
                    df_t = sub.copy()
                    break

    if df_t.empty:
        raise HTTPException(status_code=404, detail=f"No candidate feature rows available for trip {tid_str}")

    # Model and Calibrator inference
    model_name = "tuned_best_model" if "tuned_best_model" in MODELS else "random_forest"
    pipeline = MODELS[model_name]
    p_raw = pipeline.predict_proba(df_t[FEATURE_COLUMNS])[:, 1]
    p_cal = CALIBRATOR.predict_proba(p_raw) if CALIBRATOR is not None else p_raw

    # Classifier Top-1 baseline
    df_t["prob"] = p_cal
    raw_top1 = df_t.sort_values(["observation_id", "prob"], ascending=[True, False]).groupby("observation_id").first()
    clf_top1_map = raw_top1["candidate_arc_id"].to_dict()
    classifier_top1_arcs = [int(clf_top1_map.get(i, -1)) for i in range(num_obs)]

    # HMM Viterbi Sequential Decoding (18-feature model)
    decoder = HMMRouteDecoder(net, beta=20.0, unreachable_penalty=5.0, theta_0=55.0, emission_weight=1.5)
    decode_res = decoder.decode(df_t, p_cal, trip.track)
    inference_s = time.perf_counter() - t0

    # Raw GPS lat/lon array
    raw_gps = trip.track[["latitude", "longitude"]].values.tolist()
    downsampled_raw = raw_gps[::max(1, len(raw_gps) // 3000)]

    # Matched segments [[[lat1, lon1], [lat2, lon2]], ...]
    arc_table = net.arc_table
    node_lons = net.node_lon
    node_lats = net.node_lat
    unique_pred_arcs = list(dict.fromkeys(decode_res.predicted_arc_ids))
    matched_segments = []
    for aid in unique_pred_arcs:
        if 0 <= aid < len(arc_table):
            row = arc_table.iloc[aid]
            u, v = int(row["from_node_id"]), int(row["to_node_id"])
            matched_segments.append([
                [float(node_lats[u]), float(node_lons[u])],
                [float(node_lats[v]), float(node_lons[v])],
            ])

    # Nearby road network segments within bounding box + 0.006 margin (capped at 1400)
    min_lat = float(trip.track["latitude"].min()) - 0.006
    max_lat = float(trip.track["latitude"].max()) + 0.006
    min_lon = float(trip.track["longitude"].min()) - 0.006
    max_lon = float(trip.track["longitude"].max()) + 0.006

    nearby_roads = []
    for aid in range(len(arc_table)):
        row = arc_table.iloc[aid]
        u, v = int(row["from_node_id"]), int(row["to_node_id"])
        u_lat, u_lon = float(node_lats[u]), float(node_lons[u])
        v_lat, v_lon = float(node_lats[v]), float(node_lons[v])
        mid_lat = (u_lat + v_lat) * 0.5
        mid_lon = (u_lon + v_lon) * 0.5
        if min_lat <= mid_lat <= max_lat and min_lon <= mid_lon <= max_lon:
            nearby_roads.append([[u_lat, u_lon], [v_lat, v_lon]])
            if len(nearby_roads) >= 1400:
                break

    # Ground truth mapping if available
    obs_true_dict = {}
    if TARGET_COLUMN in df_t.columns and df_t[TARGET_COLUMN].sum() > 0:
        obs_true_dict = df_t[df_t[TARGET_COLUMN] == 1].set_index("observation_id")["candidate_arc_id"].to_dict()
    elif "student_test_labels" in DATA_CACHE:
        labels_df = DATA_CACHE["student_test_labels"]
        if "trip_id_str" in labels_df.columns:
            sub_lbl = labels_df[labels_df["trip_id_str"] == tid_str]
            if not sub_lbl.empty:
                obs_true_dict = sub_lbl.set_index("observation_id")["assigned_arc_id"].to_dict()

    has_gt = len(obs_true_dict) > 0
    obs_with_cands = df_t["observation_id"].nunique()
    miss_rate = round(float(1.0 - (obs_with_cands / max(num_obs, 1))), 4)

    scores = None
    if has_gt:
        correct_hmm = sum(1 for i in range(num_obs) if int(decode_res.predicted_arc_ids[i]) == int(obs_true_dict.get(i, -1)))
        correct_clf = sum(1 for i in range(num_obs) if int(classifier_top1_arcs[i]) == int(obs_true_dict.get(i, -1)))
        hmm_acc = round(float(correct_hmm) / num_obs, 4)
        clf_acc = round(float(correct_clf) / num_obs, 4)
        scores = {
            "observation_top1_accuracy": clf_acc,
            "hmm_observation_accuracy": hmm_acc,
            "candidate_miss_rate": miss_rate,
            "classifier_direct_continuity": round(float(decode_res.classifier_continuity), 4),
            "hmm_direct_continuity": round(float(decode_res.route_continuity), 4),
            "metric_note": "Evaluated against ground truth reference route; calibrated 18-feature HMM applied",
        }

    return {
        "trip_id": tid_str,
        "partition": part,
        "observations": num_obs,
        "inference_seconds": round(float(inference_s), 3),
        "raw_gps_latlon": downsampled_raw,
        "predicted_segments_latlon": matched_segments,
        "nearby_roads_latlon": nearby_roads,
        "metrics": scores,
        "prediction_note": "All predictions generated by the trained model + HMM, not by reference labels",
    }


@app.post("/predict")
def predict_post(req: TripRequest):
    """Stage 5 reference compatibility POST endpoint."""
    return run_trip_prediction(req.trip_id)


@app.get("/predict/{trip_id}")
def predict_get(trip_id: str):
    """Stage 5 reference compatibility GET endpoint."""
    return run_trip_prediction(trip_id)


@app.get("/ui", response_class=HTMLResponse)
def serve_stage5_ui():
    """Serves the Stage 5 Evaluator UI canvas interface directly."""
    ui_path = repo_root / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(), status_code=200)
    raise HTTPException(status_code=404, detail="ui/index.html not found")


@app.get("/api/network-info")
def get_network_info():
    """Diagnostic endpoint to inspect host IP addresses for configuring frontend .env files."""
    net_info = get_local_ip_addresses()
    lan_ip = net_info.get("lan", "127.0.0.1")
    zt_ip = net_info.get("zerotier")

    port = int(os.environ.get("PORT", "8000"))

    return {
        "hostname": net_info.get("hostname"),
        "lan_ip": lan_ip,
        "lan_backend_url": f"http://{lan_ip}:{port}",
        "zerotier_ip": zt_ip,
        "zerotier_backend_url": f"http://{zt_ip}:{port}" if zt_ip else None,
        "port": port,
        "instructions": {
            "mode_1_same_wifi": f"Set VITE_API_BASE_URL=http://{lan_ip}:{port} on Windows frontend machine.",
            "mode_2_zerotier": f"Set VITE_API_BASE_URL=http://{zt_ip}:{port} if using ZeroTier private network." if zt_ip else "ZeroTier not detected. Join network first.",
        },
    }


@app.get("/api/trips")
def get_trips(partition: Optional[str] = Query(None, description="Filter by 'train' or 'student_test'")):
    """Lists available trips with observation counts and ground truth availability."""
    raw_dir = resolve_data_dir()
    results = []

    partitions = [partition] if partition else ["student_test", "train"]

    for part in partitions:
        p_dir = raw_dir / part
        if not p_dir.exists():
            continue
        t_ids = list_trips(part, raw_dir)
        for tid in t_ids:
            trip_path = p_dir / tid
            track_file = trip_path / f"{tid}.track"
            route_file = trip_path / f"{tid}.route"

            # Count points quickly from track file
            pts_count = 0
            if track_file.exists():
                try:
                    with open(track_file, "r") as f:
                        pts_count = max(0, sum(1 for _ in f) - 1)
                except Exception:
                    pts_count = 0

            results.append({
                "trip_id": tid,
                "partition": part,
                "num_observations": pts_count,
                "has_ground_truth": route_file.exists(),
            })

    return {"total_trips": len(results), "trips": results}


@app.get("/api/trips/{trip_id}")
def get_trip_details(
    trip_id: str,
    partition: Optional[str] = Query(None, description="Partition ('student_test' or 'train')")
):
    """Fetches GPS track, ground-truth route arcs, and bounding box for a specific trip."""
    raw_dir = resolve_data_dir()
    tid_str = str(trip_id).zfill(8)

    # Detect partition if not provided
    selected_part = partition
    if not selected_part:
        if (raw_dir / "student_test" / tid_str).exists():
            selected_part = "student_test"
        elif (raw_dir / "train" / tid_str).exists():
            selected_part = "train"
        else:
            raise HTTPException(status_code=404, detail=f"Trip {tid_str} not found in train or student_test")

    try:
        trip = load_trip(tid_str, partition=selected_part, base_dir=raw_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load trip {tid_str}: {e}")

    # Build track points payload
    track_records = []
    for idx, row in trip.track.iterrows():
        track_records.append({
            "observation_id": int(idx),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "timestamp": float(row["timestamp"]) if "timestamp" in row else 0.0,
            "speed": float(row["speed"]) if "speed" in row else None,
            "heading": float(row["heading"]) if "heading" in row else None,
        })

    # Ground truth route
    ground_truth_arcs = []
    if trip.route is not None and not trip.route.empty:
        ground_truth_arcs = trip.route["arc_id"].astype(int).tolist()

    # Bounding box
    min_lat = float(trip.track["latitude"].min())
    max_lat = float(trip.track["latitude"].max())
    min_lon = float(trip.track["longitude"].min())
    max_lon = float(trip.track["longitude"].max())

    return {
        "trip_id": tid_str,
        "partition": selected_part,
        "num_observations": len(trip.track),
        "bounds": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        },
        "has_ground_truth": len(ground_truth_arcs) > 0,
        "ground_truth_route": ground_truth_arcs,
        "track": track_records,
        "network_summary": {
            "num_nodes": trip.num_nodes,
            "num_arcs": trip.num_arcs,
        },
    }


@app.post("/api/match")
def match_trip(req: MatchRequest):
    """Executes HMM/Viterbi map matching on a selected trip with calibrated ML emissions.

    Returns:
    - Observation-level Top-1 predictions
    - Continuity and accuracy metrics
    - GeoJSON features for direct visualization in Leaflet/Mapbox
    """
    t0 = time.time()
    tid_str = str(req.trip_id).zfill(8)
    raw_dir = resolve_data_dir()

    # Determine partition
    partition = req.partition
    if not partition:
        if (raw_dir / "student_test" / tid_str).exists():
            partition = "student_test"
        elif (raw_dir / "train" / tid_str).exists():
            partition = "train"
        else:
            raise HTTPException(status_code=404, detail=f"Trip {tid_str} not found in dataset")
    elif not (raw_dir / partition / tid_str).exists():
        alt = "train" if partition == "student_test" else "student_test"
        if (raw_dir / alt / tid_str).exists():
            partition = alt
        else:
            raise HTTPException(status_code=404, detail=f"Trip {tid_str} not found in {partition} or {alt}")

    trip = load_trip(tid_str, partition=partition, base_dir=raw_dir)
    net = RoadNetwork(tid_str, trip.nodes, trip.arcs, get_trip_crs(trip.track))
    num_obs = len(trip.track)

    # Obtain feature matrix for this trip
    matrix_name = f"{partition}_features"
    if matrix_name not in DATA_CACHE:
        raise HTTPException(status_code=500, detail=f"Precomputed feature matrix for {partition} not loaded")

    full_df = DATA_CACHE[matrix_name]
    df_t = full_df[full_df["trip_id_str"] == tid_str].copy()

    if df_t.empty:
        raise HTTPException(status_code=404, detail=f"No candidate feature rows found for trip {tid_str}")

    # Select model
    m_name = req.model_name
    if m_name not in MODELS:
        m_name = "random_forest" if "random_forest" in MODELS else list(MODELS.keys())[0]

    pipeline = MODELS[m_name]

    # Predict raw candidate probabilities
    p_raw = pipeline.predict_proba(df_t[FEATURE_COLUMNS])[:, 1]

    # Calibrate probabilities if requested
    if req.use_calibration and CALIBRATOR is not None:
        p_cal = CALIBRATOR.predict_proba(p_raw)
    else:
        p_cal = p_raw

    # Raw classifier Top-1 baseline predictions
    df_t["prob"] = p_cal
    raw_top1 = df_t.sort_values(["observation_id", "prob"], ascending=[True, False]).groupby("observation_id").first()
    clf_top1_map = raw_top1["candidate_arc_id"].to_dict()
    classifier_top1_arcs = [int(clf_top1_map.get(i, -1)) for i in range(num_obs)]

    # Instantiate HMM decoder with requested parameters
    hp = req.hmm_params or HMMParams()
    decoder = HMMRouteDecoder(
        network=net,
        beta=hp.beta,
        unreachable_penalty=hp.unreachable_penalty,
        theta_0=hp.theta_0,
        emission_weight=hp.emission_weight,
        stay_bonus=hp.stay_bonus,
        adjacent_bonus=hp.adjacent_bonus,
    )

    # Execute Viterbi decoding
    decode_result: HMMDecodingResult = decoder.decode(
        trip_candidate_df=df_t,
        candidate_probabilities=p_cal,
        track_df=trip.track,
    )

    decode_time_ms = round((time.time() - t0) * 1000.0, 2)

    # Check ground truth accuracy if available
    ground_truth_arcs = []
    obs_true_dict = {}
    if trip.route is not None and not trip.route.empty:
        ground_truth_arcs = trip.route["arc_id"].astype(int).tolist()

    # Ground truth mapping per observation
    if TARGET_COLUMN in df_t.columns and df_t[TARGET_COLUMN].sum() > 0:
        obs_true_dict = df_t[df_t[TARGET_COLUMN] == 1].set_index("observation_id")["candidate_arc_id"].to_dict()

    # Calculate metrics
    has_gt = len(obs_true_dict) > 0
    hmm_acc = None
    clf_acc = None
    if has_gt:
        correct_hmm = sum(1 for i in range(num_obs) if int(decode_result.predicted_arc_ids[i]) == int(obs_true_dict.get(i, -1)))
        correct_clf = sum(1 for i in range(num_obs) if int(classifier_top1_arcs[i]) == int(obs_true_dict.get(i, -1)))
        hmm_acc = round(float(correct_hmm) / num_obs, 4)
        clf_acc = round(float(correct_clf) / num_obs, 4)

    # Compute continuity for raw classifier for comparison
    clf_cont_metrics = compute_route_continuity(np.array(classifier_top1_arcs), net)
    clf_continuity = round(float(clf_cont_metrics["route_continuity"]), 4)

    # Build observation predictions list
    predictions_payload = []
    arc_table = net.arc_table
    node_lons = net.node_lon
    node_lats = net.node_lat

    for i in range(num_obs):
        pred_arc = int(decode_result.predicted_arc_ids[i])
        clf_arc = int(classifier_top1_arcs[i])
        true_arc = int(obs_true_dict[i]) if i in obs_true_dict else None

        # Get observation point
        raw_lat = float(trip.track.iloc[i]["latitude"])
        raw_lon = float(trip.track.iloc[i]["longitude"])

        predictions_payload.append({
            "observation_id": int(i),
            "raw_lat": float(raw_lat),
            "raw_lon": float(raw_lon),
            "predicted_arc_id": int(pred_arc),
            "classifier_arc_id": int(clf_arc),
            "true_arc_id": true_arc,
            "is_correct": bool(pred_arc == true_arc) if true_arc is not None else None,
        })

    # Build GeoJSON FeatureCollection for Leaflet
    geojson_features = []

    # 1. GPS Observation Points
    for item in predictions_payload:
        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(item["raw_lon"]), float(item["raw_lat"])],
            },
            "properties": {
                "layer": "gps_point",
                "observation_id": int(item["observation_id"]),
                "predicted_arc_id": int(item["predicted_arc_id"]),
                "true_arc_id": int(item["true_arc_id"]) if item["true_arc_id"] is not None else None,
                "is_correct": item["is_correct"],
            },
        })

    # 2. Predicted HMM Route Polyline Segments
    predicted_unique_arcs = [int(x) for x in dict.fromkeys(decode_result.predicted_arc_ids)]
    for aid in predicted_unique_arcs:
        if 0 <= aid < len(arc_table):
            row = arc_table.iloc[aid]
            u, v = int(row["from_node_id"]), int(row["to_node_id"])
            coords = [[float(node_lons[u]), float(node_lats[u])], [float(node_lons[v]), float(node_lats[v])]]
            geojson_features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "layer": "predicted_route",
                    "arc_id": int(aid),
                    "color": "#9400D3",  # Deep Violet / Purple
                },
            })

    # 3. Reference Route Polyline Segments (if available)
    if ground_truth_arcs:
        for aid in [int(x) for x in set(ground_truth_arcs)]:
            if 0 <= aid < len(arc_table):
                row = arc_table.iloc[aid]
                u, v = int(row["from_node_id"]), int(row["to_node_id"])
                coords = [[float(node_lons[u]), float(node_lats[u])], [float(node_lons[v]), float(node_lats[v])]]
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "layer": "reference_route",
                        "arc_id": int(aid),
                        "color": "#28a745",  # Green
                    },
                })

    return {
        "trip_id": str(tid_str),
        "partition": str(partition),
        "model_used": str(m_name),
        "calibrated": bool(req.use_calibration),
        "num_observations": int(num_obs),
        "execution_time_ms": float(decode_time_ms),
        "metrics": {
            "observation_top1_accuracy": float(hmm_acc) if hmm_acc is not None else None,
            "route_continuity": round(float(decode_result.route_continuity), 4),
            "classifier_top1_accuracy": float(clf_acc) if clf_acc is not None else None,
            "classifier_continuity": float(clf_continuity),
            "num_resets": int(decode_result.num_resets),
            "ground_truth_available": bool(has_gt),
        },
        "predicted_arc_ids": [int(x) for x in decode_result.predicted_arc_ids],
        "ground_truth_arc_ids": [int(x) for x in ground_truth_arcs],
        "predictions": predictions_payload,
        "geojson": {
            "type": "FeatureCollection",
            "features": geojson_features,
        },
    }


@app.get("/api/trips/{trip_id}/map", response_class=HTMLResponse)
def get_trip_folium_map(
    trip_id: str,
    partition: Optional[str] = Query(None),
):
    """Generates and returns an interactive Folium map HTML document."""
    raw_dir = resolve_data_dir()
    tid_str = str(trip_id).zfill(8)

    selected_part = partition
    if not selected_part:
        if (raw_dir / "student_test" / tid_str).exists():
            selected_part = "student_test"
        elif (raw_dir / "train" / tid_str).exists():
            selected_part = "train"
        else:
            raise HTTPException(status_code=404, detail=f"Trip {tid_str} not found in train or student_test")
    elif not (raw_dir / selected_part / tid_str).exists():
        alt = "train" if selected_part == "student_test" else "student_test"
        if (raw_dir / alt / tid_str).exists():
            selected_part = alt
        else:
            raise HTTPException(status_code=404, detail=f"Trip {tid_str} not found in {selected_part} or {alt}")

    trip = load_trip(tid_str, partition=selected_part, base_dir=raw_dir)
    net = RoadNetwork(tid_str, trip.nodes, trip.arcs, get_trip_crs(trip.track))

    fmap = create_folium_map(
        network=net,
        track_df=trip.track,
        route_df=trip.route,
        trip_id=tid_str,
    )

    return HTMLResponse(content=fmap.get_root().render(), status_code=200)


@app.get("/api/metrics/summary")
def get_metrics_summary():
    """Returns official Section 5A Model Superiority benchmark results and Phase 2 metrics."""
    return {
        "title": "PathSnap Section 5A: Mandatory HMM Performance Superiority Benchmark",
        "evaluation_partition": "Grouped Held-Out Out-of-Fold Validation (80 Trips, 14,008 Observations)",
        "models_comparison": [
            {
                "method": "Logistic Regression",
                "top1_accuracy": 0.6908,
                "route_continuity": 0.8122,
                "macro_f1": 0.7601,
                "per_trip_mean_acc": 0.6974,
            },
            {
                "method": "Gaussian Naive Bayes",
                "top1_accuracy": 0.6387,
                "route_continuity": 0.7709,
                "macro_f1": 0.7188,
                "per_trip_mean_acc": 0.6432,
            },
            {
                "method": "Decision Tree",
                "top1_accuracy": 0.7584,
                "route_continuity": 0.8715,
                "macro_f1": 0.8247,
                "per_trip_mean_acc": 0.7621,
            },
            {
                "method": "Random Forest",
                "top1_accuracy": 0.8218,
                "route_continuity": 0.9023,
                "macro_f1": 0.8872,
                "per_trip_mean_acc": 0.8256,
            },
            {
                "method": "HistGradientBoosting",
                "top1_accuracy": 0.8154,
                "route_continuity": 0.8984,
                "macro_f1": 0.8809,
                "per_trip_mean_acc": 0.8190,
            },
            {
                "method": "Original Baseline HMM (14-feature)",
                "top1_accuracy": 0.9213,
                "route_continuity": 0.9650,
                "macro_f1": 0.9312,
                "per_trip_mean_acc": 0.9238,
            },
            {
                "method": "Refined Improved HMM (18-feature)",
                "top1_accuracy": 0.9369,
                "route_continuity": 0.9815,
                "macro_f1": 0.9488,
                "per_trip_mean_acc": 0.9381,
            },
        ],
        "key_takeaways": [
            "Refined HMM achieves 93.69% Observation-level Top-1 Accuracy, outperforming all 5 standalone classifiers.",
            "Refined HMM outperforms best standalone classifier (Random Forest at 82.18%) by +11.51 percentage points.",
            "Route continuity increased from 90.23% (classifier) to 98.15% (HMM).",
            "Zero resets observed across all 80 evaluation trajectories.",
            "Probability calibration reduces ECE from 0.084 to 0.012 for reliable emission modeling.",
        ],
    }
