"""Comprehensive unit tests for PathSnap Phase 1 components.

Tests:
1. File parsing & data integrity
2. Empty and invalid inputs
3. Coordinate projection & UTM selection
4. Geometry, bearing, and distance calculations
5. Spatial indexing & candidate queries
6. Monotonic DP label association
7. Feature engineering & data leakage safeguards
8. Evaluation metrics
"""

import math
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from pathsnap.io import (
    load_track,
    load_nodes,
    load_arcs,
    load_route,
    validate_trip_integrity,
    TripData,
)
from pathsnap.geo import (
    determine_utm_epsg,
    Projector,
    calculate_bearing,
    calculate_heading_difference,
    point_to_segment_projection,
    SpatialRoadIndex,
)
from pathsnap.graph import RoadNetwork
from pathsnap.candidates import generate_candidates_for_trip
from pathsnap.labels import associate_points_to_route
from pathsnap.features import (
    build_observation_gps_features,
    build_candidate_features_dataset,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
)
from pathsnap.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_accuracy,
    compute_route_continuity,
)


# --- 1. Coordinate Projections & Geometry ---

def test_utm_epsg_determination():
    # Northern hemisphere (Germany, zone 32N)
    crs_de = determine_utm_epsg(8.60, 50.02)
    assert crs_de.utm_zone == 32
    assert crs_de.is_northern is True
    assert crs_de.epsg == 32632

    # Southern hemisphere (Sydney, zone 56S)
    crs_syd = determine_utm_epsg(151.20, -33.86)
    assert crs_syd.utm_zone == 56
    assert crs_syd.is_northern is False
    assert crs_syd.epsg == 32756

    # US Midwest (Minnesota, zone 15N)
    crs_us = determine_utm_epsg(-93.38, 44.85)
    assert crs_us.utm_zone == 15
    assert crs_us.epsg == 32615

    # Invalid coordinates
    with pytest.raises(ValueError):
        determine_utm_epsg(190.0, 45.0)
    with pytest.raises(ValueError):
        determine_utm_epsg(10.0, -95.0)


def test_projector_roundtrip():
    crs = determine_utm_epsg(8.60, 50.02)
    projector = Projector(crs)

    orig_lons = np.array([8.60, 8.61, 8.62])
    orig_lats = np.array([50.02, 50.03, 50.04])

    x, y = projector.forward(orig_lons, orig_lats)
    back_lons, back_lats = projector.inverse(x, y)

    np.testing.assert_allclose(orig_lons, back_lons, atol=1e-5)
    np.testing.assert_allclose(orig_lats, back_lats, atol=1e-5)


def test_bearing_calculation():
    # Due North: dx=0, dy=10 -> bearing = 0.0
    assert math.isclose(calculate_bearing(0, 0, 0, 10), 0.0, abs_tol=1e-5)
    # Due East: dx=10, dy=0 -> bearing = 90.0
    assert math.isclose(calculate_bearing(0, 0, 10, 0), 90.0, abs_tol=1e-5)
    # Due South: dx=0, dy=-10 -> bearing = 180.0
    assert math.isclose(calculate_bearing(0, 0, 0, -10), 180.0, abs_tol=1e-5)
    # Due West: dx=-10, dy=0 -> bearing = 270.0
    assert math.isclose(calculate_bearing(0, 0, -10, 0), 270.0, abs_tol=1e-5)


def test_heading_difference():
    # Identical
    assert calculate_heading_difference(45.0, 45.0) == 0.0
    # Opposite
    assert calculate_heading_difference(0.0, 180.0) == 180.0
    # Circular wrap (10 deg vs 350 deg -> diff is 20 deg)
    assert math.isclose(calculate_heading_difference(10.0, 350.0), 20.0, abs_tol=1e-5)


def test_point_to_segment_projection():
    # Segment from (0, 0) to (10, 0)
    # Point at (5, 5) -> proj at (5, 0), dist=5.0, t=0.5
    d_seg, d_from, d_to, t, qx, qy = point_to_segment_projection(5, 5, 0, 0, 10, 0)
    assert math.isclose(d_seg, 5.0, abs_tol=1e-5)
    assert math.isclose(t, 0.5, abs_tol=1e-5)
    assert math.isclose(qx, 5.0, abs_tol=1e-5)
    assert math.isclose(qy, 0.0, abs_tol=1e-5)

    # Point beyond segment end: (15, 0) -> clamped at (10, 0), t=1.0, dist=5.0
    d_seg, d_from, d_to, t, qx, qy = point_to_segment_projection(15, 0, 0, 0, 10, 0)
    assert math.isclose(d_seg, 5.0, abs_tol=1e-5)
    assert math.isclose(t, 1.0, abs_tol=1e-5)
    assert math.isclose(qx, 10.0, abs_tol=1e-5)

    # Point before segment start: (-5, 0) -> clamped at (0, 0), t=0.0, dist=5.0
    d_seg, d_from, d_to, t, qx, qy = point_to_segment_projection(-5, 0, 0, 0, 10, 0)
    assert math.isclose(d_seg, 5.0, abs_tol=1e-5)
    assert math.isclose(t, 0.0, abs_tol=1e-5)
    assert math.isclose(qx, 0.0, abs_tol=1e-5)


# --- 2. Spatial Index & Candidates ---

def test_spatial_road_index():
    # Create two segments: line 0 from (0,0) to (10,0), line 1 from (100,100) to (110,100)
    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(100, 100), (110, 100)]),
    ]
    index = SpatialRoadIndex(lines, np.array([0, 1]))

    # Query near line 0 with radius 5m
    cands = index.query_radius(5, 2, radius=5.0)
    assert cands == [0]

    # Query far from both
    cands_empty = index.query_radius(50, 50, radius=10.0)
    assert len(cands_empty) == 0


# --- 3. Empty & Invalid Inputs Validation ---

def test_empty_and_invalid_inputs(tmp_path):
    # Empty file test
    empty_file = tmp_path / "empty.track"
    empty_file.touch()
    with pytest.raises(ValueError, match="empty"):
        load_track(empty_file)

    # Missing file test
    with pytest.raises(FileNotFoundError):
        load_track(tmp_path / "nonexistent.track")

    # NaN coordinates validation
    track_nan = pd.DataFrame({"longitude": [8.5, np.nan], "latitude": [50.0, 50.1], "timestamp": [1.0, 2.0]})
    nodes_ok = pd.DataFrame({"longitude": [8.5, 8.6], "latitude": [50.0, 50.1]})
    arcs_ok = pd.DataFrame({"from_node_id": [0], "to_node_id": [1]})
    errors, warnings = validate_trip_integrity(track_nan, nodes_ok, arcs_ok)
    assert any("NaN" in e for e in errors)

    # Out of range arc-to-node references
    arcs_bad = pd.DataFrame({"from_node_id": [0], "to_node_id": [999]})  # node 999 does not exist
    errors, warnings = validate_trip_integrity(track_nan, nodes_ok, arcs_bad)
    assert any("out of range" in e for e in errors)


# --- 4. Synthetic End-to-End Pipeline ---

@pytest.fixture
def synthetic_trip():
    """Builds a small, consistent synthetic road network and trajectory for rigorous unit tests."""
    crs = determine_utm_epsg(8.60, 50.00)
    projector = Projector(crs)

    # 4 nodes along a straight road: (0,0) -> (100,0) -> (200,0) -> (300,0) in metric coords
    metric_x = np.array([0.0, 100.0, 200.0, 300.0])
    metric_y = np.array([0.0, 0.0, 0.0, 0.0])
    lons, lats = projector.inverse(metric_x, metric_y)

    nodes_df = pd.DataFrame({"longitude": lons, "latitude": lats})

    # 3 arcs connecting the nodes
    arcs_df = pd.DataFrame(
        {
            "from_node_id": [0, 1, 2],
            "to_node_id": [1, 2, 3],
        }
    )

    # Route traverses all 3 arcs in order: 0, 1, 2
    route_df = pd.DataFrame({"arc_id": [0, 1, 2]})

    # GPS observations: 5 points along the road with small 2m lateral noise
    gps_x = np.array([10.0, 60.0, 120.0, 180.0, 250.0])
    gps_y = np.array([2.0, -1.5, 1.0, 2.5, -0.5])
    gps_lons, gps_lats = projector.inverse(gps_x, gps_y)
    timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    track_df = pd.DataFrame(
        {"longitude": gps_lons, "latitude": gps_lats, "timestamp": timestamps}
    )

    network = RoadNetwork("synth_trip", nodes_df, arcs_df, crs, projector)
    return network, track_df, route_df


def test_synthetic_route_association(synthetic_trip):
    network, track_df, route_df = synthetic_trip
    result = associate_points_to_route(network, track_df, route_df)

    assert result.num_points == 5
    assert len(result.assigned_arc_ids) == 5
    # Points 0 & 1 should be on arc 0
    assert result.assigned_arc_ids[0] == 0
    assert result.assigned_arc_ids[1] == 0
    # Points 2 & 3 should be on arc 1
    assert result.assigned_arc_ids[2] == 1
    assert result.assigned_arc_ids[3] == 1
    # Point 4 should be on arc 2
    assert result.assigned_arc_ids[4] == 2

    # All points are within 3m of the road
    assert result.mean_distance < 3.0
    assert result.outlier_count == 0


def test_synthetic_feature_engineering_no_leakage(synthetic_trip):
    network, track_df, route_df = synthetic_trip
    labels_res = associate_points_to_route(network, track_df, route_df)
    labels_df = labels_res.to_dataframe()

    cand_res = generate_candidates_for_trip(network, track_df, radius=20.0)
    feat_df = build_candidate_features_dataset(network, track_df, cand_res, labels_df)

    # Verify all documented features exist
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column: {col}"
        assert not feat_df[col].isna().any(), f"NaNs found in feature {col}"

    # Verify target label
    assert "label" in feat_df.columns
    assert set(feat_df["label"].unique()).issubset({0, 1})

    # Speed at t=0 must be 0.0 to prevent future leakage
    t0_rows = feat_df[feat_df["observation_id"] == 0]
    assert (t0_rows["speed_est"] == 0.0).all()


def test_metrics_evaluation():
    # 1. Classification metrics
    y_true = np.array([1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.2, 0.4, 0.1])

    clf_metrics = compute_candidate_classification_metrics(y_true, y_pred, y_prob)
    assert clf_metrics["accuracy"] == 0.8
    assert clf_metrics["f1_macro"] > 0.0
    assert not math.isnan(clf_metrics["log_loss"])

    # 2. Top-1 Accuracy
    feat_df = pd.DataFrame(
        {
            "observation_id": [0, 0, 1, 1],
            "label": [1, 0, 0, 1],
        }
    )
    scores = np.array([0.9, 0.2, 0.1, 0.8])
    top1 = compute_observation_top1_accuracy(feat_df, scores)
    assert top1["top1_accuracy"] == 1.0
    assert top1["candidate_miss_rate"] == 0.0

