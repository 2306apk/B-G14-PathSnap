"""Unit tests for Phase 2 Refinement: 18-feature map matching, HMM decoding, and calibration."""

import math
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from pathsnap.features import (
    OBSERVATION_FEATURE_COLUMNS,
    TRANSITION_FEATURE_COLUMNS,
    ALL_MAPMATCHING_FEATURES,
    calculate_turn_angle_change,
    TransitionFeatures,
)
from pathsnap.geo import Projector, determine_utm_epsg
from pathsnap.graph import RoadNetwork
from pathsnap.calibration import (
    compute_expected_calibration_error,
    OutOfFoldCalibrator,
    cross_val_evaluate_calibration,
)
from pathsnap.hmm import HMMRouteDecoder, HMMDecodingResult


def test_18_features_schema_definitions():
    """Validates that exactly 18 distinct map matching features are defined with proper categorization."""
    assert len(OBSERVATION_FEATURE_COLUMNS) == 14
    assert len(TRANSITION_FEATURE_COLUMNS) == 4
    assert len(ALL_MAPMATCHING_FEATURES) == 18
    assert len(set(ALL_MAPMATCHING_FEATURES)) == 18

    # Validate the 4 transition features
    expected_trans = [
        "gps_displacement_m",
        "network_transition_distance_m",
        "transition_distance_residual_m",
        "turn_angle_change_deg",
    ]
    for feat in expected_trans:
        assert feat in TRANSITION_FEATURE_COLUMNS


def test_turn_angle_change_circular_wraparound():
    """Tests circular angular difference calculation between road bearings."""
    # Exact match
    assert calculate_turn_angle_change(45.0, 45.0) == 0.0

    # Normal turn
    assert calculate_turn_angle_change(10.0, 70.0) == 60.0

    # Wraparound across North (0 / 360)
    assert calculate_turn_angle_change(10.0, 350.0) == 20.0
    assert calculate_turn_angle_change(355.0, 5.0) == 10.0

    # Opposite directions (U-turn)
    assert calculate_turn_angle_change(0.0, 180.0) == 180.0
    assert calculate_turn_angle_change(90.0, 270.0) == 180.0
    assert calculate_turn_angle_change(58.96, 238.96) == pytest.approx(180.0, abs=1e-4)


@pytest.fixture
def synthetic_triangle_network():
    """Creates a simple synthetic 3-node, 4-arc directed road network in local UTM (meters).

    Nodes:
      0: (0, 0)
      1: (100, 0)
      2: (100, 100)

    Arcs:
      0: 0 -> 1 (heading 90 deg, East, length 100)
      1: 1 -> 0 (heading 270 deg, West, reverse of arc 0)
      2: 1 -> 2 (heading 0 deg, North, length 100)
      3: 2 -> 0 (heading 225 deg, return path)
    """
    nodes_df = pd.DataFrame({
        "longitude": [0.0, 0.0009, 0.0009],
        "latitude": [0.0, 0.0, 0.0009],
    })
    arcs_df = pd.DataFrame({
        "from_node_id": [0, 1, 1, 2],
        "to_node_id": [1, 0, 2, 0],
    })

    crs_info = determine_utm_epsg(0.0, 0.0)
    net = RoadNetwork("synthetic_test", nodes_df, arcs_df, crs_info=crs_info)
    return net


def test_transition_features_extraction(synthetic_triangle_network):
    """Tests computation of all four transition features across road segments."""
    net = synthetic_triangle_network
    dec = HMMRouteDecoder(net, beta=20.0, unreachable_penalty=5.0)

    # 1. Same road segment (arc 0 -> arc 0)
    feats_same = dec.compute_18_transition_features(prev_arc=0, curr_arc=0, d_gps=25.0, dt=1.0)
    assert feats_same.gps_displacement_m == 25.0
    assert feats_same.network_transition_distance_m == 25.0
    assert feats_same.transition_distance_residual_m == 0.0
    assert feats_same.turn_angle_change_deg == 0.0
    assert not feats_same.is_unreachable

    # 2. Adjacent right turn (arc 0 [East] -> arc 2 [North])
    # Endpoint of arc 0 is node 1; start of arc 2 is node 1.
    feats_turn = dec.compute_18_transition_features(prev_arc=0, curr_arc=2, d_gps=20.0, dt=1.0)
    assert feats_turn.gps_displacement_m == 20.0
    assert feats_turn.turn_angle_change_deg == pytest.approx(90.0, abs=1.0)
    assert not feats_turn.is_unreachable
    assert feats_turn.transition_distance_residual_m == 0.0

    # 3. U-turn on opposite lane (arc 0 [East] -> arc 1 [West])
    # Both meet at node 1, but turn angle is 180 degrees
    feats_uturn = dec.compute_18_transition_features(prev_arc=0, curr_arc=1, d_gps=10.0, dt=1.0)
    assert feats_uturn.turn_angle_change_deg == pytest.approx(180.0, abs=1.0)


def test_turn_angle_penalty_suppresses_uturns(synthetic_triangle_network):
    """Tests that the HMM scoring penalizes U-turns compared to continuing forward or turning normally."""
    net = synthetic_triangle_network
    dec = HMMRouteDecoder(net, beta=20.0, unreachable_penalty=5.0, theta_0=55.0)

    score_same = dec.compute_log_transition(prev_arc=0, curr_arc=0, d_gps=10.0, dt=1.0)
    score_right = dec.compute_log_transition(prev_arc=0, curr_arc=2, d_gps=10.0, dt=1.0)
    score_uturn = dec.compute_log_transition(prev_arc=0, curr_arc=1, d_gps=10.0, dt=1.0)

    # U-turn must have significantly worse score than staying on segment or forward turn
    assert score_same > score_right
    assert score_right > score_uturn
    assert (score_right - score_uturn) > 5.0  # Heavy U-turn penalty


def test_gap_reset_handling(synthetic_triangle_network):
    """Tests that transitions across large timestamp gaps (>30s) are reset to 0.0."""
    net = synthetic_triangle_network
    dec = HMMRouteDecoder(net, large_gap_threshold_seconds=30.0)

    # Normal gap (1.0s) -> non-zero transition
    score_normal = dec.compute_log_transition(prev_arc=0, curr_arc=2, d_gps=10.0, dt=1.0)
    assert score_normal != 0.0

    # Large gap (45.0s) -> reset to 0.0 (independent transition)
    score_gap = dec.compute_log_transition(prev_arc=0, curr_arc=2, d_gps=10.0, dt=45.0)
    assert score_gap == 0.0


def test_nested_cross_val_calibration():
    """Tests that cross_val_evaluate_calibration evaluates strictly on held-out folds."""
    rng = np.random.RandomState(42)
    n_samples = 1000
    # True uncalibrated probabilities with known miscalibration
    probs = rng.uniform(0.1, 0.9, size=n_samples)
    labels = (rng.uniform(0.0, 1.0, size=n_samples) < (probs ** 2)).astype(int)
    fold_ids = np.repeat([1, 2, 3, 4, 5], n_samples // 5)

    cal_probs, report = cross_val_evaluate_calibration(probs, labels, fold_ids, method="isotonic")

    assert len(cal_probs) == n_samples
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
    # Calibrated ECE must be positive and non-zero (since evaluation is strictly out-of-sample)
    assert report.ece_calibrated > 1e-4
    assert report.ece_calibrated < report.ece_uncalibrated
