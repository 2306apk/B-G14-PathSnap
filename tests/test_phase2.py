"""Unit tests for Phase 2 components: GroupKFold isolation, calibration, and HMM decoding.
"""

import math
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from pathsnap.calibration import (
    compute_expected_calibration_error,
    evaluate_calibration,
    OutOfFoldCalibrator,
)
from pathsnap.geo import determine_utm_epsg, Projector
from pathsnap.graph import RoadNetwork
from pathsnap.hmm import HMMRouteDecoder


def test_group_kfold_isolation():
    # Verify that GroupKFold strictly isolates trips with 0 overlap
    dummy_trips = np.repeat([f"trip_{i:02d}" for i in range(20)], 50)
    y = np.random.randint(0, 2, size=len(dummy_trips))
    X = np.random.randn(len(dummy_trips), 4)

    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=dummy_trips)):
        tr_trips = set(dummy_trips[tr_idx])
        val_trips = set(dummy_trips[val_idx])
        overlap = tr_trips.intersection(val_trips)
        assert len(overlap) == 0, f"Trip overlap found in fold {fold}: {overlap}"


def test_fold_local_preprocessing():
    # Verify scaler is strictly fit on train fold
    X_train = np.array([[10.0], [20.0], [30.0]])
    X_val = np.array([[100.0], [200.0]])

    scaler = StandardScaler()
    scaler.fit(X_train)

    assert math.isclose(scaler.mean_[0], 20.0)
    # Val data scaled using train mean (20.0), not val mean (150.0)
    val_scaled = scaler.transform(X_val)
    assert val_scaled[0, 0] > 0.0


def test_expected_calibration_error_calculation():
    # Perfectly calibrated case: prob = 0.8, all 1s
    y_true = np.ones(100, dtype=int)
    y_prob = np.full(100, 1.0)
    ece_perfect = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert math.isclose(ece_perfect, 0.0, abs_tol=1e-5)

    # Poorly calibrated case: prob = 0.9, all 0s
    y_true_bad = np.zeros(100, dtype=int)
    y_prob_bad = np.full(100, 0.9)
    ece_bad = compute_expected_calibration_error(y_true_bad, y_prob_bad, n_bins=5)
    assert ece_bad > 0.5


def test_out_of_fold_calibrator():
    oof_probs = np.linspace(0.1, 0.9, 500)
    oof_labels = (oof_probs >= 0.5).astype(int)

    calibrator = OutOfFoldCalibrator(method="isotonic")
    calibrator.fit(oof_probs, oof_labels)

    test_probs = np.array([0.2, 0.5, 0.8])
    cal_probs = calibrator.predict_proba(test_probs)

    assert len(cal_probs) == 3
    assert cal_probs[0] <= cal_probs[1] <= cal_probs[2]


def test_hmm_viterbi_continuity_and_reset():
    # Synthetic 4-node, 3-arc chain network
    crs = determine_utm_epsg(8.60, 50.00)
    projector = Projector(crs)

    mx = np.array([0.0, 100.0, 200.0, 300.0])
    my = np.array([0.0, 0.0, 0.0, 0.0])
    lons, lats = projector.inverse(mx, my)

    nodes_df = pd.DataFrame({"longitude": lons, "latitude": lats})
    arcs_df = pd.DataFrame({"from_node_id": [0, 1, 2], "to_node_id": [1, 2, 3]})
    net = RoadNetwork("synth_hmm", nodes_df, arcs_df, crs, projector)

    # GPS observations: 4 points along the road
    gx = np.array([10.0, 90.0, 110.0, 210.0])
    gy = np.array([1.0, -1.0, 2.0, -1.0])
    glons, glats = projector.inverse(gx, gy)
    timestamps = np.array([1.0, 2.0, 100.0, 101.0])  # Note: 98s gap between point 1 and 2!

    track_df = pd.DataFrame({"longitude": glons, "latitude": glats, "timestamp": timestamps})

    # Candidate table: each observation has candidates arc 0, 1, 2
    cands_records = []
    for o_id in range(4):
        for a_id in range(3):
            cands_records.append({"observation_id": o_id, "candidate_arc_id": a_id, "dist_to_arc": 5.0})
    cand_df = pd.DataFrame(cands_records)

    # Probabilities
    probs = np.array([
        0.8, 0.1, 0.1,  # Obs 0: favor arc 0
        0.7, 0.2, 0.1,  # Obs 1: favor arc 0
        0.1, 0.7, 0.2,  # Obs 2: favor arc 1
        0.1, 0.1, 0.8,  # Obs 3: favor arc 2
    ])

    decoder = HMMRouteDecoder(net, beta=25.0, large_gap_threshold_seconds=30.0)
    res = decoder.decode(cand_df, probs, track_df)

    assert len(res.predicted_arc_ids) == 4
    # Time gap between obs 1 and 2 is 98s (>30s), so a reset must have occurred
    assert res.num_resets >= 1
    # Decoded arcs should match highest probability path: [0, 0, 1, 2]
    np.testing.assert_array_equal(res.predicted_arc_ids, [0, 0, 1, 2])

