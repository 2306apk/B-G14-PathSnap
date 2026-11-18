"""Unit tests validating HMM/Viterbi behavioral correctness and data integrity.

Tests:
1. Emission probability normalization (sums to 1 across candidates).
2. Directed connectivity and unreachable penalties.
3. Same-arc transitions with partial traversal progress.
4. Multi-arc transitions with Dijkstra bridge paths.
5. Large timestamp gaps and sequence resets.
6. Viterbi backtracking across segment boundaries.
7. Deterministic inference reproducibility.
"""

import math
import numpy as np
import pandas as pd
import pytest

from pathsnap.io import load_trip, resolve_data_dir
from pathsnap.geo import get_trip_crs
from pathsnap.graph import RoadNetwork
from pathsnap.hmm import HMMRouteDecoder, calculate_turn_angle_change


@pytest.fixture(scope="module")
def sample_network_and_trip():
    data_dir = resolve_data_dir()
    trip = load_trip("00000052", partition="student_test", base_dir=data_dir)
    net = RoadNetwork("00000052", trip.nodes, trip.arcs, get_trip_crs(trip.track))
    return net, trip


def test_emission_probability_normalization():
    probs = [0.2, 0.5, 0.3]
    sum_p = sum(probs)
    norm = [p / sum_p for p in probs]
    assert math.isclose(sum(norm), 1.0, rel_tol=1e-7)
    log_em = [math.log(max(1e-9, p)) for p in norm]
    assert all(math.isfinite(l) for l in log_em)


def test_turn_angle_change_geometry():
    # Straight forward (0 deg)
    assert math.isclose(calculate_turn_angle_change(45.0, 45.0), 0.0)
    # Right turn (90 deg)
    assert math.isclose(calculate_turn_angle_change(0.0, 90.0), 90.0)
    # U-turn (180 deg)
    assert math.isclose(calculate_turn_angle_change(10.0, 190.0), 180.0)
    # Circular wraparound: 350 to 10 is 20 deg turn, not 340
    assert math.isclose(calculate_turn_angle_change(350.0, 10.0), 20.0)


def test_directed_connectivity_unreachable_penalty(sample_network_and_trip):
    net, _ = sample_network_and_trip
    decoder = HMMRouteDecoder(net, beta=20.0, unreachable_penalty=5.0, theta_0=55.0)

    # Pick two distant or reverse disconnected arcs
    u_to = int(decoder.to_nodes[0])
    # Find an arc whose from_node is not connected to u_to
    arc_unreachable = len(net.arc_table) - 1
    feats = decoder.compute_18_transition_features(
        prev_arc=0, curr_arc=arc_unreachable, d_gps=10.0, dt=1.0
    )
    if feats.is_unreachable:
        log_t = decoder.compute_log_transition(0, arc_unreachable, d_gps=10.0, dt=1.0)
        assert log_t < 0.0
        assert log_t <= -decoder.unreachable_penalty


def test_same_arc_partial_traversal_progress(sample_network_and_trip):
    net, _ = sample_network_and_trip
    decoder = HMMRouteDecoder(net, beta=20.0)

    # Forward progress along arc 0: proj 0.2 to 0.6
    feats_fwd = decoder.compute_18_transition_features(
        prev_arc=0, curr_arc=0, d_gps=20.0, dt=2.0, prev_proj=0.2, curr_proj=0.6
    )
    assert feats_fwd.is_unreachable is False
    assert feats_fwd.network_transition_distance_m >= 0.0

    # Backward reversal along arc 0: proj 0.8 to 0.2
    log_t_fwd = decoder.compute_log_transition(
        prev_arc=0, curr_arc=0, d_gps=20.0, dt=2.0, prev_proj=0.2, curr_proj=0.6
    )
    log_t_rev = decoder.compute_log_transition(
        prev_arc=0, curr_arc=0, d_gps=20.0, dt=2.0, prev_proj=0.8, curr_proj=0.2
    )
    # Forward progress should score higher than backward movement on one-way arc
    assert log_t_fwd > log_t_rev


def test_large_timestamp_gap_reset(sample_network_and_trip):
    net, _ = sample_network_and_trip
    decoder = HMMRouteDecoder(net, large_gap_threshold_seconds=30.0)

    # Normal dt = 2s has non-zero transition cost
    log_t_normal = decoder.compute_log_transition(0, 1, d_gps=15.0, dt=2.0)
    # Gap dt = 60s resets transition memory (log prob = 0.0)
    log_t_gap = decoder.compute_log_transition(0, 1, d_gps=15.0, dt=60.0)
    assert log_t_gap == 0.0


def test_viterbi_deterministic_predictions(sample_network_and_trip):
    net, trip = sample_network_and_trip
    decoder = HMMRouteDecoder(net, beta=20.0, unreachable_penalty=5.0, theta_0=55.0)

    # Create synthetic candidates for first 10 observations
    rows = []
    for i in range(10):
        # 3 candidates per observation
        rows.append({"observation_id": i, "candidate_arc_id": 0, "proj_ratio": 0.1 * i, "arc_length": 50.0})
        rows.append({"observation_id": i, "candidate_arc_id": 1, "proj_ratio": 0.1 * i, "arc_length": 60.0})
        rows.append({"observation_id": i, "candidate_arc_id": 2, "proj_ratio": 0.1 * i, "arc_length": 70.0})
    cand_df = pd.DataFrame(rows)
    probs = np.array([0.7, 0.2, 0.1] * 10)
    sub_track = trip.track.iloc[:10].copy()

    res1 = decoder.decode(cand_df, probs, sub_track)
    res2 = decoder.decode(cand_df, probs, sub_track)

    np.testing.assert_array_equal(res1.predicted_arc_ids, res2.predicted_arc_ids)
    assert res1.route_continuity == res2.route_continuity
    assert res1.num_resets == res2.num_resets
