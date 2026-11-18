"""Stage 6: test 18 real, reference-independent candidate features."""
import numpy as np
from pathsnap.candidates import SpatialCandidates, FEATURE_NAMES
from pathsnap.data import Trip


def sample_trip():
    nodes = np.array([
        [0.0, 12.0], [.0001, 12.0], [.0002, 12.0],
        [.0002, 12.0001], [.0002, 12.0002],
    ])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    track = np.array([
        [0.00002, 12.0, 1.0], [0.00010, 12.0, 2.0],
        [0.00018, 12.0, 3.0], [0.00020, 12.00005, 7.0],
        [0.00020, 12.00015, 8.0],
    ])
    return Trip('00000001', 'train', track, nodes, arcs, None)


def test_exactly_eighteen_finite_features():
    assert len(FEATURE_NAMES) == 18
    assert len(set(FEATURE_NAMES)) == 18
    batch = SpatialCandidates(sample_trip()).query(k=3)
    assert batch.features.shape == (5, 3, 18)
    assert np.isfinite(batch.features).all()


def test_new_features_are_live_and_causal():
    batch = SpatialCandidates(sample_trip()).query(k=3)
    ix = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    assert np.all(batch.features[0, :, ix['gps_time_gap_s']] == 0.0)
    assert np.all(batch.features[3, :, ix['gps_time_gap_s']] == 4.0)
    assert np.any(batch.features[3:, :, ix['gps_turn_angle_rad']] > 0.1)
    assert np.any(np.abs(batch.features[1:, :, ix['gps_acceleration_mps2']]) > 0)
    assert np.all(batch.features[:, :, ix['candidate_distance_margin_m']] >= -1e-6)
    assert np.allclose(batch.features[:, 0, ix['candidate_distance_margin_m']], 0, atol=1e-6)


def test_reference_labels_never_enter_features():
    trip = sample_trip()
    without_reference = SpatialCandidates(trip).query(k=3).features.copy()
    trip.route = np.array([0, 1, 2, 3], dtype=np.int32)
    with_reference = SpatialCandidates(trip).query(k=3).features.copy()
    np.testing.assert_array_equal(without_reference, with_reference)


def test_stride_does_not_change_causal_features():
    generator = SpatialCandidates(sample_trip())
    full = generator.query(k=3)
    subset = generator.query(obs_indices=np.array([0, 2, 4]), k=3)
    np.testing.assert_array_equal(subset.features, full.features[[0, 2, 4]])
