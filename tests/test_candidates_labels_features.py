import numpy as np

from pathsnap.candidates import candidate_coverage, generate_candidates, select_candidate_radius
from pathsnap.features import build_feature_rows, circular_difference_degrees
from pathsnap.geo import project_trip
from pathsnap.labels import associate_reference_route, monotonic_route_match


def test_monotonic_match_never_moves_backward():
    distances = np.array([[0, 10, 20], [1, 3, 20], [20, 1, 10], [30, 15, 0]], dtype=float)
    positions = monotonic_route_match(distances)
    assert np.all(np.diff(positions) >= 0)
    assert positions[0] == 0 and positions[-1] == 2


def test_reference_association_and_candidate_query(toy_trip):
    projected = project_trip(toy_trip)
    labels = associate_reference_route(projected)
    candidates = generate_candidates(projected, 20.0)
    assert len(labels) == len(toy_trip.track)
    assert labels.reference_route_position.is_monotonic_increasing
    assert candidates.observation_id.nunique() == 3
    assert (candidates.candidate_distance_m <= 20.0 + 1e-8).all()
    coverage = candidate_coverage(candidates, labels, len(toy_trip.track))
    assert coverage["no_candidate_count"] == 0


def test_candidate_query_does_not_fallback(toy_trip):
    projected = project_trip(toy_trip)
    projected.track_xy[:] += 1_000_000
    candidates = generate_candidates(projected, 5.0)
    assert candidates.empty


def test_radius_is_training_quantile_derived():
    result = select_candidate_radius(np.array([1.0, 2.0, 3.0, 9.1]))
    assert result["selected_radius_m"] == 10.0


def test_feature_calculations(toy_trip):
    projected = project_trip(toy_trip)
    labels = associate_reference_route(projected)
    candidates = generate_candidates(projected, 20.0)
    features = build_feature_rows(projected, candidates, labels)
    assert set(["target", "projection_position", "vehicle_speed_mps"]).issubset(features)
    assert features.projection_position.between(0, 1).all()
    assert features.groupby("observation_id").target.max().eq(1).all()
    assert features[features.observation_id == 0].vehicle_heading_deg.isna().all()


def test_circular_heading_difference():
    result = circular_difference_degrees(np.array([350, 10, 0]), np.array([10, 350, 180]))
    assert np.allclose(result, [20, 20, 180])
