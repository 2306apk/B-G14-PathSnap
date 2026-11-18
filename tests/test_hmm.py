import numpy as np

from pathsnap.candidates import generate_candidates
from pathsnap.geo import project_trip
from pathsnap.hmm import distance_emission_log_probability, viterbi_decode


def test_distance_emissions_are_finite():
    value = distance_emission_log_probability(np.array([0.0, 20.0]), sigma_m=20)
    assert np.allclose(value, [0.0, -0.5])


def test_viterbi_returns_one_state_per_candidate_observation(toy_trip):
    projected = project_trip(toy_trip)
    candidates = generate_candidates(projected, 20.0)
    decoded = viterbi_decode(projected, candidates)
    assert decoded.observation_id.nunique() == candidates.observation_id.nunique()
    assert decoded.predicted_arc_id.notna().all()
