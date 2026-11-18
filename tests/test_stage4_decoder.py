"""Stage 4 HMM edge cases. Checks shape, topology, reset and reference independence."""
import numpy as np
import pytest
from pathsnap.data import Trip
from pathsnap.candidates import SpatialCandidates
from pathsnap.decoder import RoadGraph, decode


def fixture_trip():
    nodes=np.array([[0.0,12.0],[0.0001,12.0],[0.0002,12.0],[0.01,12.0],[0.0101,12.0]])
    arcs=np.array([[0,1],[1,2],[3,4]],dtype=np.int32)
    track=np.array([[0.00002,12.0,1.0],[0.00012,12.0,2.0],[0.01002,12.0,100.0],[0.01008,12.0,101.0]])
    return Trip('00000000','student_test',track,nodes,arcs,None)


def test_graph_respects_directed_disconnection():
    graph=RoadGraph(fixture_trip())
    assert np.isfinite(graph.path_length(0,2,1000))
    assert not np.isfinite(graph.path_length(2,3,10000))


def test_viterbi_resets_after_large_gap_without_reference():
    trip=fixture_trip();batch=SpatialCandidates(trip).query(k=2)
    probabilities=np.exp(-batch.distance/30.)+1e-8
    decoded=decode(trip,batch,probabilities)
    assert decoded.shape==(len(trip.track),)
    assert np.all(np.any(batch.arc_ids==decoded[:,None],axis=1))


def test_viterbi_rejects_wrong_probability_shape():
    trip=fixture_trip();batch=SpatialCandidates(trip).query(k=2)
    with pytest.raises(ValueError):
        decode(trip,batch,np.ones((len(trip.track),3)))


def test_viterbi_handles_empty_samples():
    trip=fixture_trip();batch=SpatialCandidates(trip).query(k=2)
    assert decode(trip,batch,np.ones((0,2)),obs_indices=np.array([],dtype=int)).size==0
