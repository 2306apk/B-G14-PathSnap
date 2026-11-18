import numpy as np
from pathsnap.data import Trip,meter_projection
from pathsnap.candidates import SpatialCandidates,FEATURE_NAMES
from pathsnap.decoder import decode
from pathsnap.labels import approximate_reference_labels
from pathsnap.metrics import get_metrics


def mock_trip():
    # 4 connected directed segments along eastbound road; timestamps in seconds.
    lon=np.arange(5)*.0001
    nodes=np.stack([lon,np.full(5,12.)],axis=1)
    arcs=np.array([[0,1],[1,2],[2,3],[3,4]],dtype=np.int32)
    track=np.array([[x,12.+1e-6,float(i+1)] for i,x in enumerate(np.linspace(0,.0004,13))])
    return Trip('00000099','train',track,nodes,arcs,np.arange(4,dtype=np.int32))


def test_projection_is_meters_not_degrees():
    xy=meter_projection(np.array([[0,0],[.001,0.]]),np.array([0.,0.]))
    assert 100<xy[1,0]<120


def test_candidates_and_features():
    trip=mock_trip();batch=SpatialCandidates(trip).query(k=3)
    assert batch.arc_ids.shape==(13,3)
    assert batch.features.shape==(13,3,len(FEATURE_NAMES))
    assert np.all(batch.distance>=0)
    assert np.all(np.isfinite(batch.features))


def test_reference_labels_are_monotone_on_simple_route():
    trip=mock_trip();ids,error=approximate_reference_labels(trip)
    assert len(ids)==len(trip.track)
    assert ids[0]==0 and ids[-1]==3
    assert np.all(np.diff(ids)>=0)
    assert np.median(error)<1


def test_hmm_outputs_valid_arc_ids_without_reference():
    trip=mock_trip();trip.route=None
    batch=SpatialCandidates(trip).query(k=3)
    prob=np.where(batch.arc_ids==0,.01,.5).astype(float)
    out=decode(trip,batch,prob)
    assert len(out)==len(trip.track)
    assert np.all((0<=out)&(out<len(trip.arcs)))


def test_metrics_not_equal_to_binary_accuracy():
    trip=mock_trip();batch=SpatialCandidates(trip).query(k=3)
    truth,_=approximate_reference_labels(trip)
    prob=np.exp(-batch.distance/5)
    m=get_metrics(trip,batch.arc_ids,prob,truth)
    assert 0<=m['observation_top1_accuracy']<=1
    assert 0<=m['candidate_miss_rate']<=1
    assert 'candidate_binary_macro_f1' in m
