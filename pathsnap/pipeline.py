"""Inference path intentionally never loads/reference .route file."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from .data import Trip
from .candidates import SpatialCandidates, CandidateBatch, FEATURE_NAMES
from .decoder import decode


def load_model(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Model and candidate feature columns are incompatible')
    return bundle


def predict(trip: Trip, bundle: dict, run_decoder=True, candidate_k: int | None = None,
            indices: np.ndarray | None = None) -> dict:
    if indices is None:
        indices=np.arange(len(trip.track))
    indices=np.asarray(indices,dtype=int)
    generator=SpatialCandidates(trip)
    batch=generator.query(indices,k=candidate_k or bundle['candidate_k'])
    n,k=batch.arc_ids.shape
    model=bundle['estimator']
    prob=model.predict_proba(batch.features.reshape(n*k,-1))[:,1].reshape(n,k)
    classifier= batch.arc_ids[np.arange(n),np.argmax(prob,axis=1)]
    hmm=decode(trip,batch,prob,indices) if run_decoder else classifier.copy()
    return {'batch':batch,'prob':prob,'classifier':classifier,'hmm':hmm,'indices':indices}
