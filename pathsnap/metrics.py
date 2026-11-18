"""Candidate and observation-level metrics; reference alignment is approximate."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, confusion_matrix
from .data import Trip


def get_metrics(trip: Trip, arc_ids: np.ndarray, prob: np.ndarray, truth: np.ndarray,
                predicted_route: np.ndarray | None = None) -> dict:
    arc_ids = np.asarray(arc_ids)
    prob = np.asarray(prob)
    truth = np.asarray(truth)
    assert arc_ids.shape == prob.shape and len(truth) == len(arc_ids)
    predictions = arc_ids[np.arange(len(arc_ids)),np.argmax(prob,axis=1)]
    y_true = (arc_ids==truth[:,None]).ravel().astype(int)
    y_pred = (arc_ids==predictions[:,None]).ravel().astype(int)
    out = {
        'observation_top1_accuracy': float(np.mean(predictions==truth)),
        'candidate_miss_rate': float(np.mean(~np.any(arc_ids==truth[:,None],axis=1))),
        # Macro F1 is binary candidate classification. Do NOT describe it as
        # macro F1 of trip-local arc IDs or route classes.
        'candidate_binary_macro_f1': float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
        'candidate_binary_precision': float(precision_score(y_true,y_pred,zero_division=0)),
        'candidate_binary_recall': float(recall_score(y_true,y_pred,zero_division=0)),
        'candidate_binary_accuracy': float(accuracy_score(y_true,y_pred)),
        'candidate_binary_log_loss': float(log_loss(y_true,np.clip(prob.ravel(),1e-8,1-1e-8),labels=[0,1])),
        'candidate_confusion_matrix': confusion_matrix(y_true,y_pred,labels=[0,1]).tolist(),
        'classifier_direct_continuity': float(direct_continuity(trip,predictions)),
    }
    if predicted_route is not None:
        if len(predicted_route) != len(truth):
            raise ValueError('Decoded route length mismatch')
        out['hmm_observation_accuracy'] = float(np.mean(predicted_route==truth))
        out['hmm_direct_continuity'] = float(direct_continuity(trip,predicted_route))
    return out


def direct_continuity(trip: Trip, predicted: np.ndarray) -> float:
    """Same arc or immediately connected directed arcs; skipping edges counts discontinuity."""
    if len(predicted)<2:
        return 1.
    ids1,ids2=predicted[:-1],predicted[1:]
    return float(np.mean((ids1==ids2)|(trip.arcs[ids1,1]==trip.arcs[ids2,0])))
