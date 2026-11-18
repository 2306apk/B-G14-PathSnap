"""Evaluation metrics for candidate classification and route reconstruction.

Implements observation-level Top-1 accuracy, candidate miss rate, binary classification metrics,
and topological route continuity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

from pathsnap.graph import RoadNetwork


def compute_candidate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Computes binary classification metrics for candidate pairs."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    f1_binary = float(f1_score(y_t, y_p, average="binary", zero_division=0))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1_macro,
        "f1_binary": f1_binary,
    }

    if y_prob is not None:
        # Clip probabilities to avoid log(0)
        eps = 1e-15
        p_clipped = np.clip(y_prob, eps, 1.0 - eps)
        try:
            metrics["log_loss"] = float(log_loss(y_t, p_clipped))
        except Exception:
            metrics["log_loss"] = float("nan")

    return metrics


def compute_observation_top1_accuracy(
    features_df: pd.DataFrame,
    predicted_scores: np.ndarray,
) -> Dict[str, float]:
    """Computes observation-level Top-1 accuracy and candidate miss rate.

    For each GPS observation:
    Select candidate with maximum predicted score.
    Observation is correct if chosen candidate has label == 1.
    """
    df = features_df[["observation_id", "label"]].copy()
    df["score"] = np.asarray(predicted_scores, dtype=np.float64)

    # Group by observation_id and pick candidate with max score
    top_candidates = df.sort_values(["observation_id", "score"], ascending=[True, False]).groupby(
        "observation_id"
    ).first()

    total_observations = len(top_candidates)
    correct_observations = int(top_candidates["label"].sum())
    top1_accuracy = (correct_observations / total_observations) if total_observations > 0 else 0.0

    # Candidate miss rate: observations where NO candidate had label == 1
    has_positive = df.groupby("observation_id")["label"].max()
    missed_count = int((has_positive == 0).sum())
    miss_rate = (missed_count / total_observations) if total_observations > 0 else 0.0

    return {
        "top1_accuracy": top1_accuracy,
        "candidate_miss_rate": miss_rate,
        "total_evaluated_observations": total_observations,
        "correct_observations": correct_observations,
        "missed_observations": missed_count,
    }


def compute_route_continuity(
    predicted_arc_ids: List[int] | np.ndarray,
    network: RoadNetwork,
) -> Dict[str, float]:
    """Calculates route continuity for a sequence of predicted road arcs.

    Two consecutive predicted arcs (a_t, a_{t+1}) are continuous if:
    1. They are the same arc (vehicle stayed on the arc).
    2. to_node(a_t) == from_node(a_{t+1}) (direct network connection).
    """
    arcs = np.asarray(predicted_arc_ids, dtype=np.int64)
    total_transitions = len(arcs) - 1
    if total_transitions <= 0:
        return {"route_continuity": 1.0, "direct_connections": 0, "total_transitions": 0}

    to_nodes = network.arc_table["to_node_id"].values
    from_nodes = network.arc_table["from_node_id"].values

    continuous_count = 0
    for t in range(total_transitions):
        a1 = arcs[t]
        a2 = arcs[t + 1]
        if a1 == a2:
            continuous_count += 1
        elif to_nodes[a1] == from_nodes[a2]:
            continuous_count += 1

    continuity_ratio = continuous_count / total_transitions
    return {
        "route_continuity": float(continuity_ratio),
        "continuous_transitions": continuous_count,
        "total_transitions": total_transitions,
    }

