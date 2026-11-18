"""Candidate, observation-level, and route-continuity metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    prediction = (probability >= 0.5).astype(int)
    return {
        "candidate_accuracy": float(accuracy_score(y_true, prediction)),
        "candidate_macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "candidate_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "candidate_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "candidate_log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def observation_top1(features: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    scored = features[["trip_id", "observation_id", "arc_id", "reference_arc_id"]].copy()
    scored["probability"] = probability
    top = scored.sort_values(
        ["trip_id", "observation_id", "probability", "arc_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["trip_id", "observation_id"])
    top["correct"] = top["arc_id"] == top["reference_arc_id"]
    return top


def observation_metrics(features: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    top = observation_top1(features, probability)
    return {"observation_top1_accuracy": float(top["correct"].mean()), "scored_observations": int(len(top))}


def route_continuity(arcs: pd.DataFrame, predicted_arc_ids: np.ndarray) -> float:
    ids = np.asarray(predicted_arc_ids, dtype=np.int64)
    if len(ids) < 2:
        return 1.0
    selected = arcs.iloc[ids]
    connected = (
        (ids[:-1] == ids[1:])
        | (selected["to_node_id"].to_numpy()[:-1] == selected["from_node_id"].to_numpy()[1:])
    )
    return float(connected.mean())
