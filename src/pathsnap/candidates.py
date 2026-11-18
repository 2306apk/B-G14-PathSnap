"""Trip-local spatial candidate generation and training-derived radius selection."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .geo import ProjectedTrip


DEFAULT_RADIUS_GRID_M = (10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0)


def radius_recall_table(
    reference_distances_m: np.ndarray,
    radii_m: tuple[float, ...] = DEFAULT_RADIUS_GRID_M,
) -> pd.DataFrame:
    distances = np.asarray(reference_distances_m, dtype=float)
    if distances.size == 0 or not np.isfinite(distances).all():
        raise ValueError("Reference distances must be non-empty and finite")
    return pd.DataFrame(
        {
            "radius_m": radii_m,
            "recall": [float(np.mean(distances <= radius)) for radius in radii_m],
            "miss_rate": [float(np.mean(distances > radius)) for radius in radii_m],
            "miss_count": [int(np.sum(distances > radius)) for radius in radii_m],
            "observation_count": len(distances),
        }
    )


def select_candidate_radius(reference_distances_m: np.ndarray) -> dict[str, float | int]:
    """Choose the training 99th percentile rounded upward to the next 5 metres."""
    distances = np.asarray(reference_distances_m, dtype=float)
    p95 = float(np.quantile(distances, 0.95))
    p99 = float(np.quantile(distances, 0.99))
    radius = max(5.0, math.ceil(p99 / 5.0) * 5.0)
    return {
        "p95_reference_distance_m": p95,
        "p99_reference_distance_m": p99,
        "selected_radius_m": radius,
        "selected_recall": float(np.mean(distances <= radius)),
        "selected_miss_rate": float(np.mean(distances > radius)),
        "observation_count": int(len(distances)),
    }


def generate_candidates(projected: ProjectedTrip, radius_m: float) -> pd.DataFrame:
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    points = shapely.points(projected.track_xy)
    tree = STRtree(projected.arc_geometries)
    pairs = tree.query(points, predicate="dwithin", distance=float(radius_m))
    if pairs.shape[1] == 0:
        return pd.DataFrame(
            columns=["observation_id", "arc_id", "candidate_distance_m", "candidate_rank", "candidate_count"]
        ).astype({"observation_id": "int64", "arc_id": "int64"})
    observation_ids = pairs[0].astype(np.int64)
    arc_ids = pairs[1].astype(np.int64)
    distances = shapely.distance(points[observation_ids], projected.arc_geometries[arc_ids]).astype(float)
    frame = pd.DataFrame(
        {"observation_id": observation_ids, "arc_id": arc_ids, "candidate_distance_m": distances}
    ).sort_values(["observation_id", "candidate_distance_m", "arc_id"], kind="stable")
    frame["candidate_rank"] = frame.groupby("observation_id").cumcount() + 1
    frame["candidate_count"] = frame.groupby("observation_id")["arc_id"].transform("size")
    return frame.reset_index(drop=True)


def candidate_coverage(
    candidates: pd.DataFrame, labels: pd.DataFrame, observation_count: int
) -> dict[str, int | float]:
    candidate_keys = pd.MultiIndex.from_frame(candidates[["observation_id", "arc_id"]])
    label_keys = pd.MultiIndex.from_arrays(
        [labels["observation_id"], labels["reference_arc_id"]]
    )
    reference_present = label_keys.isin(candidate_keys)
    observations_with_candidates = candidates["observation_id"].nunique()
    return {
        "observation_count": int(observation_count),
        "observations_with_candidates": int(observations_with_candidates),
        "no_candidate_count": int(observation_count - observations_with_candidates),
        "no_candidate_rate": float((observation_count - observations_with_candidates) / observation_count),
        "reference_candidate_miss_count": int((~reference_present).sum()),
        "reference_candidate_miss_rate": float((~reference_present).mean()),
    }
