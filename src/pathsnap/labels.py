"""Point-to-reference-route association and label generation.

Implements order-aware, monotonic sequence alignment to associate each GPS observation
with its most plausible reference route arc while preserving chronological route progression.
Explicitly accounts for reference route uncertainty, noise, and detours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pathsnap.graph import RoadNetwork


@dataclass
class RouteAssociationResult:
    """Stores point-to-route association outcomes for a single trip."""

    trip_id: str
    num_points: int
    num_route_arcs: int
    assigned_arc_ids: np.ndarray
    route_indices: np.ndarray
    point_to_ref_distances: np.ndarray
    confidences: np.ndarray
    is_confident: np.ndarray
    is_outlier: np.ndarray
    mean_distance: float
    median_distance: float
    p95_distance: float
    p99_distance: float
    outlier_count: int

    def to_dataframe(self) -> pd.DataFrame:
        """Converts observation-level association results to a DataFrame."""
        return pd.DataFrame(
            {
                "observation_id": np.arange(self.num_points, dtype=np.int64),
                "assigned_arc_id": self.assigned_arc_ids,
                "route_index": self.route_indices,
                "point_to_ref_dist": self.point_to_ref_distances,
                "confidence": self.confidences,
                "is_confident": self.is_confident,
                "is_outlier": self.is_outlier,
            }
        )


def associate_points_to_route(
    network: RoadNetwork,
    track_df: pd.DataFrame,
    route_df: pd.DataFrame,
    max_search_lookahead: int = 25,
    arc_jump_cost: float = 0.5,
    outlier_distance_threshold: float = 50.0,
    confidence_scale: float = 15.0,
) -> RouteAssociationResult:
    """Associates GPS observations with reference route arcs using monotonic DP.

    Minimizes perpendicular point-to-arc distance subject to monotonic forward progress
    along the reference route sequence.

    Parameters:
    - network: RoadNetwork instance with precomputed coordinates and arc tables.
    - track_df: GPS track DataFrame with 'longitude', 'latitude', 'timestamp'.
    - route_df: Reference route DataFrame with 'arc_id'.
    - max_search_lookahead: Maximum number of route arcs to advance per 1s step.
    - arc_jump_cost: Penalty per arc skipped without observations.
    - outlier_distance_threshold: Distance in meters beyond which a point is deemed an outlier.
    - confidence_scale: Scale factor for confidence exponential decay exp(-d / scale).
    """
    if route_df is None or route_df.empty:
        raise ValueError("Cannot perform route association without a non-empty route DataFrame.")

    # 1. Project GPS observations to metric coordinates
    tx, ty = network.projector.forward(track_df["longitude"].values, track_df["latitude"].values)
    timestamps = track_df["timestamp"].values
    route_arcs = route_df["arc_id"].values.astype(np.int64)

    N = len(tx)
    K = len(route_arcs)

    # 2. Extract segment coordinates for route arcs
    r_x1 = network.arc_table.loc[route_arcs, "x1"].values
    r_y1 = network.arc_table.loc[route_arcs, "y1"].values
    r_x2 = network.arc_table.loc[route_arcs, "x2"].values
    r_y2 = network.arc_table.loc[route_arcs, "y2"].values

    r_dx = r_x2 - r_x1
    r_dy = r_y2 - r_y1
    r_l2 = np.maximum(1e-12, r_dx * r_dx + r_dy * r_dy)

    # Helper function for vectorized distance from point (px, py) to slice of route segments [k_start:k_end]
    def compute_segment_dists(px: float, py: float, k_start: int, k_end: int) -> np.ndarray:
        dx = r_dx[k_start:k_end]
        dy = r_dy[k_start:k_end]
        l2 = r_l2[k_start:k_end]
        x1 = r_x1[k_start:k_end]
        y1 = r_y1[k_start:k_end]

        t = ((px - x1) * dx + (py - y1) * dy) / l2
        t_clamped = np.clip(t, 0.0, 1.0)
        proj_x = x1 + t_clamped * dx
        proj_y = y1 + t_clamped * dy
        return np.hypot(px - proj_x, py - proj_y)

    # 3. Dynamic Programming
    dp = np.full((N, K), np.inf, dtype=np.float64)
    backptr = np.zeros((N, K), dtype=np.int32)

    # Initialization: allow starting on the first few route arcs
    init_k_max = min(15, K)
    dists_0 = compute_segment_dists(tx[0], ty[0], 0, init_k_max)
    dp[0, :init_k_max] = dists_0 + np.arange(init_k_max) * arc_jump_cost

    for i in range(1, N):
        dt = max(1.0, float(timestamps[i] - timestamps[i - 1]))
        # Scale lookahead with time gap if GPS sampling was interrupted
        step_lookahead = min(K - 1, int(max_search_lookahead + dt * 2.0))

        valid_j = np.where(np.isfinite(dp[i - 1]))[0]
        if len(valid_j) == 0:
            min_j, max_j = 0, K - 1
        else:
            min_j = int(valid_j[0])
            max_j = int(valid_j[-1])

        k_start = min_j
        k_end = min(K, max_j + step_lookahead + 1)
        if k_end <= k_start:
            k_end = min(K, k_start + 1)

        dists_i = compute_segment_dists(tx[i], ty[i], k_start, k_end)

        for rel_k, k in enumerate(range(k_start, k_end)):
            dist = dists_i[rel_k]
            j_low = max(0, k - step_lookahead)
            prev_costs = dp[i - 1, j_low : k + 1]
            if len(prev_costs) == 0:
                continue

            # Jump penalties
            jumps = (k - np.arange(j_low, k + 1)) * arc_jump_cost
            total_costs = prev_costs + jumps
            best_rel_j = int(np.argmin(total_costs))
            best_cost = total_costs[best_rel_j]

            if np.isfinite(best_cost):
                dp[i, k] = best_cost + dist
                backptr[i, k] = j_low + best_rel_j

    # 4. Backtracking to recover optimal path
    best_final_k = int(np.argmin(dp[N - 1]))
    path = [best_final_k]
    for i in range(N - 1, 0, -1):
        path.append(int(backptr[i, path[-1]]))
    path.reverse()

    route_indices = np.array(path, dtype=np.int64)
    assigned_arc_ids = route_arcs[route_indices]

    # 5. Compute point-to-ref distances and confidences
    point_dists = np.zeros(N, dtype=np.float64)
    for i in range(N):
        k = route_indices[i]
        d = compute_segment_dists(tx[i], ty[i], k, k + 1)[0]
        point_dists[i] = d

    confidences = np.exp(-point_dists / confidence_scale)
    is_confident = point_dists <= 30.0
    is_outlier = point_dists > outlier_distance_threshold

    return RouteAssociationResult(
        trip_id=network.trip_id,
        num_points=N,
        num_route_arcs=K,
        assigned_arc_ids=assigned_arc_ids,
        route_indices=route_indices,
        point_to_ref_distances=point_dists,
        confidences=confidences,
        is_confident=is_confident,
        is_outlier=is_outlier,
        mean_distance=float(np.mean(point_dists)),
        median_distance=float(np.median(point_dists)),
        p95_distance=float(np.percentile(point_dists, 95)),
        p99_distance=float(np.percentile(point_dists, 99)),
        outlier_count=int(np.sum(is_outlier)),
    )

