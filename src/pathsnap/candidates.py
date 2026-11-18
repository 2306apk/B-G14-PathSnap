"""Candidate road-segment generation, spatial radius filtering, and recall analysis.

Finds plausible road arcs within a training-justified radius R for each GPS observation.
Preserves candidate ranks, counts, and explicit handling of candidate misses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from pathsnap.geo import point_to_segment_projection
from pathsnap.graph import RoadNetwork


@dataclass
class CandidateGenerationResult:
    """Stores candidate arcs and summary statistics for a trip."""

    trip_id: str
    radius: float
    num_observations: int
    candidates_df: pd.DataFrame
    num_zero_candidate_points: int
    mean_candidates_per_point: float
    median_candidates_per_point: float
    max_candidates_per_point: int


def generate_candidates_for_trip(
    network: RoadNetwork,
    track_df: pd.DataFrame,
    radius: float = 50.0,
    max_candidates_per_point: Optional[int] = None,
) -> CandidateGenerationResult:
    """Generates candidate road arcs within distance `radius` (meters) for each GPS observation.

    For each candidate, computes:
    - candidate_arc_id: trip-local arc ID
    - candidate_rank: 1-based rank by perpendicular distance (1 = closest)
    - dist_to_arc: perpendicular distance in meters
    - dist_to_from_node: distance to from-node
    - dist_to_to_node: distance to to-node
    - proj_ratio: projection parameter t in [0.0, 1.0] (progress along segment)
    - proj_x, proj_y: metric coordinates of projection point on the segment
    - total_candidates: count of candidates found for this GPS observation
    - dist_to_alternative: distance difference to nearest alternative candidate
    """
    # Project spherical GPS coordinates (lon, lat) to planar metric coordinates (tx, ty)
    tx, ty = network.projector.forward(track_df["longitude"].values, track_df["latitude"].values)
    num_points = len(tx)

    records: List[Dict[str, Any]] = []
    zero_cand_count = 0
    cand_counts: List[int] = []

    # Pre-extract endpoint coordinates into flat numpy arrays for fast vectorized lookups
    arc_x1 = network.arc_table["x1"].values
    arc_y1 = network.arc_table["y1"].values
    arc_x2 = network.arc_table["x2"].values
    arc_y2 = network.arc_table["y2"].values

    # Iterate through each GPS point in the trajectory
    for obs_idx in range(num_points):
        px, py = float(tx[obs_idx]), float(ty[obs_idx])
        
        # Step 1: Query spatial R-tree (STRtree) index for road segments within the search radius (e.g. 50m)
        cand_arc_ids = network.query_candidate_arcs(px, py, radius)
        count = len(cand_arc_ids)
        cand_counts.append(count)

        # Track observations where GPS drift exceeded the search radius (zero candidates found)
        if count == 0:
            zero_cand_count += 1
            continue

        # Step 2: Compute exact perpendicular distances, node distances, and projection ratio [0, 1]
        cand_details: List[Tuple[int, float, float, float, float, float, float]] = []
        for arc_id in cand_arc_ids:
            x1 = arc_x1[arc_id]
            y1 = arc_y1[arc_id]
            x2 = arc_x2[arc_id]
            y2 = arc_y2[arc_id]
            # Projects point (px, py) orthogonally onto directed segment (x1, y1) -> (x2, y2)
            d_seg, d_from, d_to, t_clamped, qx, qy = point_to_segment_projection(
                px, py, x1, y1, x2, y2
            )
            cand_details.append((arc_id, d_seg, d_from, d_to, t_clamped, qx, qy))

        # Step 3: Sort candidate arcs by perpendicular distance ascending (closest road first)
        cand_details.sort(key=lambda item: item[1])

        # Optional cap on max candidates per observation
        if max_candidates_per_point is not None:
            cand_details = cand_details[:max_candidates_per_point]

        actual_k = len(cand_details)
        dists = [item[1] for item in cand_details]

        # Step 4: Build feature records with proximity rank and alternative margin
        for rank_idx, (arc_id, d_seg, d_from, d_to, t_clamped, qx, qy) in enumerate(cand_details):
            # Calculate distance margin to nearest competitor (measures choice ambiguity)
            if actual_k == 1:
                dist_alt = radius - d_seg  # Only one road nearby; margin is distance to search perimeter
            elif rank_idx == 0:
                dist_alt = dists[1] - d_seg  # Difference between 1st and 2nd closest candidate
            else:
                dist_alt = d_seg - dists[0]  # Difference between this candidate and the best candidate

            records.append(
                {
                    "observation_id": obs_idx,
                    "candidate_arc_id": arc_id,
                    "candidate_rank": rank_idx + 1,
                    "dist_to_arc": d_seg,
                    "dist_to_from_node": d_from,
                    "dist_to_to_node": d_to,
                    "proj_ratio": t_clamped,
                    "proj_x": qx,
                    "proj_y": qy,
                    "total_candidates": actual_k,
                    "dist_to_alternative": dist_alt,
                }
            )

    candidates_df = pd.DataFrame(records)
    # Ensure standard schema even if no candidates were found
    if candidates_df.empty:
        candidates_df = pd.DataFrame(
            columns=[
                "observation_id",
                "candidate_arc_id",
                "candidate_rank",
                "dist_to_arc",
                "dist_to_from_node",
                "dist_to_to_node",
                "proj_ratio",
                "proj_x",
                "proj_y",
                "total_candidates",
                "dist_to_alternative",
            ]
        )

    return CandidateGenerationResult(
        trip_id=network.trip_id,
        radius=radius,
        num_observations=num_points,
        candidates_df=candidates_df,
        num_zero_candidate_points=zero_cand_count,
        mean_candidates_per_point=float(np.mean(cand_counts)) if cand_counts else 0.0,
        median_candidates_per_point=float(np.median(cand_counts)) if cand_counts else 0.0,
        max_candidates_per_point=int(np.max(cand_counts)) if cand_counts else 0,
    )


def evaluate_candidate_radius_sweep(
    network: RoadNetwork,
    track_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    candidate_radii: List[float] = [10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0],
) -> pd.DataFrame:
    """Evaluates candidate recall and miss rates across different radii on a trip.

    Recall = fraction of observations where the assigned reference arc is in the candidate set.
    """
    assigned_arcs = labels_df.set_index("observation_id")["assigned_arc_id"].to_dict()
    num_obs = len(track_df)

    results: List[Dict[str, Any]] = []

    for r in candidate_radii:
        cand_res = generate_candidates_for_trip(network, track_df, radius=r)
        df = cand_res.candidates_df

        # Check for each observation whether true assigned arc is present
        cand_grouped = df.groupby("observation_id")["candidate_arc_id"].apply(set).to_dict()

        hits = 0
        misses = 0
        for obs_id in range(num_obs):
            true_arc = assigned_arcs.get(obs_id)
            if true_arc is None:
                continue
            cand_set = cand_grouped.get(obs_id, set())
            if true_arc in cand_set:
                hits += 1
            else:
                misses += 1

        total = hits + misses
        recall = (hits / total) if total > 0 else 0.0
        miss_rate = 1.0 - recall

        results.append(
            {
                "radius_m": r,
                "recall": recall,
                "miss_rate": miss_rate,
                "hits": hits,
                "misses": misses,
                "mean_candidates": cand_res.mean_candidates_per_point,
                "median_candidates": cand_res.median_candidates_per_point,
                "max_candidates": cand_res.max_candidates_per_point,
                "zero_cand_obs": cand_res.num_zero_candidate_points,
            }
        )

    return pd.DataFrame(results)

