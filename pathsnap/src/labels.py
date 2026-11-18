"""
Reference-arc ground-truth assignment module for PathSnap.
Assigns each GPS point in a trip a reference arc from the trip's .route sequence.
Enforces monotonic progress along the route and flags ambiguous or distant assignments.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from src.data_io import TripData
from src.geo import TripSpatialIndex


@dataclass
class LabelResult:
    trip_id: str
    reference_arcs: np.ndarray      # 1D array of arc_ids for each GPS point
    reference_distances: np.ndarray # distance (m) from GPS point to assigned reference arc
    is_ambiguous: np.ndarray        # boolean flag if assignment was ambiguous or distant
    confidently_labelled_count: int
    ambiguous_count: int


def assign_reference_arcs_monotonic(
    track_df: pd.DataFrame,
    sindex: TripSpatialIndex,
    route_arc_ids: np.ndarray,
    search_window: int = 12,
    ambiguity_dist_gap: float = 1.0,
    max_reasonable_dist: float = 40.0
) -> LabelResult:
    """
    Assigns each GPS point a reference arc from the route sequence.
    
    Algorithm:
    - Maintains the current progress index `curr_idx` along route_arc_ids.
    - For each GPS point, evaluates candidate route arcs in the window:
        [max(0, curr_idx - 1), min(len(route_arc_ids), curr_idx + search_window)]
      (allowing a 1-arc lookback for jitter near transitions, and forward window).
    - Selects the route arc minimizing distance to the GPS point.
    - If the minimum distance is > max_reasonable_dist or the distance difference between
      the 1st and 2nd best route candidate is < ambiguity_dist_gap, flags the point as ambiguous.
    - Advances `curr_idx` monotonically to the chosen arc's position in the route.
    """
    n_points = len(track_df)
    n_route = len(route_arc_ids)
    
    ref_arcs = np.zeros(n_points, dtype=np.int64)
    ref_dists = np.zeros(n_points, dtype=np.float64)
    is_ambiguous = np.zeros(n_points, dtype=bool)
    
    if n_route == 0 or n_points == 0:
        return LabelResult(
            trip_id="",
            reference_arcs=ref_arcs,
            reference_distances=ref_dists,
            is_ambiguous=is_ambiguous,
            confidently_labelled_count=0,
            ambiguous_count=0
        )
        
    # Pre-extract Shapely geometries for route arcs
    route_geoms = [sindex.arc_geometries[a] for a in route_arc_ids]
    
    curr_route_pos = 0
    
    for i in range(n_points):
        px = track_df["x"].iloc[i]
        py = track_df["y"].iloc[i]
        pt = Point(px, py)
        
        # Define window of route indices to consider
        win_start = max(0, curr_route_pos - 1)
        win_end = min(n_route, curr_route_pos + search_window)
        
        # Calculate distance to each route arc in the window
        dists = [pt.distance(route_geoms[idx]) for idx in range(win_start, win_end)]
        sorted_order = np.argsort(dists)
        
        best_local_idx = sorted_order[0]
        best_route_pos = win_start + best_local_idx
        best_arc_id = route_arc_ids[best_route_pos]
        best_dist = dists[best_local_idx]
        
        ref_arcs[i] = best_arc_id
        ref_dists[i] = best_dist
        
        # Check ambiguity:
        # Ambiguous if distant, or if runner-up is very close in distance (< ambiguity_dist_gap)
        ambiguous = False
        if best_dist > max_reasonable_dist:
            ambiguous = True
        elif len(sorted_order) > 1:
            second_best_dist = dists[sorted_order[1]]
            if (second_best_dist - best_dist) < ambiguity_dist_gap:
                ambiguous = True
                
        is_ambiguous[i] = ambiguous
        
        # Advance monotonic progress (do not regress behind best_route_pos)
        if best_route_pos > curr_route_pos:
            curr_route_pos = best_route_pos
            
    ambig_count = int(np.sum(is_ambiguous))
    confident_count = n_points - ambig_count
    
    return LabelResult(
        trip_id=str(track_df["crs"].iloc[0] if "crs" in track_df else ""),
        reference_arcs=ref_arcs,
        reference_distances=ref_dists,
        is_ambiguous=is_ambiguous,
        confidently_labelled_count=confident_count,
        ambiguous_count=ambig_count
    )
