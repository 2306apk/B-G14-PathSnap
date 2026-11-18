"""
Candidate arc generation module for PathSnap.
For each GPS point, queries road-network arcs within radius R metres using Shapely STRtree.
Computes candidate miss rate against the assigned reference arc and handles edge cases (zero candidates).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from shapely.geometry import Point
from src.geo import TripSpatialIndex


def generate_candidates_for_trip(
    track_df: pd.DataFrame,
    sindex: TripSpatialIndex,
    radius: float = 35.0,
    reference_arcs: Optional[np.ndarray] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Generates candidate arc indices for each GPS point in track_df within radius (metres).
    
    Parameters:
      track_df: DataFrame with projected metric 'x' and 'y' columns.
      sindex: TripSpatialIndex containing the trip's road network STRtree.
      radius: Search radius in metres.
      reference_arcs: Optional array of ground-truth reference arc_ids for miss rate calculation.
      
    Returns:
      candidates_list: List of 1D numpy arrays, where candidate_list[i] is the array of candidate arc_ids for point i.
      metrics: Summary dictionary (num_points, zero_candidate_count, miss_count, miss_rate, avg_candidates, etc.)
    """
    n_points = len(track_df)
    candidates_list = []
    
    zero_candidate_indices = []
    miss_indices = []
    has_reference = reference_arcs is not None and len(reference_arcs) == n_points
    
    xs = track_df["x"].values
    ys = track_df["y"].values
    
    for i in range(n_points):
        x = xs[i]
        y = ys[i]
        
        # Query STRtree within radius
        cand = sindex.query_radius(x, y, radius)
        
        if len(cand) == 0:
            zero_candidate_indices.append(i)
            # If zero candidates within radius, as a fallback find the single nearest arc in the network
            pt = Point(x, y)
            nearest_idx = sindex.tree.nearest(pt)
            cand = np.array([nearest_idx], dtype=np.int64)
            
        candidates_list.append(cand)
        
        if has_reference:
            ref_arc = reference_arcs[i]
            if ref_arc >= 0 and ref_arc not in cand:
                miss_indices.append(i)
                
    counts = np.array([len(c) for c in candidates_list])
    
    metrics = {
        "n_points": n_points,
        "radius": radius,
        "zero_candidate_count": len(zero_candidate_indices),
        "zero_candidate_pct": float(len(zero_candidate_indices) / n_points * 100) if n_points > 0 else 0.0,
        "candidate_count_min": int(np.min(counts)) if n_points > 0 else 0,
        "candidate_count_median": float(np.median(counts)) if n_points > 0 else 0.0,
        "candidate_count_p90": float(np.percentile(counts, 90)) if n_points > 0 else 0.0,
        "candidate_count_max": int(np.max(counts)) if n_points > 0 else 0,
        "candidate_count_mean": float(np.mean(counts)) if n_points > 0 else 0.0,
        "candidate_miss_count": len(miss_indices) if has_reference else None,
        "candidate_miss_rate": float(len(miss_indices) / n_points * 100) if (has_reference and n_points > 0) else None,
    }
    
    return candidates_list, metrics
