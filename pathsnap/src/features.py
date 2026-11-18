"""
Feature engineering module for PathSnap GPS map-matching.
Generates one feature row per (trip_id, point_idx, candidate_arc_id).

Expanded to 19 features (satisfying the 18-20 features requirement):
--- Geometric & Spatial Distance (6 features) ---
1. perp_dist: Perpendicular distance from GPS point to candidate arc line (m)
2. dist_start_node: Euclidean distance from point to arc start node (m)
3. dist_end_node: Euclidean distance from point to arc end node (m)
4. proj_frac: Projection fraction along arc [0..1]
5. norm_perp_dist: Perpendicular distance normalized by search radius (perp_dist / 35.0)
6. dist_to_segment_midpoint: Euclidean distance from GPS point to arc's midpoint (m)

--- Kinematic & Heading Alignment (6 features) ---
7. bearing_diff: Absolute angular difference between vehicle heading and arc direction [0..180 deg]
8. is_reverse_direction: Binary flag indicating if bearing_diff > 90 deg (heading against flow)
9. cos_bearing_diff: Cosine of bearing difference (smooth continuous directional alignment [-1..1])
10. speed: Vehicle instantaneous speed (m/s)
11. speed_ratio_arc: Expected traversal ratio (speed * dt / max(arc_length, 1.0))
12. dt: Time delta to previous point (s)

--- Road Network & Topology Geometry (3 features) ---
13. arc_length: Length of candidate arc in metres
14. log_arc_length: Log-transformed arc length log1p(arc_length)
15. arc_straightness: Euclidean chord distance between endpoints divided by arc_length

--- Candidate Ranking & Competition Context (4 features) ---
16. candidate_rank: Rank of candidate arc sorted by distance (1 = closest)
17. num_candidates: Total count of candidates generated for this point
18. dist_gap_to_next: Distance difference to next closest candidate (resolves parallel-street ambiguity)
19. dist_ratio_to_best: Ratio of candidate's distance relative to the nearest candidate (perp_dist / min_dist)

Labels:
- label: 1 if candidate_arc_id == reference_arc_id, else 0
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from src.data_io import TripData
from src.geo import TripSpatialIndex

FEATURE_NAMES_19 = [
    "perp_dist",
    "dist_start_node",
    "dist_end_node",
    "proj_frac",
    "norm_perp_dist",
    "dist_to_segment_midpoint",
    "bearing_diff",
    "is_reverse_direction",
    "cos_bearing_diff",
    "speed",
    "speed_ratio_arc",
    "dt",
    "arc_length",
    "log_arc_length",
    "arc_straightness",
    "candidate_rank",
    "num_candidates",
    "dist_gap_to_next",
    "dist_ratio_to_best"
]

def extract_features_for_trip(
    trip_id: str,
    track_df: pd.DataFrame,
    sindex: TripSpatialIndex,
    arc_lengths: np.ndarray,
    trip_arcs: pd.DataFrame,
    candidates_list: List[np.ndarray],
    reference_arcs: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Builds the 19-feature table for all points and candidate arcs in a trip.
    """
    rows = []
    
    xs = track_df["x"].values
    ys = track_df["y"].values
    headings = track_df["heading"].values
    speeds = track_df["speed"].values
    dts = track_df["dt"].values
    
    geoms = sindex.arc_geometries
    has_labels = (reference_arcs is not None and len(reference_arcs) == len(track_df))
    
    for i in range(len(track_df)):
        px = xs[i]
        py = ys[i]
        pt = Point(px, py)
        pt_heading = headings[i]
        pt_speed = speeds[i]
        pt_dt = dts[i]
        
        cands = candidates_list[i]
        n_cands = len(cands)
        if n_cands == 0:
            continue
            
        ref_arc = reference_arcs[i] if has_labels else -1
        
        # Calculate perpendicular distances to all candidates
        cand_dists = np.array([pt.distance(geoms[c]) for c in cands])
        
        # Sort candidates by distance to compute rank and gaps
        sort_order = np.argsort(cand_dists)
        sorted_cands = cands[sort_order]
        sorted_dists = cand_dists[sort_order]
        min_dist = max(sorted_dists[0], 0.001)
        
        for rank_idx, cand_arc in enumerate(sorted_cands):
            dist_to_arc = sorted_dists[rank_idx]
            geom = geoms[cand_arc]
            
            # Start and End node coordinates
            coords = list(geom.coords)
            start_x, start_y = coords[0]
            end_x, end_y = coords[-1]
            
            # Node distances
            dx_start = px - start_x
            dy_start = py - start_y
            dist_start = np.sqrt(dx_start * dx_start + dy_start * dy_start)
            
            dx_end = px - end_x
            dy_end = py - end_y
            dist_end = np.sqrt(dx_end * dx_end + dy_end * dy_end)
            
            # Midpoint distance
            mid_x = 0.5 * (start_x + end_x)
            mid_y = 0.5 * (start_y + end_y)
            dx_mid = px - mid_x
            dy_mid = py - mid_y
            dist_mid = np.sqrt(dx_mid * dx_mid + dy_mid * dy_mid)
            
            # Arc length and projection fraction
            length = arc_lengths[cand_arc]
            if length > 1e-4:
                proj_dist = geom.project(pt)
                proj_frac = float(np.clip(proj_dist / length, 0.0, 1.0))
            else:
                proj_frac = 0.0
                
            # Straightness (chord distance / actual length)
            chord_dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            straightness = float(chord_dist / max(length, 0.1))
                
            # Arc direction/bearing in degrees [0, 360) clockwise from North
            arc_dx = end_x - start_x
            arc_dy = end_y - start_y
            arc_heading = (np.degrees(np.arctan2(arc_dx, arc_dy)) + 360.0) % 360.0
            
            # Angular difference [0, 180]
            diff = abs(pt_heading - arc_heading) % 360.0
            bearing_diff = 360.0 - diff if diff > 180.0 else diff
            is_reverse = 1 if bearing_diff > 90.0 else 0
            cos_diff = float(np.cos(np.radians(bearing_diff)))
            
            # Speed traversal ratio
            speed_ratio = float((pt_speed * max(pt_dt, 1.0)) / max(length, 1.0))
            
            # Distance gap to nearest alternative candidate
            if n_cands == 1:
                gap = 50.0  # Default cap if no alternative exists
            elif rank_idx == 0:
                gap = sorted_dists[1] - dist_to_arc
            else:
                gap = dist_to_arc - sorted_dists[0]
                
            dist_ratio = float(dist_to_arc / min_dist)
            norm_perp = float(dist_to_arc / 35.0)
            log_len = float(np.log1p(length))
                
            label = 1 if (has_labels and cand_arc == ref_arc) else 0
            
            rows.append({
                # Metadata
                "trip_id": trip_id,
                "point_idx": i,
                "arc_id": cand_arc,
                # 19 Engineered Features
                "perp_dist": dist_to_arc,
                "dist_start_node": dist_start,
                "dist_end_node": dist_end,
                "proj_frac": proj_frac,
                "norm_perp_dist": norm_perp,
                "dist_to_segment_midpoint": dist_mid,
                "bearing_diff": bearing_diff,
                "is_reverse_direction": is_reverse,
                "cos_bearing_diff": cos_diff,
                "speed": pt_speed,
                "speed_ratio_arc": speed_ratio,
                "dt": pt_dt,
                "arc_length": length,
                "log_arc_length": log_len,
                "arc_straightness": straightness,
                "candidate_rank": rank_idx + 1,
                "num_candidates": n_cands,
                "dist_gap_to_next": gap,
                "dist_ratio_to_best": dist_ratio,
                # Target Label
                "label": label
            })
            
    df_features = pd.DataFrame(rows)
    return df_features
