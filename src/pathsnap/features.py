"""Feature extraction, schema definitions, and leakage prevention for candidate road-segment classification.

Builds one row per (GPS-observation, candidate-arc) pair, strictly using current and past observations
to avoid temporal data leakage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from pathsnap.geo import (
    calculate_bearing,
    calculate_heading_difference,
)
from pathsnap.graph import RoadNetwork
from pathsnap.candidates import CandidateGenerationResult


OBSERVATION_FEATURE_COLUMNS = [
    "dist_to_arc",
    "dist_to_from_node",
    "dist_to_to_node",
    "proj_ratio",
    "gps_heading",
    "arc_bearing",
    "heading_diff",
    "is_reverse_direction",
    "speed_est",
    "time_gap",
    "arc_length",
    "candidate_rank",
    "candidate_count",
    "dist_to_alternative",
]

TRANSITION_FEATURE_COLUMNS = [
    "gps_displacement_m",
    "network_transition_distance_m",
    "transition_distance_residual_m",
    "turn_angle_change_deg",
]

# Backward compatibility: FEATURE_COLUMNS refers to the 14 classifier observation features
FEATURE_COLUMNS = OBSERVATION_FEATURE_COLUMNS

# Complete 18-feature map matching pipeline
ALL_MAPMATCHING_FEATURES = OBSERVATION_FEATURE_COLUMNS + TRANSITION_FEATURE_COLUMNS

METADATA_COLUMNS = [
    "trip_id",
    "observation_id",
    "candidate_arc_id",
    "timestamp",
    "assigned_arc_id",
]

TARGET_COLUMN = "label"

FEATURE_SCHEMA: Dict[str, Dict[str, str]] = {
    "dist_to_arc": {
        "unit": "meters",
        "description": "Perpendicular Euclidean distance from the projected GPS observation to the candidate road segment.",
        "rationale": "Primary geometric proximity indicator. True segments have small perpendicular distances.",
    },
    "dist_to_from_node": {
        "unit": "meters",
        "description": "Euclidean distance from the GPS observation to the candidate arc's starting node (from_node).",
        "rationale": "Captures proximity to intersection entry and relative position along the road.",
    },
    "dist_to_to_node": {
        "unit": "meters",
        "description": "Euclidean distance from the GPS observation to the candidate arc's ending node (to_node).",
        "rationale": "Captures proximity to intersection exit and endpoint convergence.",
    },
    "proj_ratio": {
        "unit": "ratio [0, 1]",
        "description": "Normalized projection position along the directed segment from 0 (from_node) to 1 (to_node).",
        "rationale": "Indicates whether the observation projects onto the middle of the segment or clamps at endpoints.",
    },
    "gps_heading": {
        "unit": "degrees [0, 360)",
        "description": "Instantaneous vehicle travel heading computed from previous to current GPS observation.",
        "rationale": "Reflects vehicle orientation without looking ahead into the future (leakage-free).",
    },
    "arc_bearing": {
        "unit": "degrees [0, 360)",
        "description": "Directed compass bearing of the road segment from from_node to to_node clockwise from True North.",
        "rationale": "Represents the permissible direction of travel along this road arc.",
    },
    "heading_diff": {
        "unit": "degrees [0, 180]",
        "description": "Circular angular difference between the vehicle GPS heading and the candidate arc bearing.",
        "rationale": "Discriminates between opposing lanes, parallel streets, and perpendicular cross-streets.",
    },
    "is_reverse_direction": {
        "unit": "binary {0, 1}",
        "description": "Indicator flag equal to 1 if heading_diff > 90 degrees, indicating counter-flow direction.",
        "rationale": "Vehicles rarely travel backwards along one-way road segments; strongly filters reverse arcs.",
    },
    "speed_est": {
        "unit": "m/s",
        "description": "Estimated instantaneous vehicle speed calculated as distance(P_{t-1}, P_t) / delta_t.",
        "rationale": "Differentiates slow/stopped maneuvers (where heading is noisy) from high-speed arterial travel.",
    },
    "time_gap": {
        "unit": "seconds",
        "description": "Elapsed time delta between the previous and current GPS observation.",
        "rationale": "Flags GPS sampling interruptions or tunnels where dead-reckoning uncertainty increases.",
    },
    "arc_length": {
        "unit": "meters",
        "description": "Total Euclidean length of the candidate road segment.",
        "rationale": "Short intersection connector arcs vs long freeway segments have different prior probabilities.",
    },
    "candidate_rank": {
        "unit": "integer [1, K]",
        "description": "Rank of the candidate arc among all candidates for this observation, sorted by ascending dist_to_arc.",
        "rationale": "Relative rank provides an invariant ranking feature robust across different road densities.",
    },
    "candidate_count": {
        "unit": "count",
        "description": "Total number of candidate road arcs within the search radius R for this observation.",
        "rationale": "Reflects local road network density; complex intersections have higher candidate counts.",
    },
    "dist_to_alternative": {
        "unit": "meters",
        "description": "Distance margin to the nearest competing candidate arc.",
        "rationale": "Measures confidence: a large margin indicates an unambiguous choice with no close competitors.",
    },
    # --- Four Transition-Level Features (Completing the 18-Feature System) ---
    "gps_displacement_m": {
        "unit": "meters",
        "category": "transition",
        "description": "Euclidean/geodesic distance between consecutive GPS observations in metres on projected UTM plane.",
        "rationale": "Quantifies physical vehicle movement between time steps t-1 and t.",
    },
    "network_transition_distance_m": {
        "unit": "meters",
        "category": "transition",
        "description": "Directed road-network distance required to travel between previous and current candidate road arcs.",
        "rationale": "Measures true topological path feasibility along directed road network edges.",
    },
    "transition_distance_residual_m": {
        "unit": "meters",
        "category": "transition",
        "description": "Absolute difference between network transition distance and GPS displacement.",
        "rationale": "Primary movement plausibility indicator: true path transitions have near-zero residual.",
    },
    "turn_angle_change_deg": {
        "unit": "degrees [0, 180]",
        "category": "transition",
        "description": "Circular angular change in heading direction between consecutive candidate road arcs.",
        "rationale": "Penalizes sharp turns and unphysical U-turns into oncoming traffic.",
    },
}


@dataclass
class TransitionFeatures:
    """Stores the four transition-level features for a candidate arc pair (a_{t-1}, a_t)."""

    gps_displacement_m: float
    network_transition_distance_m: float
    transition_distance_residual_m: float
    turn_angle_change_deg: float
    is_unreachable: bool


def calculate_turn_angle_change(bearing1: float, bearing2: float) -> float:
    """Calculates circular angular change between two directed arc bearings wrapped to [0, 180]."""
    diff = abs(bearing1 - bearing2)
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)


def build_observation_gps_features(
    network: RoadNetwork, track_df: pd.DataFrame
) -> pd.DataFrame:
    """Computes point-level kinematic features (heading, speed, time gap) using past observations only.

    Important Leakage Prevention Rule:
    Strictly uses information from timestamps t and t-1. No future observations (t+1, t+2, etc.)
    are accessed, ensuring the machine learning classifier cannot cheat via temporal lookahead.
    """
    # Project GPS coordinates to local metric UTM
    tx, ty = network.projector.forward(track_df["longitude"].values, track_df["latitude"].values)
    timestamps = track_df["timestamp"].values
    num_pts = len(tx)

    speeds = np.zeros(num_pts, dtype=np.float64)
    headings = np.zeros(num_pts, dtype=np.float64)
    time_gaps = np.ones(num_pts, dtype=np.float64)  # Default 1.0s fallback

    last_valid_heading = 0.0

    # Sequential pass from t=1 to end
    for i in range(1, num_pts):
        dt = float(timestamps[i] - timestamps[i - 1])
        if dt <= 0.0:
            dt = 1e-3  # Guard against division by zero in duplicate timestamps
        time_gaps[i] = dt

        # Euclidean displacement between consecutive GPS observations
        dx = tx[i] - tx[i - 1]
        dy = ty[i] - ty[i - 1]
        dist_step = math.hypot(dx, dy)
        speeds[i] = dist_step / dt

        # Only update heading if vehicle has moved at least 0.5 meters;
        # stationary vehicles suffer from random GPS noise that rotates the bearing erratically.
        if dist_step >= 0.5:
            last_valid_heading = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

        headings[i] = last_valid_heading

    # Anchor the initial point (i=0): initial speed is 0, heading inherits from i=1
    if num_pts > 1:
        headings[0] = headings[1]
    speeds[0] = 0.0
    time_gaps[0] = 1.0

    return pd.DataFrame(
        {
            "observation_id": np.arange(num_pts, dtype=np.int64),
            "timestamp": timestamps,
            "x": tx,
            "y": ty,
            "gps_heading": headings,
            "speed_est": speeds,
            "time_gap": time_gaps,
        }
    )


def build_candidate_features_dataset(
    network: RoadNetwork,
    track_df: pd.DataFrame,
    candidate_res: CandidateGenerationResult,
    labels_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Builds the complete feature DataFrame with one row per (observation, candidate) pair.

    Includes target column `label`:
    - 1 if candidate matches the assigned reference route arc.
    - 0 otherwise.
    If labels_df is None (e.g. unlabelled test set), target is set to NaN.
    """
    cand_df = candidate_res.candidates_df
    if cand_df.empty:
        return pd.DataFrame(columns=METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])

    # 1. Compute observation-level kinematics (speed, heading, delta t)
    obs_kinematics = build_observation_gps_features(network, track_df).set_index("observation_id")

    # 2. Extract road segment attributes (bearings, lengths) from network table
    arc_table = network.arc_table.set_index("arc_id")

    # 3. Vectorized feature assembly
    obs_ids = cand_df["observation_id"].values
    cand_arc_ids = cand_df["candidate_arc_id"].values

    # Merge kinematic features for each candidate's corresponding observation
    cand_headings = obs_kinematics.loc[obs_ids, "gps_heading"].values
    cand_speeds = obs_kinematics.loc[obs_ids, "speed_est"].values
    cand_time_gaps = obs_kinematics.loc[obs_ids, "time_gap"].values
    cand_timestamps = obs_kinematics.loc[obs_ids, "timestamp"].values

    # Merge road attributes for each candidate arc
    cand_arc_bearings = arc_table.loc[cand_arc_ids, "bearing"].values
    cand_arc_lengths = arc_table.loc[cand_arc_ids, "length"].values

    # Angular difference between travel direction and permissible road bearing (wrapped to [0, 180])
    diff_raw = np.abs(cand_headings - cand_arc_bearings) % 360.0
    heading_diffs = np.minimum(diff_raw, 360.0 - diff_raw)
    
    # Reverse direction flag: 1 if vehicle appears to be moving counter to one-way traffic (>90 deg)
    is_reverse = (heading_diffs > 90.0).astype(np.int8)

    # 4. Target label generation (binary ground-truth matching)
    assigned_arc_ids: Optional[np.ndarray] = None
    labels: np.ndarray

    if labels_df is not None and not labels_df.empty:
        labels_map = labels_df.set_index("observation_id")["assigned_arc_id"].to_dict()
        true_assigned = np.array([labels_map.get(oid, -1) for oid in obs_ids], dtype=np.int64)
        assigned_arc_ids = true_assigned
        # Binary classification target: 1 = matches ground truth, 0 = incorrect candidate
        labels = (cand_arc_ids == true_assigned).astype(np.int8)
    else:
        # Unlabelled evaluation split
        assigned_arc_ids = np.full(len(cand_df), -1, dtype=np.int64)
        labels = np.full(len(cand_df), np.nan, dtype=np.float32)

    # Compile the final structured dataset
    features_df = pd.DataFrame(
        {
            "trip_id": network.trip_id,
            "observation_id": obs_ids,
            "candidate_arc_id": cand_arc_ids,
            "timestamp": cand_timestamps,
            "assigned_arc_id": assigned_arc_ids,
            "dist_to_arc": cand_df["dist_to_arc"].values,
            "dist_to_from_node": cand_df["dist_to_from_node"].values,
            "dist_to_to_node": cand_df["dist_to_to_node"].values,
            "proj_ratio": cand_df["proj_ratio"].values,
            "gps_heading": cand_headings,
            "arc_bearing": cand_arc_bearings,
            "heading_diff": heading_diffs,
            "is_reverse_direction": is_reverse,
            "speed_est": cand_speeds,
            "time_gap": cand_time_gaps,
            "arc_length": cand_arc_lengths,
            "candidate_rank": cand_df["candidate_rank"].values,
            "candidate_count": cand_df["total_candidates"].values,
            "dist_to_alternative": cand_df["dist_to_alternative"].values,
            "label": labels,
        }
    )

    return features_df
