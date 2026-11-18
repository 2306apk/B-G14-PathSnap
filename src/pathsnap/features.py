"""Causal point/candidate feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from .geo import ProjectedTrip, planar_bearing_degrees


FEATURE_COLUMNS = [
    "point_arc_distance_m",
    "from_node_distance_m",
    "to_node_distance_m",
    "projection_position",
    "vehicle_heading_deg",
    "arc_bearing_deg",
    "heading_difference_deg",
    "reverse_direction",
    "vehicle_speed_mps",
    "time_gap_s",
    "arc_length_m",
    "candidate_rank",
    "candidate_count",
    "nearest_alternative_distance_m",
]


def circular_difference_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def observation_dynamics(projected: ProjectedTrip) -> pd.DataFrame:
    n = len(projected.track_xy)
    delta_xy = np.diff(projected.track_xy, axis=0)
    timestamps = projected.trip.track["timestamp_s"].to_numpy(dtype=float)
    gaps = np.diff(timestamps)
    headings = np.full(n, np.nan)
    speeds = np.full(n, np.nan)
    time_gaps = np.full(n, np.nan)
    if n > 1:
        headings[1:] = planar_bearing_degrees(np.zeros_like(delta_xy), delta_xy)
        speeds[1:] = np.divide(
            np.linalg.norm(delta_xy, axis=1), gaps, out=np.full(n - 1, np.nan), where=gaps > 0
        )
        time_gaps[1:] = gaps
    return pd.DataFrame(
        {
            "observation_id": np.arange(n),
            "vehicle_heading_deg": headings,
            "vehicle_speed_mps": speeds,
            "time_gap_s": time_gaps,
        }
    )


def build_feature_rows(
    projected: ProjectedTrip,
    candidates: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        columns = ["trip_id", "partition", "observation_id", "arc_id", *FEATURE_COLUMNS]
        if labels is not None:
            columns += ["reference_arc_id", "target"]
        return pd.DataFrame(columns=columns)
    frame = candidates.copy()
    obs = frame["observation_id"].to_numpy(dtype=np.int64)
    arc = frame["arc_id"].to_numpy(dtype=np.int64)
    points_xy = projected.track_xy[obs]
    endpoints = projected.trip.arcs.iloc[arc][["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    from_xy = projected.nodes_xy[endpoints[:, 0]]
    to_xy = projected.nodes_xy[endpoints[:, 1]]
    geoms = projected.arc_geometries[arc]
    points = shapely.points(points_xy)
    lengths = projected.arc_lengths_m[arc]
    frame["point_arc_distance_m"] = frame.pop("candidate_distance_m")
    frame["from_node_distance_m"] = np.linalg.norm(points_xy - from_xy, axis=1)
    frame["to_node_distance_m"] = np.linalg.norm(points_xy - to_xy, axis=1)
    located = shapely.line_locate_point(geoms, points)
    frame["projection_position"] = np.divide(
        located, lengths, out=np.zeros_like(located, dtype=float), where=lengths > 0
    ).clip(0.0, 1.0)
    frame = frame.merge(observation_dynamics(projected), on="observation_id", how="left", validate="many_to_one")
    frame["arc_bearing_deg"] = projected.arc_bearings_deg[arc]
    frame["heading_difference_deg"] = circular_difference_degrees(
        frame["vehicle_heading_deg"].to_numpy(), frame["arc_bearing_deg"].to_numpy()
    )
    frame["reverse_direction"] = (frame["heading_difference_deg"] > 90.0).astype("int8")
    frame["arc_length_m"] = lengths
    grouped = frame.groupby("observation_id")["point_arc_distance_m"]
    nearest = grouped.transform("min")
    sorted_distance = frame.groupby("observation_id")["point_arc_distance_m"].transform(
        lambda values: values.nsmallest(2).iloc[-1] if len(values) > 1 else np.nan
    )
    frame["nearest_alternative_distance_m"] = np.where(
        frame["candidate_rank"] == 1, sorted_distance, nearest
    )
    frame.insert(0, "partition", projected.trip.partition)
    frame.insert(0, "trip_id", projected.trip.trip_id)
    if labels is not None:
        frame = frame.merge(
            labels[["observation_id", "reference_arc_id", "reference_route_position", "reference_distance_m", "label_confidence", "label_uncertain"]],
            on="observation_id",
            how="left",
            validate="many_to_one",
        )
        frame["target"] = (frame["arc_id"] == frame["reference_arc_id"]).astype("int8")
    return frame
