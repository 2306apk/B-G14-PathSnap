"""Order-aware association of GPS observations to an ordered reference route."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from .geo import ProjectedTrip


def point_route_distances(
    track_xy: np.ndarray, route_geometries: np.ndarray, chunk_size: int = 512
) -> np.ndarray:
    """Return an observation-by-route-position distance matrix in metres."""
    if len(track_xy) == 0 or len(route_geometries) == 0:
        raise ValueError("Track and route must both be non-empty")
    result = np.empty((len(track_xy), len(route_geometries)), dtype=np.float32)
    for start in range(0, len(track_xy), chunk_size):
        stop = min(len(track_xy), start + chunk_size)
        points = shapely.points(track_xy[start:stop])
        result[start:stop] = shapely.distance(points[:, None], route_geometries[None, :])
    return result


def monotonic_route_match(
    distances_m: np.ndarray,
    skip_penalty_m: float = 5.0,
    endpoint_penalty_m: float = 2.0,
) -> np.ndarray:
    """Dynamic program allowing route position to stay or advance monotonically.

    The objective is summed point-to-arc distance plus a per-skipped-route-position
    penalty. Endpoint penalties discourage, but do not force, incomplete route use.
    """
    distances = np.asarray(distances_m, dtype=float)
    if distances.ndim != 2 or min(distances.shape) == 0:
        raise ValueError("distances_m must be a non-empty 2D matrix")
    n_obs, n_route = distances.shape
    previous = distances[0] + endpoint_penalty_m * np.arange(n_route)
    back = np.empty((n_obs, n_route), dtype=np.int32)
    back[0] = -1
    route_positions = np.arange(n_route, dtype=float)
    for i in range(1, n_obs):
        transformed = previous - skip_penalty_m * route_positions
        prefix_value = np.minimum.accumulate(transformed)
        prefix_arg = np.empty(n_route, dtype=np.int32)
        best_index = 0
        for j in range(n_route):
            if transformed[j] < transformed[best_index]:
                best_index = j
            prefix_arg[j] = best_index
        advance_cost = np.full(n_route, np.inf)
        advance_from = np.zeros(n_route, dtype=np.int32)
        if n_route > 1:
            advance_cost[1:] = prefix_value[:-1] + skip_penalty_m * route_positions[1:]
            advance_from[1:] = prefix_arg[:-1]
        stay = previous
        use_advance = advance_cost < stay
        best_previous = np.where(use_advance, advance_cost, stay)
        back[i] = np.where(use_advance, advance_from, np.arange(n_route)).astype(np.int32)
        previous = distances[i] + best_previous
    terminal = previous + endpoint_penalty_m * (n_route - 1 - route_positions)
    positions = np.empty(n_obs, dtype=np.int32)
    positions[-1] = int(np.argmin(terminal))
    for i in range(n_obs - 1, 0, -1):
        positions[i - 1] = back[i, positions[i]]
    return positions


def associate_reference_route(
    projected: ProjectedTrip,
    skip_penalty_m: float = 5.0,
    endpoint_penalty_m: float = 2.0,
) -> pd.DataFrame:
    route_arc_ids = projected.trip.route["arc_id"].to_numpy(dtype=np.int64)
    route_geometries = projected.arc_geometries[route_arc_ids]
    distances = point_route_distances(projected.track_xy, route_geometries)
    positions = monotonic_route_match(distances, skip_penalty_m, endpoint_penalty_m)
    row = np.arange(len(positions))
    assigned_distance = distances[row, positions].astype(float)
    if distances.shape[1] > 1:
        nearest_two = np.partition(distances, 1, axis=1)[:, :2]
        second_distance = nearest_two.max(axis=1).astype(float)
    else:
        second_distance = np.full(len(positions), np.inf)
    margin = second_distance - assigned_distance
    confidence = (1.0 / (1.0 + assigned_distance / 20.0)) * np.clip(margin / 20.0, 0.0, 1.0)
    return pd.DataFrame(
        {
            "observation_id": row,
            "reference_arc_id": route_arc_ids[positions],
            "reference_route_position": positions,
            "reference_distance_m": assigned_distance,
            "nearest_route_distance_m": distances.min(axis=1).astype(float),
            "distance_margin_m": margin,
            "label_confidence": confidence,
            "label_uncertain": (assigned_distance > 30.0) | (margin < 5.0),
        }
    )
