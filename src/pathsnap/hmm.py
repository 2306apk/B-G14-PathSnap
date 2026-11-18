"""Separate HMM-style Viterbi decoder for candidate sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import ProjectedTrip


def distance_emission_log_probability(distances_m: np.ndarray, sigma_m: float = 20.0) -> np.ndarray:
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")
    distances = np.asarray(distances_m, dtype=float)
    return -0.5 * (distances / sigma_m) ** 2


def _transition_cost(
    projected: ProjectedTrip,
    previous_arcs: np.ndarray,
    current_arcs: np.ndarray,
    disconnected_penalty: float,
    endpoint_scale_m: float,
) -> np.ndarray:
    arcs = projected.trip.arcs
    prev_endpoints = arcs.iloc[previous_arcs][["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    curr_endpoints = arcs.iloc[current_arcs][["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    same = previous_arcs[:, None] == current_arcs[None, :]
    directed = prev_endpoints[:, 1, None] == curr_endpoints[None, :, 0]
    shares_node = (
        (prev_endpoints[:, 0, None] == curr_endpoints[None, :, 0])
        | (prev_endpoints[:, 0, None] == curr_endpoints[None, :, 1])
        | (prev_endpoints[:, 1, None] == curr_endpoints[None, :, 0])
        | (prev_endpoints[:, 1, None] == curr_endpoints[None, :, 1])
    )
    prev_to_xy = projected.nodes_xy[prev_endpoints[:, 1]]
    curr_from_xy = projected.nodes_xy[curr_endpoints[:, 0]]
    endpoint_distance = np.linalg.norm(prev_to_xy[:, None, :] - curr_from_xy[None, :, :], axis=2)
    cost = disconnected_penalty + endpoint_distance / endpoint_scale_m
    cost = np.where(shares_node, 1.0, cost)
    cost = np.where(directed, 0.25, cost)
    cost = np.where(same, 0.0, cost)
    return cost


def viterbi_decode(
    projected: ProjectedTrip,
    candidates: pd.DataFrame,
    emission_log_probability: np.ndarray | None = None,
    sigma_m: float = 20.0,
    transition_weight: float = 1.0,
    disconnected_penalty: float = 6.0,
    endpoint_scale_m: float = 50.0,
    gap_reset_s: float = 60.0,
) -> pd.DataFrame:
    """Decode observations that have candidates; gaps/no-candidate spans reset the chain."""
    if candidates.empty:
        return pd.DataFrame(columns=["observation_id", "predicted_arc_id", "viterbi_score"])
    ordered = candidates.sort_values(["observation_id", "candidate_rank"]).reset_index(drop=True)
    if emission_log_probability is None:
        distance_column = "point_arc_distance_m" if "point_arc_distance_m" in ordered else "candidate_distance_m"
        emissions = distance_emission_log_probability(ordered[distance_column].to_numpy(), sigma_m)
    else:
        emissions = np.asarray(emission_log_probability, dtype=float)
        if len(emissions) != len(ordered):
            raise ValueError("Emission vector length does not match candidates")
    ordered["_emission"] = emissions
    groups = [(int(obs), group) for obs, group in ordered.groupby("observation_id", sort=True)]
    timestamps = projected.trip.track["timestamp_s"].to_numpy(dtype=float)
    decoded: list[dict[str, float | int]] = []
    segment: list[tuple[int, pd.DataFrame]] = []

    def flush(items: list[tuple[int, pd.DataFrame]]) -> None:
        if not items:
            return
        scores: list[np.ndarray] = []
        backpointers: list[np.ndarray] = []
        arcs_by_step: list[np.ndarray] = []
        for step, (_, group) in enumerate(items):
            arcs = group["arc_id"].to_numpy(dtype=np.int64)
            emission = group["_emission"].to_numpy(dtype=float)
            arcs_by_step.append(arcs)
            if step == 0:
                scores.append(emission)
                backpointers.append(np.full(len(arcs), -1, dtype=np.int32))
                continue
            transition = _transition_cost(
                projected, arcs_by_step[-2], arcs, disconnected_penalty, endpoint_scale_m
            )
            options = scores[-1][:, None] - transition_weight * transition
            best_previous = np.argmax(options, axis=0)
            scores.append(emission + options[best_previous, np.arange(len(arcs))])
            backpointers.append(best_previous.astype(np.int32))
        states = np.empty(len(items), dtype=np.int32)
        states[-1] = int(np.argmax(scores[-1]))
        for step in range(len(items) - 1, 0, -1):
            states[step - 1] = backpointers[step][states[step]]
        for step, ((observation_id, _), state) in enumerate(zip(items, states)):
            decoded.append(
                {
                    "observation_id": observation_id,
                    "predicted_arc_id": int(arcs_by_step[step][state]),
                    "viterbi_score": float(scores[step][state]),
                }
            )

    previous_observation: int | None = None
    for observation_id, group in groups:
        reset = False
        if previous_observation is not None:
            reset = observation_id != previous_observation + 1
            if not reset:
                reset = timestamps[observation_id] - timestamps[previous_observation] > gap_reset_s
        if reset:
            flush(segment)
            segment = []
        segment.append((observation_id, group))
        previous_observation = observation_id
    flush(segment)
    return pd.DataFrame(decoded).sort_values("observation_id").reset_index(drop=True)
