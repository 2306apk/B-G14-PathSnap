"""Hidden Markov Model (HMM) and Viterbi sequential decoder for PathSnap route reconstruction.

Integrates the 18-feature map matching architecture:
- 14 observation-level classifier emission features.
- 4 transition-level features (gps_displacement_m, network_transition_distance_m,
  transition_distance_residual_m, turn_angle_change_deg).
- Calibrated unreachable penalties, turn-angle constraints, and sequence gap resets.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import numpy as np
import pandas as pd

from pathsnap.graph import RoadNetwork
from pathsnap.features import (
    TransitionFeatures,
    calculate_turn_angle_change,
    OBSERVATION_FEATURE_COLUMNS,
    TRANSITION_FEATURE_COLUMNS,
)


@dataclass
class HMMDecodingResult:
    """Stores the outcome of HMM Viterbi route decoding for a single trip."""

    trip_id: str
    num_observations: int
    predicted_arc_ids: np.ndarray
    classifier_top1_arc_ids: np.ndarray
    route_continuity: float
    classifier_continuity: float
    num_resets: int
    execution_time_seconds: float


class HMMRouteDecoder:
    """Sequential route decoder combining classifier emission evidence with 4-feature transition scoring."""

    def __init__(
        self,
        network: RoadNetwork,
        beta: float = 20.0,
        unreachable_penalty: float = 5.0,
        theta_0: float = 55.0,
        emission_weight: float = 1.5,
        transition_weight: float = 1.0,
        stay_bonus: float = 1.5,
        adjacent_bonus: float = 1.5,
        large_gap_threshold_seconds: float = 30.0,
    ):
        """Initializes the HMM decoder with configurable transition and emission parameters.

        Parameters:
        - network: RoadNetwork instance with precomputed graph and arc table.
        - beta: Scale parameter for distance difference exponential distribution (meters).
        - unreachable_penalty: Bounded penalty (in log space) for transitions between disconnected arcs.
        - theta_0: Characteristic scale for turn-angle penalty (degrees).
        - emission_weight: Scaling multiplier for classifier log-emission probabilities.
        - transition_weight: Scaling multiplier for graph transition log-probabilities.
        - stay_bonus: Log-probability bonus for staying on the same road segment.
        - adjacent_bonus: Log-probability bonus for moving to a directly connected downstream segment.
        - large_gap_threshold_seconds: Threshold for resetting transition memory across GPS gaps.
        """
        self.network = network
        self.beta = max(1.0, beta)
        self.unreachable_penalty = float(unreachable_penalty)
        self.theta_0 = max(10.0, float(theta_0))
        self.emission_weight = float(emission_weight)
        self.transition_weight = float(transition_weight)
        self.stay_bonus = float(stay_bonus)
        self.adjacent_bonus = float(adjacent_bonus)
        self.large_gap_threshold = float(large_gap_threshold_seconds)

        # Pre-extract endpoint nodes and bearings for O(1) lookups
        self.from_nodes = network.arc_table["from_node_id"].values
        self.to_nodes = network.arc_table["to_node_id"].values
        self.arc_lengths = network.arc_table["length"].values
        self.arc_bearings = network.arc_table["bearing"].values

        # Bounded Dijkstra distance cache: u_node -> dict of {v_node: distance}
        self._dijkstra_cache: Dict[int, Dict[int, float]] = {}

    def _get_network_distance(self, u_node: int, v_node: int, max_dist: float) -> float:
        """Computes shortest path length between u_node and v_node using bounded Dijkstra."""
        if u_node == v_node:
            return 0.0

        if u_node not in self._dijkstra_cache:
            try:
                # Bounded Dijkstra explores only small local radius around u_node
                cutoff_radius = max(150.0, max_dist * 2.5)
                reachable = nx.single_source_dijkstra_path_length(
                    self.network.graph, source=u_node, cutoff=cutoff_radius, weight="weight"
                )
                self._dijkstra_cache[u_node] = reachable
            except (nx.NodeNotFound, KeyError):
                self._dijkstra_cache[u_node] = {}

        return self._dijkstra_cache[u_node].get(v_node, -1.0)

    def compute_18_transition_features(
        self,
        prev_arc: int,
        curr_arc: int,
        d_gps: float,
        dt: float,
        prev_proj: Optional[float] = None,
        curr_proj: Optional[float] = None,
    ) -> TransitionFeatures:
        """Calculates the four transition-level features for candidate transition (prev_arc -> curr_arc).

        Features:
        - Feature 15: gps_displacement_m
        - Feature 16: network_transition_distance_m
        - Feature 17: transition_distance_residual_m
        - Feature 18: turn_angle_change_deg
        """
        f15_gps_disp = float(d_gps)

        # Feature 18: Turn angle change between arc bearings
        b_prev = float(self.arc_bearings[prev_arc])
        b_curr = float(self.arc_bearings[curr_arc])
        f18_turn_angle = calculate_turn_angle_change(b_prev, b_curr)

        # Case 1: Same road segment
        if prev_arc == curr_arc:
            if prev_proj is not None and curr_proj is not None:
                arc_len = float(self.arc_lengths[prev_arc])
                f16_net_dist = max(0.0, (curr_proj - prev_proj) * arc_len)
                f17_residual = abs(f16_net_dist - f15_gps_disp)
            else:
                f16_net_dist = f15_gps_disp
                f17_residual = 0.0
            return TransitionFeatures(
                gps_displacement_m=f15_gps_disp,
                network_transition_distance_m=f16_net_dist,
                transition_distance_residual_m=f17_residual,
                turn_angle_change_deg=f18_turn_angle,
                is_unreachable=False,
            )

        u_to = int(self.to_nodes[prev_arc])
        v_from = int(self.from_nodes[curr_arc])

        # Case 2: Directly connected adjacent segments meeting at an intersection node
        if u_to == v_from:
            if prev_proj is not None and curr_proj is not None:
                p_len = float(self.arc_lengths[prev_arc])
                c_len = float(self.arc_lengths[curr_arc])
                f16_net_dist = max(1.0, (1.0 - prev_proj) * p_len + curr_proj * c_len)
            else:
                f16_net_dist = max(1.0, f15_gps_disp)
            f17_residual = abs(f16_net_dist - f15_gps_disp)
            return TransitionFeatures(
                gps_displacement_m=f15_gps_disp,
                network_transition_distance_m=f16_net_dist,
                transition_distance_residual_m=f17_residual,
                turn_angle_change_deg=f18_turn_angle,
                is_unreachable=False,
            )

        # Case 3: Network shortest path
        net_dist = self._get_network_distance(u_to, v_from, d_gps)
        if net_dist < 0.0:
            return TransitionFeatures(
                gps_displacement_m=f15_gps_disp,
                network_transition_distance_m=float("inf"),
                transition_distance_residual_m=float("inf"),
                turn_angle_change_deg=f18_turn_angle,
                is_unreachable=True,
            )

        if prev_proj is not None and curr_proj is not None:
            p_len = float(self.arc_lengths[prev_arc])
            c_len = float(self.arc_lengths[curr_arc])
            f16_net_dist = float((1.0 - prev_proj) * p_len + net_dist + curr_proj * c_len)
        else:
            f16_net_dist = float(net_dist)
        f17_residual = abs(f16_net_dist - f15_gps_disp)
        return TransitionFeatures(
            gps_displacement_m=f15_gps_disp,
            network_transition_distance_m=f16_net_dist,
            transition_distance_residual_m=f17_residual,
            turn_angle_change_deg=f18_turn_angle,
            is_unreachable=False,
        )

    def compute_log_transition(
        self,
        prev_arc: int,
        curr_arc: int,
        d_gps: float,
        dt: float,
        prev_proj: Optional[float] = None,
        curr_proj: Optional[float] = None,
    ) -> float:
        """Calculates log transition probability using the four transition-level features."""
        # Sequence reset across large timestamp gaps
        if dt > self.large_gap_threshold:
            return 0.0

        feats = self.compute_18_transition_features(
            prev_arc, curr_arc, d_gps, dt, prev_proj, curr_proj
        )

        # Feature 18: Turn angle penalty
        turn_penalty = (feats.turn_angle_change_deg / self.theta_0) ** 2

        # Case 1: Same road segment
        if prev_arc == curr_arc:
            backward_pen = 3.0 if (prev_proj is not None and curr_proj is not None and curr_proj + 0.15 < prev_proj) else 0.0
            return self.stay_bonus - (feats.gps_displacement_m / 60.0) - backward_pen

        # Case 2: Unreachable transition in directed graph
        if feats.is_unreachable:
            # Calibrated unreachable penalty; does not crush strong classifier emissions
            return -self.unreachable_penalty - 0.5 * turn_penalty

        # Case 3: Reachable path
        # Feature 17: Distance residual penalty
        dist_cost = feats.transition_distance_residual_m / self.beta

        # Speed plausibility check (penalizes speeds exceeding highway limits > 65 m/s = 234 km/h)
        v_net = feats.network_transition_distance_m / max(dt, 0.1)
        speed_penalty = max(0.0, (v_net - 65.0) / 20.0)

        # Adjacency bonus applied only for forward connections (turn_angle < 80 deg)
        u_to = self.to_nodes[prev_arc]
        v_from = self.from_nodes[curr_arc]
        adj = self.adjacent_bonus if (u_to == v_from and feats.turn_angle_change_deg < 80.0) else 0.0

        return adj - dist_cost - turn_penalty - speed_penalty

    def decode(
        self,
        trip_candidate_df: pd.DataFrame,
        candidate_probabilities: np.ndarray,
        track_df: pd.DataFrame,
    ) -> HMMDecodingResult:
        """Executes Viterbi sequential decoding over the trip trajectory.

        Parameters:
        - trip_candidate_df: DataFrame containing candidate rows for the trip.
        - candidate_probabilities: 1D array of calibrated P(y=1) scores matching trip_candidate_df.
        - track_df: Raw GPS track DataFrame for this trip with 'timestamp', 'longitude', 'latitude'.
        """
        t_start = time.perf_counter()

        df = trip_candidate_df.copy()
        df["prob"] = np.asarray(candidate_probabilities, dtype=np.float64)

        # Projected UTM GPS coordinates
        tx, ty = self.network.projector.forward(
            track_df["longitude"].values, track_df["latitude"].values
        )
        timestamps = track_df["timestamp"].values
        num_obs = len(track_df)

        grouped = df.groupby("observation_id")
        obs_candidates: Dict[int, List[int]] = {
            int(obs_id): group["candidate_arc_id"].values.astype(int).tolist()
            for obs_id, group in grouped
        }
        obs_probs: Dict[int, List[float]] = {
            int(obs_id): group["prob"].values.astype(float).tolist()
            for obs_id, group in grouped
        }
        has_proj = "proj_ratio" in df.columns
        obs_projs: Dict[int, List[float]] = {}
        if has_proj:
            obs_projs = {
                int(obs_id): group["proj_ratio"].values.astype(float).tolist()
                for obs_id, group in grouped
            }

        # Pointwise Classifier-Only Top-1 baseline
        classifier_top1 = np.full(num_obs, -1, dtype=np.int64)
        for i in range(num_obs):
            cands = obs_candidates.get(i, [])
            probs = obs_probs.get(i, [])
            if len(cands) > 0:
                classifier_top1[i] = cands[int(np.argmax(probs))]

        # Viterbi Trellis
        viterbi_states: List[Dict[int, Tuple[float, Optional[int]]]] = []
        num_resets = 0

        # Step 0
        cands_0 = obs_candidates.get(0, [])
        probs_0 = obs_probs.get(0, [])
        state_0: Dict[int, Tuple[float, Optional[int]]] = {}

        if len(cands_0) > 0:
            sum_p = sum(probs_0) + 1e-12
            for arc, p in zip(cands_0, probs_0):
                log_em = math.log(max(1e-9, p / sum_p))
                state_0[arc] = (self.emission_weight * log_em, None)
            max_s0 = max(v[0] for v in state_0.values())
            state_0 = {k: (v[0] - max_s0, v[1]) for k, v in state_0.items()}
        viterbi_states.append(state_0)

        # Forward Viterbi passes
        for i in range(1, num_obs):
            curr_cands = obs_candidates.get(i, [])
            curr_probs = obs_probs.get(i, [])
            prev_state = viterbi_states[i - 1]

            dt = float(timestamps[i] - timestamps[i - 1])
            dx = float(tx[i] - tx[i - 1])
            dy = float(ty[i] - ty[i - 1])
            d_gps = math.hypot(dx, dy)

            # Check for 0 candidates
            if len(curr_cands) == 0:
                viterbi_states.append(prev_state)
                continue

            # Check for sequence gap or empty previous state
            if len(prev_state) == 0 or dt > self.large_gap_threshold:
                num_resets += 1
                curr_state: Dict[int, Tuple[float, Optional[int]]] = {}
                sum_p = sum(curr_probs) + 1e-12
                for arc, p in zip(curr_cands, curr_probs):
                    log_em = math.log(max(1e-9, p / sum_p))
                    curr_state[arc] = (self.emission_weight * log_em, None)
                if len(curr_state) > 0:
                    max_sc = max(v[0] for v in curr_state.values())
                    curr_state = {k: (v[0] - max_sc, v[1]) for k, v in curr_state.items()}
                viterbi_states.append(curr_state)
                continue

            sum_p = sum(curr_probs) + 1e-12
            log_emissions = [math.log(max(1e-9, p / sum_p)) for p in curr_probs]

            curr_proj_list = obs_projs.get(i, [None] * len(curr_cands)) if has_proj else [None] * len(curr_cands)
            prev_cands_list = obs_candidates.get(i - 1, [])
            prev_proj_list = obs_projs.get(i - 1, [None] * len(prev_cands_list)) if has_proj else [None] * len(prev_cands_list)
            prev_proj_map = dict(zip(prev_cands_list, prev_proj_list)) if has_proj else {}

            curr_state = {}
            for rel_idx, curr_arc in enumerate(curr_cands):
                log_em = self.emission_weight * log_emissions[rel_idx]
                curr_proj_val = curr_proj_list[rel_idx] if rel_idx < len(curr_proj_list) else None
                best_score = -float("inf")
                best_arc = None

                for prev_arc, (prev_score, _) in prev_state.items():
                    prev_proj_val = prev_proj_map.get(prev_arc)
                    log_trans = self.transition_weight * self.compute_log_transition(
                        prev_arc, curr_arc, d_gps, dt, prev_proj_val, curr_proj_val
                    )
                    score = prev_score + log_trans + log_em
                    if score > best_score:
                        best_score = score
                        best_arc = prev_arc

                curr_state[curr_arc] = (best_score, best_arc)

            if len(curr_state) > 0:
                max_sc = max(v[0] for v in curr_state.values())
                curr_state = {k: (v[0] - max_sc, v[1]) for k, v in curr_state.items()}

            viterbi_states.append(curr_state)

        # Backtracking
        decoded_path: List[int] = []
        last_state = viterbi_states[-1]
        best_final_arc = (
            max(last_state.keys(), key=lambda a: last_state[a][0])
            if len(last_state) > 0
            else int(classifier_top1[-1])
        )
        curr_arc_ptr = best_final_arc
        decoded_path.append(curr_arc_ptr)

        for i in range(num_obs - 1, 0, -1):
            state = viterbi_states[i]
            if curr_arc_ptr in state and state[curr_arc_ptr][1] is not None:
                curr_arc_ptr = state[curr_arc_ptr][1]
            else:
                prev_st = viterbi_states[i - 1]
                curr_arc_ptr = (
                    max(prev_st.keys(), key=lambda a: prev_st[a][0])
                    if len(prev_st) > 0
                    else int(classifier_top1[i - 1])
                )
            decoded_path.append(curr_arc_ptr)

        decoded_path.reverse()
        predicted_arcs = np.array(decoded_path, dtype=np.int64)

        from pathsnap.metrics import compute_route_continuity
        hmm_cont = compute_route_continuity(predicted_arcs, self.network)["route_continuity"]
        clf_cont = compute_route_continuity(classifier_top1, self.network)["route_continuity"]

        t_elapsed = time.perf_counter() - t_start

        return HMMDecodingResult(
            trip_id=self.network.trip_id,
            num_observations=num_obs,
            predicted_arc_ids=predicted_arcs,
            classifier_top1_arc_ids=classifier_top1,
            route_continuity=float(hmm_cont),
            classifier_continuity=float(clf_cont),
            num_resets=num_resets,
            execution_time_seconds=round(t_elapsed, 3),
        )

    def extract_decoded_transition_features(
        self, predicted_arc_ids: np.ndarray, track_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extracts the sequence of four transition features along the decoded Viterbi path."""
        tx, ty = self.network.projector.forward(
            track_df["longitude"].values, track_df["latitude"].values
        )
        timestamps = track_df["timestamp"].values
        num_obs = len(track_df)

        records = []
        for i in range(num_obs):
            if i == 0:
                records.append({
                    "observation_id": 0,
                    "arc_id": predicted_arc_ids[0],
                    "gps_displacement_m": 0.0,
                    "network_transition_distance_m": 0.0,
                    "transition_distance_residual_m": 0.0,
                    "turn_angle_change_deg": 0.0,
                    "is_unreachable": False,
                })
            else:
                prev_arc = int(predicted_arc_ids[i - 1])
                curr_arc = int(predicted_arc_ids[i])
                dt = float(timestamps[i] - timestamps[i - 1])
                dx = float(tx[i] - tx[i - 1])
                dy = float(ty[i] - ty[i - 1])
                d_gps = math.hypot(dx, dy)

                feats = self.compute_18_transition_features(prev_arc, curr_arc, d_gps, dt)
                records.append({
                    "observation_id": i,
                    "arc_id": curr_arc,
                    "gps_displacement_m": feats.gps_displacement_m,
                    "network_transition_distance_m": feats.network_transition_distance_m,
                    "transition_distance_residual_m": feats.transition_distance_residual_m,
                    "turn_angle_change_deg": feats.turn_angle_change_deg,
                    "is_unreachable": feats.is_unreachable,
                })

        return pd.DataFrame(records)
