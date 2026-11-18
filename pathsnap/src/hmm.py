"""
Hidden Markov Model (HMM) and Viterbi Decoding for PathSnap Route Reconstruction.

Section 5 Answers Implementation:
1. Hidden States: Candidate road segments (arc_ids) generated within radius R of each GPS point.
2. Emission Probabilities: P(z_t | s_t) derived from the tuned classifier predicted probabilities.
3. Transition Probabilities: P(s_t | s_{t-1}) based on network reachability, shortest-path distance vs. Euclidean travel distance.
4. Handling missing candidates / gaps: Continuity fallbacks and uniform priors when candidates are sparse.
5. Disconnected road segments: Large transition penalties to prevent illegal topological teleportation.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import Point

class MapMatchingHMM:
    def __init__(
        self,
        beta: float = 12.0,            # Sharper transition scale penalizing illegal detour jumps
        disconnected_penalty: float = -35.0, # Stronger log penalty preventing network teleportation
        same_arc_bonus: float = 2.5     # Higher prior bonus for staying on legitimate continuous road segments
    ):
        self.beta = beta
        self.disconnected_penalty = disconnected_penalty
        self.same_arc_bonus = same_arc_bonus

    def build_graph(self, arcs_df: pd.DataFrame) -> nx.DiGraph:
        """Constructs directed graph from trip arcs with length as weight."""
        G = nx.DiGraph()
        # arcs_df has columns: arc_id, from_node, to_node, and optionally length
        for row in arcs_df.itertuples():
            # row has arc_id, from_node, to_node, length
            arc_id = int(row.arc_id)
            u = int(row.from_node)
            v = int(row.to_node)
            length = float(getattr(row, "length", 10.0))
            G.add_edge(u, v, arc_id=arc_id, weight=max(length, 1.0))
        return G

    def decode_trip(
        self,
        trip_features_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        prob_col: str = "pred_prob"
    ) -> List[int]:
        """
        Runs Viterbi dynamic programming algorithm to find the most probable sequence of arcs.
        Returns: List of chosen arc_ids, one per GPS point index.
        """
        # Group candidates by point_idx
        grouped = trip_features_df.groupby("point_idx")
        point_indices = sorted(grouped.groups.keys())
        if not point_indices:
            return []

        # Graph node mapping for arcs: arc_id -> (from_node, to_node, length)
        arc_info = {}
        for row in arcs_df.itertuples():
            arc_info[int(row.arc_id)] = (
                int(row.from_node),
                int(row.to_node),
                float(getattr(row, "length", 10.0))
            )

        # Pre-build NetworkX graph for shortest path queries
        G = self.build_graph(arcs_df)

        # Viterbi tables
        # V[t][i] = max log-probability ending at candidate state i for point t
        # backpointer[t][i] = index of previous state j at t-1 that maximized V[t][i]
        
        # Step 0: Initialize t = 0
        p0 = point_indices[0]
        group0 = grouped.get_group(p0)
        cands_0 = group0["arc_id"].values.astype(int)
        probs_0 = np.clip(group0[prob_col].values, 1e-6, 1.0 - 1e-6)
        
        # Initial log emission probabilities
        v_prev = np.log(probs_0)
        states_prev = cands_0
        
        # Keep track of backpointers and states per step
        all_states = [states_prev]
        backpointers = []

        # Cache for pairwise shortest path distances
        sp_cache = {}

        def get_shortest_path_dist(u_node: int, v_node: int) -> float:
            pair = (u_node, v_node)
            if pair in sp_cache:
                return sp_cache[pair]
            try:
                d = nx.shortest_path_length(G, source=u_node, target=v_node, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = -1.0  # Disconnected
            sp_cache[pair] = d
            return d

        # Step t: 1 to T-1
        for step_idx in range(1, len(point_indices)):
            pt = point_indices[step_idx]
            group_t = grouped.get_group(pt)
            cands_t = group_t["arc_id"].values.astype(int)
            probs_t = np.clip(group_t[prob_col].values, 1e-6, 1.0 - 1e-6)
            log_emissions = np.log(probs_t)

            n_prev = len(states_prev)
            n_curr = len(cands_t)
            v_curr = np.zeros(n_curr)
            bp_curr = np.zeros(n_curr, dtype=int)

            # Compute Euclidean distance between consecutive points (speed * dt)
            euclid_d = float(group_t["speed"].values[0] * max(group_t["dt"].values[0], 1.0))
            if euclid_d <= 0.0:
                euclid_d = 1.0

            for j in range(n_curr):
                arc_curr = cands_t[j]
                from_curr, to_curr, len_curr = arc_info.get(arc_curr, (-1, -1, 10.0))

                best_log_prob = -1e9
                best_prev_idx = 0

                for i in range(n_prev):
                    arc_prev = states_prev[i]
                    from_prev, to_prev, len_prev = arc_info.get(arc_prev, (-1, -1, 10.0))

                    # Calculate transition probability
                    if arc_prev == arc_curr:
                        # Remained on the same road arc
                        log_trans = self.same_arc_bonus
                    elif to_prev == from_curr:
                        # Direct connected topological neighbor
                        log_trans = 0.0
                    else:
                        # Route distance via network graph
                        route_dist = get_shortest_path_dist(to_prev, from_curr)
                        if route_dist < 0:
                            # Disconnected segments
                            log_trans = self.disconnected_penalty
                        else:
                            # Exponential transition probability based on path discrepancy |route_dist - euclid_d|
                            diff = abs(route_dist - euclid_d)
                            log_trans = -diff / self.beta

                    total_score = v_prev[i] + log_trans

                    if total_score > best_log_prob:
                        best_log_prob = total_score
                        best_prev_idx = i

                v_curr[j] = best_log_prob + log_emissions[j]
                bp_curr[j] = best_prev_idx

            v_prev = v_curr
            states_prev = cands_t
            all_states.append(states_prev)
            backpointers.append(bp_curr)

        # Traceback best path
        best_last_idx = int(np.argmax(v_prev))
        chosen_path = [all_states[-1][best_last_idx]]

        curr_idx = best_last_idx
        for step_idx in range(len(backpointers) - 1, -1, -1):
            curr_idx = backpointers[step_idx][curr_idx]
            chosen_path.append(all_states[step_idx][curr_idx])

        chosen_path.reverse()
        return chosen_path
