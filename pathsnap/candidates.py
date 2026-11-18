"""Spatial candidate generation and per-candidate features. No reference-route inputs."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .data import Trip, project_trip

FEATURE_NAMES = [
    'distance_m','log_distance_m','segment_length_m','log_segment_length',
    'heading_alignment','abs_heading_alignment','position_along_segment',
    'endpoint_proximity','gps_speed_mps','gps_motion_m','node_degree',
    'distance_rank','road_dx','road_dy'
]


@dataclass
class CandidateBatch:
    arc_ids: np.ndarray   # (GPS, candidate_count)
    distance: np.ndarray  # (GPS, candidate_count) metres
    fraction: np.ndarray  # fractional position on directed segment
    features: np.ndarray  # (GPS, candidate_count, num_features)


class SpatialCandidates:
    def __init__(self, trip: Trip):
        self.trip = trip
        self.gps_xy, self.node_xy = project_trip(trip)
        a = trip.arcs
        self.starts = self.node_xy[a[:, 0]]
        self.ends = self.node_xy[a[:, 1]]
        self.vectors = self.ends - self.starts
        self.length = np.linalg.norm(self.vectors, axis=1)
        self.length_sq = np.maximum(self.length * self.length, 1e-8)
        self.mid_tree = cKDTree((self.starts + self.ends) * 0.5)
        self.node_tree = cKDTree(self.node_xy)
        self.node_degree = np.bincount(a.reshape(-1), minlength=len(self.node_xy))
        # Efficient, array-based mapping node ID -> incident arc IDs.
        joined_nodes = a.reshape(-1)
        joined_arcs = np.repeat(np.arange(len(a), dtype=np.int32), 2)
        order = np.argsort(joined_nodes, kind='stable')
        self.incident_nodes = joined_nodes[order]
        self.incident_arcs = joined_arcs[order]

    def query(self, obs_indices: np.ndarray | None = None, k: int = 8,
              mid_k: int = 24, node_k: int = 8) -> CandidateBatch:
        if obs_indices is None:
            obs_indices = np.arange(len(self.gps_xy))
        obs_indices = np.asarray(obs_indices, dtype=int)
        points = self.gps_xy[obs_indices]
        n_arcs = len(self.trip.arcs)
        k = min(max(1, k), n_arcs)
        _, nearby_mid = self.mid_tree.query(points, k=min(mid_k, n_arcs), workers=-1)
        _, nearby_node = self.node_tree.query(points, k=min(node_k, len(self.node_xy)), workers=-1)
        nearby_mid = np.asarray(nearby_mid).reshape(len(points), -1)
        nearby_node = np.asarray(nearby_node).reshape(len(points), -1)
        indices = np.empty((len(points), k), dtype=np.int32)
        distances = np.empty((len(points), k), dtype=float)
        fractions = np.empty((len(points), k), dtype=float)
        for i, p in enumerate(points):
            ids = list(nearby_mid[i])
            for node in nearby_node[i]:
                lower = np.searchsorted(self.incident_nodes, node, side='left')
                upper = np.searchsorted(self.incident_nodes, node, side='right')
                ids.extend(self.incident_arcs[lower:upper])
            ids = np.unique(np.asarray(ids, dtype=np.int32))
            if len(ids) < k:
                # Tiny synthetic graphs can have fewer nearby arcs than requested.
                ids = np.arange(n_arcs, dtype=np.int32)
            vec = self.vectors[ids]
            t = np.clip(np.sum((p - self.starts[ids]) * vec, axis=1) / self.length_sq[ids], 0, 1)
            projection = self.starts[ids] + t[:, None] * vec
            d = np.linalg.norm(p - projection, axis=1)
            # Sort strictly by actual point-to-segment distance, not midpoint distance.
            choose = np.argsort(d, kind='stable')[:k]
            indices[i] = ids[choose]
            distances[i] = d[choose]
            fractions[i] = t[choose]
        feats = self._features(obs_indices, indices, distances, fractions)
        return CandidateBatch(indices, distances, fractions, feats)

    def _features(self, obs_indices, ids, d, fraction):
        track_xy = self.gps_xy
        ts = self.trip.track[:, 2]
        # Causal GPS velocity: only previous observations, never the reference route.
        dx = np.diff(track_xy, axis=0, prepend=track_xy[:1])
        dt = np.diff(ts, prepend=ts[:1]); dt[0] = 1.0
        dt = np.maximum(dt, 1e-6)
        speed = np.linalg.norm(dx, axis=1) / dt
        heading = dx / np.maximum(np.linalg.norm(dx, axis=1, keepdims=True), 1e-8)
        road_unit = self.vectors[ids] / np.maximum(self.length[ids, None], 1e-8)
        dot = np.sum(heading[obs_indices, None, :] * road_unit, axis=2)
        deg = self.node_degree[self.trip.arcs[ids, 0]] + self.node_degree[self.trip.arcs[ids, 1]]
        n, k = ids.shape
        def col(x):
            return np.broadcast_to(np.asarray(x), (n, k))
        features = np.stack([
            d, np.log1p(d), self.length[ids], np.log1p(self.length[ids]),
            dot, np.abs(dot), fraction, np.minimum(fraction, 1-fraction),
            col(speed[obs_indices, None]), col(np.linalg.norm(dx[obs_indices], axis=1)[:, None]),
            deg, col(np.arange(k)[None, :]), road_unit[:, :, 0], road_unit[:, :, 1]
        ], axis=-1)
        return features.astype(np.float32)
