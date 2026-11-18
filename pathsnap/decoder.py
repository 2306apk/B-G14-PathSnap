"""HMM emission + road-network-aware Viterbi decoding for candidate arc IDs.

States: each GPS observation's candidate directed road arcs.
Emissions: log normalized binary classifier probabilities.
Transitions: use road-network shortest paths between arc endpoints, plus
movement plausibility and a reset for large GPS gaps / disconnected roads.
"""
from __future__ import annotations
from collections import defaultdict
from heapq import heappush, heappop
import numpy as np
from .data import Trip, project_trip
from .candidates import CandidateBatch


class RoadGraph:
    def __init__(self, trip: Trip):
        _, nodes = project_trip(trip)
        self.nodes = nodes
        self.arcs = trip.arcs
        self.lengths = np.linalg.norm(nodes[trip.arcs[:,0]]-nodes[trip.arcs[:,1]],axis=1)
        self.links = defaultdict(list)
        self._path_cache = {}
        for arc_id,(u,v) in enumerate(trip.arcs):
            self.links[int(u)].append((int(v), max(0.1,float(self.lengths[arc_id]))))

    def path_length(self, origin: int, target: int, limit_m: int) -> float:
        """Bounded directed Dijkstra; inf means no reachable path within cutoff."""
        key=(origin,target,limit_m)
        if key in self._path_cache:
            return self._path_cache[key]
        answer=self._uncached_path_length(origin,target,limit_m)
        if len(self._path_cache)>40000:
            self._path_cache.clear()
        self._path_cache[key]=answer
        return answer

    def _uncached_path_length(self, origin: int, target: int, limit_m: int) -> float:
        if origin == target:
            return 0.0
        # Straight-line distance is an admissible lower bound in projected metres.
        euclid = float(np.linalg.norm(self.nodes[origin]-self.nodes[target]))
        if euclid > limit_m * 1.01:
            return float('inf')
        best = {origin:0.0}
        heap = [(0.0,origin)]
        while heap:
            d,u = heappop(heap)
            if d > limit_m:
                break
            if u == target:
                return d
            if d > best[u]:
                continue
            for v,w in self.links.get(u, ()):
                new_d = d+w
                if new_d < best.get(v,float('inf')) and new_d <= limit_m:
                    best[v] = new_d
                    heappush(heap,(new_d,v))
        return float('inf')


def decode(trip: Trip, batch: CandidateBatch, probabilities: np.ndarray,
           obs_indices: np.ndarray | None = None, gap_reset_seconds: float = 45,
           reset_penalty: float = -12.) -> np.ndarray:
    """Return one arc ID per GPS sample, never reference labels.

    In disconnected/no-path cases, an explicit low-scoring reset keeps a sequence
    decodable; gap resets are independent when GPS timestamps are far apart.
    """
    if obs_indices is None:
        obs_indices = np.arange(len(trip.track))
    obs_indices = np.asarray(obs_indices,dtype=int)
    if not len(obs_indices):
        return np.empty(0,dtype=np.int32)
    ids = batch.arc_ids
    if probabilities.shape != ids.shape or len(obs_indices) != len(ids):
        raise ValueError('Batch, GPS indices and probability arrays are misaligned')
    graph = RoadGraph(trip)
    gps_xy,_ = project_trip(trip)
    prob = np.maximum(np.asarray(probabilities,dtype=float), 1e-8)
    prob /= np.maximum(np.sum(prob,axis=1,keepdims=True),1e-8)
    emit = np.log(prob)
    score = emit[0].copy()
    back = np.zeros(ids.shape,dtype=np.int16)
    for t in range(1,len(ids)):
        dt = float(trip.track[obs_indices[t],2]-trip.track[obs_indices[t-1],2])
        gps_m = float(np.linalg.norm(gps_xy[obs_indices[t]]-gps_xy[obs_indices[t-1]]))
        if dt > gap_reset_seconds or dt <= 0:
            # Recover independent state sequence after a substantial dropout.
            back[t] = int(np.argmax(score))
            score = emit[t] + np.max(score)
            continue
        limit = int(min(2000, max(100, gps_m*2.4 + 100, dt*50)))
        transitions = np.full((ids.shape[1],ids.shape[1]), reset_penalty,dtype=float)
        for i,a in enumerate(ids[t-1]):
            a = int(a)
            for j,b in enumerate(ids[t]):
                b = int(b)
                if a==b:
                    # Same arc: small progress, including stationary vehicle.
                    road_move = max(0.0,(batch.fraction[t,j]-batch.fraction[t-1,i])*graph.lengths[a])
                    backward_penalty = 3.0 if batch.fraction[t,j] + .15 < batch.fraction[t-1,i] else 0.
                else:
                    end_node = int(trip.arcs[a,1]); start_node = int(trip.arcs[b,0])
                    bridge = graph.path_length(end_node,start_node,limit)
                    if not np.isfinite(bridge):
                        continue
                    road_move = (1.0-batch.fraction[t-1,i])*graph.lengths[a] + bridge + batch.fraction[t,j]*graph.lengths[b]
                    backward_penalty = 0.
                mismatch = abs(road_move-gps_m)/(25.0 + gps_m*0.5)
                max_move = max(55.*dt, gps_m*2+40)
                implausible = max(0.,road_move-max_move)/30.
                transitions[i,j] = -mismatch - implausible - backward_penalty
        candidate = score[:,None]+transitions
        best_prev = np.argmax(candidate,axis=0)
        back[t] = best_prev
        score = emit[t]+candidate[best_prev,np.arange(ids.shape[1])]
        # Relative scores avoid unbounded negative accumulation, don't alter Viterbi argmax.
        score -= np.max(score)
    selected = np.empty(len(ids),dtype=np.int32)
    state = int(np.argmax(score))
    for t in range(len(ids)-1,-1,-1):
        selected[t] = int(ids[t,state])
        state = int(back[t,state])
    return selected
