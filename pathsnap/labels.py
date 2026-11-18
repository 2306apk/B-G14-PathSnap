"""Project ordered reference-route geometry to GPS samples for approximate labels.

The .route file is an ordered arc sequence, *not* pointwise GPS ground truth.
Labels constructed here are explicitly pseudo-aligned and may be imperfect near loops,
intersections and reference-route gaps. Never import/use this in runtime prediction.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .data import Trip, project_trip


def approximate_reference_labels(trip: Trip, obs_indices: np.ndarray | None = None,
                                 sample_step_m: float = 25, beam_k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    if trip.route is None or not len(trip.route):
        raise ValueError(f'{trip.trip_id} has no reference route for evaluation/alignment')
    if obs_indices is None:
        obs_indices = np.arange(len(trip.track))
    obs_indices = np.asarray(obs_indices, dtype=int)
    gps_xy, nodes_xy = project_trip(trip)
    start = nodes_xy[trip.arcs[trip.route, 0]]
    end = nodes_xy[trip.arcs[trip.route, 1]]
    segment = end - start
    length = np.linalg.norm(segment, axis=1)
    cumdist = np.r_[0., np.cumsum(length)]
    # Sample along every route arc, including long arcs whose midpoint is too far from GPS.
    samples, route_idx = [], []
    for i, (p, v, meters) in enumerate(zip(start, segment, length)):
        chunks = max(1, int(np.ceil(meters / sample_step_m)))
        ts = np.linspace(0, 1, chunks + 1)
        samples.append(p[None, :] + ts[:, None] * v[None, :])
        route_idx.extend([i] * len(ts))
    sample_points = np.concatenate(samples, axis=0)
    route_idx = np.asarray(route_idx, dtype=np.int32)
    tree = cKDTree(sample_points)
    _, nearest = tree.query(gps_xy[obs_indices], k=min(max(beam_k*2, 16),len(sample_points)), workers=-1)
    nearest = np.asarray(nearest).reshape(len(obs_indices), -1)
    # Build top candidates via exact point-to-segment distances.
    choices, geomdist, progress = [], [], []
    for point, neighbors in zip(gps_xy[obs_indices], nearest):
        cand = np.unique(route_idx[neighbors])
        cand = np.unique(np.clip(np.concatenate((cand-1, cand, cand+1)), 0, len(length)-1))
        v = segment[cand]
        t = np.clip(np.sum((point - start[cand]) * v, axis=1) / np.maximum(length[cand]**2, 1e-8), 0, 1)
        projected = start[cand] + t[:, None]*v
        d = np.linalg.norm(point - projected, axis=1)
        order = np.argsort(d)[:beam_k]
        choices.append(cand[order])
        geomdist.append(d[order])
        progress.append(cumdist[cand[order]] + t[order]*length[cand[order]])

    # Ordered Viterbi: prefer moving forward on the provided reference arc sequence.
    back = []
    previous = None
    prev_progress = None
    for i, (cand, dist, along) in enumerate(zip(choices, geomdist, progress)):
        emit = -(np.minimum(dist, 300) / 25.) ** 2
        if previous is None:
            score = emit
            prev_choice = np.full(len(cand), -1, dtype=np.int32)
        else:
            dt = max(1., trip.track[obs_indices[i],2]-trip.track[obs_indices[i-1],2])
            gps_move = float(np.linalg.norm(gps_xy[obs_indices[i]] - gps_xy[obs_indices[i-1]]))
            delta = along[None,:] - prev_progress[:,None]
            # Soft penalty for backwards movement (looping route arcs allowed via ordered indices).
            backward = 15.0 * (delta < -5)
            # A jump far longer than plausible driving speed is strongly discouraged.
            huge = np.maximum(delta - max(60.0, 55.0*dt), 0) / 18.
            mismatch = np.abs(np.maximum(delta,0) - gps_move) / (40.0 + 2.0 * dt)
            scores = previous[:,None] - backward - huge - mismatch
            prev_choice = np.argmax(scores,axis=0).astype(np.int32)
            score = emit + scores[prev_choice,np.arange(len(cand))]
        back.append(prev_choice)
        previous = score
        prev_progress = along
    selected = np.empty(len(choices),dtype=np.int32)
    idx = int(np.argmax(previous))
    for i in range(len(choices)-1,-1,-1):
        selected[i] = idx
        idx = int(back[i][idx]) if i>0 else -1
    route_positions = np.array([choices[i][selected[i]] for i in range(len(selected))],dtype=np.int32)
    label_arc_ids = trip.route[route_positions]
    alignment_error_m = np.array([geomdist[i][selected[i]] for i in range(len(selected))])
    return label_arc_ids, alignment_error_m
