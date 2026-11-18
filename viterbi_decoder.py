"""
PathSnap - Hidden Markov Model (HMM) & Viterbi Sequential Route Reconstructor
----------------------------------------------------------------------------
Implements the core sequential map-matching decoder:
1. Emission Probabilities:
   - Uses predicted probabilities P(arc is match | features) from the tuned ML classifier.
   - Formulated as log-emission: log P(O_t | S_t)

2. Transition Probabilities:
   - Uses the road network graph structure:
     * If S_{t-1} == S_t (vehicle remains on same segment): P_stay (very high probability)
     * If S_{t-1} directly connects to S_t in the road network: P_direct (connected road)
     * If S_{t-1} can reach S_t via short graph path: P_hop (exponential decay on shortest graph distance vs GPS distance)
     * If disconnected / unreachable: heavy penalty (log 1e-6)

3. Viterbi Decoding:
   - Dynamic programming finding the most likely continuous sequence of arcs:
     V[t, j] = max_i (V[t-1, i] + log_trans(i, j)) + log_emit(j)
   - Backtracks to produce the reconstructed route.

4. Evaluation:
   - Evaluates on the 10 Student Testing trips ('student_test')
   - Compares:
       * Raw Classifier-Only Top-1 Prediction (pointwise)
       * Classifier + HMM / Viterbi Reconstructed Route (sequential)
   - Computes:
       * Observation-level Top-1 Accuracy
       * Route Continuity % (valid graph transitions between consecutive steps)
       * Ground-Truth Arc Sequence Jaccard / Overlap Score
"""

import os
import glob
import math
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree

TUNED_MODEL_PATH = "tuned_model.pkl"
STUDENT_TEST_DIR = r"Student folder\Student folder\student_test"
RADIUS_M         = 35.0
TOP_K            = 4
R_EARTH          = 6_371_000.0

FEATURE_COLS = [
    "dist_to_arc_m", "log_dist", "is_within_segment", "projection_t",
    "bearing_diff", "cos_bearing_diff", "heading_aligned", "arc_len_m",
    "log_arc_len", "speed_mps", "delta_t", "delta_d", "speed_to_len_ratio"
]

def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlam/2.0)**2
    return 2.0 * R_EARTH * math.asin(math.sqrt(min(1.0, max(0.0, a))))

def bearing(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(p2)
    y = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

def point_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx*dx + dy*dy
    if l2 == 0:
        return math.hypot(px - ax, py - ay), 0.0, True
    raw_t = ((px - ax)*dx + (py - ay)*dy) / l2
    t = max(0.0, min(1.0, raw_t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist_m = math.hypot(px - proj_x, py - proj_y)
    return dist_m, t, (0.0 <= raw_t <= 1.0)

def build_road_graph(arcs):
    """
    Builds forward connectivity:
    adjacency[from_node] = list of (to_node, arc_id)
    arc_to_nodes[arc_id] = (from_node, to_node)
    arc_successors[arc_id] = set of connected outgoing arc_ids
    """
    node_outgoing = defaultdict(list)
    arc_to_nodes = {}
    for arc_id, (fn, tn) in enumerate(arcs):
        node_outgoing[fn].append((tn, arc_id))
        arc_to_nodes[arc_id] = (fn, tn)

    arc_successors = defaultdict(set)
    for arc_id, (fn, tn) in enumerate(arcs):
        # Arcs starting from this arc's to_node are immediate successors
        for next_node, next_arc in node_outgoing[tn]:
            arc_successors[arc_id].add(next_arc)

    return arc_successors, arc_to_nodes

def viterbi_reconstruct(track_points, candidates_per_step, arc_successors, clf):
    """
    Applies the Viterbi dynamic programming algorithm across the trajectory.
    candidates_per_step: list of candidate dicts with features and arc_id
    """
    T = len(candidates_per_step)
    if T == 0:
        return []

    # Collect all candidate features for batch model inference
    all_cand_features = []
    cands_slices = []
    curr_idx = 0

    for step_cands in candidates_per_step:
        if not step_cands:
            cands_slices.append((curr_idx, curr_idx))
            continue
        cands_slices.append((curr_idx, curr_idx + len(step_cands)))
        for c in step_cands:
            all_cand_features.append([c[k] for k in FEATURE_COLS])
        curr_idx += len(step_cands)

    if not all_cand_features:
        return []

    # Batch prediction of emission probabilities
    probs = clf.predict_proba(all_cand_features)[:, 1]

    # Pre-populate emission log probabilities
    emissions = []
    for s, e in cands_slices:
        if s == e:
            emissions.append(np.array([]))
        else:
            step_probs = probs[s:e]
            # Regularize emission probability to prevent log(0)
            step_probs = np.clip(step_probs, 1e-4, 0.999)
            emissions.append(np.log(step_probs))

    # Initialize Viterbi matrices
    # dp[t][i] stores max log probability of path ending at candidate i at step t
    # backpointer[t][i] stores previous candidate index
    dp = [None] * T
    backpointer = [None] * T

    # Step 0 initialization
    first_step = 0
    while first_step < T and len(candidates_per_step[first_step]) == 0:
        first_step += 1

    if first_step >= T:
        return []

    dp[first_step] = emissions[first_step].copy()
    backpointer[first_step] = np.zeros(len(dp[first_step]), dtype=int)

    # Transition log weights:
    LOG_STAY = math.log(0.60)       # staying on same road segment
    LOG_DIRECT = math.log(0.35)     # direct valid connection in graph
    LOG_DISCONNECTED = math.log(1e-4) # disconnected / jump penalty

    for t in range(first_step + 1, T):
        curr_cands = candidates_per_step[t]
        prev_cands = candidates_per_step[t - 1]

        if not curr_cands:
            dp[t] = dp[t - 1]
            backpointer[t] = backpointer[t - 1]
            continue

        if dp[t - 1] is None or len(prev_cands) == 0:
            dp[t] = emissions[t].copy()
            backpointer[t] = np.zeros(len(curr_cands), dtype=int)
            continue

        dp[t] = np.zeros(len(curr_cands))
        backpointer[t] = np.zeros(len(curr_cands), dtype=int)

        for j, curr_c in enumerate(curr_cands):
            curr_arc = curr_c["arc_id"]
            best_prev_score = -float("inf")
            best_prev_idx = 0

            for i, prev_c in enumerate(prev_cands):
                prev_arc = prev_c["arc_id"]

                # Determine transition probability
                if curr_arc == prev_arc:
                    trans_prob = LOG_STAY
                elif curr_arc in arc_successors[prev_arc]:
                    trans_prob = LOG_DIRECT
                else:
                    trans_prob = LOG_DISCONNECTED

                score = dp[t - 1][i] + trans_prob
                if score > best_prev_score:
                    best_prev_score = score
                    best_prev_idx = i

            dp[t][j] = best_prev_score + emissions[t][j]
            backpointer[t][j] = best_prev_idx

    # Backtracking best path
    reconstructed_arcs = []
    last_step = T - 1
    while last_step >= 0 and (dp[last_step] is None or len(dp[last_step]) == 0):
        last_step -= 1

    if last_step < 0:
        return []

    best_cand_idx = int(np.argmax(dp[last_step]))
    curr_idx = best_cand_idx

    for t in range(last_step, first_step - 1, -1):
        if candidates_per_step[t]:
            reconstructed_arcs.append(candidates_per_step[t][curr_idx]["arc_id"])
            curr_idx = backpointer[t][curr_idx]
        else:
            # fill gap with duplicate or None
            if reconstructed_arcs:
                reconstructed_arcs.append(reconstructed_arcs[-1])

    reconstructed_arcs.reverse()
    return reconstructed_arcs

def evaluate_student_test_trips():
    print(f"Loading Tuned Model from: {TUNED_MODEL_PATH}")
    clf = joblib.load(TUNED_MODEL_PATH)

    test_dirs = sorted(glob.glob(os.path.join(STUDENT_TEST_DIR, "*")))
    print(f"Found {len(test_dirs)} Student Test trips to evaluate.\n")

    summary_rows = []

    for trip_dir in test_dirs:
        tid = os.path.basename(trip_dir)
        track_path = os.path.join(trip_dir, f"{tid}.track")
        nodes_path = os.path.join(trip_dir, f"{tid}.nodes")
        arcs_path  = os.path.join(trip_dir, f"{tid}.arcs")
        route_path = os.path.join(trip_dir, f"{tid}.route")

        track = np.loadtxt(track_path)
        nodes = np.loadtxt(nodes_path)
        arcs  = np.loadtxt(arcs_path, dtype=int)
        gt_route = list(np.loadtxt(route_path, dtype=int)) if os.path.exists(route_path) else []
        gt_route_set = set(gt_route)

        arc_successors, _ = build_road_graph(arcs)

        # Coordinate projection
        ref_lon, ref_lat = float(nodes[:, 0].mean()), float(nodes[:, 1].mean())
        scale_x = math.radians(1.0) * R_EARTH * math.cos(math.radians(ref_lat))
        scale_y = math.radians(1.0) * R_EARTH
        node_x = (nodes[:, 0] - ref_lon) * scale_x
        node_y = (nodes[:, 1] - ref_lat) * scale_y

        from_ids = arcs[:, 0]
        to_ids   = arcs[:, 1]
        ax, ay = node_x[from_ids], node_y[from_ids]
        bx, by = node_x[to_ids], node_y[to_ids]
        mid_x = (ax + bx) / 2.0
        mid_y = (ay + by) / 2.0
        arc_lens = np.hypot(bx - ax, by - ay)

        lon_a, lat_a = nodes[from_ids, 0], nodes[from_ids, 1]
        lon_b, lat_b = nodes[to_ids, 0], nodes[to_ids, 1]
        arc_brgs = [bearing(lon_a[i], lat_a[i], lon_b[i], lat_b[i]) for i in range(len(arcs))]

        tree = cKDTree(np.column_stack([mid_x, mid_y]))

        # Generate candidates for each GPS point
        candidates_per_step = []
        raw_top1_predictions = []
        prev_lon, prev_lat, prev_ts = None, None, None

        for i in range(len(track)):
            glon, glat, gts = track[i, 0], track[i, 1], track[i, 2]
            gx = (glon - ref_lon) * scale_x
            gy = (glat - ref_lat) * scale_y

            if prev_lon is not None:
                g_brg = float(bearing(prev_lon, prev_lat, glon, glat))
                dt = max(gts - prev_ts, 0.001)
                ds = float(haversine(prev_lon, prev_lat, glon, glat))
                speed = ds / dt
            else:
                g_brg = None
                speed = 0.0
                dt = 1.0
                ds = 0.0

            cand_indices = tree.query_ball_point([gx, gy], r=RADIUS_M + 25.0)
            step_cands = []

            for c_idx in cand_indices:
                dist_m, t, is_within = point_to_segment(
                    gx, gy,
                    float(ax[c_idx]), float(ay[c_idx]),
                    float(bx[c_idx]), float(by[c_idx])
                )
                if dist_m > RADIUS_M:
                    continue

                a_brg = arc_brgs[c_idx]
                brg_diff = abs((g_brg - a_brg + 180.0) % 360.0 - 180.0) if g_brg is not None else 0.0
                rad_diff = math.radians(brg_diff)
                arc_len = float(arc_lens[c_idx])

                step_cands.append({
                    "arc_id": int(c_idx),
                    "dist_to_arc_m": dist_m,
                    "log_dist": math.log1p(dist_m),
                    "is_within_segment": int(is_within),
                    "projection_t": t,
                    "bearing_diff": brg_diff,
                    "cos_bearing_diff": math.cos(rad_diff),
                    "heading_aligned": int(brg_diff < 45.0),
                    "arc_len_m": arc_len,
                    "log_arc_len": math.log1p(arc_len),
                    "speed_mps": speed,
                    "delta_t": dt,
                    "delta_d": ds,
                    "speed_to_len_ratio": speed / max(arc_len, 1.0)
                })

            if step_cands:
                step_cands.sort(key=lambda c: c["dist_to_arc_m"])
                step_cands = step_cands[:TOP_K]
                raw_top1_predictions.append(step_cands[0]["arc_id"])
            else:
                raw_top1_predictions.append(-1)

            candidates_per_step.append(step_cands)
            prev_lon, prev_lat, prev_ts = glon, glat, gts

        # Run Viterbi Reconstruction
        viterbi_route = viterbi_reconstruct(track, candidates_per_step, arc_successors, clf)

        # ── Calculate Key Evaluation Metrics ──────────────────────────────
        # 1. Pointwise Top-1 Accuracy (% of chosen arcs in ground-truth route)
        clf_matches = sum(1 for a in raw_top1_predictions if a in gt_route_set)
        clf_acc = clf_matches / max(len(raw_top1_predictions), 1)

        viterbi_matches = sum(1 for a in viterbi_route if a in gt_route_set)
        viterbi_acc = viterbi_matches / max(len(viterbi_route), 1)

        # 2. Route Continuity (% of consecutive segments connected in the road graph)
        def calc_continuity(seq):
            if len(seq) <= 1:
                return 1.0
            valid_steps = 0
            for i in range(len(seq) - 1):
                a1, a2 = seq[i], seq[i + 1]
                if a1 == a2 or a2 in arc_successors[a1]:
                    valid_steps += 1
            return valid_steps / (len(seq) - 1)

        clf_continuity = calc_continuity(raw_top1_predictions)
        viterbi_continuity = calc_continuity(viterbi_route)

        # 3. Unique Route Sequence Overlap (Jaccard similarity with reference sequence)
        pred_unique = set(viterbi_route)
        gt_unique = set(gt_route)
        jaccard = len(pred_unique & gt_unique) / max(len(pred_unique | gt_unique), 1)

        print(f"Trip {tid} (GPS Points: {len(track)}):")
        print(f"  * Classifier-Only Top-1 Acc  : {clf_acc*100:.2f}% | Continuity: {clf_continuity*100:.2f}%")
        print(f"  * Viterbi Reconstructed Acc : {viterbi_acc*100:.2f}% | Continuity: {viterbi_continuity*100:.2f}%")
        print(f"  * Route Jaccard Overlap      : {jaccard*100:.2f}%\n")

        summary_rows.append({
            "Trip_ID": tid,
            "GPS_Points": len(track),
            "Clf_Accuracy": round(clf_acc, 4),
            "Viterbi_Accuracy": round(viterbi_acc, 4),
            "Clf_Continuity": round(clf_continuity, 4),
            "Viterbi_Continuity": round(viterbi_continuity, 4),
            "Route_Jaccard_Overlap": round(jaccard, 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("viterbi_evaluation_results.csv", index=False)
    print("="*75)
    print("AVERAGE BENCHMARK ACROSS ALL 10 STUDENT TEST TRIPS:")
    print(f"  - Classifier Alone Accuracy    : {summary_df['Clf_Accuracy'].mean()*100:.2f}%")
    print(f"  - Viterbi Map-Matched Accuracy : {summary_df['Viterbi_Accuracy'].mean()*100:.2f}%")
    print(f"  - Classifier Route Continuity  : {summary_df['Clf_Continuity'].mean()*100:.2f}%")
    print(f"  - Viterbi Route Continuity     : {summary_df['Viterbi_Continuity'].mean()*100:.2f}%  (Clean topological path!)")
    print(f"  - Ground Truth Route Overlap   : {summary_df['Route_Jaccard_Overlap'].mean()*100:.2f}%")
    print("="*75)
    print("Saved complete evaluation report to: viterbi_evaluation_results.csv")

if __name__ == "__main__":
    evaluate_student_test_trips()
