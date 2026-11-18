"""
HMM & Viterbi Route Reconstruction Evaluation.
Compares:
1. Classifier-only Top-1 Predictions vs.
2. Classifier + HMM Viterbi Decoded Route
Metrics: Top-1 Accuracy, Route Continuity (percentage of connected consecutive steps), and Trip Completion.
"""

import os
import sys
from pathlib import Path

# Add pathsnap to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.data_io import PathSnapDataLoader
from src.hmm import MapMatchingHMM

FEATURE_COLS = [
    'perp_dist', 'dist_start_node', 'dist_end_node', 'proj_frac',
    'bearing_diff', 'is_reverse_direction', 'speed', 'dt',
    'arc_length', 'candidate_rank', 'num_candidates', 'dist_gap_to_next'
]

def evaluate_route_continuity(chosen_arcs: list, arcs_df: pd.DataFrame) -> float:
    """Computes % of consecutive steps where to_node(arc_{t}) == from_node(arc_{t+1}) or same arc."""
    if len(chosen_arcs) <= 1:
        return 100.0
    
    arc_to_nodes = arcs_df.set_index('arc_id')[['from_node', 'to_node']].to_dict(orient='index')
    
    connected_steps = 0
    total_steps = len(chosen_arcs) - 1
    
    for t in range(total_steps):
        a_prev = chosen_arcs[t]
        a_curr = chosen_arcs[t+1]
        
        if a_prev == a_curr:
            connected_steps += 1
            continue
            
        nodes_prev = arc_to_nodes.get(a_prev)
        nodes_curr = arc_to_nodes.get(a_curr)
        
        if nodes_prev and nodes_curr:
            if nodes_prev['to_node'] == nodes_curr['from_node']:
                connected_steps += 1
                
    return (connected_steps / total_steps) * 100.0

def main():
    print("Loading datasets and best hyperparameters...")
    train_df = pd.read_parquet('pathsnap/data/processed/train_features.parquet')
    test_df = pd.read_parquet('pathsnap/data/processed/student_test_features.parquet')
    
    with open('pathsnap/outputs/hyperparameter/best_params.json', 'r') as f:
        best_cfg = json.load(f)['best_params']
        
    print(f"Training tuned HistGradientBoostingClassifier with params: {best_cfg}")
    clf = HistGradientBoostingClassifier(**best_cfg, random_state=42)
    clf.fit(train_df[FEATURE_COLS].to_numpy(), train_df['label'].to_numpy())
    
    # Predict probabilities on test
    test_probs = clf.predict_proba(test_df[FEATURE_COLS].to_numpy())[:, 1]
    test_df['pred_prob'] = test_probs
    
    hmm = MapMatchingHMM(beta=12.0, disconnected_penalty=-35.0, same_arc_bonus=2.5)
    
    loader = PathSnapDataLoader('Student folder')
    trip_ids = sorted(test_df['trip_id'].unique())
    print(f"\nDecoding {len(trip_ids)} student_test trips with HMM/Viterbi...")
    
    records = []
    
    for tid in trip_ids:
        t0 = time.time()
        sub_df = test_df[test_df['trip_id'] == tid].copy()
        
        # Load trip network arcs
        trip_data = loader.load_trip('student_test', tid)
        arcs_df = trip_data.arcs
        
        # 1. Baseline Classifier-Only (Greedy Top-1)
        top1_idx = sub_df.groupby('point_idx')['pred_prob'].idxmax()
        top1_df = sub_df.loc[top1_idx].sort_values('point_idx')
        clf_chosen_arcs = top1_df['arc_id'].tolist()
        clf_acc = float((top1_df['label'] == 1).mean() * 100.0)
        clf_continuity = evaluate_route_continuity(clf_chosen_arcs, arcs_df)
        
        # 2. Classifier + HMM Viterbi Decoding
        viterbi_chosen_arcs = hmm.decode_trip(sub_df, arcs_df, prob_col='pred_prob')
        
        # Map viterbi choices back to evaluate accuracy
        # create mapping from point_idx to chosen arc
        pts = sorted(sub_df['point_idx'].unique())
        pt_to_arc = dict(zip(pts, viterbi_chosen_arcs))
        
        # Ground truth mapping
        gt_mapping = sub_df[sub_df['label'] == 1].set_index('point_idx')['arc_id'].to_dict()
        correct_viterbi = sum(1 for p in pts if p in gt_mapping and pt_to_arc[p] == gt_mapping[p])
        viterbi_acc = (correct_viterbi / len(pts)) * 100.0
        viterbi_continuity = evaluate_route_continuity(viterbi_chosen_arcs, arcs_df)
        
        decode_time = time.time() - t0
        print(f"Trip {tid}: Top-1 Acc: {clf_acc:.2f}% -> HMM Acc: {viterbi_acc:.2f}% | Continuity: {clf_continuity:.1f}% -> {viterbi_continuity:.1f}% ({decode_time:.2f}s)")
        
        records.append({
            'trip_id': tid,
            'points': len(pts),
            'clf_top1_acc': clf_acc,
            'hmm_viterbi_acc': viterbi_acc,
            'clf_continuity': clf_continuity,
            'hmm_continuity': viterbi_continuity,
            'acc_gain': viterbi_acc - clf_acc,
            'continuity_gain': viterbi_continuity - clf_continuity,
            'decode_seconds': round(decode_time, 2)
        })
        
    res_df = pd.DataFrame(records)
    os.makedirs('pathsnap/outputs/hmm', exist_ok=True)
    res_df.to_csv('pathsnap/outputs/hmm/viterbi_comparison.csv', index=False)
    
    mean_clf_acc = res_df['clf_top1_acc'].mean()
    mean_hmm_acc = res_df['hmm_viterbi_acc'].mean()
    mean_clf_cont = res_df['clf_continuity'].mean()
    mean_hmm_cont = res_df['hmm_continuity'].mean()
    
    summary = {
        'mean_clf_top1_acc': round(mean_clf_acc, 2),
        'mean_hmm_viterbi_acc': round(mean_hmm_acc, 2),
        'mean_clf_continuity': round(mean_clf_cont, 2),
        'mean_hmm_continuity': round(mean_hmm_cont, 2),
        'total_student_test_trips': len(trip_ids)
    }
    with open('pathsnap/outputs/hmm/hmm_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*80)
    print("HMM & VITERBI EVALUATION COMPLETE")
    print(f"Observation Top-1 Accuracy: {mean_clf_acc:.2f}% -> {mean_hmm_acc:.2f}% (Gain: {mean_hmm_acc - mean_clf_acc:+.2f}%)")
    print(f"Route Continuity         : {mean_clf_cont:.2f}% -> {mean_hmm_cont:.2f}% (Gain: {mean_hmm_cont - mean_clf_cont:+.2f}%)")
    print("="*80)

if __name__ == '__main__':
    main()
