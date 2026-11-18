"""
Pipeline runner script to execute candidate generation and feature extraction across
the train and student_test partitions, generating clean Parquet and CSV files in data/processed/.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path("pathsnap").resolve()))
from src.data_io import PathSnapDataLoader
from src.preprocess import preprocess_trip
from src.labels import assign_reference_arcs_monotonic
from src.candidates import generate_candidates_for_trip
from src.features import extract_features_for_trip

RADIUS = 35.0  # Selected based on 95th-98th percentile of point-to-route distance on TRAIN data

def run_pipeline():
    data_root = Path("Student folder")
    loader = PathSnapDataLoader(data_root)
    
    processed_dir = Path("pathsnap/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("pathsnap/outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chosen radius configuration as required by rule 3 (fitted parameters saved to disk)
    config = {
        "candidate_radius_metres": RADIUS,
        "large_gap_threshold_seconds": 5.0,
        "search_window": 12,
        "ambiguity_dist_gap_metres": 1.0,
        "crs_strategy": "per_trip_local_utm"
    }
    with open(processed_dir / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved pipeline_config.json")
    
    train_trips = loader.list_trips("train")
    student_trips = loader.list_trips("student_test")
    
    print(f"Processing Train Partition ({len(train_trips)} trips)...")
    train_dfs = []
    train_candidate_metrics = []
    
    for idx, trip_id in enumerate(train_trips):
        trip = loader.load_trip("train", trip_id)
        track, sindex, arc_lengths = preprocess_trip(trip)
        labels = assign_reference_arcs_monotonic(track, sindex, trip.route["arc_id"].values)
        cands, metrics = generate_candidates_for_trip(
            track, sindex, radius=RADIUS, reference_arcs=labels.reference_arcs
        )
        metrics["trip_id"] = trip_id
        train_candidate_metrics.append(metrics)
        
        feats = extract_features_for_trip(
            trip_id=trip_id,
            track_df=track,
            sindex=sindex,
            arc_lengths=arc_lengths,
            trip_arcs=trip.arcs,
            candidates_list=cands,
            reference_arcs=labels.reference_arcs
        )
        train_dfs.append(feats)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(train_trips):
            print(f"  Processed {idx + 1}/{len(train_trips)} train trips...")
            
    df_train_all = pd.concat(train_dfs, ignore_index=True)
    df_train_all.to_parquet(processed_dir / "train_features.parquet", index=False)
    # Also save small sample CSV for quick inspection
    df_train_all.head(2000).to_csv(processed_dir / "train_features_sample.csv", index=False)
    print(f"Train features shape: {df_train_all.shape}")
    
    print(f"\nProcessing Student Test Partition ({len(student_trips)} trips)...")
    test_dfs = []
    test_candidate_metrics = []
    
    for idx, trip_id in enumerate(student_trips):
        trip = loader.load_trip("student_test", trip_id)
        track, sindex, arc_lengths = preprocess_trip(trip)
        # For student_test, we generate candidates; route is available for offline evaluation
        labels = assign_reference_arcs_monotonic(track, sindex, trip.route["arc_id"].values)
        cands, metrics = generate_candidates_for_trip(
            track, sindex, radius=RADIUS, reference_arcs=labels.reference_arcs
        )
        metrics["trip_id"] = trip_id
        test_candidate_metrics.append(metrics)
        
        feats = extract_features_for_trip(
            trip_id=trip_id,
            track_df=track,
            sindex=sindex,
            arc_lengths=arc_lengths,
            trip_arcs=trip.arcs,
            candidates_list=cands,
            reference_arcs=labels.reference_arcs
        )
        test_dfs.append(feats)
        
    df_test_all = pd.concat(test_dfs, ignore_index=True)
    df_test_all.to_parquet(processed_dir / "student_test_features.parquet", index=False)
    df_test_all.head(2000).to_csv(processed_dir / "student_test_features_sample.csv", index=False)
    print(f"Student test features shape: {df_test_all.shape}")
    
    # Save candidate evaluation report
    df_train_cand_metrics = pd.DataFrame(train_candidate_metrics)
    df_train_cand_metrics.to_csv(plots_dir / "train_candidate_metrics.csv", index=False)
    
    total_train_pts = df_train_cand_metrics["n_points"].sum()
    total_train_misses = df_train_cand_metrics["candidate_miss_count"].sum()
    overall_miss_rate = total_train_misses / total_train_pts * 100
    
    print("\n=== OVERALL CANDIDATE GENERATION PERFORMANCE (TRAIN) ===")
    print(f"Radius: {RADIUS} m")
    print(f"Total points: {total_train_pts}")
    print(f"Total misses (reference arc not in candidates): {total_train_misses} ({overall_miss_rate:.2f}%)")
    print(f"Hit rate: {100.0 - overall_miss_rate:.2f}%")
    print(f"Zero candidates rate: {df_train_cand_metrics['zero_candidate_pct'].mean():.2f}%")
    print(f"Mean candidate count per point: {df_train_cand_metrics['candidate_count_mean'].mean():.2f}")
    print(f"Class imbalance (label=1 ratio in train): {df_train_all['label'].mean():.4f} (1:{(1-df_train_all['label'].mean())/df_train_all['label'].mean():.1f})")
    
    # Generate distribution plots
    plt.figure(figsize=(10, 5))
    sns.histplot(df_train_all["perp_dist"], bins=50, kde=True, color="royalblue")
    plt.title(f"Perpendicular Distance to Candidates (Radius={RADIUS}m)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(plots_dir / "candidate_distance_distribution.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    feature_cols = [
        "perp_dist", "dist_start_node", "dist_end_node", "proj_frac",
        "bearing_diff", "is_reverse_direction", "speed", "dt", "arc_length",
        "candidate_rank", "num_candidates", "dist_gap_to_next", "label"
    ]
    corr = df_train_all[feature_cols].sample(n=min(50000, len(df_train_all)), random_state=42).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 7})
    plt.title("Feature Correlation Heatmap (Train sample)")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_correlations.png", dpi=300)
    plt.close()
    print("Saved feature plots in outputs/plots/")

if __name__ == "__main__":
    run_pipeline()
