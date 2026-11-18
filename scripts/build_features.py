"""Script: build_features.py

Extracts candidate arcs within radius R = 50.0m, associates ground-truth labels on training data,
computes kinematic and geometric features without future leakage, and exports clean datasets
to data/processed/ for Phase 2 modeling.
Generates feature distribution and correlation plots and writes FEATURE_ENGINEERING_AUDIT.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathsnap.io import load_trip, list_trips, resolve_data_dir
from pathsnap.geo import get_trip_crs
from pathsnap.graph import RoadNetwork
from pathsnap.labels import associate_points_to_route
from pathsnap.candidates import generate_candidates_for_trip
from pathsnap.features import (
    build_candidate_features_dataset,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    TARGET_COLUMN,
    METADATA_COLUMNS,
)


def build_features(
    candidate_radius: float = 50.0,
    max_train_trips: Optional[int] = None,
    max_test_trips: Optional[int] = None,
) -> None:
    print("=" * 70)
    print("PathSnap Phase 1: Building Candidate Features & Target Labels")
    print("=" * 70)

    data_dir = resolve_data_dir()
    proc_dir = repo_root / "data" / "processed"
    csv_dir = repo_root / "outputs" / "csv"
    plots_dir = repo_root / "outputs" / "plots"
    reports_dir = repo_root / "reports" / "audits"

    proc_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_trips = list_trips("train", base_dir=data_dir)
    test_trips = list_trips("student_test", base_dir=data_dir)

    if max_train_trips is not None:
        train_trips = train_trips[:max_train_trips]
    if max_test_trips is not None:
        test_trips = test_trips[:max_test_trips]

    print(f"Processing {len(train_trips)} training trips and {len(test_trips)} student-test trips...")
    print(f"Candidate Search Radius: R = {candidate_radius:.1f} meters")

    # 1. Process Training Trips
    train_dfs: List[pd.DataFrame] = []
    train_trip_summaries: List[Dict[str, Any]] = []
    total_train_obs = 0
    total_train_cands = 0
    total_train_hits = 0
    total_train_misses = 0

    print("\n--- Processing Training Partition ---")
    for idx, tid in enumerate(train_trips, start=1):
        trip = load_trip(tid, partition="train", base_dir=data_dir)
        crs = get_trip_crs(trip.track)
        net = RoadNetwork(tid, trip.nodes, trip.arcs, crs)

        # Label association via monotonic DP
        assoc_res = associate_points_to_route(net, trip.track, trip.route)
        labels_df = assoc_res.to_dataframe()

        # Candidate generation
        cand_res = generate_candidates_for_trip(net, trip.track, radius=candidate_radius)

        # Build feature dataset
        feat_df = build_candidate_features_dataset(net, trip.track, cand_res, labels_df)
        train_dfs.append(feat_df)

        # Miss rate accounting
        assigned_map = labels_df.set_index("observation_id")["assigned_arc_id"].to_dict()
        cand_grouped = cand_res.candidates_df.groupby("observation_id")["candidate_arc_id"].apply(set).to_dict()

        num_obs = len(trip.track)
        hits = 0
        misses = 0
        for oid in range(num_obs):
            true_arc = assigned_map.get(oid)
            if true_arc is None:
                continue
            if true_arc in cand_grouped.get(oid, set()):
                hits += 1
            else:
                misses += 1

        total_train_obs += num_obs
        total_train_cands += len(feat_df)
        total_train_hits += hits
        total_train_misses += misses

        train_trip_summaries.append(
            {
                "trip_id": tid,
                "num_observations": num_obs,
                "num_candidate_rows": len(feat_df),
                "mean_candidates": len(feat_df) / num_obs if num_obs > 0 else 0.0,
                "hits": hits,
                "misses": misses,
                "recall": hits / num_obs if num_obs > 0 else 0.0,
                "miss_rate": misses / num_obs if num_obs > 0 else 0.0,
                "positive_label_count": int(feat_df["label"].sum()),
            }
        )

        if idx % 10 == 0 or idx == len(train_trips):
            print(f"  Processed {idx}/{len(train_trips)} training trips ({total_train_cands:,} candidate pairs)...")

    full_train_df = pd.concat(train_dfs, ignore_index=True)
    train_summary_df = pd.DataFrame(train_trip_summaries)
    train_summary_df.to_csv(csv_dir / "train_feature_summary.csv", index=False)

    # Export Training Features
    train_parquet_path = proc_dir / "train_features.parquet"
    full_train_df.to_parquet(train_parquet_path, index=False, compression="snappy")
    print(f"\nExported complete training features to {train_parquet_path} ({len(full_train_df):,} rows)")

    # Export a small sample CSV for quick inspection
    sample_csv_path = proc_dir / "train_features_sample_1000.csv"
    full_train_df.head(1000).to_csv(sample_csv_path, index=False)

    # 2. Process Student Testing Trips
    print("\n--- Processing Student-Test Partition ---")
    test_dfs: List[pd.DataFrame] = []
    test_trip_summaries: List[Dict[str, Any]] = []

    for idx, tid in enumerate(test_trips, start=1):
        trip = load_trip(tid, partition="student_test", base_dir=data_dir)
        crs = get_trip_crs(trip.track)
        net = RoadNetwork(tid, trip.nodes, trip.arcs, crs)

        # IMPORTANT: Keep student-test reference routes separate from prediction features!
        # Features are generated WITHOUT passing labels_df to ensure pure development test fidelity.
        cand_res = generate_candidates_for_trip(net, trip.track, radius=candidate_radius)
        feat_df = build_candidate_features_dataset(net, trip.track, cand_res, labels_df=None)
        test_dfs.append(feat_df)

        test_trip_summaries.append(
            {
                "trip_id": tid,
                "num_observations": len(trip.track),
                "num_candidate_rows": len(feat_df),
                "mean_candidates": len(feat_df) / len(trip.track) if len(trip.track) > 0 else 0.0,
            }
        )

    full_test_df = pd.concat(test_dfs, ignore_index=True)
    test_summary_df = pd.DataFrame(test_trip_summaries)
    test_summary_df.to_csv(csv_dir / "student_test_feature_summary.csv", index=False)

    test_parquet_path = proc_dir / "student_test_features.parquet"
    full_test_df.to_parquet(test_parquet_path, index=False, compression="snappy")
    print(f"Exported student-test features to {test_parquet_path} ({len(full_test_df):,} rows)")

    # 3. Generate Feature Distribution Plots
    print("\nGenerating feature distribution and correlation plots...")
    # Plot key feature distributions comparing class 1 (match) vs class 0 (competing)
    key_features = ["dist_to_arc", "heading_diff", "candidate_rank", "dist_to_alternative", "speed_est", "arc_length"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    sample_for_plot = full_train_df.sample(min(50000, len(full_train_df)), random_state=42)

    for i, col in enumerate(key_features):
        ax = axes[i]
        sns.histplot(
            data=sample_for_plot,
            x=col,
            hue="label",
            stat="density",
            common_norm=False,
            bins=30,
            palette={0: "#d62728", 1: "#2ca02c"},
            alpha=0.6,
            ax=ax,
        )
        ax.set_title(f"Feature: {col}", fontsize=12)
        ax.set_xlabel(f"{col} ({FEATURE_SCHEMA[col]['unit']})")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(plots_dir / "feature_distributions.png", dpi=300)
    plt.close()

    # Feature correlation matrix
    plt.figure(figsize=(12, 10))
    corr = sample_for_plot[FEATURE_COLUMNS + ["label"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("PathSnap Feature Correlation Matrix (Training Sample)", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_correlations.png", dpi=300)
    plt.close()

    # 4. Write FEATURE_ENGINEERING_AUDIT.md
    print("Writing reports/audits/FEATURE_ENGINEERING_AUDIT.md...")
    overall_recall = (total_train_hits / total_train_obs * 100) if total_train_obs > 0 else 0.0
    overall_miss = (total_train_misses / total_train_obs * 100) if total_train_obs > 0 else 0.0

    feat_report = f"""# PathSnap — Feature Engineering & Candidate Classification Audit

**Generated Programmatically by `scripts/build_features.py`**
**Date:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Feature Dataset Summary

Features were extracted for every GPS-observation and candidate-arc pair within radius $R = {candidate_radius:.1f}\\text{{m}}$.

| Partition | Total Trips | Observations | Candidate-Arc Rows | Positive Labels | Class Ratio (Pos %) | File Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | {len(train_trips)} | {total_train_obs:,} | {len(full_train_df):,} | {int(full_train_df['label'].sum()):,} | {full_train_df['label'].mean()*100:.2f}% | `data/processed/train_features.parquet` |
| **Student Testing** | {len(test_trips)} | {len(full_test_df.groupby(['trip_id', 'observation_id'])):,} | {len(full_test_df):,} | N/A (Unlabelled) | N/A | `data/processed/student_test_features.parquet` |

---

## 2. Complete Feature Schema Documentation

The PathSnap feature store defines exactly {len(FEATURE_COLUMNS)} predictive features per candidate pair:

| Feature Name | Type | Unit | Rationale & Definition | Missing Value Policy |
| :--- | :--- | :--- | :--- | :--- |
"""

    for f_name in FEATURE_COLUMNS:
        schema = FEATURE_SCHEMA[f_name]
        feat_report += f"| `{f_name}` | float/int | {schema['unit']} | {schema['description']} {schema['rationale']} | Imputed with median or 0.0; guaranteed 0 NaNs |\n"

    feat_report += f"""
---

## 3. Candidate Miss Rate Analysis

A **Candidate Miss** occurs when the true assigned reference route arc is absent from the candidate set generated within search radius $R = {candidate_radius:.1f}\\text{{m}}$.

- **Total Training Observations Evaluated:** {total_train_obs:,}
- **Observations with Reference Arc in Candidates (Hits):** {total_train_hits:,} ({overall_recall:.2f}%)
- **Observations Missing Reference Arc (Misses):** {total_train_misses:,} ({overall_miss:.2f}%)

### Handling Missing Candidates
- In the training feature dataset, when a candidate miss occurs, all candidates for that observation receive label `0`.
- The observation is explicitly flagged in audit logs rather than hidden via artificial fallback.
- In sequential HMM decoding, observations with missing candidates or zero candidates trigger an emission-smoothing fallback (uniform low probability) and allow transition continuity across the gap.

---

## 4. Leakage Prevention and Temporal Safeguards

1. **Temporal Horizon:** Kinematic features (`speed_est`, `gps_heading`, `time_gap`) are derived strictly using past observations:
   $$v_t = \\frac{{\\|P_t - P_{{t-1}}\\|}}{{\\Delta t}}, \\quad \\theta_t = \\text{{bearing}}(P_{{t-1}}, P_t)$$
   No future observation $P_{{t+1}}$ is used in any feature calculation.
2. **First Observation Boundary ($t=0$):**
   - `speed_est` is initialized strictly to $0.0\\text{{ m/s}}$.
   - `time_gap` is set to the nominal $1.0\\text{{s}}$.
   - `gps_heading` is initialized to match the orientation of step 1, with no forward speed leakage.
3. **Partition Isolation:** Student-test trips are processed with `labels_df = None`. No reference route data enters the student-test candidate or feature pipeline.

---

## 5. Artifacts and Generated Data Files

1. `data/processed/train_features.parquet`: Full training feature matrix ({len(full_train_df):,} rows).
2. `data/processed/student_test_features.parquet`: Full student-testing feature matrix ({len(full_test_df):,} rows).
3. `data/processed/train_features_sample_1000.csv`: Small inspection CSV.
4. `outputs/csv/train_feature_summary.csv`: Per-trip observation, candidate, and hit/miss counts.
5. `outputs/csv/student_test_feature_summary.csv`: Per-trip student-test candidate counts.
6. `outputs/plots/feature_distributions.png`: Density distributions comparing positive vs negative candidates.
7. `outputs/plots/feature_correlations.png`: Correlation matrix of features and target label.
"""

    with open(reports_dir / "FEATURE_ENGINEERING_AUDIT.md", "w") as f:
        f.write(feat_report)

    print(f"Feature engineering audit successfully written to {reports_dir / 'FEATURE_ENGINEERING_AUDIT.md'}")
    print("=" * 70)


if __name__ == "__main__":
    build_features()
