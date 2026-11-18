#!/usr/bin/env python3
"""Build one row per observation/candidate pair for both supplied partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pathsnap.candidates import candidate_coverage, generate_candidates
from pathsnap.features import FEATURE_COLUMNS, build_feature_rows
from pathsnap.geo import project_trip
from pathsnap.io import discover_trips, load_trip
from pathsnap.visualize import save_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    processed = Path(args.processed_root); output = Path(args.output_root)
    radius_config = json.loads((processed / "candidate_radius.json").read_text())
    radius = float(radius_config["selected_radius_m"])
    summaries = []
    feature_samples = []
    count_samples = []
    for partition in ("train", "student_test"):
        target_dir = processed / "features" / partition; target_dir.mkdir(parents=True, exist_ok=True)
        for trip_dir in discover_trips(args.data_root, partition):
            trip = load_trip(trip_dir, partition); projected = project_trip(trip)
            labels = pd.read_parquet(processed / "labels" / partition / f"{trip.trip_id}.parquet")
            candidates = generate_candidates(projected, radius)
            coverage = candidate_coverage(candidates, labels, len(trip.track))
            features = build_feature_rows(projected, candidates, labels)
            features.to_parquet(target_dir / f"{trip.trip_id}.parquet", index=False)
            summaries.append({"partition": partition, "trip_id": trip.trip_id, "radius_m": radius, "candidate_rows": len(features), **coverage})
            if len(features):
                feature_samples.append(features[FEATURE_COLUMNS].sample(min(2000, len(features)), random_state=int(trip.trip_id)))
                count_samples.append(candidates[["observation_id", "candidate_count"]].drop_duplicates())
    summary = pd.DataFrame(summaries)
    (output / "csv").mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "csv" / "candidate_coverage.csv", index=False)
    sample = pd.concat(feature_samples, ignore_index=True)
    counts = pd.concat(count_samples, ignore_index=True)
    plt.figure(figsize=(8, 5)); plt.hist(counts.candidate_count.clip(upper=counts.candidate_count.quantile(.99)), bins=50, color="#6a1b9a")
    plt.xlabel("Candidates per observation (clipped at p99)"); plt.ylabel("Observations"); plt.title("Candidate-count distribution")
    save_figure(output / "plots" / "candidate_count_distribution.png")
    sample.hist(figsize=(16, 14), bins=40); plt.suptitle("Feature distributions (deterministic per-trip sample)", y=1.01)
    save_figure(output / "plots" / "feature_distributions.png")
    train = summary[summary.partition == "train"]
    test = summary[summary.partition == "student_test"]
    report = f"""# Feature Engineering Audit

Status: **implemented and executed**. Candidate radius: {radius:.1f} m.

## Row and label definition

One row represents one trip-local `(GPS observation, candidate arc)` pair. `target=1` exactly when candidate `arc_id` equals the monotonic reference assignment; all other candidates are 0. Observations whose assigned arc is absent remain explicit misses and create no artificial positive.

Training candidate rows: {int(train.candidate_rows.sum()):,}. Student-test candidate rows: {int(test.candidate_rows.sum()):,}.
Training reference miss rate: {train.reference_candidate_miss_count.sum() / train.observation_count.sum():.6%}. Student-test reference miss rate: {test.reference_candidate_miss_count.sum() / test.observation_count.sum():.6%}.
Training no-candidate rate: {train.no_candidate_count.sum() / train.observation_count.sum():.6%}. Student-test no-candidate rate: {test.no_candidate_count.sum() / test.observation_count.sum():.6%}.

## Features (causal; no future observations)

| Feature | Definition | Unit / missing handling |
|---|---|---|
| point_arc_distance_m | perpendicular Shapely point-to-line distance | m |
| from_node_distance_m | Euclidean metric distance to from-node | m |
| to_node_distance_m | Euclidean metric distance to to-node | m |
| projection_position | line projection distance / arc length, clipped [0,1] | ratio; zero for zero-length arc |
| vehicle_heading_deg | `atan2(delta_easting, delta_northing)` from previous GPS point | degrees; first point NaN |
| arc_bearing_deg | same bearing formula from arc endpoints | degrees |
| heading_difference_deg | `abs((vehicle-arc+180) mod 360 - 180)` | degrees; first point NaN |
| reverse_direction | 1 when heading difference > 90 | binary; first point 0 after comparison |
| vehicle_speed_mps | previous-to-current metric displacement / positive time gap | m/s; first/non-positive gap NaN |
| time_gap_s | current timestamp minus previous timestamp | s; first point NaN |
| arc_length_m | straight endpoint geometry length | m |
| candidate_rank | ascending point-to-arc distance, deterministic arc-ID tie break | ordinal |
| candidate_count | number of candidates for the observation | count |
| nearest_alternative_distance_m | second-nearest distance for rank 1; nearest distance otherwise | m; NaN when only one candidate |

Missing numeric values are retained in Parquet and imputed from training-fold medians inside model pipelines. No scaler or imputer is fit on student-test data.

## Executed command

`python scripts/build_features.py --data-root "../Student folder"`
"""
    Path("reports/audits/FEATURE_ENGINEERING_AUDIT.md").write_text(report)
    print(summary.groupby("partition").sum(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
