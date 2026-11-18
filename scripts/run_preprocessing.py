#!/usr/bin/env python3
"""Project trips, generate monotonic labels, and derive the candidate radius."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.candidates import radius_recall_table, select_candidate_radius
from pathsnap.geo import project_trip
from pathsnap.io import discover_trips, load_trip
from pathsnap.labels import associate_reference_route
from pathsnap.visualize import save_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    processed = Path(args.processed_root); output = Path(args.output_root)
    all_summary = []
    train_distances: list[tuple[str, np.ndarray]] = []
    for partition in ("train", "student_test"):
        label_dir = processed / "labels" / partition; label_dir.mkdir(parents=True, exist_ok=True)
        for trip_dir in discover_trips(args.data_root, partition):
            trip = load_trip(trip_dir, partition); projected = project_trip(trip)
            labels = associate_reference_route(projected)
            labels.insert(0, "trip_id", trip.trip_id); labels.insert(1, "partition", partition)
            labels.to_parquet(label_dir / f"{trip.trip_id}.parquet", index=False)
            if partition == "train":
                train_distances.append((trip.trip_id, labels.reference_distance_m.to_numpy()))
            all_summary.append({
                "partition": partition, "trip_id": trip.trip_id, "epsg": projected.metadata.epsg,
                "observations": len(labels), "route_positions": len(trip.route),
                "reference_distance_mean_m": labels.reference_distance_m.mean(),
                "reference_distance_p95_m": labels.reference_distance_m.quantile(.95),
                "reference_distance_p99_m": labels.reference_distance_m.quantile(.99),
                "uncertain_labels": int(labels.label_uncertain.sum()),
            })
    summary = pd.DataFrame(all_summary)
    (output / "csv").mkdir(parents=True, exist_ok=True); summary.to_csv(output / "csv" / "label_association_summary.csv", index=False)
    trip_medians = np.array([np.median(values) for _, values in train_distances])
    q1, q3 = np.quantile(trip_medians, [0.25, 0.75])
    extreme_fence = float(q3 + 5.0 * (q3 - q1))
    excluded_trips = [trip_id for (trip_id, values) in train_distances if np.median(values) > extreme_fence]
    calibration_distances = np.concatenate(
        [values for trip_id, values in train_distances if trip_id not in excluded_trips]
    )
    all_train_distances = np.concatenate([values for _, values in train_distances])
    radius = select_candidate_radius(calibration_distances)
    selected_radius = float(radius["selected_radius_m"])
    radius.update(
        {
            "calibration_trip_count": len(train_distances) - len(excluded_trips),
            "excluded_extreme_route_mismatch_trips": excluded_trips,
            "trip_median_extreme_fence_m": extreme_fence,
            "all_training_p95_reference_distance_m": float(np.quantile(all_train_distances, 0.95)),
            "all_training_p99_reference_distance_m": float(np.quantile(all_train_distances, 0.99)),
            "all_training_selected_recall": float(np.mean(all_train_distances <= selected_radius)),
            "all_training_selected_miss_rate": float(np.mean(all_train_distances > selected_radius)),
        }
    )
    radius_grid = tuple(
        sorted({10, 20, 30, 40, 50, 75, 100, 150, radius["selected_radius_m"]})
    )
    all_recall = radius_recall_table(all_train_distances, radius_grid)
    all_recall.insert(0, "scope", "all_80_training_trips")
    calibration_recall = radius_recall_table(calibration_distances, radius_grid)
    calibration_recall.insert(0, "scope", "robust_78_trip_calibration")
    recall = pd.concat([all_recall, calibration_recall], ignore_index=True)
    recall.to_csv(output / "csv" / "candidate_radius_recall.csv", index=False)
    (processed / "candidate_radius.json").write_text(json.dumps(radius, indent=2) + "\n")
    plt.figure(figsize=(8, 5)); plt.hist(np.clip(all_train_distances, 0, np.quantile(all_train_distances, .999)), bins=80, color="#2e7d32")
    plt.axvline(radius["p95_reference_distance_m"], color="orange", label="95th percentile")
    plt.axvline(radius["p99_reference_distance_m"], color="red", label="99th percentile")
    plt.xlabel("Assigned point-to-reference-arc distance (m)"); plt.ylabel("Training observations"); plt.legend()
    plt.title("Training reference distance distribution"); save_figure(output / "plots" / "candidate_distance_distribution.png")
    report = f"""# Preprocessing Audit

Status: **implemented and executed** for 80 training and 10 student-test trips.

## Projection and geometry

- WGS84 longitude/latitude is validated before projection.
- Each trip uses the UTM zone selected from its median combined GPS/network coordinate; exact EPSG metadata is in `outputs/csv/trip_crs.csv`.
- `pyproj.Transformer(..., always_xy=True)` preserves longitude/latitude order.
- Every arc is represented by a straight metric line between its from/to nodes because intermediate geometry is unavailable.
- Each trip has its own arc array and STRtree; arc IDs never cross trip boundaries.

## Reference-route labelling

- A monotonic dynamic program minimizes projected point-to-route-arc distance.
- Staying on an arc is free; skipping a route position costs 5 m; soft start/end omission costs 2 m per position.
- Outputs retain route position, assigned distance, distance margin, confidence, and an uncertainty flag.
- These labels are reference-derived and are not claimed as independently verified physical ground truth.

## Candidate radius

- All-training assigned distances: p95={radius['all_training_p95_reference_distance_m']:.6f} m; p99={radius['all_training_p99_reference_distance_m']:.6f} m.
- Trips with extreme reference mismatch are detected from training only when their median distance exceeds Q3 + 5*IQR ({radius['trip_median_extreme_fence_m']:.6f} m). Excluded from radius calibration only: {radius['excluded_extreme_route_mismatch_trips']}.
- Robust calibration p95={radius['p95_reference_distance_m']:.6f} m; p99={radius['p99_reference_distance_m']:.6f} m across {radius['calibration_trip_count']} trips.
- Selected radius={radius['selected_radius_m']:.1f} m (robust training p99 rounded upward to 5 m).
- Calibration recall={radius['selected_recall']:.6%}; miss rate={radius['selected_miss_rate']:.6%}. Across all 80 trips, recall={radius['all_training_selected_recall']:.6%}; miss rate={radius['all_training_selected_miss_rate']:.6%}.
- Excluded trips are still processed and included in coverage reporting; this rule only prevents anomalous reference geometry from inflating every candidate set.
- Student-test data and labels were not used to choose the radius.

## Leakage safeguards

Train and student-test labels are stored in separate directories. Radius fitting reads training labels only. Prediction functions do not accept reference-route data.

## Executed command

`python scripts/run_preprocessing.py --data-root "../Student folder"`
"""
    Path("reports/audits/PREPROCESSING_AUDIT.md").write_text(report)
    print(json.dumps(radius, indent=2))


if __name__ == "__main__":
    main()
