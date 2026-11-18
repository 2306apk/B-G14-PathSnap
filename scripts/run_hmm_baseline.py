#!/usr/bin/env python3
"""Compare classifier top-1 with Viterbi decoding on student-test trips."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pathsnap.geo import project_trip
from pathsnap.hmm import viterbi_decode
from pathsnap.io import discover_trips, load_trip
from pathsnap.metrics import route_continuity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--model", default=None, help="Joblib artifact from run_experiments.py")
    args = parser.parse_args()
    processed = Path(args.processed_root)
    artifact = joblib.load(args.model) if args.model else None
    rows = []
    for trip_dir in discover_trips(args.data_root, "student_test"):
        trip = load_trip(trip_dir, "student_test"); projected = project_trip(trip)
        features = pd.read_parquet(processed / "features" / "student_test" / f"{trip.trip_id}.parquet")
        if artifact:
            probability = artifact["pipeline"].predict_proba(features[artifact["feature_columns"]])[:, 1]
            emission = np.log(np.clip(probability, 1e-9, 1.0))
            emission_name = Path(args.model).stem
        else:
            probability = np.exp(-0.5 * (features.point_arc_distance_m.to_numpy() / 20.0) ** 2)
            emission = np.log(np.clip(probability, 1e-9, 1.0))
            emission_name = "distance_baseline"
        scored = features[["observation_id", "arc_id", "reference_arc_id", "candidate_rank", "point_arc_distance_m"]].copy()
        scored["probability"] = probability
        classifier = scored.sort_values(["observation_id", "probability", "arc_id"], ascending=[True, False, True]).drop_duplicates("observation_id")
        decoded = viterbi_decode(projected, features, emission_log_probability=emission)
        evaluated = decoded.merge(features[["observation_id", "reference_arc_id"]].drop_duplicates(), on="observation_id", how="left")
        rows.append({
            "trip_id": trip.trip_id, "emission_model": emission_name, "scored_observations": len(classifier),
            "classifier_top1_accuracy": float((classifier.arc_id == classifier.reference_arc_id).mean()),
            "viterbi_top1_accuracy": float((evaluated.predicted_arc_id == evaluated.reference_arc_id).mean()),
            "classifier_route_continuity": route_continuity(trip.arcs, classifier.arc_id.to_numpy()),
            "viterbi_route_continuity": route_continuity(trip.arcs, evaluated.predicted_arc_id.to_numpy()),
        })
    result = pd.DataFrame(rows)
    result.to_csv("outputs/csv/hmm_baseline_results.csv", index=False)
    print(result.to_string(index=False))
    print("MEAN")
    print(result.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
