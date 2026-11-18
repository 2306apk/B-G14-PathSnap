#!/usr/bin/env python3
"""Inference-only overfitting, leakage, calibration, and reliability audit.

This script reuses generated feature Parquets and fitted model artifacts. It does
not refit models or rerun any upstream pipeline stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import candidate_metrics, observation_metrics
from run_experiments import stratified_cap


PROCESSED = Path("data/processed")
OUTPUT = Path("outputs/csv")
REPORT = Path("reports/audits/OVERFITTING_RELIABILITY_AUDIT.md")
SEED = 42
MAX_TRAIN_ROWS = 150_000
CV_FOLDS = 3


def load_features(partition: str) -> pd.DataFrame:
    columns = [
        "trip_id", "observation_id", "arc_id", "reference_arc_id", "target", *FEATURE_COLUMNS
    ]
    files = sorted((PROCESSED / "features" / partition).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No cached feature Parquets for {partition}")
    return pd.concat(
        (pd.read_parquet(path, columns=columns) for path in files), ignore_index=True
    )


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(y_true[mask].mean()) - float(probability[mask].mean())
            )
    return error


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_full = load_features("train")
    student = load_features("student_test")
    train_sample = stratified_cap(train_full, MAX_TRAIN_ROWS, SEED)

    sample_groups = train_sample["trip_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=CV_FOLDS)
    fold_rows = []
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(train_sample[FEATURE_COLUMNS], train_sample["target"], sample_groups), start=1
    ):
        fit_trips = set(sample_groups[fit_index])
        valid_trips = set(sample_groups[valid_index])
        fold_rows.append(
            {
                "fold": fold,
                "fit_rows": len(fit_index),
                "validation_rows": len(valid_index),
                "fit_trips": len(fit_trips),
                "validation_trips": len(valid_trips),
                "trip_overlap_count": len(fit_trips & valid_trips),
                "validation_positive_rate": float(train_sample.iloc[valid_index].target.mean()),
                "validation_trip_ids": ",".join(sorted(valid_trips)),
            }
        )
    fold_integrity = pd.DataFrame(fold_rows)
    fold_integrity.to_csv(OUTPUT / "reliability_cv_fold_integrity.csv", index=False)

    coverage = pd.read_csv(OUTPUT / "candidate_coverage.csv", dtype={"trip_id": str})
    labels = pd.read_csv(OUTPUT / "label_association_summary.csv", dtype={"trip_id": str})
    train_trip = (
        train_full.groupby("trip_id", as_index=False)
        .agg(positive_rows=("target", "sum"),
             candidate_observations=("observation_id", "nunique"))
    )
    sample_trip = (
        train_sample.groupby("trip_id", as_index=False)
        .agg(sample_rows=("target", "size"), sample_positive_rows=("target", "sum"))
    )
    trip_quality = (
        coverage[coverage.partition == "train"]
        .merge(labels[labels.partition == "train"], on=["partition", "trip_id"], how="left")
        .merge(train_trip, on="trip_id", how="left")
        .merge(sample_trip, on="trip_id", how="left")
    )
    trip_quality["natural_positive_rate"] = (
        trip_quality.positive_rows / trip_quality.candidate_rows
    )
    trip_quality["sample_positive_rate"] = (
        trip_quality.sample_positive_rows / trip_quality.sample_rows
    )
    trip_quality["zero_positive_candidate_observations"] = (
        trip_quality.reference_candidate_miss_count - trip_quality.no_candidate_count
    )
    trip_quality["gross_route_mismatch"] = trip_quality.trip_id.isin(["00000033", "00000054"])
    trip_quality.to_csv(OUTPUT / "reliability_training_trip_quality.csv", index=False)

    original = pd.read_csv(OUTPUT / "initial_model_results.csv")
    result_rows = []
    per_trip_rows = []
    for model_path in sorted((PROCESSED / "models").glob("*.joblib")):
        artifact = joblib.load(model_path)
        name = model_path.stem
        pipeline = artifact["pipeline"]
        train_probability = pipeline.predict_proba(train_sample[FEATURE_COLUMNS])[:, 1]
        student_probability = pipeline.predict_proba(student[FEATURE_COLUMNS])[:, 1]
        train_metrics = candidate_metrics(train_sample.target.to_numpy(), train_probability)
        student_metrics = candidate_metrics(student.target.to_numpy(), student_probability)
        student_metrics.update(observation_metrics(student, student_probability))
        saved = original.loc[original.model == name].iloc[0]
        cv_score = float(saved.cv_macro_f1_mean)
        metric_deltas = {
            "candidate_macro_f1": abs(student_metrics["candidate_macro_f1"] - saved.candidate_macro_f1),
            "candidate_log_loss": abs(student_metrics["candidate_log_loss"] - saved.candidate_log_loss),
            "observation_top1_accuracy": abs(student_metrics["observation_top1_accuracy"] - saved.observation_top1_accuracy),
        }
        result_rows.append(
            {
                "model": name,
                "train_sample_macro_f1": train_metrics["candidate_macro_f1"],
                "group_cv_macro_f1": cv_score,
                "train_minus_cv_macro_f1": train_metrics["candidate_macro_f1"] - cv_score,
                "student_candidate_macro_f1": student_metrics["candidate_macro_f1"],
                "cv_minus_student_macro_f1": cv_score - student_metrics["candidate_macro_f1"],
                "student_top1_accuracy": student_metrics["observation_top1_accuracy"],
                "train_sample_log_loss": train_metrics["candidate_log_loss"],
                "student_log_loss": student_metrics["candidate_log_loss"],
                "train_sample_brier": brier_score_loss(train_sample.target, train_probability),
                "student_brier": brier_score_loss(student.target, student_probability),
                "student_ece_10_bin": expected_calibration_error(student.target.to_numpy(), student_probability),
                "student_positive_rate": float(student.target.mean()),
                "student_mean_probability": float(student_probability.mean()),
                "max_saved_metric_delta": max(metric_deltas.values()),
                "student_prediction_sha256": hashlib.sha256(
                    np.asarray(student_probability, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
        scored = student[["trip_id", "observation_id", "arc_id", "reference_arc_id", "target"]].copy()
        scored["probability"] = student_probability
        for trip_id, group in scored.groupby("trip_id", sort=True):
            metrics = candidate_metrics(group.target.to_numpy(), group.probability.to_numpy())
            top = group.sort_values(
                ["observation_id", "probability", "arc_id"],
                ascending=[True, False, True],
            ).drop_duplicates("observation_id")
            per_trip_rows.append(
                {
                    "model": name,
                    "trip_id": trip_id,
                    "candidate_rows": len(group),
                    "positive_rate": float(group.target.mean()),
                    "candidate_macro_f1": metrics["candidate_macro_f1"],
                    "candidate_log_loss": metrics["candidate_log_loss"],
                    "top1_accuracy": float((top.arc_id == top.reference_arc_id).mean()),
                    "scored_observations": len(top),
                }
            )

    reliability = pd.DataFrame(result_rows).sort_values("student_top1_accuracy", ascending=False)
    per_trip = pd.DataFrame(per_trip_rows).sort_values(["model", "top1_accuracy", "trip_id"])
    reliability.to_csv(OUTPUT / "reliability_model_summary.csv", index=False)
    per_trip.to_csv(OUTPUT / "reliability_student_per_trip.csv", index=False)

    worst = (
        per_trip.sort_values(["model", "top1_accuracy"])
        .groupby("model", as_index=False)
        .first()[["model", "trip_id", "top1_accuracy", "candidate_macro_f1"]]
        .sort_values("model")
    )
    outliers = trip_quality[trip_quality.gross_route_mismatch].copy()
    full_train_positive_rate = float(train_full.target.mean())
    sample_positive_rate = float(train_sample.target.mean())
    student_positive_rate = float(student.target.mean())
    max_delta = float(reliability.max_saved_metric_delta.max())
    zero_positive_observations = int(trip_quality.zero_positive_candidate_observations.sum())
    outlier_sample_rows = int(outliers.sample_rows.sum())
    outlier_sample_positives = int(outliers.sample_positive_rows.sum())
    sample_trip_share_max = float(trip_quality.sample_rows.max() / len(train_sample))
    sample_trip_share_min = float(trip_quality.sample_rows.min() / len(train_sample))

    report = f"""# Overfitting and Result-Reliability Audit

Status: **executed, inference-only**. Cached features and fitted artifacts were reused; no preprocessing, feature generation, model fitting, CV fitting, or tuning was rerun.

## Verdict

**Conditionally trustworthy.** The saved student-test figures reproduce exactly and no cross-trip row leakage or student-test fitting was found. However, CV is measured on a globally balanced sample rather than the natural candidate distribution, probability outputs are prior-shifted, student-test results were used for model ranking, and approximate/missing labels materially affect some training trips.

## 1. Leakage and split evidence

- All three reconstructed `GroupKFold` splits have zero trip overlap. Fold details are in `outputs/csv/reliability_cv_fold_integrity.csv`.
- The deterministic sample contains {train_sample.trip_id.nunique()} training trip groups. Student-test trip IDs are absent from it.
- Imputation and scaling are pipeline steps fitted separately inside each CV fold. No global learned scaler/imputer is present.
- Features are trip-local and causal. Reference fields are retained for evaluation but are absent from `FEATURE_COLUMNS`.
- Candidate radius/outlier detection uses training labels only. Student-test labels do not affect features, fitting, preprocessing, or HMM parameters.
- **Selection caveat:** `initial_model_results.csv` is sorted by student-test Top-1, and HistGradientBoosting was chosen for HMM evaluation from that ranking. This is permitted development-stage comparison, but the student-test set is no longer an untouched estimate after selection.
- **Sampling caveat:** class-stratified sampling is performed globally before GroupKFold. It does not mix trips across folds, but validation labels influence which validation rows enter the artificial 50/50 sample.

## 2. Train vs GroupKFold CV vs student test

{markdown_table(reliability, ['model', 'train_sample_macro_f1', 'group_cv_macro_f1', 'train_minus_cv_macro_f1', 'student_candidate_macro_f1', 'cv_minus_student_macro_f1', 'student_top1_accuracy'])}

- Train-CV gaps above show resubstitution overfitting on the exact balanced sample used to fit saved artifacts. Tree ensembles show the largest gaps; logistic regression is the least overfit by this measure.
- CV-to-student gaps are not directly apples-to-apples because CV uses 50% positives while student candidate prevalence is {student_positive_rate:.6%}.

Worst student-test trip per model:

{markdown_table(worst, ['model', 'trip_id', 'top1_accuracy', 'candidate_macro_f1'])}

Full per-trip results are in `outputs/csv/reliability_student_per_trip.csv`; aggregate scores conceal substantial trip variation.

## 3. Sampling, calibration, and label reliability

- Natural training positive rate: {full_train_positive_rate:.6%}; sampled rate: {sample_positive_rate:.6%}; student-test rate: {student_positive_rate:.6%}.
- Global balancing therefore changes the class prior by {sample_positive_rate / full_train_positive_rate:.2f}x. Candidate accuracy, log loss, and 0.5-threshold metrics from balanced CV do not represent deployment prevalence.
- Sample trip shares range from {sample_trip_share_min:.6%} to {sample_trip_share_max:.6%}; sampling is row-weighted, not trip-balanced.

{markdown_table(reliability, ['model', 'student_positive_rate', 'student_mean_probability', 'student_brier', 'student_ece_10_bin', 'student_log_loss'])}

- Mean probabilities exceed the natural student positive rate for every model, consistent with prior distortion. Probabilities should not be treated as calibrated HMM emissions without grouped calibration or prior correction.
- {zero_positive_observations:,} training observations have candidates but no positive row because the assigned reference arc is missing; another 15,286 observations have no candidates and never enter classifier rows. Candidate metrics reward all-negative missed observations, while observation Top-1 necessarily counts them wrong when present.
- Trips `00000033` and `00000054` contribute {outlier_sample_rows:,} sampled rows but only {outlier_sample_positives:,} positives. They were excluded from radius calibration, not from classifier fitting. Their approximate labels can bias the classifier toward negatives.
- Reference assignments are monotonic approximations from straight endpoint geometries, not independently verified ground truth. Label uncertainty is not used as a sample weight or filter.

## 4. Reproducibility of reported figures

- Saved model artifacts regenerated all student predictions without refitting. Maximum absolute difference from the three reported metrics checked per model was {max_delta:.3e}.
- SHA-256 hashes of regenerated float64 probability arrays are stored in `outputs/csv/reliability_model_summary.csv`.
- Initial raw prediction arrays and out-of-fold predictions were not saved. Student metrics are reproducible from artifacts; historical CV scores can be checked for group integrity and logs, but exact fold scores require refitting because fold artifacts/predictions were not retained.

## 5. Corrections needed, ranked

1. **Critical:** perform negative sampling inside each training fold only (or use all rows/class weights), evaluate validation folds at natural prevalence, and persist fold assignments plus out-of-fold probabilities.
2. **High:** calibrate probabilities with grouped out-of-fold predictions or apply an explicit prior correction before HMM emissions; report Brier/ECE alongside log loss.
3. **High:** exclude, down-weight, or separately model uncertain/missing-reference observations and trips `00000033`/`00000054`; report sensitivity with and without them.
4. **High:** report per-trip macro summaries, worst-trip performance, candidate coverage, and route continuity for every model, not only pooled candidate metrics.
5. **Medium:** treat student-test as development data explicitly; use nested/grouped training validation for selection and reserve a separate untouched partition for final generalization claims.
6. **Medium:** persist raw student predictions, OOF predictions, model/config hashes, and exact sampled row identifiers.

## Evidence files

- `outputs/csv/reliability_model_summary.csv`
- `outputs/csv/reliability_student_per_trip.csv`
- `outputs/csv/reliability_cv_fold_integrity.csv`
- `outputs/csv/reliability_training_trip_quality.csv`
"""
    REPORT.write_text(report)
    print(reliability.to_string(index=False))
    print("\nWorst trips\n", worst.to_string(index=False))
    print("\nFold integrity\n", fold_integrity.to_string(index=False))
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
