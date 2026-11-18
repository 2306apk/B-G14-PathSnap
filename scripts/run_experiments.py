#!/usr/bin/env python3
"""Run five probability-compatible initial models with group-aware CV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import candidate_metrics, observation_metrics
from pathsnap.models import candidate_models
from pathsnap.visualize import save_figure


def load_partition(root: Path, partition: str) -> pd.DataFrame:
    files = sorted((root / "features" / partition).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature files for {partition}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def stratified_cap(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    positives = frame[frame.target == 1]
    negatives = frame[frame.target == 0]
    positive_count = min(len(positives), max_rows // 2)
    negative_count = min(len(negatives), max_rows - positive_count)
    if positive_count + negative_count < max_rows:
        positive_count = min(len(positives), max_rows - negative_count)
    selected_positive = positives.sample(positive_count, random_state=seed)
    selected_negative = negatives.sample(negative_count, random_state=seed)
    return pd.concat([selected_positive, selected_negative]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--max-train-rows", type=int, default=150000)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.processed_root)
    train_full = load_partition(root, "train")
    student = load_partition(root, "student_test")
    train = stratified_cap(train_full, args.max_train_rows, args.seed)
    x = train[FEATURE_COLUMNS]; y = train.target.to_numpy(); groups = train.trip_id.to_numpy()
    x_student = student[FEATURE_COLUMNS]; y_student = student.target.to_numpy()
    splitter = GroupKFold(n_splits=args.cv_folds)
    results = []
    model_dir = root / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    for name, pipeline in candidate_models(args.seed).items():
        fold_f1 = []
        cv_start = time.perf_counter()
        for fit_index, valid_index in splitter.split(x, y, groups):
            fold_model = clone(pipeline)
            fold_model.fit(x.iloc[fit_index], y[fit_index])
            probability = fold_model.predict_proba(x.iloc[valid_index])[:, 1]
            fold_f1.append(candidate_metrics(y[valid_index], probability)["candidate_macro_f1"])
        cv_seconds = time.perf_counter() - cv_start
        fit_start = time.perf_counter(); pipeline.fit(x, y); fit_seconds = time.perf_counter() - fit_start
        infer_start = time.perf_counter(); student_probability = pipeline.predict_proba(x_student)[:, 1]; infer_seconds = time.perf_counter() - infer_start
        metrics = candidate_metrics(y_student, student_probability)
        metrics.update(observation_metrics(student, student_probability))
        row = {
            "model": name, "seed": args.seed, "train_rows_used": len(train),
            "cv_folds": args.cv_folds, "cv_macro_f1_mean": float(np.mean(fold_f1)),
            "cv_macro_f1_std": float(np.std(fold_f1)), "cv_fold_scores": json.dumps(fold_f1),
            "cv_seconds": cv_seconds, "fit_seconds": fit_seconds, "student_inference_seconds": infer_seconds,
            **metrics,
        }
        results.append(row)
        joblib.dump({"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS, "metadata": row}, model_dir / f"{name}.joblib")
        print(name, json.dumps(row, indent=2))
    result = pd.DataFrame(results).sort_values("observation_top1_accuracy", ascending=False)
    Path("outputs/csv").mkdir(parents=True, exist_ok=True)
    result.to_csv("outputs/csv/initial_model_results.csv", index=False)
    plt.figure(figsize=(9, 5)); plt.bar(result.model, result.cv_macro_f1_mean, yerr=result.cv_macro_f1_std)
    plt.xticks(rotation=25, ha="right"); plt.ylabel("Group-CV candidate Macro F1"); plt.title("Initial five-model comparison")
    save_figure("outputs/plots/model_cv_macro_f1.png")
    plt.figure(figsize=(9, 5)); plt.bar(result.model, result.observation_top1_accuracy)
    plt.xticks(rotation=25, ha="right"); plt.ylabel("Student-test observation Top-1 accuracy"); plt.title("Initial five-model student-test comparison")
    save_figure("outputs/plots/model_student_top1.png")
    (model_dir / "initial_ranking.json").write_text(result.to_json(orient="records", indent=2) + "\n")


if __name__ == "__main__":
    main()
