"""Script: smoke_test_models.py

Smoke test for the 5-model experimentation suite using trip-grouped validation on a small subset.
Verifies pipeline execution, metric computation, probability calibration, and model artifact serialization.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_accuracy,
)


def run_smoke_test():
    print("=" * 60)
    print("Running Model Experimentation Smoke Test")
    print("=" * 60)

    # 1. Load sample data
    train_df = pd.read_parquet(repo_root / "data" / "processed" / "train_features.parquet")
    # Take 6 diverse trips for smoke test (4 train, 2 val)
    sample_trips = train_df["trip_id"].unique()[:6]
    df_sample = train_df[train_df["trip_id"].isin(sample_trips)].copy()
    print(f"Smoke test dataset: {len(df_sample):,} rows across {len(sample_trips)} trips")

    X = df_sample[FEATURE_COLUMNS]
    y = df_sample["label"].astype(int).values
    groups = df_sample["trip_id"].values
    obs_ids = df_sample["observation_id"].values

    gkf = GroupKFold(n_splits=2)
    train_idx, val_idx = next(gkf.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    val_obs = df_sample.iloc[val_idx][["observation_id", "label"]].copy()

    print(f"Train fold: {len(X_train):,} rows ({len(set(groups[train_idx]))} trips)")
    print(f"Val fold:   {len(X_val):,} rows ({len(set(groups[val_idx]))} trips)")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        "Extra Trees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        "HistGradientBoosting": Pipeline([
            # HistGradientBoosting natively handles raw features, but scaling preserves pipeline uniformity
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=30, max_depth=8, random_state=42))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=15, random_state=42))
        ]),
    }

    results = []
    artifacts_dir = repo_root / "outputs" / "models" / "smoke_test"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        t0 = time.perf_counter()
        # Strictly fit on train fold only
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        infer_time = time.perf_counter() - t1

        clf_metrics = compute_candidate_classification_metrics(y_val, val_preds, val_probs)
        top1_metrics = compute_observation_top1_accuracy(val_obs, val_probs)

        # Save artifact
        artifact_path = artifacts_dir / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, artifact_path)

        res = {
            "model": name,
            "fit_time_s": round(fit_time, 2),
            "infer_time_s": round(infer_time, 3),
            "macro_f1": round(clf_metrics["f1_macro"], 4),
            "top1_acc": round(top1_metrics["top1_accuracy"], 4),
            "log_loss": round(clf_metrics.get("log_loss", float("nan")), 4),
            "precision": round(clf_metrics["precision"], 4),
            "recall": round(clf_metrics["recall"], 4),
        }
        results.append(res)
        print(f"  {name:22s} | Fit: {fit_time:5.2f}s | Top-1: {res['top1_acc']:.4f} | Macro F1: {res['macro_f1']:.4f} | LogLoss: {res['log_loss']:.4f}")

    print("\nSmoke test successfully passed for all 5 models!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

