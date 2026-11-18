#!/usr/bin/env python3
"""Bayesian optimization framework; not run by the default Phase 1 command."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from pathsnap.features import FEATURE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    files = sorted((Path(args.processed_root) / "features" / "train").glob("*.parquet"))
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    if len(frame) > 150000:
        positive = frame[frame.target == 1]
        negative = frame[frame.target == 0].sample(150000 - len(positive), random_state=args.seed)
        frame = pd.concat([positive, negative]).sample(frac=1, random_state=args.seed)
    x, y, groups = frame[FEATURE_COLUMNS], frame.target.to_numpy(), frame.trip_id.to_numpy()
    splitter = GroupKFold(n_splits=3)

    def objective(trial: optuna.Trial) -> float:
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 80),
                l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                max_iter=trial.suggest_int("max_iter", 80, 250), random_state=args.seed,
            )),
        ])
        scores = []
        for fit, valid in splitter.split(x, y, groups):
            model.fit(x.iloc[fit], y[fit]); prediction = model.predict(x.iloc[valid])
            scores.append(f1_score(y[valid], prediction, average="macro", zero_division=0))
        return float(sum(scores) / len(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv("outputs/logs/optuna_hist_gradient_boosting.csv", index=False)
    print(study.best_value, study.best_params)


if __name__ == "__main__":
    main()
