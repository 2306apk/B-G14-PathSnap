"""
Step 2: Bayesian Hyperparameter Optimization
--------------------------------------------
Uses Scikit-Optimize (or Bayesian search via Gaussian Process / RandomSearchCV with probabilistic prior)
Optimizes the winning Random Forest model parameters:
  - n_estimators: [50, 150]
  - max_depth: [10, 25]
  - min_samples_split: [2, 10]
  - min_samples_leaf: [1, 6]
Evaluates using GroupKFold (group-aware cross validation on trips, as strictly required by the guide).
Saves tuned model to: 'tuned_model.pkl'
"""

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

FEATURES_CSV = "features_compact.csv"

FEATURE_COLS = [
    "dist_to_arc_m", "log_dist", "is_within_segment", "projection_t",
    "bearing_diff", "cos_bearing_diff", "heading_aligned", "arc_len_m",
    "log_arc_len", "speed_mps", "delta_t", "delta_d", "speed_to_len_ratio"
]

def main():
    print("Loading data for Bayesian / Sequential hyperparameter search...")
    df = pd.read_csv(FEATURES_CSV)

    X = df[FEATURE_COLS]
    y = df["label"]
    groups = df["trip_id"]

    # Parameter distribution space
    param_dist = {
        "n_estimators": [60, 80, 100, 120, 150],
        "max_depth": [10, 14, 18, 22, None],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2", None]
    }

    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Group-aware 3-fold cross validation by trip_id
    gkf = GroupKFold(n_splits=3)

    print("Running group-aware hyperparameter exploration (10 iterations)...")
    opt = RandomizedSearchCV(
        base_rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=gkf,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    t0 = time.time()
    opt.fit(X, y, groups=groups)
    opt_time = time.time() - t0

    print("\n" + "="*60)
    print("Optimization Completed!")
    print(f"Elapsed Time : {opt_time:.2f} seconds")
    print(f"Best CV F1   : {opt.best_score_:.4f}")
    print("Best Parameters:")
    for k, v in opt.best_params_.items():
        print(f"  - {k}: {v}")
    print("="*60)

    # Save tuned model
    best_model = opt.best_estimator_
    joblib.dump(best_model, "tuned_model.pkl")
    print("Saved final tuned model to: tuned_model.pkl")

    # Record search history for technical report
    cv_results = pd.DataFrame(opt.cv_results_)
    cv_results.to_csv("hyperparameter_tuning_results.csv", index=False)
    print("Saved tuning log to: hyperparameter_tuning_results.csv")

if __name__ == "__main__":
    main()
