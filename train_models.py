"""
Step 1: Train & Benchmark 5 Diverse ML Models for PathSnap
---------------------------------------------------------
Requirements from Guide:
- 5 different algorithm families:
  1. Linear / Probabilistic: Logistic Regression
  2. Tree-based: Decision Tree
  3. Ensemble: Random Forest
  4. Boosting: HistGradientBoosting (Fast GBDT)
  5. Neural Network: Multi-Layer Perceptron (MLP)
- Metrics computed:
  Accuracy, Precision, Recall, F1-Score, Log Loss, Training & Inference time.
- Group-aware train / validation split (splits by trip_id to prevent data leakage).
- Exports comparison table to 'models_benchmark.csv'
- Identifies and saves candidate winner for Bayesian Tuning & HMM.
"""

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

FEATURES_CSV = "features_compact.csv"

FEATURE_COLS = [
    "dist_to_arc_m",
    "log_dist",
    "is_within_segment",
    "projection_t",
    "bearing_diff",
    "cos_bearing_diff",
    "heading_aligned",
    "arc_len_m",
    "log_arc_len",
    "speed_mps",
    "delta_t",
    "delta_d",
    "speed_to_len_ratio"
]

def main():
    print("Loading compact features dataset...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df):,} samples with {len(FEATURE_COLS)} features.")

    # 1. Leakage-free Group Split by trip_id (80% train trips, 20% val trips)
    unique_trips = df["trip_id"].unique()
    np.random.seed(42)
    val_trips = np.random.choice(unique_trips, size=int(0.2 * len(unique_trips)), replace=False)
    val_mask = df["trip_id"].isin(val_trips)

    X_train = df.loc[~val_mask, FEATURE_COLS]
    y_train = df.loc[~val_mask, "label"]
    X_val   = df.loc[val_mask, FEATURE_COLS]
    y_val   = df.loc[val_mask, "label"]

    print(f"Train set: {len(X_train):,} samples ({len(unique_trips)-len(val_trips)} trips)")
    print(f"Val set  : {len(X_val):,} samples ({len(val_trips)} trips)")

    # 2. Define 5 Algorithms across 5 Families
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=14, n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100, max_depth=10, random_state=42
        ),
        "Neural Network (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, random_state=42))
        ])
    }

    results = []
    trained_models = {}

    print("\n" + "="*80)
    print(f"{'Model':<24} | {'Acc':<7} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'LogLoss':<7} | {'Train(s)':<8}")
    print("="*80)

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        infer_time = time.time() - t1

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        loss = log_loss(y_val, y_prob)

        print(f"{name:<24} | {acc:<7.4f} | {prec:<7.4f} | {rec:<7.4f} | {f1:<7.4f} | {loss:<7.4f} | {train_time:<8.2f}")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "Log_Loss": round(loss, 4),
            "Train_Time_s": round(train_time, 2),
            "Infer_Time_s": round(infer_time, 3)
        })
        trained_models[name] = clf

    print("="*80)
    res_df = pd.DataFrame(results)
    res_df.to_csv("models_benchmark.csv", index=False)
    print("Exported comparison results to: models_benchmark.csv")

    # Pick the best by F1 score
    best_row = res_df.sort_values(by="F1_Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    print(f"\nWinning Base Model: {best_name} (F1: {best_row['F1_Score']:.4f})")

    # Save baseline winner
    joblib.dump(trained_models[best_name], "best_base_model.pkl")
    print("Saved baseline model to: best_base_model.pkl")

if __name__ == "__main__":
    main()
