"""
Model Training and Evaluation Script for PathSnap.
Models:
1. Logistic Regression
2. Gaussian Naive Bayes
3. Decision Tree
4. Random Forest
5. HistGradientBoosting (Gradient Boosting)

Evaluates:
- Binary Candidate-level Accuracy, Macro F1, Log Loss
- Observation-level (GPS point) Top-1 Accuracy on Train & Test
"""

import os
import time
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'perp_dist', 'dist_start_node', 'dist_end_node', 'proj_frac',
    'bearing_diff', 'is_reverse_direction', 'speed', 'dt',
    'arc_length', 'candidate_rank', 'num_candidates', 'dist_gap_to_next'
]

def compute_observation_top1(df_subset, proba_col):
    """
    Computes Observation-level Top-1 Accuracy:
    For each GPS observation (trip_id, point_idx), picks the candidate with the highest predicted probability.
    Checks if that chosen candidate has label == 1.
    """
    idx_max = df_subset.groupby(['trip_id', 'point_idx'])[proba_col].idxmax()
    best_picks = df_subset.loc[idx_max]
    top1_acc = (best_picks['label'] == 1).mean()
    return float(top1_acc)

def main():
    print("Loading preprocessed feature data...")
    train_path = 'pathsnap/data/processed/train_features.parquet'
    test_path = 'pathsnap/data/processed/student_test_features.parquet'
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Train samples: {len(train_df):,}, Test samples: {len(test_df):,}")
    
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df['label'].to_numpy()
    
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df['label'].to_numpy()
    
    # Scale features for models that require scaling (Logistic Regression, GaussianNB)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42),
            True  # use scaled
        ),
        'Gaussian Naive Bayes': (
            GaussianNB(),
            True  # use scaled
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42),
            False
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=15, n_jobs=-1, random_state=42),
            False
        ),
        'HistGradientBoosting': (
            HistGradientBoostingClassifier(max_iter=150, max_depth=8, min_samples_leaf=20, random_state=42),
            False
        )
    }
    
    results = []
    os.makedirs('pathsnap/outputs/models', exist_ok=True)
    
    # For reporting training observation top-1 without extreme memory overhead,
    # evaluate observation top-1 on a representative 100k subset of train points as well as full test
    sample_train_trips = train_df['trip_id'].unique()[:10]
    eval_train_df = train_df[train_df['trip_id'].isin(sample_train_trips)].copy()
    eval_test_df = test_df.copy()
    
    print("\n" + "="*80)
    print("STARTING MODEL TRAINING & BENCHMARKING")
    print("="*80)
    
    for name, (clf, use_scaled) in models.items():
        print(f"\nTraining [{name}]...")
        t0 = time.time()
        
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t0
        print(f"Fit completed in {fit_time:.2f}s")
        
        # Inference & probabilities
        t_infer = time.time()
        test_probs = clf.predict_proba(X_te)[:, 1]
        inference_time = time.time() - t_infer
        
        test_preds = (test_probs >= 0.5).astype(int)
        
        # Test candidate metrics
        test_acc = accuracy_score(y_test, test_preds)
        test_macro_f1 = f1_score(y_test, test_preds, average='macro')
        test_loss = log_loss(y_test, np.clip(test_probs, 1e-15, 1 - 1e-15))
        
        # Observation-level Top-1 on Test
        eval_test_df['prob'] = test_probs
        test_obs_top1 = compute_observation_top1(eval_test_df, 'prob')
        
        # Train metrics on representative sample
        X_eval_tr = scaler.transform(eval_train_df[FEATURE_COLS]) if use_scaled else eval_train_df[FEATURE_COLS]
        train_probs = clf.predict_proba(X_eval_tr)[:, 1]
        train_preds = (train_probs >= 0.5).astype(int)
        y_eval_tr = eval_train_df['label'].to_numpy()
        
        train_acc = accuracy_score(y_eval_tr, train_preds)
        train_macro_f1 = f1_score(y_eval_tr, train_preds, average='macro')
        train_loss = log_loss(y_eval_tr, np.clip(train_probs, 1e-15, 1 - 1e-15))
        
        eval_train_df['prob'] = train_probs
        train_obs_top1 = compute_observation_top1(eval_train_df, 'prob')
        
        # Confusion matrix for test
        cm = confusion_matrix(y_test, test_preds).tolist()
        
        print(f"Results for [{name}]:")
        print(f"  Train: Candidate Acc: {train_acc*100:.2f}% | Top-1 Obs Acc: {train_obs_top1*100:.2f}% | Macro F1: {train_macro_f1:.4f}")
        print(f"  Test : Candidate Acc: {test_acc*100:.2f}% | Top-1 Obs Acc: {test_obs_top1*100:.2f}% | Macro F1: {test_macro_f1:.4f} | Log Loss: {test_loss:.4f}")
        
        results.append({
            'Model': name,
            'Train Candidate Acc (%)': round(train_acc * 100, 2),
            'Train Top-1 Obs Acc (%)': round(train_obs_top1 * 100, 2),
            'Train Macro F1': round(train_macro_f1, 4),
            'Test Candidate Acc (%)': round(test_acc * 100, 2),
            'Test Top-1 Obs Acc (%)': round(test_obs_top1 * 100, 2),
            'Test Macro F1': round(test_macro_f1, 4),
            'Test Log Loss': round(test_loss, 4),
            'Fit Time (s)': round(fit_time, 2),
            'Inference Time (s)': round(inference_time, 2),
            'Confusion Matrix (Test)': cm
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('pathsnap/outputs/models/model_comparison.csv', index=False)
    with open('pathsnap/outputs/models/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*80)
    print("ALL 5 MODELS EVALUATED SUCCESSFULLY. Results saved to pathsnap/outputs/models/")
    print("="*80)

if __name__ == '__main__':
    main()
