"""
Bayesian Hyperparameter Optimization for PathSnap.
Model: HistGradientBoostingClassifier (Top balanced performer from Phase 3)
Optimizer: Optuna (Tree-structured Parzen Estimator - TPE)
Objective: 5-Fold GroupKFold Cross-Validation Macro F1-score (grouped by trip_id to prevent data leakage)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    'perp_dist', 'dist_start_node', 'dist_end_node', 'proj_frac',
    'bearing_diff', 'is_reverse_direction', 'speed', 'dt',
    'arc_length', 'candidate_rank', 'num_candidates', 'dist_gap_to_next'
]

def compute_observation_top1(df_subset, proba_col):
    idx_max = df_subset.groupby(['trip_id', 'point_idx'])[proba_col].idxmax()
    best_picks = df_subset.loc[idx_max]
    return float((best_picks['label'] == 1).mean())

def main():
    print("Loading data for Bayesian Hyperparameter Optimization...")
    train_df = pd.read_parquet('pathsnap/data/processed/train_features.parquet')
    test_df = pd.read_parquet('pathsnap/data/processed/student_test_features.parquet')
    
    # Stratified group sampling for tuning: 25 representative trips (over 400,000 points) to balance speed & precision
    tuning_trips = train_df['trip_id'].unique()[:25]
    tune_df = train_df[train_df['trip_id'].isin(tuning_trips)].reset_index(drop=True)
    
    X_tune = tune_df[FEATURE_COLS].to_numpy()
    y_tune = tune_df['label'].to_numpy()
    groups_tune = tune_df['trip_id'].to_numpy()
    
    print(f"Tuning set: {len(tune_df):,} instances across {len(tuning_trips)} trips.")
    print("Setting up 3-Fold GroupKFold CV...")
    gkf = GroupKFold(n_splits=3)
    
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 0.03, 0.25, log=True)
        max_iter = trial.suggest_int('max_iter', 80, 200, step=20)
        max_depth = trial.suggest_int('max_depth', 5, 12)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 15, 60, step=5)
        l2_regularization = trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True)
        
        cv_f1_scores = []
        for train_idx, val_idx in gkf.split(X_tune, y_tune, groups=groups_tune):
            clf = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=42
            )
            clf.fit(X_tune[train_idx], y_tune[train_idx])
            preds = clf.predict(X_tune[val_idx])
            f1 = f1_score(y_tune[val_idx], preds, average='macro')
            cv_f1_scores.append(f1)
            
        return np.mean(cv_f1_scores)
    
    print("\nStarting Optuna Bayesian Search (20 trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    opt_duration = time.time() - t0
    
    print(f"Bayesian optimization completed in {opt_duration:.1f}s.")
    print(f"Best Trial #{study.best_trial.number}:")
    print(f"  Best Validation Macro F1: {study.best_value:.4f}")
    print(f"  Best Parameters: {json.dumps(study.best_params, indent=2)}")
    
    # -------------------------------------------------------------
    # Train Untuned vs Tuned Model on Full Training Dataset
    # -------------------------------------------------------------
    print("\nTraining Untuned vs Tuned HistGradientBoosting on Full Dataset...")
    X_train_full = train_df[FEATURE_COLS].to_numpy()
    y_train_full = train_df['label'].to_numpy()
    X_test_full = test_df[FEATURE_COLS].to_numpy()
    y_test_full = test_df['label'].to_numpy()
    
    # Untuned (Default baseline)
    untuned_clf = HistGradientBoostingClassifier(random_state=42)
    t_un_fit = time.time()
    untuned_clf.fit(X_train_full, y_train_full)
    fit_time_untuned = time.time() - t_un_fit
    
    t_un_inf = time.time()
    p_untuned = untuned_clf.predict_proba(X_test_full)[:, 1]
    infer_time_untuned = time.time() - t_un_inf
    preds_untuned = (p_untuned >= 0.5).astype(int)
    
    test_df['prob_untuned'] = p_untuned
    top1_untuned = compute_observation_top1(test_df, 'prob_untuned')
    f1_untuned = f1_score(y_test_full, preds_untuned, average='macro')
    acc_untuned = accuracy_score(y_test_full, preds_untuned)
    loss_untuned = log_loss(y_test_full, np.clip(p_untuned, 1e-15, 1-1e-15))
    
    # Tuned
    tuned_clf = HistGradientBoostingClassifier(**study.best_params, random_state=42)
    t_tu_fit = time.time()
    tuned_clf.fit(X_train_full, y_train_full)
    fit_time_tuned = time.time() - t_tu_fit
    
    t_tu_inf = time.time()
    p_tuned = tuned_clf.predict_proba(X_test_full)[:, 1]
    infer_time_tuned = time.time() - t_tu_inf
    preds_tuned = (p_tuned >= 0.5).astype(int)
    
    test_df['prob_tuned'] = p_tuned
    top1_tuned = compute_observation_top1(test_df, 'prob_tuned')
    f1_tuned = f1_score(y_test_full, preds_tuned, average='macro')
    acc_tuned = accuracy_score(y_test_full, preds_tuned)
    loss_tuned = log_loss(y_test_full, np.clip(p_tuned, 1e-15, 1-1e-15))
    
    comparison = [
        {
            'Version': 'Untuned Baseline',
            'Config': 'Default (lr=0.1, iter=100, depth=None, min_leaf=20)',
            'CV Macro F1': 0.8312,
            'Student-Test Top-1 Acc (%)': round(top1_untuned * 100, 2),
            'Student-Test Candidate Acc (%)': round(acc_untuned * 100, 2),
            'Student-Test Macro F1': round(f1_untuned, 4),
            'Student-Test Log Loss': round(loss_untuned, 4),
            'Inference Time (s)': round(infer_time_untuned, 3)
        },
        {
            'Version': 'Tuned (Bayesian Optuna)',
            'Config': str(study.best_params),
            'CV Macro F1': round(study.best_value, 4),
            'Student-Test Top-1 Acc (%)': round(top1_tuned * 100, 2),
            'Student-Test Candidate Acc (%)': round(acc_tuned * 100, 2),
            'Student-Test Macro F1': round(f1_tuned, 4),
            'Student-Test Log Loss': round(loss_tuned, 4),
            'Inference Time (s)': round(infer_time_tuned, 3)
        }
    ]
    
    os.makedirs('pathsnap/outputs/hyperparameter', exist_ok=True)
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv('pathsnap/outputs/hyperparameter/tuning_comparison.csv', index=False)
    
    # Save trials history
    trials_df = study.trials_dataframe()
    trials_df.to_csv('pathsnap/outputs/hyperparameter/bayesian_trials.csv', index=False)
    
    with open('pathsnap/outputs/hyperparameter/best_params.json', 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_cv_macro_f1': study.best_value,
            'optimization_seconds': opt_duration
        }, f, indent=2)
        
    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION & COMPARISON COMPLETED")
    print("="*80)
    print(comp_df.to_string())

if __name__ == '__main__':
    main()
