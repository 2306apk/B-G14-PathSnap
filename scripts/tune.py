#!/usr/bin/env python3
"""Bayesian-style TPE sequential hyperparameter optimisation with group-aware CV."""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import joblib
import optuna
from pathsnap.experiments import get_estimator,build_samples,pack_samples,cv_score,evaluate_model,MODEL_NAMES
from pathsnap.candidates import FEATURE_NAMES


def suggest(trial,model):
    if model=='logistic':
        return {'C':trial.suggest_float('C',1e-3,20,log=True)}
    if model=='gaussian_nb':
        return {'var_smoothing':trial.suggest_float('var_smoothing',1e-11,1e-7,log=True)}
    if model=='decision_tree':
        return {'max_depth':trial.suggest_int('max_depth',5,24),'min_samples_leaf':trial.suggest_int('min_samples_leaf',2,25)}
    if model=='random_forest':
        return {'max_depth':trial.suggest_int('max_depth',6,24),
                'n_estimators':trial.suggest_int('n_estimators',35,120),
                'min_samples_leaf':trial.suggest_int('min_samples_leaf',2,16)}
    if model=='hist_gradient_boosting':
        return {'max_iter':trial.suggest_int('max_iter',45,170),
                'max_leaf_nodes':trial.suggest_int('max_leaf_nodes',8,45),
                'learning_rate':trial.suggest_float('learning_rate',.025,.22,log=True),
                'l2_regularization':trial.suggest_float('l2_regularization',1e-4,10,log=True)}
    raise ValueError(model)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='data')
    ap.add_argument('--model',choices=MODEL_NAMES,default=None)
    ap.add_argument('--trials',type=int,default=15,help='Number of NEW trials; resumes existing SQLite study')
    ap.add_argument('--candidate-k',type=int,default=8)
    ap.add_argument('--train-stride',type=int,default=8)
    ap.add_argument('--test-stride',type=int,default=1)
    ap.add_argument('--max-train-trips',type=int,default=None,help='SMOKE ONLY')
    ap.add_argument('--max-test-trips',type=int,default=None,help='SMOKE ONLY')
    ap.add_argument('--cv-folds',type=int,default=3)
    ap.add_argument('--out-dir',default='results')
    ap.add_argument('--artifact-dir',default='artifacts')
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    artifacts=Path(args.artifact_dir);artifacts.mkdir(parents=True,exist_ok=True)
    sel_path=out/'selected_model.txt'
    model=args.model or (sel_path.read_text().strip() if sel_path.exists() else 'hist_gradient_boosting')
    print('Bayesian TPE tuning model:',model,flush=True)
    train=build_samples(args.data_root,'train',args.candidate_k,args.train_stride,args.max_train_trips)
    X,arcs,truth,groups,y=pack_samples(train)
    t0=time.perf_counter()
    def objective(trial):
        params=suggest(trial,model)
        folds=cv_score(model,X,arcs,truth,groups,args.cv_folds,params=params)
        trial.set_user_attr('fold_scores',[round(x['top1_accuracy'],6) for x in folds])
        trial.set_user_attr('train_seconds',sum(x['train_seconds'] for x in folds))
        return float(np.mean([x['top1_accuracy'] for x in folds]))
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f'pathsnap_{model}_k{args.candidate_k}_stride{args.train_stride}',
        storage='sqlite:///'+str((out/'optuna_study.sqlite3').resolve()), load_if_exists=True)
    study.optimize(objective,n_trials=args.trials)
    study.trials_dataframe().to_csv(out/'optuna_trials.csv',index=False)
    best_params=dict(study.best_params)
    optimized=get_estimator(model,params=best_params)
    optimized.fit(X.reshape(-1,len(FEATURE_NAMES)),y.ravel())
    joblib.dump({'estimator':optimized,'name':model+'_tuned','feature_names':FEATURE_NAMES,
                 'candidate_k':X.shape[1],'train_stride':args.train_stride,
                 'seed':42,'model_params':best_params},artifacts/'best_tuned.joblib',compress=3)
    baseline_path=artifacts/f'{model}.joblib'
    baseline=joblib.load(baseline_path)['estimator'] if baseline_path.exists() else get_estimator(model)
    if not baseline_path.exists():baseline.fit(X.reshape(-1,len(FEATURE_NAMES)),y.ravel())
    baseline_cv=cv_score(model,X,arcs,truth,groups,args.cv_folds)
    baseline_cv_score=float(np.mean([f['top1_accuracy'] for f in baseline_cv]))
    test=build_samples(args.data_root,'student_test',args.candidate_k,args.test_stride,args.max_test_trips)
    original,_=evaluate_model(baseline,test)
    tuned,_=evaluate_model(optimized,test)
    summary={
        'model':model,'objective':'group-aware CV observation-level Top-1 accuracy',
        'optimizer':'Optuna TPESampler (tree-structured Parzen estimator)',
        'best_cv_accuracy':study.best_value,'baseline_cv_accuracy':baseline_cv_score,
        'best_params':best_params,'trials':len(study.trials),'total_seconds':time.perf_counter()-t0,
        'baseline_student_test':original,'tuned_student_test':tuned,
        'limited_smoke_test':bool(args.max_train_trips or args.max_test_trips or args.test_stride!=1),
        'warning':'Approximate GPS-to-reference-route alignment; no per-GPS true arc labels were provided'
    }
    (out/'tuning_summary.json').write_text(json.dumps(summary,indent=2))
    print('Tuning complete:',json.dumps({k:summary[k] for k in ('best_params','best_cv_accuracy','baseline_cv_accuracy','limited_smoke_test')},indent=2))

if __name__=='__main__':main()
