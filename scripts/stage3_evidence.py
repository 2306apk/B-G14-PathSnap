#!/usr/bin/env python3
"""Reproducible Stage-3 evaluation evidence. Never trains on student-testing trips.

Requires Stage 2 and Stage 3 experiment outputs produced by this repository.
Stage 3 CV uses the same training trips, candidate_k, stride, grouped folds, and
accuracy/F1 definitions as the Stage 2 baseline. All labels are approximate.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import math
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathsnap.experiments import build_samples, cv_score, pack_samples, evaluate_model


def read_csv(p: Path) -> list[dict]:
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict]):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot_comparison(out: Path, rows: list[dict], metric: str, filename: str, percent=False):
    selected=[r for r in rows if r['metric'] in metric.split(',')]
    labels=[r['metric'].replace('_',' ').replace('candidate binary','candidate') for r in selected]
    baseline=np.array([float(r['baseline']) for r in selected]);tuned=np.array([float(r['tuned']) for r in selected])
    if percent:
        baseline*=100;tuned*=100
    x=np.arange(len(selected));width=.35
    fig,ax=plt.subplots(figsize=(max(7,len(selected)*2.9),5))
    ax.bar(x-width/2,baseline,width,label='Baseline')
    ax.bar(x+width/2,tuned,width,label='Tuned')
    ax.set_xticks(x,labels,rotation=10,ha='right');ax.set_ylabel('Percent (%)' if percent else 'Value')
    ax.set_title('PathSnap Stage 3 — measured baseline vs tuned')
    ax.legend();fig.tight_layout();fig.savefig(out/filename,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default='data')
    p.add_argument('--stage2-results',default='results/stage2')
    p.add_argument('--stage3-results',default='results/stage3')
    p.add_argument('--stage3-artifacts',default='artifacts/stage3')
    args=p.parse_args()
    s2,s3,a3=map(Path,[args.stage2_results,args.stage3_results,args.stage3_artifacts]);s3.mkdir(parents=True,exist_ok=True)
    summary=json.loads((s3/'tuning_summary.json').read_text())
    m=json.loads((s2/'run_manifest.json').read_text())
    assert summary['model']=='hist_gradient_boosting'
    assert not summary['limited_smoke_test'] and m['run_type']=='FULL TRAIN/TEST'
    assert m['train_trips']==80 and m['student_test_trips']==10 and m['cv_folds']==3
    assert m['train_stride']==12 and m['student_test_stride']==1 and m['candidate_k']==8
    assert (s3/'optuna_trials.csv').exists()
    trial_rows=read_csv(s3/'optuna_trials.csv')
    completed=sorted((r for r in trial_rows if r['state']=='COMPLETE'),key=lambda r:int(r['number']))
    assert len(completed)>=15, 'Expected at least 15 completed TPE trials'
    cv_base=read_csv(s2/'cv_hist_gradient_boosting.csv')
    assert len(cv_base)==3
    print('Recomputing tuned model group-CV metrics on all 80 training trips...',flush=True)
    train=build_samples(args.data_root,'train',k=8,stride=12)
    X,arcs,truth,groups,_=pack_samples(train)
    cv_tuned=cv_score('hist_gradient_boosting',X,arcs,truth,groups,n_splits=3,params=summary['best_params'])
    write_csv(s3/'cv_hist_gradient_boosting_tuned.csv',cv_tuned)
    baseline=joblib.load(a3/'hist_gradient_boosting.joblib')['estimator']
    tuned=joblib.load(a3/'best_tuned.joblib')['estimator']
    print('Benchmarking classifier inference on the same 10 student-testing trips...',flush=True)
    test=build_samples(args.data_root,'student_test',k=8,stride=1)
    orig,orig_trips=evaluate_model(baseline,test)
    opt,opt_trips=evaluate_model(tuned,test)
    assert orig['observations']==opt['observations']==19160
    rows=[
        dict(metric='cv_top1_accuracy',baseline=float(np.mean([float(f['top1_accuracy']) for f in cv_base])),tuned=float(np.mean([f['top1_accuracy'] for f in cv_tuned])),unit='fraction'),
        dict(metric='cv_candidate_binary_macro_f1',baseline=float(np.mean([float(f['candidate_binary_macro_f1']) for f in cv_base])),tuned=float(np.mean([f['candidate_binary_macro_f1'] for f in cv_tuned])),unit='fraction'),
    ]
    for name in ('observation_top1_accuracy','candidate_binary_macro_f1','candidate_binary_log_loss','classifier_direct_continuity'):
        rows.append(dict(metric='student_'+name,baseline=float(orig[name]),tuned=float(opt[name]),unit='fraction'))
    rows.append(dict(metric='student_inference_seconds_total',baseline=sum(float(r['inference_seconds']) for r in orig_trips),tuned=sum(float(r['inference_seconds']) for r in opt_trips),unit='seconds'))
    write_csv(s3/'baseline_vs_tuned.csv',rows)
    # Optuna history; no fabricated training/validation loss curves.
    trials=np.array([int(r['number']) for r in completed]);values=np.array([float(r['value']) for r in completed]);running=np.maximum.accumulate(values)
    fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(trials,values,'o-',label='Trial group-CV Top-1');ax.plot(trials,running,'-',label='Best so far');ax.set(xlabel='TPE trial number',ylabel='Trip-group CV Top-1 accuracy',title='Actual Bayesian optimisation trials');ax.legend();fig.tight_layout();fig.savefig(s3/'optuna_history.png',dpi=160);plt.close(fig)
    plot_comparison(s3,rows,'cv_top1_accuracy,cv_candidate_binary_macro_f1,student_observation_top1_accuracy,student_candidate_binary_macro_f1','baseline_tuned_accuracy_f1.png',percent=True)
    plot_comparison(s3,rows,'student_candidate_binary_log_loss','baseline_tuned_log_loss.png')
    plot_comparison(s3,rows,'student_inference_seconds_total','baseline_tuned_inference.png')
    for r in rows:
        assert all(math.isfinite(float(r[k])) for k in ('baseline','tuned'))
        print(f"{r['metric']}: {float(r['baseline']):.6f} -> {float(r['tuned']):.6f} ({r['unit']})")
    print('STAGE 3 REPORT EVIDENCE: PASS',flush=True)

if __name__=='__main__':main()
