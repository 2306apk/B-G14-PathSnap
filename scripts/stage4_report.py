#!/usr/bin/env python3
"""Validate and visualise measured classifier-vs-HMM route reconstruction results."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def chart(path: Path, names: list[str], baseline: list[float], hmm: list[float], title: str, ylabel: str):
    x=np.arange(len(names));w=.37
    fig,ax=plt.subplots(figsize=(max(9,1.25*len(names)),5))
    ax.bar(x-w/2,baseline,width=w,label='Classifier-only');ax.bar(x+w/2,hmm,width=w,label='HMM / Viterbi')
    ax.set_xticks(x,names,rotation=30,ha='right');ax.set_ylabel(ylabel);ax.set_title(title);ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',default='results/stage4/decoder_comparison.csv');p.add_argument('--out-dir',default='results/stage4');args=p.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    with Path(args.csv).open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==10, f'Expected all 10 student-testing trips, found {len(rows)}'
    ids=[r['trip_id'] for r in rows];assert len(set(ids))==len(ids)
    count=np.array([int(r['observations']) for r in rows],dtype=int)
    assert count.sum()==19160, f'Expected all 19,160 observations; got {count.sum()}'
    keys=('observation_top1_accuracy','hmm_observation_accuracy','classifier_direct_continuity','hmm_direct_continuity','candidate_miss_rate')
    for r in rows:
        for k in keys:
            v=float(r[k]);assert math.isfinite(v) and 0<=v<=1,(r['trip_id'],k,v)
        assert math.isfinite(float(r['pipeline_seconds'])) and float(r['pipeline_seconds'])>=0
    w=lambda k: float(np.average([float(r[k]) for r in rows],weights=count))
    baseline_acc=w('observation_top1_accuracy');hmm_acc=w('hmm_observation_accuracy')
    baseline_cont=w('classifier_direct_continuity');hmm_cont=w('hmm_direct_continuity')
    delta=[dict(trip_id=r['trip_id'],observations=int(r['observations']),accuracy_delta=float(r['hmm_observation_accuracy'])-float(r['observation_top1_accuracy']),continuity_delta=float(r['hmm_direct_continuity'])-float(r['classifier_direct_continuity'])) for r in rows]
    improved=[x['trip_id'] for x in delta if x['accuracy_delta']>1e-12]
    declined=[x['trip_id'] for x in delta if x['accuracy_delta']< -1e-12]
    ties=[x['trip_id'] for x in delta if abs(x['accuracy_delta'])<=1e-12]
    summary={'partition':'student_test','trip_count':10,'observations':int(count.sum()),'metric_note':'Observation-level Top-1 is scored using approximate GPS-to-reference-route alignment; direct continuity counts same or directly adjacent road arcs only.',
             'classifier_top1_accuracy':baseline_acc,'hmm_top1_accuracy':hmm_acc,'top1_absolute_percentage_point_change':(hmm_acc-baseline_acc)*100,
             'classifier_direct_continuity':baseline_cont,'hmm_direct_continuity':hmm_cont,'continuity_absolute_percentage_point_change':(hmm_cont-baseline_cont)*100,
             'trips_improved':improved,'trips_declined':declined,'trips_tied':ties,'pipeline_total_seconds':float(sum(float(r['pipeline_seconds']) for r in rows)),'by_trip':delta}
    (out/'decoder_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    chart(out/'hmm_accuracy_per_trip.png',ids,[100*float(r['observation_top1_accuracy']) for r in rows],[100*float(r['hmm_observation_accuracy']) for r in rows],'Observation-level Top-1: classifier vs HMM','Top-1 accuracy (%)')
    chart(out/'hmm_continuity_per_trip.png',ids,[100*float(r['classifier_direct_continuity']) for r in rows],[100*float(r['hmm_direct_continuity']) for r in rows],'Direct arc continuity: classifier vs HMM','Direct continuity (%)')
    chart(out/'hmm_weighted_summary.png',['Top-1','Direct continuity'],[100*baseline_acc,100*baseline_cont],[100*hmm_acc,100*hmm_cont],'Observation-weighted metrics, 10 student-testing trips','Percent (%)')
    print('HMM comparison (observation-weighted; reference is approximate):')
    print(f'Top-1 accuracy: {baseline_acc:.4%} -> {hmm_acc:.4%} ({(hmm_acc-baseline_acc)*100:+.2f} pp)')
    print(f'Direct continuity: {baseline_cont:.4%} -> {hmm_cont:.4%} ({(hmm_cont-baseline_cont)*100:+.2f} pp)')
    print('Trips improved:',improved,'declined:',declined,'tied:',ties)
    print('STAGE 4 RESULT VERIFICATION: PASS')

if __name__=='__main__':main()
