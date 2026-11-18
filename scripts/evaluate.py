#!/usr/bin/env python3
"""Evaluate classifier vs HMM on student testing. Never touches evaluator data."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from pathsnap.data import load_trip,list_trips
from pathsnap.pipeline import load_model,predict
from pathsnap.labels import approximate_reference_labels
from pathsnap.metrics import get_metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='data')
    ap.add_argument('--model',default='artifacts/best_tuned.joblib')
    ap.add_argument('--partition',choices=['train','student_test'],default='student_test')
    ap.add_argument('--max-trips',type=int,default=None,help='SMOKE ONLY')
    ap.add_argument('--stride',type=int,default=1,help='1 = evaluate ALL observations; >1 SMOKE ONLY')
    ap.add_argument('--out',default='results/decoder_comparison.csv')
    args=ap.parse_args()
    bundle=load_model(args.model)
    trip_ids=list_trips(args.data_root,args.partition)
    if args.max_trips:trip_ids=trip_ids[:args.max_trips]
    rows=[]
    for tid in trip_ids:
        print('Evaluating trip',tid,flush=True)
        trip=load_trip(args.data_root,args.partition,tid,include_reference=False)
        idx=np.arange(0,len(trip.track),args.stride)
        t0=time.perf_counter()
        result=predict(trip,bundle,indices=idx)
        runtime=time.perf_counter()-t0
        # Reference file is read only AFTER inference has finished.
        reference=load_trip(args.data_root,args.partition,tid,include_reference=True)
        truth,error=approximate_reference_labels(reference,idx)
        scores=get_metrics(trip,result['batch'].arc_ids,result['prob'],truth,result['hmm'])
        row={'trip_id':tid,'observations':len(idx),'median_reference_alignment_error_m':float(np.median(error)),
             'pipeline_seconds':runtime,**scores}
        row['candidate_confusion_matrix']=json.dumps(row['candidate_confusion_matrix'])
        rows.append(row)
        print(f'  classifier Top1 {row["observation_top1_accuracy"]:.4f} -> HMM {row["hmm_observation_accuracy"]:.4f}; direct continuity {row["classifier_direct_continuity"]:.4f} -> {row["hmm_direct_continuity"]:.4f}; runtime {runtime:.1f}s',flush=True)
    if not rows:raise ValueError('No trips found')
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as fd:
        writer=csv.DictWriter(fd,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('Summary weighted by observations:')
    for metric in ('observation_top1_accuracy','hmm_observation_accuracy','classifier_direct_continuity','hmm_direct_continuity','candidate_miss_rate'):
        print(metric, float(np.average([r[metric] for r in rows],weights=[r['observations'] for r in rows])))
    print('Results:',path)

if __name__=='__main__':main()
