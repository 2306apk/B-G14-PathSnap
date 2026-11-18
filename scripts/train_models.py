#!/usr/bin/env python3
"""Run five genuine experiments with group-aware CV on the 80 training trips."""
import argparse,csv,json,time
from pathlib import Path
import joblib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathsnap.experiments import MODEL_NAMES,build_samples,pack_samples,cv_score,evaluate_model,get_estimator
from pathsnap.candidates import FEATURE_NAMES


def save_csv(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as fd:
        writer=csv.DictWriter(fd,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='data')
    ap.add_argument('--train-stride',type=int,default=8)
    ap.add_argument('--test-stride',type=int,default=1)
    ap.add_argument('--candidate-k',type=int,default=8)
    ap.add_argument('--max-train-trips',type=int,default=None,help='SMOKE TEST ONLY: never report limited-run metrics as final results')
    ap.add_argument('--max-test-trips',type=int,default=None,help='SMOKE TEST ONLY')
    ap.add_argument('--cv-folds',type=int,default=3)
    ap.add_argument('--out-dir',default='results')
    ap.add_argument('--artifact-dir',default='artifacts')
    args=ap.parse_args()
    if args.train_stride<1 or args.test_stride<1:
        ap.error('Stride must be >=1')
    out=Path(args.out_dir);out.mkdir(exist_ok=True,parents=True)
    Path(args.artifact_dir).mkdir(parents=True,exist_ok=True)
    train=build_samples(args.data_root,'train',args.candidate_k,args.train_stride,args.max_train_trips)
    test=build_samples(args.data_root,'student_test',args.candidate_k,args.test_stride,args.max_test_trips)
    if len(train)<args.cv_folds:ap.error('Not enough training trips for group CV')
    X,arcs,truth,groups,y=pack_samples(train)
    k=X.shape[1]
    print('Train sampled observations',len(X),'positives',int(y.sum()),'missing labels',int(np.sum(y.sum(axis=1)==0)),flush=True)
    results=[];trip_rows=[]
    for name in MODEL_NAMES:
        print('\nMODEL',name,flush=True)
        folds=cv_score(name,X,arcs,truth,groups,args.cv_folds)
        estimator=get_estimator(name)
        started=time.perf_counter()
        estimator.fit(X.reshape(-1,len(FEATURE_NAMES)),y.ravel())
        train_s=time.perf_counter()-started
        student,per_trip=evaluate_model(estimator,test)
        bundle={'estimator':estimator,'name':name,'feature_names':FEATURE_NAMES,'candidate_k':k,
                'train_stride':args.train_stride,'seed':42,'model_params':estimator.get_params()}
        joblib.dump(bundle,Path(args.artifact_dir)/f'{name}.joblib',compress=3)
        row={'model':name,'cv_top1_accuracy':float(np.mean([f['top1_accuracy'] for f in folds])),
             'cv_candidate_macro_f1':float(np.mean([f['candidate_binary_macro_f1'] for f in folds])),
             'cv_top1_std':float(np.std([f['top1_accuracy'] for f in folds])),
             'student_top1_accuracy':student['observation_top1_accuracy'],
             'student_candidate_macro_f1':student['candidate_binary_macro_f1'],
             'student_candidate_precision':student['candidate_binary_precision'],
             'student_candidate_recall':student['candidate_binary_recall'],
             'student_candidate_log_loss':student['candidate_binary_log_loss'],
             'student_candidate_miss_rate':student['candidate_miss_rate'],
             'student_route_continuity':student['classifier_direct_continuity'],
             'training_seconds':train_s,'student_inference_seconds':sum(r['inference_seconds'] for r in per_trip)}
        results.append(row)
        for record in per_trip:
            record['model']=name
            record['candidate_confusion_matrix']=json.dumps(record['candidate_confusion_matrix'])
            trip_rows.append(record)
        save_csv(out/("cv_"+name+".csv"),folds)
        save_csv(out/'model_comparison.csv',results)
        print('CV top1',row['cv_top1_accuracy'],'student top1',row['student_top1_accuracy'],flush=True)
    save_csv(out/'per_trip_comparison.csv',trip_rows)
    best=max(results,key=lambda r:(r['cv_top1_accuracy'],r['cv_candidate_macro_f1']))['model']
    (out/'selected_model.txt').write_text(best+'\n')
    (out/'run_manifest.json').write_text(json.dumps({
        'run_type':'LIMITED SMOKE TEST - NOT FINAL' if args.max_train_trips or args.max_test_trips or args.test_stride!=1 else 'FULL TRAIN/TEST',
        'train_trips':len(train),'student_test_trips':len(test),'train_stride':args.train_stride,
        'student_test_stride':args.test_stride,'candidate_k':k,'cv_folds':args.cv_folds,
        'primary_selection_metric':'group-CV observation-level Top-1 accuracy',
        'label_limitation':'GPS point labels are approximate alignment against ordered reference route; raw data supplies no pointwise true labels',
        'selected_model':best},indent=2))
    for metric,title,filename in [
        ('cv_candidate_macro_f1','Train CV candidate macro F1','cv_macro_f1.png'),
        ('student_top1_accuracy','Student-test observation Top-1','student_top1.png'),
        ('student_candidate_macro_f1','Student-test candidate macro F1','student_macro_f1.png'),
        ('student_candidate_log_loss','Student-test candidate log loss','student_log_loss.png')]:
        plt.figure(figsize=(9,4.5))
        plt.bar([x['model'] for x in results],[x[metric] for x in results])
        plt.xticks(rotation=17,ha='right');plt.ylabel(title);plt.title('Measured PathSnap experiment: '+title)
        plt.tight_layout();plt.savefig(out/filename,dpi=160);plt.close()
    print('\nBEST (based on training-only group CV):',best)
    print('Saved results to',out, 'artifacts to',args.artifact_dir)

if __name__=='__main__':main()
