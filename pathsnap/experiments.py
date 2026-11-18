"""Train-only grouped CV and student-test evaluation support.

Reference route alignment produces *approximate* pointwise labels; disclose this.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from .data import load_trip,list_trips
from .candidates import SpatialCandidates,FEATURE_NAMES
from .labels import approximate_reference_labels
from .metrics import get_metrics


MODEL_NAMES=('logistic','gaussian_nb','decision_tree','random_forest','hist_gradient_boosting')


def get_estimator(name: str, random_seed: int=42, params: dict | None = None):
    params=params or {}
    if name=='logistic':
        opts={'max_iter':350,'class_weight':'balanced','random_state':random_seed};opts.update(params)
        return make_pipeline(StandardScaler(),LogisticRegression(**opts))
    if name=='gaussian_nb':
        return make_pipeline(StandardScaler(),GaussianNB(**params))
    if name=='decision_tree':
        opts={'max_depth':14,'min_samples_leaf':8,'class_weight':'balanced','random_state':random_seed};opts.update(params)
        return DecisionTreeClassifier(**opts)
    if name=='random_forest':
        opts={'n_estimators':60,'max_depth':14,'min_samples_leaf':6,'class_weight':'balanced_subsample','n_jobs':-1,'random_state':random_seed};opts.update(params)
        return RandomForestClassifier(**opts)
    if name=='hist_gradient_boosting':
        opts={'max_iter':90,'max_leaf_nodes':21,'learning_rate':.1,'class_weight':'balanced','random_state':random_seed};opts.update(params)
        return HistGradientBoostingClassifier(**opts)
    raise ValueError(f'Unknown model: {name}')


@dataclass
class TripSamples:
    trip_id: str
    trip: object
    batch: object
    truth: np.ndarray
    alignment_error_m: np.ndarray
    obs_indices: np.ndarray


def build_samples(data_root: str, partition: str, k: int=8,
                  stride: int=8, max_trips: int | None = None) -> list[TripSamples]:
    out=[]
    trip_ids=list_trips(data_root,partition)
    if max_trips is not None:
        trip_ids=trip_ids[:max_trips]
    for i,trip_id in enumerate(trip_ids,1):
        trip=load_trip(data_root,partition,trip_id,include_reference=True)
        idx=np.arange(0,len(trip.track),stride,dtype=int)
        candidates=SpatialCandidates(trip).query(idx,k=k)
        truth,error=approximate_reference_labels(trip,idx)
        out.append(TripSamples(trip_id,trip,candidates,truth,error,idx))
        print(f'Prepared {partition} trip {i}/{len(trip_ids)}: {trip_id} -> {len(idx)} samples; candidate miss {np.mean(~np.any(candidates.arc_ids==truth[:,None],axis=1)):.2%}; alignment median {np.median(error):.1f}m',flush=True)
    return out


def pack_samples(samples: list[TripSamples]):
    if not samples:
        raise ValueError('No trips available for training. Check --data-root.')
    X=np.concatenate([s.batch.features for s in samples],axis=0)
    arcs=np.concatenate([s.batch.arc_ids for s in samples],axis=0)
    truth=np.concatenate([s.truth for s in samples],axis=0)
    groups=np.concatenate([np.full(len(s.truth),i,dtype=int) for i,s in enumerate(samples)])
    y=(arcs==truth[:,None]).astype(np.uint8)
    return X,arcs,truth,groups,y


def cv_score(name: str,X: np.ndarray,arcs: np.ndarray,truth: np.ndarray,
             groups: np.ndarray,n_splits: int=3,seed: int=42,params: dict | None=None):
    n,k,_=X.shape
    if len(np.unique(groups))<n_splits:
        raise ValueError('GroupKFold requires at least n_splits distinct trips')
    Y=(arcs==truth[:,None]).astype(np.uint8)
    cv=GroupKFold(n_splits=n_splits)
    results=[]
    for fold,(itr,ival) in enumerate(cv.split(np.arange(n),groups=groups),1):
        model=get_estimator(name,seed,params)
        start=time.perf_counter()
        model.fit(X[itr].reshape(-1,X.shape[2]),Y[itr].ravel())
        training=time.perf_counter()-start
        start=time.perf_counter()
        p=model.predict_proba(X[ival].reshape(-1,X.shape[2]))[:,1].reshape(-1,k)
        inference=time.perf_counter()-start
        predicted=arcs[ival,np.argmax(p,axis=1)]
        yy=Y[ival].ravel()
        yyhat=(arcs[ival]==predicted[:,None]).ravel().astype(int)
        results.append({
            'fold':fold,'top1_accuracy':float(np.mean(predicted==truth[ival])),
            'candidate_binary_macro_f1':float(f1_score(yy,yyhat,average='macro',zero_division=0)),
            'train_seconds':training,'inference_seconds':inference
        })
    return results


def evaluate_model(estimator,samples: list[TripSamples]) -> tuple[dict,list[dict]]:
    rows=[]; total=0
    for s in samples:
        n,k,_=s.batch.features.shape
        started=time.perf_counter()
        prob=estimator.predict_proba(s.batch.features.reshape(-1,len(FEATURE_NAMES)))[:,1].reshape(n,k)
        elapsed=time.perf_counter()-started
        metrics=get_metrics(s.trip,s.batch.arc_ids,prob,s.truth)
        metrics.update({'trip_id':s.trip_id,'observations':n,'inference_seconds':elapsed,'median_alignment_error_m':float(np.median(s.alignment_error_m))})
        rows.append(metrics);total+=n
    keys=[key for key in rows[0].keys() if key not in ('trip_id','observations','candidate_confusion_matrix') and isinstance(rows[0][key],(float,int))]
    weighted={key:float(np.average([r[key] for r in rows],weights=[r['observations'] for r in rows])) for key in keys}
    weighted['observations']=total
    return weighted,rows
