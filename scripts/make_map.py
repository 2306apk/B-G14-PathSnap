#!/usr/bin/env python3
"""Make Folium HTML layered map for TRAIN/STUDENT data only."""
import argparse
from pathlib import Path
import numpy as np
import folium
from pathsnap.data import load_trip
from pathsnap.pipeline import load_model,predict


def segment_latlon(trip,ids):
    nodes=trip.nodes
    for aid in ids:
        u,v=trip.arcs[int(aid)]
        yield nodes[u,[1,0]].tolist(),nodes[v,[1,0]].tolist()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='data')
    ap.add_argument('--model',default='artifacts/best_tuned.joblib')
    ap.add_argument('--partition',choices=['train','student_test'],default='student_test')
    ap.add_argument('--trip-id',required=True)
    ap.add_argument('--out',default='results/folium_trip.html')
    args=ap.parse_args()
    trip=load_trip(args.data_root,args.partition,args.trip_id,include_reference=False)
    bundle=load_model(args.model)
    r=predict(trip,bundle)
    center=[float(np.mean(trip.track[:,1])),float(np.mean(trip.track[:,0]))]
    m=folium.Map(location=center,zoom_start=13,tiles=None,control_scale=True)
    # Road geometry is drawn in an actual metric bounding region; not arbitrary neighbors.
    road=folium.FeatureGroup(name='Nearby road network',show=False)
    boundary=(trip.track[:,0].min()-.008,trip.track[:,0].max()+.008,trip.track[:,1].min()-.008,trip.track[:,1].max()+.008)
    a=trip.arcs
    mid=(trip.nodes[a[:,0]]+trip.nodes[a[:,1]])*.5
    select=np.where((mid[:,0]>=boundary[0])&(mid[:,0]<=boundary[1])&(mid[:,1]>=boundary[2])&(mid[:,1]<=boundary[3]))[0]
    for i,(p1,p2) in enumerate(segment_latlon(trip,select[:3000])):
        folium.PolyLine([p1,p2],weight=1,color='#999999',opacity=.4).add_to(road)
    road.add_to(m)
    folium.PolyLine(trip.track[:,[1,0]].tolist(),color='#2563eb',weight=2.5,opacity=.7,tooltip='Raw GPS').add_to(folium.FeatureGroup(name='GPS',show=True).add_to(m))
    pred=folium.FeatureGroup(name='HMM matched arcs',show=True)
    for p1,p2 in segment_latlon(trip,np.unique(r['hmm'])):
        folium.PolyLine([p1,p2],color='#dc2626',weight=4,opacity=.8).add_to(pred)
    pred.add_to(m)
    # Report maps may include student_test reference. Never send references to evaluator UI.
    ground=load_trip(args.data_root,args.partition,args.trip_id,include_reference=True)
    if ground.route is not None:
        g=folium.FeatureGroup(name='Reference route',show=False)
        for p1,p2 in segment_latlon(trip,ground.route):
            folium.PolyLine([p1,p2],color='#16a34a',weight=3,opacity=.7).add_to(g)
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    folium.Marker(location=center,popup='Blue: GPS · Red: predicted · Green: reference · Gray: road network').add_to(m)
    m.fit_bounds([[trip.track[:,1].min(),trip.track[:,0].min()],[trip.track[:,1].max(),trip.track[:,0].max()]])
    path=Path(args.out);path.parent.mkdir(exist_ok=True,parents=True)
    m.save(str(path));print('Saved Folium map:',path)

if __name__=='__main__':main()
