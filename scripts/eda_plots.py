"""Reproducible dataset visualisations without invented quantities."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathsnap.data import load_trip


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default='data')
    p.add_argument('--inventory',default='results/inventory.csv')
    p.add_argument('--partition',choices=['train','student_test'],default='train')
    p.add_argument('--trip-id',default='00000005')
    p.add_argument('--out-dir',default='results/eda')
    args=p.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.inventory,dtype={'trip_id':str})
    for col,title,file in [('observations','GPS observations per trip','gps_counts.png'),
                           ('road_arcs','Road arcs per trip','road_arcs.png'),
                           ('time_gaps_gt_1s','Irregular GPS intervals per trip','time_gaps.png')]:
        plt.figure(figsize=(8,4.5))
        for part,group in df.groupby('partition'):
            plt.hist(group[col],bins=20,alpha=.5,label=part)
        plt.xlabel(title);plt.ylabel('Trip count');plt.legend();plt.tight_layout()
        plt.savefig(out/file,dpi=160);plt.close()
    trip=load_trip(args.data_root,args.partition,args.trip_id)
    plt.figure(figsize=(9,6))
    gps=trip.track
    plt.plot(gps[:,0],gps[:,1],'-',lw=.8,alpha=.8,label='Raw GPS')
    a=trip.arcs; nodes=trip.nodes
    # Draw a genuine subset of roads near the track's geography.
    mid=(nodes[a[:,0]]+nodes[a[:,1]])*.5
    box=[gps[:,0].min()-.002,gps[:,0].max()+.002,gps[:,1].min()-.002,gps[:,1].max()+.002]
    ids=np.where((mid[:,0]>=box[0])&(mid[:,0]<=box[1])&(mid[:,1]>=box[2])&(mid[:,1]<=box[3]))[0][:1500]
    for i in ids:
        u,v=a[i];plt.plot(nodes[[u,v],0],nodes[[u,v],1],color='gray',lw=.45,alpha=.35)
    plt.xlabel('Longitude');plt.ylabel('Latitude');plt.title(f'{args.partition} trip {args.trip_id}: GPS + road arcs')
    plt.tight_layout();plt.savefig(out/'trip_gps_roads.png',dpi=170);plt.close()
    print('Saved real EDA charts to',out)

if __name__=='__main__':main()
