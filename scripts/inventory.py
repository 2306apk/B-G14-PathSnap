#!/usr/bin/env python3
"""Real EDA inventory; no synthetic values. Report each row from raw files."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from pathsnap.data import list_trips,load_trip


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='data')
    ap.add_argument('--out',default='results/inventory.csv')
    args=ap.parse_args()
    rows=[]
    for split in ('train','student_test'):
        for tid in list_trips(args.data_root,split):
            trip=load_trip(args.data_root,split,tid)
            delta=np.diff(trip.track[:,2]); arcs=trip.arcs
            n_components=None
            rows.append({
                'partition':split,'trip_id':tid,'observations':len(trip.track),
                'road_nodes':len(trip.nodes),'road_arcs':len(arcs),
                'reference_arcs':len(trip.route) if trip.route is not None else 0,
                'gps_nans':int(np.isnan(trip.track).sum()),
                'time_gaps_gt_1s':int(np.sum(delta>1)),
                'largest_gap_s':float(delta.max()) if len(delta) else 0,
                'gps_min_lon':float(trip.track[:,0].min()),'gps_max_lon':float(trip.track[:,0].max()),
                'gps_min_lat':float(trip.track[:,1].min()),'gps_max_lat':float(trip.track[:,1].max()),
                'route_arc_valid':bool(trip.route is None or np.all(trip.route<len(arcs))),
                'route_connected_fraction':float(np.mean(arcs[trip.route[:-1],1]==arcs[trip.route[1:],0])) if trip.route is not None and len(trip.route)>1 else None,
            })
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as fd:
        writer=csv.DictWriter(fd,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for s in ('train','student_test'):
        these=[r for r in rows if r['partition']==s]
        print(s, 'trips=',len(these),'GPS=',sum(r['observations'] for r in these),'arcs=',sum(r['road_arcs'] for r in these))
    print('Saved',path)

if __name__=='__main__':main()
