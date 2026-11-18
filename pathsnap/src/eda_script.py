import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

sys.path.insert(0, str(Path("pathsnap").resolve()))
from src.data_io import PathSnapDataLoader

def run_eda():
    data_root = Path("Student folder")
    loader = PathSnapDataLoader(data_root)
    
    out_dir = Path("pathsnap/outputs/eda")
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir = Path("pathsnap/outputs/maps")
    map_dir.mkdir(parents=True, exist_ok=True)
    
    partitions = ["train", "student_test"]
    
    inventory_records = []
    timestamp_gap_records = []
    extent_records = []
    network_records = []
    route_check_records = []
    
    # Store first trip's nodes to check if network is shared across trips
    train_trips = loader.list_trips("train")
    student_trips = loader.list_trips("student_test")
    
    first_trip = loader.load_trip("train", train_trips[0])
    first_nodes_shape = first_trip.nodes.shape
    first_arcs_shape = first_trip.arcs.shape
    
    # We will sample several trips to see if nodes/arcs files are identical or per-trip
    network_identities = []
    
    for partition in partitions:
        trips = loader.list_trips(partition)
        for trip_id in trips:
            trip = loader.load_trip(partition, trip_id)
            
            # 1. Inventory & file sizes
            trip_path = data_root / partition / trip_id
            track_sz = (trip_path / f"{trip_id}.track").stat().st_size
            nodes_sz = (trip_path / f"{trip_id}.nodes").stat().st_size
            arcs_sz = (trip_path / f"{trip_id}.arcs").stat().st_size
            route_sz = (trip_path / f"{trip_id}.route").stat().st_size
            
            n_points = len(trip.track)
            n_nodes = len(trip.nodes)
            n_arcs = len(trip.arcs)
            n_route_arcs = len(trip.route)
            
            inventory_records.append({
                "partition": partition,
                "trip_id": trip_id,
                "n_points": n_points,
                "n_nodes": n_nodes,
                "n_arcs": n_arcs,
                "n_route_arcs": n_route_arcs,
                "track_size_kb": track_sz / 1024,
                "nodes_size_kb": nodes_sz / 1024,
                "arcs_size_kb": arcs_sz / 1024,
                "route_size_kb": route_sz / 1024,
            })
            
            # Check network shape comparison
            network_identities.append({
                "trip_id": trip_id,
                "n_nodes": n_nodes,
                "n_arcs": n_arcs,
                "same_as_first": (n_nodes == first_nodes_shape[0] and n_arcs == first_arcs_shape[0])
            })
            
            # 2. Timestamp analysis
            ts = trip.track["timestamp"].values
            is_sorted = np.all(np.diff(ts) >= 0)
            is_strictly_increasing = np.all(np.diff(ts) > 0)
            n_duplicates = len(ts) - len(np.unique(ts))
            diffs = np.diff(ts)
            
            gaps_1s = np.sum(np.isclose(diffs, 1.0, atol=1e-3))
            gaps_other = len(diffs) - gaps_1s
            pct_1s = (gaps_1s / len(diffs) * 100) if len(diffs) > 0 else 100.0
            
            timestamp_gap_records.append({
                "trip_id": trip_id,
                "partition": partition,
                "is_sorted": is_sorted,
                "is_strictly_increasing": is_strictly_increasing,
                "n_duplicates": n_duplicates,
                "min_gap": float(np.min(diffs)) if len(diffs) > 0 else 0,
                "median_gap": float(np.median(diffs)) if len(diffs) > 0 else 0,
                "max_gap": float(np.max(diffs)) if len(diffs) > 0 else 0,
                "pct_1s": pct_1s,
                "count_1s": int(gaps_1s),
                "count_other": int(gaps_other)
            })
            
            # 3. Geographic extent
            lon_min, lon_max = trip.track["lon"].min(), trip.track["lon"].max()
            lat_min, lat_max = trip.track["lat"].min(), trip.track["lat"].max()
            node_lon_min, node_lon_max = trip.nodes["lon"].min(), trip.nodes["lon"].max()
            node_lat_min, node_lat_max = trip.nodes["lat"].min(), trip.nodes["lat"].max()
            
            extent_records.append({
                "trip_id": trip_id,
                "partition": partition,
                "track_lon_min": lon_min,
                "track_lon_max": lon_max,
                "track_lat_min": lat_min,
                "track_lat_max": lat_max,
                "node_lon_min": node_lon_min,
                "node_lon_max": node_lon_max,
                "node_lat_min": node_lat_min,
                "node_lat_max": node_lat_max,
            })
            
            # 4. Network stats (connected components, isolated nodes)
            # Arcs reference node ids
            nodes_in_arcs = set(trip.arcs["from_node"]).union(set(trip.arcs["to_node"]))
            isolated_nodes = len(trip.nodes) - len(nodes_in_arcs)
            
            network_records.append({
                "trip_id": trip_id,
                "partition": partition,
                "n_nodes": n_nodes,
                "n_arcs": n_arcs,
                "isolated_nodes": isolated_nodes,
                "max_node_in_arcs": max(trip.arcs["from_node"].max(), trip.arcs["to_node"].max()),
            })
            
            # 5. Route validation
            # Are all route arc_ids valid?
            route_arcs = trip.route["arc_id"].values
            valid_arcs = np.all((route_arcs >= 0) & (route_arcs < n_arcs))
            
            # Consecutive route continuity check: does arc_i's to_node == arc_{i+1}'s from_node (or connected)?
            # Let's map arc_id -> (from_node, to_node)
            arcs_df = trip.arcs.set_index("arc_id")
            route_from_nodes = arcs_df.loc[route_arcs, "from_node"].values
            route_to_nodes = arcs_df.loc[route_arcs, "to_node"].values
            
            # Check strict connectivity: to_node[i] == from_node[i+1]
            consecutive_matches = (route_to_nodes[:-1] == route_from_nodes[1:])
            pct_consecutive_connected = np.mean(consecutive_matches) * 100 if len(consecutive_matches) > 0 else 100.0
            
            route_check_records.append({
                "trip_id": trip_id,
                "partition": partition,
                "all_arcs_valid": bool(valid_arcs),
                "n_route_arcs": len(route_arcs),
                "pct_strictly_connected": float(pct_consecutive_connected),
                "disconnected_transitions": int(np.sum(~consecutive_matches))
            })

    # Save summary tables
    df_inv = pd.DataFrame(inventory_records)
    df_ts = pd.DataFrame(timestamp_gap_records)
    df_ext = pd.DataFrame(extent_records)
    df_net = pd.DataFrame(network_records)
    df_route = pd.DataFrame(route_check_records)
    
    df_inv.to_csv(out_dir / "inventory_summary.csv", index=False)
    df_ts.to_csv(out_dir / "timestamp_summary.csv", index=False)
    df_ext.to_csv(out_dir / "extent_summary.csv", index=False)
    df_net.to_csv(out_dir / "network_summary.csv", index=False)
    df_route.to_csv(out_dir / "route_summary.csv", index=False)
    
    print("=== SUMMARY STATISTICS ===")
    print(f"Total trips processed: {len(df_inv)} (Train: {len(df_inv[df_inv['partition']=='train'])}, Student Test: {len(df_inv[df_inv['partition']=='student_test'])})")
    print(f"Total GPS points: {df_inv['n_points'].sum()} (Train: {df_inv[df_inv['partition']=='train']['n_points'].sum()}, Student Test: {df_inv[df_inv['partition']=='student_test']['n_points'].sum()})")
    print(f"Points per trip: min={df_inv['n_points'].min()}, median={df_inv['n_points'].median()}, max={df_inv['n_points'].max()}, mean={df_inv['n_points'].mean():.1f}")
    
    # Network check: are node counts varying across trips?
    unique_node_counts = df_inv["n_nodes"].nunique()
    unique_arc_counts = df_inv["n_arcs"].nunique()
    print(f"Unique node counts across trips: {unique_node_counts}, unique arc counts: {unique_arc_counts}")
    print(f"Sample node counts: {df_inv['n_nodes'].value_counts().head(5).to_dict()}")
    
    # Timestamps
    total_1s = df_ts["count_1s"].sum()
    total_other = df_ts["count_other"].sum()
    pct_total_1s = total_1s / (total_1s + total_other) * 100
    print(f"Overall timestamp 1s gaps: {pct_total_1s:.2f}% (Total 1s: {total_1s}, Other: {total_other})")
    print(f"Any duplicate timestamps?: {df_ts['n_duplicates'].sum() > 0} (Total duplicate points: {df_ts['n_duplicates'].sum()})")
    print(f"Trips with non-sorted timestamps: {(~df_ts['is_sorted']).sum()}")
    
    # Extent
    print("Geographic bounds across all trips:")
    print(f"Longitude range: [{df_ext['track_lon_min'].min():.6f}, {df_ext['track_lon_max'].max():.6f}]")
    print(f"Latitude range: [{df_ext['track_lat_min'].min():.6f}, {df_ext['track_lat_max'].max():.6f}]")
    
    # Route connectivity
    print(f"All route arc_ids valid across all trips: {df_route['all_arcs_valid'].all()}")
    print(f"Mean consecutive route connectivity: {df_route['pct_strictly_connected'].mean():.2f}%")
    print(f"Trips with 100% strictly connected route arcs: {(df_route['pct_strictly_connected'] == 100.0).sum()}/{len(df_route)}")
    
    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df_inv["n_points"], bins=30, ax=axes[0, 0], color="skyblue", kde=True)
    axes[0, 0].set_title("Distribution of GPS Points per Trip")
    axes[0, 0].set_xlabel("Number of Points")
    
    sns.boxplot(data=df_inv, x="partition", y="n_points", ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Points per Trip by Partition")
    
    # Gaps histogram across all trips
    all_gaps = []
    for trip_id in train_trips[:15]:
        t = loader.load_trip("train", trip_id)
        all_gaps.extend(np.diff(t.track["timestamp"].values))
    all_gaps = np.array(all_gaps)
    
    non_1s = all_gaps[~np.isclose(all_gaps, 1.0, atol=1e-3)]
    sns.histplot(non_1s[non_1s < 20], bins=40, ax=axes[1, 0], color="salmon")
    axes[1, 0].set_title("Distribution of Non-1s Gaps (< 20s)")
    axes[1, 0].set_xlabel("Gap (seconds)")
    
    sns.histplot(df_route["pct_strictly_connected"], bins=20, ax=axes[1, 1], color="lightgreen")
    axes[1, 1].set_title("Route Arc Consecutive Connectivity (%)")
    axes[1, 1].set_xlabel("% Consecutive Arcs Connected")
    
    plt.tight_layout()
    plt.savefig(out_dir / "eda_distributions.png", dpi=300)
    plt.close()
    print("Saved eda_distributions.png")
    
    # Folium Map generation for Trip 00000001
    sample_trip = loader.load_trip("train", "00000001")
    center_lat = sample_trip.track["lat"].mean()
    center_lon = sample_trip.track["lon"].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="cartodbpositron")
    
    # Add route arcs in green
    route_arcs_df = sample_trip.arcs.set_index("arc_id").loc[sample_trip.route["arc_id"].values]
    nodes_df = sample_trip.nodes.set_index("node_id")
    
    route_layer = folium.FeatureGroup(name="Ground-Truth Route Arcs", show=True)
    for _, arc in route_arcs_df.iterrows():
        fn = nodes_df.loc[arc["from_node"]]
        tn = nodes_df.loc[arc["to_node"]]
        coords = [[fn["lat"], fn["lon"]], [tn["lat"], tn["lon"]]]
        folium.PolyLine(
            locations=coords,
            color="#2ecc71",
            weight=4,
            opacity=0.8,
            tooltip=f"Arc {arc.name}"
        ).add_to(route_layer)
    route_layer.add_to(m)
    
    # Add GPS points in blue
    gps_layer = folium.FeatureGroup(name="GPS Points (1 Hz)", show=True)
    for i, pt in sample_trip.track.iterrows():
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=3,
            color="#2980b9",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.8,
            tooltip=f"Point {i}: t={pt['timestamp']}s"
        ).add_to(gps_layer)
    gps_layer.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    # Add simple HTML legend
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 180px; height: 90px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
     padding: 10px; border-radius: 5px;">
     <b>PathSnap Trip 00000001</b><br>
     <i style="background:#2980b9; width:10px; height:10px; border-radius:50%; display:inline-block;"></i> GPS Points<br>
     <i style="background:#2ecc71; width:20px; height:4px; display:inline-block;"></i> Reference Route
     </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    map_path = map_dir / "trip_00000001_map.html"
    m.save(str(map_path))
    print(f"Saved Folium map to {map_path}")

if __name__ == "__main__":
    run_eda()
