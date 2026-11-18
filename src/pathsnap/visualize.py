"""Visualisation utilities for GPS trajectories, road networks, distributions, and interactive Folium maps.

Ensures strict compliance with Folium coordinate ordering [lat, lon], contrasting layer colors,
and reproducible static plot exports.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import folium
from folium import plugins
import matplotlib
# Ensure headless non-GUI backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathsnap.graph import RoadNetwork


# Apply clean scientific styling
sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_trajectory_and_network(
    network: RoadNetwork,
    track_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str | Path] = None,
    trip_id: str = "trip",
) -> None:
    """Plots projected 2D road network arcs, GPS trajectory, and reference route."""
    plt.figure(figsize=(10, 10))

    # 1. Plot road network arcs in bounding box of trajectory
    tx, ty = network.projector.forward(track_df["longitude"].values, track_df["latitude"].values)
    margin = 500.0  # 500m buffer
    min_x, max_x = np.min(tx) - margin, np.max(tx) + margin
    min_y, max_y = np.min(ty) - margin, np.max(ty) + margin

    arc_df = network.arc_table
    in_box = (
        (arc_df["x1"] >= min_x)
        & (arc_df["x1"] <= max_x)
        & (arc_df["y1"] >= min_y)
        & (arc_df["y1"] <= max_y)
    )
    sub_arcs = arc_df[in_box]

    for _, row in sub_arcs.iterrows():
        plt.plot(
            [row["x1"], row["x2"]],
            [row["y1"], row["y2"]],
            color="#bbbbbb",
            linewidth=0.8,
            alpha=0.6,
            zorder=1,
        )

    # 2. Plot Reference Route if available
    if route_df is not None and not route_df.empty:
        r_arcs = route_df["arc_id"].values
        for r_id in r_arcs:
            row = arc_df.iloc[r_id]
            plt.plot(
                [row["x1"], row["x2"]],
                [row["y1"], row["y2"]],
                color="#2ca02c",  # Green
                linewidth=2.5,
                alpha=0.85,
                zorder=2,
            )

    # 3. Plot GPS observations
    plt.scatter(
        tx,
        ty,
        color="#1f77b4",  # Blue
        s=12,
        alpha=0.8,
        zorder=3,
        label="GPS Observations (noisy)",
    )
    plt.plot(
        tx,
        ty,
        color="#1f77b4",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        zorder=3,
    )

    plt.title(f"PathSnap: Trip {trip_id} Trajectory & Local Road Network", fontsize=14, pad=12)
    plt.xlabel("UTM Easting (meters)")
    plt.ylabel("UTM Northing (meters)")
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.gca().set_aspect("equal")
    plt.legend(loc="upper right")
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()


def plot_timestamp_gap_distribution(
    time_gaps: np.ndarray,
    output_path: Optional[str | Path] = None,
    title: str = "GPS Timestamp Gap Distribution",
) -> None:
    """Plots histogram and cumulative distribution of consecutive GPS time gaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Log scale histogram
    gaps = np.asarray(time_gaps)
    ax1.hist(gaps, bins=50, color="#3470a3", edgecolor="black", alpha=0.8)
    ax1.set_title(f"{title} (Histogram)")
    ax1.set_xlabel("Time gap $\Delta t$ (seconds)")
    ax1.set_ylabel("Observation Count")
    ax1.set_yscale("log")

    # Zoomed bar chart of common gaps (1s vs non-1s)
    exact_1s = np.sum(gaps == 1.0)
    non_1s = len(gaps) - exact_1s
    gt_10s = np.sum(gaps > 10.0)
    gt_60s = np.sum(gaps > 60.0)

    categories = ["Exact 1s", "Non-1s Total", "Gaps > 10s", "Gaps > 60s"]
    counts = [exact_1s, non_1s, gt_10s, gt_60s]
    bars = ax2.bar(categories, counts, color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"])
    ax2.set_title("Sampling Gap Regularity Breakdown")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_yscale("log")

    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.1,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()


def plot_candidate_distance_distribution(
    distances: np.ndarray,
    output_path: Optional[str | Path] = None,
    title: str = "Point-to-Reference-Arc Distance Distribution",
) -> None:
    """Plots the empirical distribution of distances from GPS observations to reference arcs."""
    plt.figure(figsize=(8, 5))
    d = np.asarray(distances)
    p50 = np.median(d)
    p95 = np.percentile(d, 95)
    p99 = np.percentile(d, 99)

    sns.histplot(d[d <= 100], bins=50, kde=True, color="#1f77b4", edgecolor="black")
    plt.axvline(p50, color="orange", linestyle="--", linewidth=1.5, label=f"Median ({p50:.1f}m)")
    plt.axvline(p95, color="red", linestyle="--", linewidth=1.5, label=f"95th Pct ({p95:.1f}m)")
    plt.axvline(p99, color="purple", linestyle="--", linewidth=1.5, label=f"99th Pct ({p99:.1f}m)")

    plt.title(f"{title} (up to 100m)")
    plt.xlabel("Perpendicular Distance (meters)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()


def create_folium_map(
    network: RoadNetwork,
    track_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame] = None,
    output_html_path: Optional[str | Path] = None,
    trip_id: str = "trip",
    max_network_arcs_to_render: int = 5000,
) -> folium.Map:
    """Generates an interactive Folium map adhering to strict requirements:

    1. Original GPS observations: BLUE circle markers and connecting polyline.
    2. Reference route: GREEN polyline.
    3. Relevant road-network arcs: neutral contrasting color (#555555).
    4. Distinguishable colors, clear legend, layer controls, and appropriate extent.
    5. CRITICAL: Folium expects [latitude, longitude] ordering!
    """
    # 1. Compute trajectory center and bounds
    mean_lat = float(track_df["latitude"].mean())
    mean_lon = float(track_df["longitude"].mean())

    fmap = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Layer groups
    fg_network = folium.FeatureGroup(name="Road Network (Nearby Arcs)", show=True)
    fg_route = folium.FeatureGroup(name="Reference Ground Truth Route (Green)", show=True)
    fg_gps = folium.FeatureGroup(name="Noisy GPS Observations (Blue)", show=True)

    # 2. Road Network Arcs within trajectory bounding box
    tx, ty = network.projector.forward(track_df["longitude"].values, track_df["latitude"].values)
    buf = 400.0  # 400m margin
    min_x, max_x = np.min(tx) - buf, np.max(tx) + buf
    min_y, max_y = np.min(ty) - buf, np.max(ty) + buf

    arc_df = network.arc_table
    in_box = (
        (arc_df["x1"] >= min_x)
        & (arc_df["x1"] <= max_x)
        & (arc_df["y1"] >= min_y)
        & (arc_df["y1"] <= max_y)
    )
    relevant_arcs = arc_df[in_box]
    if len(relevant_arcs) > max_network_arcs_to_render:
        relevant_arcs = relevant_arcs.sample(max_network_arcs_to_render, random_state=42)

    # Convert node coordinates to WGS84 for folium
    node_lons = network.node_lon
    node_lats = network.node_lat

    from_ids = relevant_arcs["from_node_id"].values
    to_ids = relevant_arcs["to_node_id"].values
    arc_ids = relevant_arcs["arc_id"].values

    for i in range(len(relevant_arcs)):
        u = from_ids[i]
        v = to_ids[i]
        a_id = arc_ids[i]
        p1 = [float(node_lats[u]), float(node_lons[u])]
        p2 = [float(node_lats[v]), float(node_lons[v])]
        folium.PolyLine(
            locations=[p1, p2],
            color="#777777",
            weight=1.5,
            opacity=0.5,
            tooltip=f"Arc ID: {a_id} ({u}->{v})",
        ).add_to(fg_network)

    # 3. Reference Route (Green)
    if route_df is not None and not route_df.empty:
        r_arcs = route_df["arc_id"].values
        route_coords: List[List[float]] = []

        all_from = network.arc_table["from_node_id"].values
        all_to = network.arc_table["to_node_id"].values

        for a_id in r_arcs:
            u = all_from[a_id]
            v = all_to[a_id]
            p1 = [float(node_lats[u]), float(node_lons[u])]
            p2 = [float(node_lats[v]), float(node_lons[v])]
            if not route_coords:
                route_coords.append(p1)
            route_coords.append(p2)

        folium.PolyLine(
            locations=route_coords,
            color="#28a745",  # Green
            weight=4.0,
            opacity=0.9,
            tooltip=f"Trip {trip_id} Reference Ground-Truth Route ({len(r_arcs)} arcs)",
        ).add_to(fg_route)

    # 4. GPS Trajectory (Blue)
    gps_coords = []
    for idx, row in track_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        t = float(row["timestamp"])
        coord = [lat, lon]
        gps_coords.append(coord)

        # Draw circle markers for discrete GPS pings
        folium.CircleMarker(
            location=coord,
            radius=3,
            color="#0056b3",
            fill=True,
            fill_color="#007bff",
            fill_opacity=0.8,
            weight=1,
            popup=f"Obs #{idx}<br>Time: {t:.1f}s<br>Lat: {lat:.5f}<br>Lon: {lon:.5f}",
        ).add_to(fg_gps)

    folium.PolyLine(
        locations=gps_coords,
        color="#007bff",  # Blue
        weight=2.0,
        opacity=0.7,
        dash_array="4, 6",
        tooltip=f"Trip {trip_id} Raw GPS Path ({len(gps_coords)} points)",
    ).add_to(fg_gps)

    # Add all layer groups to map
    fg_network.add_to(fmap)
    fg_route.add_to(fmap)
    fg_gps.add_to(fmap)

    # Add LayerControl
    folium.LayerControl(collapsed=False).add_to(fmap)

    # Add interactive fullscreen plugin
    plugins.Fullscreen(position="topright").add_to(fmap)

    # Save HTML if requested
    if output_html_path is not None:
        p = Path(output_html_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(p))

    return fmap

