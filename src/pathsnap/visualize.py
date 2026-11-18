"""Static and interactive visualisations generated only from loaded data."""

from __future__ import annotations

from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geo import ProjectedTrip


def save_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_trajectory(projected: ProjectedTrip, path: str | Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot(projected.track_xy[:, 0], projected.track_xy[:, 1], color="#1769aa", linewidth=1)
    plt.scatter(projected.track_xy[:, 0], projected.track_xy[:, 1], s=3, alpha=0.5)
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(f"GPS trajectory - trip {projected.trip.trip_id} (EPSG:{projected.metadata.epsg})")
    save_figure(path)


def create_folium_map(projected: ProjectedTrip, path: str | Path) -> None:
    track = projected.trip.track
    center = [float(track["latitude"].median()), float(track["longitude"].median())]
    map_object = folium.Map(location=center, tiles="OpenStreetMap", control_scale=True)
    network_layer = folium.FeatureGroup(name="Relevant road network", show=True)
    route_layer = folium.FeatureGroup(name="Reference route", show=True)
    gps_layer = folium.FeatureGroup(name="Original GPS observations", show=True)
    nodes = projected.trip.nodes
    relevant_arc_ids = np.unique(projected.trip.route["arc_id"].to_numpy(dtype=np.int64))
    route_node_ids = projected.trip.arcs.iloc[relevant_arc_ids][["from_node_id", "to_node_id"]].to_numpy()
    relevant_nodes = np.unique(route_node_ids)
    lon_min, lon_max = nodes.iloc[relevant_nodes]["longitude"].min(), nodes.iloc[relevant_nodes]["longitude"].max()
    lat_min, lat_max = nodes.iloc[relevant_nodes]["latitude"].min(), nodes.iloc[relevant_nodes]["latitude"].max()
    network_mask = (
        nodes.iloc[projected.trip.arcs["from_node_id"].to_numpy()]["longitude"].to_numpy() >= lon_min - 0.01
    ) & (
        nodes.iloc[projected.trip.arcs["from_node_id"].to_numpy()]["longitude"].to_numpy() <= lon_max + 0.01
    ) & (
        nodes.iloc[projected.trip.arcs["from_node_id"].to_numpy()]["latitude"].to_numpy() >= lat_min - 0.01
    ) & (
        nodes.iloc[projected.trip.arcs["from_node_id"].to_numpy()]["latitude"].to_numpy() <= lat_max + 0.01
    )
    for arc_id in np.flatnonzero(network_mask)[:: max(1, int(network_mask.sum() / 5000))]:
        arc = projected.trip.arcs.iloc[arc_id]
        a = nodes.iloc[int(arc.from_node_id)]
        b = nodes.iloc[int(arc.to_node_id)]
        folium.PolyLine([[a.latitude, a.longitude], [b.latitude, b.longitude]], color="#777777", weight=1, opacity=0.35).add_to(network_layer)
    route_coordinates: list[list[float]] = []
    for arc_id in projected.trip.route["arc_id"].to_numpy(dtype=np.int64):
        arc = projected.trip.arcs.iloc[arc_id]
        a = nodes.iloc[int(arc.from_node_id)]
        b = nodes.iloc[int(arc.to_node_id)]
        route_coordinates.extend([[a.latitude, a.longitude], [b.latitude, b.longitude]])
    folium.PolyLine(route_coordinates, color="green", weight=5, opacity=0.8, tooltip="Reference route").add_to(route_layer)
    gps_coordinates = track[["latitude", "longitude"]].to_numpy().tolist()
    folium.PolyLine(gps_coordinates, color="blue", weight=2, opacity=0.7, tooltip="GPS trajectory").add_to(gps_layer)
    stride = max(1, len(track) // 500)
    for obs_id, row in track.iloc[::stride].iterrows():
        folium.CircleMarker(
            [row.latitude, row.longitude], radius=2, color="blue", fill=True,
            tooltip=f"GPS observation {obs_id}",
        ).add_to(gps_layer)
    network_layer.add_to(map_object)
    route_layer.add_to(map_object)
    gps_layer.add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    all_lat = track["latitude"].to_numpy()
    all_lon = track["longitude"].to_numpy()
    map_object.fit_bounds([[float(all_lat.min()), float(all_lon.min())], [float(all_lat.max()), float(all_lon.max())]])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(path)
