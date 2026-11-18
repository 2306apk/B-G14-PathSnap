"""Per-trip metric projection and straight endpoint arc geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Transformer

from .io import TripData


@dataclass(frozen=True)
class ProjectionMetadata:
    epsg: int
    crs_name: str
    utm_zone: int
    hemisphere: str
    central_longitude: float
    center_longitude: float
    center_latitude: float
    longitude_min: float
    longitude_max: float
    latitude_min: float
    latitude_max: float


@dataclass(frozen=True)
class ProjectedTrip:
    trip: TripData
    metadata: ProjectionMetadata
    transformer: Transformer
    track_xy: np.ndarray
    nodes_xy: np.ndarray
    arc_geometries: np.ndarray
    arc_lengths_m: np.ndarray
    arc_bearings_deg: np.ndarray


def select_local_utm(longitudes: np.ndarray, latitudes: np.ndarray) -> ProjectionMetadata:
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    if longitudes.size == 0 or latitudes.size == 0:
        raise ValueError("Cannot choose a CRS from empty coordinates")
    if not (np.isfinite(longitudes).all() and np.isfinite(latitudes).all()):
        raise ValueError("Coordinates must be finite")
    lon = float(np.median(longitudes))
    lat = float(np.median(latitudes))
    zone = min(60, max(1, int(np.floor((lon + 180.0) / 6.0)) + 1))
    north = lat >= 0
    epsg = (32600 if north else 32700) + zone
    crs = CRS.from_epsg(epsg)
    return ProjectionMetadata(
        epsg=epsg,
        crs_name=crs.name,
        utm_zone=zone,
        hemisphere="north" if north else "south",
        central_longitude=zone * 6.0 - 183.0,
        center_longitude=lon,
        center_latitude=lat,
        longitude_min=float(np.min(longitudes)),
        longitude_max=float(np.max(longitudes)),
        latitude_min=float(np.min(latitudes)),
        latitude_max=float(np.max(latitudes)),
    )


def project_lonlat(
    longitudes: np.ndarray, latitudes: np.ndarray, transformer: Transformer
) -> np.ndarray:
    x, y = transformer.transform(longitudes, latitudes)
    return np.column_stack([x, y]).astype(float)


def build_arc_geometries(nodes_xy: np.ndarray, arcs: pd.DataFrame) -> np.ndarray:
    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    if endpoints.size == 0:
        return np.asarray([], dtype=object)
    if (endpoints < 0).any() or (endpoints >= len(nodes_xy)).any():
        raise ValueError("Arc endpoint references a missing node")
    coordinates = np.stack([nodes_xy[endpoints[:, 0]], nodes_xy[endpoints[:, 1]]], axis=1)
    return shapely.linestrings(coordinates)


def planar_bearing_degrees(start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    delta = np.asarray(end_xy, dtype=float) - np.asarray(start_xy, dtype=float)
    return (np.degrees(np.arctan2(delta[..., 0], delta[..., 1])) + 360.0) % 360.0


def project_trip(trip: TripData) -> ProjectedTrip:
    combined_lon = np.concatenate(
        [trip.track["longitude"].to_numpy(), trip.nodes["longitude"].to_numpy()]
    )
    combined_lat = np.concatenate(
        [trip.track["latitude"].to_numpy(), trip.nodes["latitude"].to_numpy()]
    )
    metadata = select_local_utm(combined_lon, combined_lat)
    transformer = Transformer.from_crs(4326, metadata.epsg, always_xy=True)
    track_xy = project_lonlat(
        trip.track["longitude"].to_numpy(), trip.track["latitude"].to_numpy(), transformer
    )
    nodes_xy = project_lonlat(
        trip.nodes["longitude"].to_numpy(), trip.nodes["latitude"].to_numpy(), transformer
    )
    geometries = build_arc_geometries(nodes_xy, trip.arcs)
    endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    lengths = shapely.length(geometries).astype(float)
    bearings = planar_bearing_degrees(nodes_xy[endpoints[:, 0]], nodes_xy[endpoints[:, 1]])
    return ProjectedTrip(
        trip=trip,
        metadata=metadata,
        transformer=transformer,
        track_xy=track_xy,
        nodes_xy=nodes_xy,
        arc_geometries=geometries,
        arc_lengths_m=lengths,
        arc_bearings_deg=bearings,
    )
