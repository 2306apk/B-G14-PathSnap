"""Geographical projection, spatial transformations, metric calculations, and spatial indexing.

Adheres strictly to the requirement that no global projection (e.g. EPSG:28992 or EPSG:32631)
is assumed, calculating a verified local UTM projection per trip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree


@dataclass
class TripCRS:
    """CRS metadata for a specific trip."""

    epsg: int
    utm_zone: int
    is_northern: bool
    center_lon: float
    center_lat: float
    description: str


def determine_utm_epsg(lon: float, lat: float) -> TripCRS:
    """Calculates the standard UTM EPSG code for a given WGS84 coordinate.

    UTM Zone = floor((lon + 180) / 6) + 1
    EPSG = 32600 + zone for Northern Hemisphere (lat >= 0)
    EPSG = 32700 + zone for Southern Hemisphere (lat < 0)
    """
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude out of valid range [-180, 180]: {lon}")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of valid range [-90, 90]: {lat}")

    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    # Standard UTM zone bounding: 1 to 60
    zone = max(1, min(60, zone))

    is_northern = lat >= 0.0
    epsg = (32600 + zone) if is_northern else (32700 + zone)
    hemisphere = "N" if is_northern else "S"

    # Validate with pyproj
    crs = pyproj.CRS.from_epsg(epsg)

    return TripCRS(
        epsg=epsg,
        utm_zone=zone,
        is_northern=is_northern,
        center_lon=lon,
        center_lat=lat,
        description=f"WGS 84 / UTM zone {zone}{hemisphere} (EPSG:{epsg})",
    )


def get_trip_crs(track_df: pd.DataFrame) -> TripCRS:
    """Determines the appropriate local UTM CRS based on the median center of a trajectory."""
    med_lon = float(track_df["longitude"].median())
    med_lat = float(track_df["latitude"].median())
    return determine_utm_epsg(med_lon, med_lat)


class Projector:
    """Handles bidirectional projection between WGS84 (lon, lat) and local metric coordinates (x, y)."""

    def __init__(self, crs_info: TripCRS):
        self.crs_info = crs_info
        self.transformer_to_metric = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{crs_info.epsg}", always_xy=True
        )
        self.transformer_to_wgs84 = pyproj.Transformer.from_crs(
            f"EPSG:{crs_info.epsg}", "EPSG:4326", always_xy=True
        )

    def forward(self, lons: np.ndarray | List[float], lats: np.ndarray | List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts WGS84 (lon, lat) to metric (x, y) in meters."""
        x, y = self.transformer_to_metric.transform(np.asarray(lons), np.asarray(lats))
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def inverse(self, x: np.ndarray | List[float], y: np.ndarray | List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts metric (x, y) in meters to WGS84 (lon, lat)."""
        lon, lat = self.transformer_to_wgs84.transform(np.asarray(x), np.asarray(y))
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def calculate_bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculates compass bearing in degrees [0, 360) from (x1, y1) to (x2, y2).

    In UTM metric coordinates:
    x increases Eastwards, y increases Northwards.
    Bearing is measured clockwise from True North:
    bearing = (atan2(dx, dy) * 180 / pi) % 360
    """
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    angle_deg = math.degrees(math.atan2(dx, dy))
    return (angle_deg + 360.0) % 360.0


def calculate_heading_difference(heading1: float, heading2: float) -> float:
    """Calculates the minimum absolute circular difference between two angles in [0, 180] degrees."""
    diff = abs(heading1 - heading2) % 360.0
    return min(diff, 360.0 - diff)


def point_to_segment_projection(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float, float, float]:
    """Computes perpendicular projection of point P onto segment AB.

    Returns:
    (dist_to_segment, dist_to_from_node, dist_to_to_node, proj_ratio_t, proj_x, proj_y)
    """
    dx = x2 - x1
    dy = y2 - y1
    dist_from = math.hypot(px - x1, py - y1)
    dist_to = math.hypot(px - x2, py - y2)

    l2 = dx * dx + dy * dy
    if l2 < 1e-12:  # Degenerate zero-length segment
        return dist_from, dist_from, dist_to, 0.0, x1, y1

    # Unclamped projection parameter t
    t = ((px - x1) * dx + (py - y1) * dy) / l2
    # Clamped to segment [0, 1]
    t_clamped = max(0.0, min(1.0, t))

    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    dist_seg = math.hypot(px - proj_x, py - proj_y)

    return dist_seg, dist_from, dist_to, t_clamped, proj_x, proj_y


class SpatialRoadIndex:
    """Encapsulates spatial indexing using Shapely STRtree for fast candidate arc retrieval."""

    def __init__(self, arc_geometries: List[LineString], arc_ids: np.ndarray):
        """Initializes STRtree on preconstructed shapely LineString geometries."""
        self.geometries = np.array(arc_geometries, dtype=object)
        self.arc_ids = np.asarray(arc_ids, dtype=np.int64)
        self.tree = STRtree(self.geometries)

    def query_radius(self, px: float, py: float, radius: float) -> List[int]:
        """Queries all arc IDs within distance `radius` of point (px, py).

        Uses shapely STRtree.query with buffer for initial candidate filtering,
        followed by exact distance check to eliminate bounding-box false positives.
        """
        pt = Point(px, py)
        search_box = pt.buffer(radius)
        candidate_indices = self.tree.query(search_box)

        # Exact distance filtering
        matched_arc_ids: List[int] = []
        for idx in candidate_indices:
            geom = self.geometries[idx]
            if pt.distance(geom) <= radius:
                matched_arc_ids.append(int(self.arc_ids[idx]))

        return matched_arc_ids

