"""
Geographic projection, coordinate transformation, and spatial indexing utilities.
Enforces metric calculations using local UTM projections per trip via pyproj and Shapely STRtree.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


def determine_local_utm_crs(lon: float, lat: float) -> str:
    """
    Determines the EPSG code for the optimal local UTM zone based on WGS84 (lon, lat).
    
    UTM zones span 6 degrees of longitude:
      Zone = floor((lon + 180) / 6) + 1
      Northern hemisphere: EPSG 32601 - 32660
      Southern hemisphere: EPSG 32701 - 32760
    """
    # Wrap longitude to [-180, 180)
    lon_norm = ((lon + 180) % 360) - 180
    zone = int(np.floor((lon_norm + 180) / 6)) + 1
    # Clamp zone between 1 and 60
    zone = max(1, min(60, zone))
    epsg = (32600 if lat >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


def project_coordinates(
    lons: np.ndarray,
    lats: np.ndarray,
    target_crs: str,
    source_crs: str = "EPSG:4326"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects arrays of longitudes and latitudes from WGS84 to a target metric CRS.
    always_xy=True ensures (lon, lat) -> (x, y) order regardless of PROJ axis conventions.
    """
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(lons, lats)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def build_arc_geometries(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    target_crs: str
) -> Tuple[List[LineString], np.ndarray]:
    """
    Projects node coordinates into target_crs and constructs Shapely LineString geometries
    for each arc from from_node to to_node.
    
    Returns:
      arc_geometries: List of LineString objects corresponding to each row in arcs_df.
      arc_lengths: 1D array of arc lengths in metres.
    """
    # Project all nodes
    x_nodes, y_nodes = project_coordinates(
        nodes_df["lon"].values,
        nodes_df["lat"].values,
        target_crs=target_crs
    )
    
    # Fast array indexing for arc coordinates
    from_indices = arcs_df["from_node"].values
    to_indices = arcs_df["to_node"].values
    
    x1, y1 = x_nodes[from_indices], y_nodes[from_indices]
    x2, y2 = x_nodes[to_indices], y_nodes[to_indices]
    
    # Calculate Euclidean lengths in projected metric space
    dx = x2 - x1
    dy = y2 - y1
    arc_lengths = np.sqrt(dx * dx + dy * dy)
    
    # Construct Shapely LineStrings
    coords = np.stack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ], axis=1)  # shape: (n_arcs, 2, 2)
    
    arc_geometries = [LineString(c) for c in coords]
    return arc_geometries, arc_lengths


class TripSpatialIndex:
    """
    Spatial index wrapper holding projected arc geometries and an STRtree for fast candidate querying.
    """

    def __init__(self, arc_geometries: List[LineString], target_crs: str):
        self.arc_geometries = np.array(arc_geometries, dtype=object)
        self.target_crs = target_crs
        self.tree = STRtree(self.arc_geometries)

    def query_radius(self, x: float, y: float, radius: float) -> np.ndarray:
        """
        Queries all arc indices within radius (in metres) of point (x, y).
        Uses a bounding box query followed by exact distance checking.
        """
        pt = Point(x, y)
        buffer_geom = pt.buffer(radius)
        # Query candidates whose bbox overlaps buffer
        candidate_indices = self.tree.query(buffer_geom, predicate="intersects")
        return candidate_indices
