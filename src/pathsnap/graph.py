"""Road-network graph representation and topological quality analysis.

Builds independent, per-trip directed road networks and spatial indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from pathsnap.geo import TripCRS, Projector, calculate_bearing, SpatialRoadIndex


@dataclass
class NetworkQualityReport:
    """Summary metrics of road network topology and reference route connectivity."""

    num_nodes: int
    num_arcs: int
    num_isolated_nodes: int
    num_weakly_connected_components: int
    largest_component_node_count: int
    largest_component_node_fraction: float
    num_strongly_connected_components: int
    avg_in_degree: float
    avg_out_degree: float
    max_in_degree: int
    max_out_degree: int
    route_total_arcs: int
    route_exact_connections: int
    route_exact_connection_ratio: float
    route_disconnections: List[Tuple[int, int, int, int]]  # (step_idx, arc1, arc2, gap_nodes)


class RoadNetwork:
    """Independent directed road-network graph and spatial query structure for a trip.
    
    Encapsulates:
    - Node coordinates projected to the trip's local UTM metric plane.
    - Directed road segments (arcs) with lengths and compass bearings.
    - Spatial STRtree index for high-speed radius candidate queries.
    - NetworkX DiGraph for shortest path traversal and connectivity checks.
    """

    def __init__(
        self,
        trip_id: str,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        crs_info: TripCRS,
        projector: Optional[Projector] = None,
    ):
        self.trip_id = trip_id
        self.crs_info = crs_info
        self.projector = projector or Projector(crs_info)

        # -------------------------------------------------------------
        # 1. Project node coordinates from WGS84 (lon, lat) to metric UTM (x, y)
        # -------------------------------------------------------------
        lons = nodes_df["longitude"].values
        lats = nodes_df["latitude"].values
        nx_coords, ny_coords = self.projector.forward(lons, lats)

        self.node_x = nx_coords
        self.node_y = ny_coords
        self.node_lon = lons
        self.node_lat = lats
        self.num_nodes = len(nodes_df)

        # -------------------------------------------------------------
        # 2. Build arc geometries, metric lengths, and compass bearings
        # -------------------------------------------------------------
        from_nodes = arcs_df["from_node_id"].values
        to_nodes = arcs_df["to_node_id"].values
        self.num_arcs = len(arcs_df)

        x1 = self.node_x[from_nodes]
        y1 = self.node_y[from_nodes]
        x2 = self.node_x[to_nodes]
        y2 = self.node_y[to_nodes]

        dx = x2 - x1
        dy = y2 - y1
        lengths = np.hypot(dx, dy)

        # Vectorized compass bearing calculation clockwise from True North [0, 360)
        bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

        # Tabular representation of all directed road segments
        self.arc_table = pd.DataFrame(
            {
                "arc_id": np.arange(self.num_arcs, dtype=np.int64),
                "from_node_id": from_nodes,
                "to_node_id": to_nodes,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length": lengths,
                "bearing": bearings,
            }
        )

        # Preconstruct Shapely LineString geometries for 2D spatial indexing (STRtree)
        self.arc_geometries = [
            LineString([(x1[i], y1[i]), (x2[i], y2[i])]) for i in range(self.num_arcs)
        ]
        self.spatial_index = SpatialRoadIndex(self.arc_geometries, self.arc_table["arc_id"].values)

        # -------------------------------------------------------------
        # 3. Construct directed graph (NetworkX DiGraph)
        # -------------------------------------------------------------
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.num_nodes))
        for arc_id in range(self.num_arcs):
            u = int(from_nodes[arc_id])
            v = int(to_nodes[arc_id])
            # Directed edge from u to v weighted by road length in meters
            self.graph.add_edge(u, v, arc_id=arc_id, weight=float(lengths[arc_id]))

    def query_candidate_arcs(self, px: float, py: float, radius: float) -> List[int]:
        """Returns candidate arc IDs within radius (meters) of projected point (px, py)."""
        return self.spatial_index.query_radius(px, py, radius)

    def get_arc_endpoints(self, arc_id: int) -> Tuple[float, float, float, float]:
        """Returns (x1, y1, x2, y2) for a given arc ID in metric coordinates."""
        row = self.arc_table.iloc[arc_id]
        return float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])

    def analyze_quality(self, route_df: Optional[pd.DataFrame] = None) -> NetworkQualityReport:
        """Computes comprehensive topological quality metrics for the trip network."""
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())

        isolated_count = sum(
            1 for n in range(self.num_nodes) if in_degrees[n] == 0 and out_degrees[n] == 0
        )

        wcc = list(nx.weakly_connected_components(self.graph))
        num_wcc = len(wcc)
        largest_wcc_size = max(len(c) for c in wcc) if wcc else 0
        largest_wcc_fraction = largest_wcc_size / self.num_nodes if self.num_nodes > 0 else 0.0

        scc = list(nx.strongly_connected_components(self.graph))
        num_scc = len(scc)

        in_vals = list(in_degrees.values())
        out_vals = list(out_degrees.values())
        avg_in = float(np.mean(in_vals)) if in_vals else 0.0
        avg_out = float(np.mean(out_vals)) if out_vals else 0.0
        max_in = int(np.max(in_vals)) if in_vals else 0
        max_out = int(np.max(out_vals)) if out_vals else 0

        route_total = 0
        route_exact = 0
        disconnections: List[Tuple[int, int, int, int]] = []

        if route_df is not None and not route_df.empty:
            route_arcs = route_df["arc_id"].values
            route_total = len(route_arcs)
            from_nodes = self.arc_table["from_node_id"].values
            to_nodes = self.arc_table["to_node_id"].values

            for step in range(route_total - 1):
                a1 = route_arcs[step]
                a2 = route_arcs[step + 1]
                u_to = to_nodes[a1]
                v_from = from_nodes[a2]
                if u_to == v_from:
                    route_exact += 1
                else:
                    # Check graph distance
                    try:
                        shortest_len = nx.shortest_path_length(self.graph, u_to, v_from)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        shortest_len = -1
                    disconnections.append((step, int(a1), int(a2), shortest_len))

        ratio = (route_exact / (route_total - 1)) if route_total > 1 else 1.0

        return NetworkQualityReport(
            num_nodes=self.num_nodes,
            num_arcs=self.num_arcs,
            num_isolated_nodes=isolated_count,
            num_weakly_connected_components=num_wcc,
            largest_component_node_count=largest_wcc_size,
            largest_component_node_fraction=largest_wcc_fraction,
            num_strongly_connected_components=num_scc,
            avg_in_degree=avg_in,
            avg_out_degree=avg_out,
            max_in_degree=max_in,
            max_out_degree=max_out,
            route_total_arcs=route_total,
            route_exact_connections=route_exact,
            route_exact_connection_ratio=ratio,
            route_disconnections=disconnections,
        )

