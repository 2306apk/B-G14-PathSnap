"""
Data loading and dataset validation module for PathSnap.
Strictly enforces academic integrity and dataset sanity rules:
- Prohibits loading or referencing evaluator_test partition.
- Validates row/column shapes, data types, missing files, and NaNs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


FORBIDDEN_PARTITIONS = {"evaluator_test", "evaluator", "test_evaluator"}


@dataclass
class TripData:
    trip_id: str
    partition: str
    track: pd.DataFrame  # cols: [lon, lat, timestamp]
    nodes: pd.DataFrame  # cols: [node_id, lon, lat]
    arcs: pd.DataFrame   # cols: [arc_id, from_node, to_node]
    route: pd.DataFrame  # cols: [route_order, arc_id]


class PathSnapDataLoader:
    """
    DataLoader for PathSnap GPS map-matching dataset.
    """

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {self.data_root}")

    def list_partitions(self) -> List[str]:
        """Lists available partitions under data_root, strictly excluding forbidden ones."""
        partitions = []
        for p in self.data_root.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                if p.name.lower() in FORBIDDEN_PARTITIONS:
                    raise PermissionError(
                        f"Academic integrity violation: Access to forbidden partition '{p.name}' is strictly denied."
                    )
                partitions.append(p.name)
        return sorted(partitions)

    def list_trips(self, partition: str) -> List[str]:
        """List all trip_ids for a valid partition."""
        if partition.lower() in FORBIDDEN_PARTITIONS:
            raise PermissionError(
                f"Academic integrity violation: Access to forbidden partition '{partition}' is strictly denied."
            )
        part_dir = self.data_root / partition
        if not part_dir.exists():
            raise FileNotFoundError(f"Partition directory does not exist: {part_dir}")

        trip_dirs = [d.name for d in part_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        return sorted(trip_dirs)

    def load_trip(self, partition: str, trip_id: str) -> TripData:
        """
        Loads track, nodes, arcs, route for a specific trip.
        Performs thorough validation on columns, nulls, and expected types.
        """
        if partition.lower() in FORBIDDEN_PARTITIONS:
            raise PermissionError(
                f"Academic integrity violation: Access to forbidden partition '{partition}' is strictly denied."
            )

        trip_dir = self.data_root / partition / trip_id
        if not trip_dir.exists():
            raise FileNotFoundError(f"Trip directory does not exist: {trip_dir}")

        track_file = trip_dir / f"{trip_id}.track"
        nodes_file = trip_dir / f"{trip_id}.nodes"
        arcs_file = trip_dir / f"{trip_id}.arcs"
        route_file = trip_dir / f"{trip_id}.route"

        for f in [track_file, nodes_file, arcs_file, route_file]:
            if not f.exists():
                raise FileNotFoundError(f"Missing required trip file: {f}")

        # 1. Load .track: lon, lat, timestamp (s)
        track = pd.read_csv(
            track_file,
            sep=r"\s+",
            header=None,
            names=["lon", "lat", "timestamp"],
            dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64},
        )
        if track.isnull().any().any():
            raise ValueError(f"NaN values detected in {track_file}")
        if len(track.columns) != 3:
            raise ValueError(f"Expected 3 columns in {track_file}, got {len(track.columns)}")

        # 2. Load .nodes: lon, lat (0-based index = node_id)
        nodes = pd.read_csv(
            nodes_file,
            sep=r"\s+",
            header=None,
            names=["lon", "lat"],
            dtype={"lon": np.float64, "lat": np.float64},
        )
        if nodes.isnull().any().any():
            raise ValueError(f"NaN values detected in {nodes_file}")
        if len(nodes.columns) != 2:
            raise ValueError(f"Expected 2 columns in {nodes_file}, got {len(nodes.columns)}")
        nodes.index.name = "node_id"
        nodes = nodes.reset_index()

        # 3. Load .arcs: from_node_id, to_node_id (0-based index = arc_id)
        arcs = pd.read_csv(
            arcs_file,
            sep=r"\s+",
            header=None,
            names=["from_node", "to_node"],
            dtype={"from_node": np.int64, "to_node": np.int64},
        )
        if arcs.isnull().any().any():
            raise ValueError(f"NaN values detected in {arcs_file}")
        if len(arcs.columns) != 2:
            raise ValueError(f"Expected 2 columns in {arcs_file}, got {len(arcs.columns)}")
        arcs.index.name = "arc_id"
        arcs = arcs.reset_index()

        # 4. Load .route: chronological ground-truth arc_ids
        route = pd.read_csv(
            route_file,
            sep=r"\s+",
            header=None,
            names=["arc_id"],
            dtype={"arc_id": np.int64},
        )
        if route.isnull().any().any():
            raise ValueError(f"NaN values detected in {route_file}")
        if len(route.columns) != 1:
            raise ValueError(f"Expected 1 column in {route_file}, got {len(route.columns)}")
        route.index.name = "route_order"
        route = route.reset_index()

        return TripData(
            trip_id=trip_id,
            partition=partition,
            track=track,
            nodes=nodes,
            arcs=arcs,
            route=route,
        )
