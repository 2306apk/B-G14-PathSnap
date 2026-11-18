"""Input/Output and validation utilities for PathSnap raw data files.

Handles reading, validation, and indexing for .track, .nodes, .arcs, and .route files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


TRACK_COLUMNS = ["longitude", "latitude", "timestamp"]
NODES_COLUMNS = ["longitude", "latitude"]
ARCS_COLUMNS = ["from_node_id", "to_node_id"]
ROUTE_COLUMNS = ["arc_id"]


@dataclass
class TripData:
    """Encapsulates all raw data tables and metadata for a single trip."""

    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: Optional[pd.DataFrame] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_observations(self) -> int:
        return len(self.track) if self.track is not None else 0

    @property
    def num_nodes(self) -> int:
        return len(self.nodes) if self.nodes is not None else 0

    @property
    def num_arcs(self) -> int:
        return len(self.arcs) if self.arcs is not None else 0

    @property
    def num_route_arcs(self) -> int:
        return len(self.route) if self.route is not None else 0


def resolve_data_dir(custom_path: Optional[str | Path] = None) -> Path:
    """Resolves the root dataset directory containing 'train' and 'student_test' partitions.

    Checks:
    1. custom_path argument if provided.
    2. PATHSNAP_DATA_DIR environment variable.
    3. 'data/raw' relative to current working directory or repo root.
    4. 'Student folder' in current or parent directories.
    """
    if custom_path is not None:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified dataset directory not found: {custom_path}")

    env_path = os.environ.get("PATHSNAP_DATA_DIR")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        Path("data/raw"),
        Path("../data/raw"),
        Path("Student folder"),
        Path("../Student folder"),
        Path("../../Student folder"),
        Path(__file__).resolve().parent.parent.parent / "data" / "raw",
        Path(__file__).resolve().parent.parent.parent.parent / "Student folder",
    ]

    for cand in candidates:
        if cand.exists() and (cand / "train").exists():
            return cand.resolve()

    # Default fallback
    return Path("data/raw").resolve()


def list_trips(partition: str = "train", base_dir: Optional[str | Path] = None) -> List[str]:
    """Lists all available trip IDs in a given partition ('train' or 'student_test')."""
    root = resolve_data_dir(base_dir)
    part_dir = root / partition
    if not part_dir.exists():
        raise FileNotFoundError(f"Partition directory does not exist: {part_dir}")

    trips = [
        d.name
        for d in part_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    return sorted(trips)


def load_track(filepath: str | Path) -> pd.DataFrame:
    """Loads and validates a .track file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Track file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Track file is empty: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=TRACK_COLUMNS,
        dtype={"longitude": np.float64, "latitude": np.float64, "timestamp": np.float64},
    )

    if df.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in track file, got {df.shape[1]}")

    return df


def load_nodes(filepath: str | Path) -> pd.DataFrame:
    """Loads and validates a .nodes file. Row index corresponds to node_id."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Nodes file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Nodes file is empty: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=NODES_COLUMNS,
        dtype={"longitude": np.float64, "latitude": np.float64},
    )

    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in nodes file, got {df.shape[1]}")

    df.index.name = "node_id"
    return df


def load_arcs(filepath: str | Path) -> pd.DataFrame:
    """Loads and validates an .arcs file. Row index corresponds to arc_id."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arcs file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Arcs file is empty: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ARCS_COLUMNS,
        dtype={"from_node_id": np.int64, "to_node_id": np.int64},
    )

    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in arcs file, got {df.shape[1]}")

    df.index.name = "arc_id"
    return df


def load_route(filepath: str | Path) -> pd.DataFrame:
    """Loads and validates a .route file. Chronological sequence of arc IDs."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Route file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Route file is empty: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ROUTE_COLUMNS,
        dtype={"arc_id": np.int64},
    )

    if df.shape[1] != 1:
        raise ValueError(f"Expected 1 column in route file, got {df.shape[1]}")

    df.index.name = "route_step"
    return df


def validate_trip_integrity(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    route: Optional[pd.DataFrame] = None,
) -> Tuple[List[str], List[str]]:
    """Performs rigorous integrity checks on trip tables."""
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Track checks
    if track.empty:
        errors.append("Track table is empty.")
    else:
        if track["longitude"].isna().any() or track["latitude"].isna().any():
            errors.append("Track contains NaN coordinates.")
        if (track["longitude"] < -180).any() or (track["longitude"] > 180).any():
            errors.append("Track longitude out of valid WGS84 range [-180, 180].")
        if (track["latitude"] < -90).any() or (track["latitude"] > 90).any():
            errors.append("Track latitude out of valid WGS84 range [-90, 90].")
        if not track["timestamp"].is_monotonic_increasing:
            warnings.append("Track timestamps are not strictly monotonically increasing.")

    # 2. Nodes checks
    num_nodes = len(nodes)
    if num_nodes == 0:
        errors.append("Nodes table is empty.")
    else:
        if nodes["longitude"].isna().any() or nodes["latitude"].isna().any():
            errors.append("Nodes table contains NaN coordinates.")
        if (nodes["longitude"] < -180).any() or (nodes["longitude"] > 180).any():
            errors.append("Nodes longitude out of valid WGS84 range [-180, 180].")
        if (nodes["latitude"] < -90).any() or (nodes["latitude"] > 90).any():
            errors.append("Nodes latitude out of valid WGS84 range [-90, 90].")

    # 3. Arcs checks
    num_arcs = len(arcs)
    if num_arcs == 0:
        errors.append("Arcs table is empty.")
    else:
        min_from, max_from = arcs["from_node_id"].min(), arcs["from_node_id"].max()
        min_to, max_to = arcs["to_node_id"].min(), arcs["to_node_id"].max()
        if min_from < 0 or max_from >= num_nodes:
            errors.append(f"Arcs from_node_id out of range [0, {num_nodes-1}]: min={min_from}, max={max_from}")
        if min_to < 0 or max_to >= num_nodes:
            errors.append(f"Arcs to_node_id out of range [0, {num_nodes-1}]: min={min_to}, max={max_to}")

    # 4. Route checks
    if route is not None and not route.empty:
        min_arc, max_arc = route["arc_id"].min(), route["arc_id"].max()
        if min_arc < 0 or max_arc >= num_arcs:
            errors.append(f"Route arc_id out of range [0, {num_arcs-1}]: min={min_arc}, max={max_arc}")

    return errors, warnings


def load_trip(
    trip_id: str,
    partition: str = "train",
    base_dir: Optional[str | Path] = None,
    validate: bool = True,
) -> TripData:
    """Loads all data files (.track, .nodes, .arcs, .route) for a single trip.

    Preserves 0-based indexing on nodes and arcs tables.
    """
    root = resolve_data_dir(base_dir)
    trip_dir = root / partition / trip_id

    if not trip_dir.exists():
        raise FileNotFoundError(f"Trip folder not found: {trip_dir}")

    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    track = load_track(track_path)
    nodes = load_nodes(nodes_path)
    arcs = load_arcs(arcs_path)
    route = load_route(route_path) if route_path.exists() else None

    errors: List[str] = []
    warnings: List[str] = []

    if validate:
        errors, warnings = validate_trip_integrity(track, nodes, arcs, route)

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        errors=errors,
        warnings=warnings,
    )

