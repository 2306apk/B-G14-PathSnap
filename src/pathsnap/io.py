"""Strict loaders for PathSnap's headerless, trip-local files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FILE_COLUMNS = {
    "track": ["longitude", "latitude", "timestamp_s"],
    "nodes": ["longitude", "latitude"],
    "arcs": ["from_node_id", "to_node_id"],
    "route": ["arc_id"],
}


class DataValidationError(ValueError):
    """Raised when a raw trip violates the documented file contract."""


@dataclass(frozen=True)
class TripData:
    trip_id: str
    partition: str
    directory: Path
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame


def resolve_data_root(value: str | Path | None = None) -> Path:
    """Resolve data root from an argument or PATHSNAP_DATA_ROOT."""
    root = Path(value or os.environ.get("PATHSNAP_DATA_ROOT", "data/raw"))
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def discover_trips(data_root: str | Path, partition: str) -> list[Path]:
    partition_dir = resolve_data_root(data_root) / partition
    if not partition_dir.is_dir():
        raise FileNotFoundError(f"Partition directory does not exist: {partition_dir}")
    return sorted(path for path in partition_dir.iterdir() if path.is_dir())


def expected_paths(trip_dir: str | Path) -> dict[str, Path]:
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    return {kind: trip_dir / f"{trip_id}.{kind}" for kind in FILE_COLUMNS}


def _read_raw(path: Path, kind: str) -> pd.DataFrame:
    if kind not in FILE_COLUMNS:
        raise KeyError(f"Unknown PathSnap file kind: {kind}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise DataValidationError(f"Empty file: {path}")
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    except Exception as exc:
        raise DataValidationError(f"Could not parse {path}: {exc}") from exc
    columns = FILE_COLUMNS[kind]
    if frame.shape[1] != len(columns):
        raise DataValidationError(
            f"{path} has {frame.shape[1]} columns; expected {len(columns)}"
        )
    frame.columns = columns
    frame.index.name = f"{kind[:-1] if kind.endswith('s') else kind}_row_id"
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[columns].isna().any().any():
        bad = frame[columns].isna().sum().to_dict()
        raise DataValidationError(f"Non-numeric or missing values in {path}: {bad}")
    if kind in {"arcs", "route"}:
        values = frame[columns].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise DataValidationError(f"Non-integer identifiers in {path}")
        frame = frame.astype({column: "int64" for column in columns})
    else:
        frame = frame.astype({column: "float64" for column in columns})
    return frame


def validate_trip(trip: TripData) -> None:
    n_nodes = len(trip.nodes)
    n_arcs = len(trip.arcs)
    endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()
    if (endpoints < 0).any() or (endpoints >= n_nodes).any():
        raise DataValidationError(f"Invalid arc-to-node reference in trip {trip.trip_id}")
    route_ids = trip.route["arc_id"].to_numpy()
    if (route_ids < 0).any() or (route_ids >= n_arcs).any():
        raise DataValidationError(f"Invalid route arc ID in trip {trip.trip_id}")
    for name, frame in (("track", trip.track), ("nodes", trip.nodes)):
        lon = frame["longitude"].to_numpy()
        lat = frame["latitude"].to_numpy()
        if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
            raise DataValidationError(f"Non-finite {name} coordinates in {trip.trip_id}")
        if ((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)).any():
            raise DataValidationError(f"Invalid WGS84 {name} coordinates in {trip.trip_id}")


def load_trip(trip_dir: str | Path, partition: str | None = None) -> TripData:
    trip_dir = Path(trip_dir).expanduser().resolve()
    paths = expected_paths(trip_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing trip files: {missing}")
    trip = TripData(
        trip_id=trip_dir.name,
        partition=partition or trip_dir.parent.name,
        directory=trip_dir,
        track=_read_raw(paths["track"], "track"),
        nodes=_read_raw(paths["nodes"], "nodes"),
        arcs=_read_raw(paths["arcs"], "arcs"),
        route=_read_raw(paths["route"], "route"),
    )
    validate_trip(trip)
    return trip


def iter_trips(data_root: str | Path, partitions: Iterable[str]) -> Iterable[TripData]:
    for partition in partitions:
        for trip_dir in discover_trips(data_root, partition):
            yield load_trip(trip_dir, partition)


def file_inventory(trip_dir: str | Path) -> dict[str, int | str | bool]:
    paths = expected_paths(trip_dir)
    result: dict[str, int | str | bool] = {"trip_id": Path(trip_dir).name}
    for kind, path in paths.items():
        result[f"{kind}_present"] = path.is_file()
        result[f"{kind}_bytes"] = path.stat().st_size if path.is_file() else 0
    return result
