"""Read raw whitespace-separated track/nodes/arcs/route trip files safely."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray  # lon, lat, seconds; shape (n,3)
    nodes: np.ndarray  # lon, lat; shape (m,2)
    arcs: np.ndarray   # from/to node index; shape (e,2)
    route: np.ndarray | None  # ordered arc ids; NOT per-GPS point

    def validate(self) -> None:
        if self.track.shape[1] != 3 or self.nodes.shape[1] != 2 or self.arcs.shape[1] != 2:
            raise ValueError(f"Malformed raw array in {self.trip_id}")
        if not (len(self.track) and len(self.nodes) and len(self.arcs)):
            raise ValueError(f"Empty trip data in {self.trip_id}")
        if not np.isfinite(self.track).all() or not np.isfinite(self.nodes).all():
            raise ValueError(f"Non-finite coordinates/timestamps in {self.trip_id}")
        if np.any(np.diff(self.track[:, 2]) <= 0):
            raise ValueError(f"Non-increasing timestamps in {self.trip_id}")
        if np.any(self.arcs < 0) or np.any(self.arcs >= len(self.nodes)):
            raise ValueError(f"Arcs reference nonexistent node in {self.trip_id}")
        if self.route is not None and (np.any(self.route < 0) or np.any(self.route >= len(self.arcs))):
            raise ValueError(f"Reference route contains invalid arc IDs in {self.trip_id}")
        if (np.abs(self.track[:, 1]) > 90).any() or (np.abs(self.track[:, 0]) > 180).any():
            raise ValueError(f"Implausible GPS coordinates in {self.trip_id}")


def load_trip(root: str | Path, partition: str, trip_id: str, include_reference: bool = True) -> Trip:
    if partition not in {"train", "student_test", "evaluator_test"}:
        raise ValueError("Unknown partition")
    if not (len(trip_id) == 8 and trip_id.isdigit()):
        raise ValueError("Trip ID must be exactly eight digits")
    base = Path(root) / partition / trip_id
    if not base.is_dir():
        raise FileNotFoundError(base)
    def read(ext: str, dtype) -> np.ndarray:
        return np.loadtxt(base / f"{trip_id}.{ext}", dtype=dtype, ndmin=2)
    track = read("track", np.float64)
    nodes = read("nodes", np.float64)
    arcs = read("arcs", np.int32)
    route_path = base / f"{trip_id}.route"
    route = np.loadtxt(route_path, dtype=np.int32, ndmin=1).ravel() if include_reference and route_path.exists() else None
    trip = Trip(trip_id=trip_id, partition=partition, track=track, nodes=nodes, arcs=arcs, route=route)
    trip.validate()
    return trip


def list_trips(root: str | Path, partition: str) -> list[str]:
    parent = Path(root) / partition
    if not parent.is_dir():
        return []
    return sorted(p.name for p in parent.iterdir() if p.is_dir() and len(p.name) == 8 and p.name.isdigit())


def meter_projection(lonlat: np.ndarray, origin_lonlat: np.ndarray) -> np.ndarray:
    """Local equirectangular metres, suitable for per-trip small regional networks.

    Each trip is reprojected independently, so degrees are never treated as metres.
    Very large / polar networks would require a proper projected CRS.
    """
    ref_lon, ref_lat = np.asarray(origin_lonlat, dtype=float)
    x = (lonlat[..., 0] - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
    y = (lonlat[..., 1] - ref_lat) * 110574.0
    return np.stack((x, y), axis=-1)


def project_trip(trip: Trip) -> tuple[np.ndarray, np.ndarray]:
    origin = np.median(trip.track[:, :2], axis=0)
    return meter_projection(trip.track[:, :2], origin), meter_projection(trip.nodes, origin)
