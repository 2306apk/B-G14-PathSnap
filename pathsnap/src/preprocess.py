"""
Trajectory preprocessing and kinematics module for PathSnap.
Cleans raw GPS track data, projects to metres, and computes motion features:
- dt (s), step distance (m), speed (m/s), heading/bearing (degrees 0..360),
- flags large time gaps without dropping rows.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from src.data_io import TripData
from src.geo import determine_local_utm_crs, project_coordinates, build_arc_geometries, TripSpatialIndex


def clean_track(track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans GPS track data:
    - Sorts strictly by timestamp.
    - Drops duplicate timestamps if any (keeping first).
    - Checks for invalid coordinate values (NaNs, out-of-range lon/lat).
    - Resets index.
    """
    df = track_df.copy()
    
    # Assert valid geographic boundaries
    valid_coords = (
        (df["lon"] >= -180.0) & (df["lon"] <= 180.0) &
        (df["lat"] >= -90.0) & (df["lat"] <= 90.0) &
        df["lon"].notnull() & df["lat"].notnull() & df["timestamp"].notnull()
    )
    df = df[valid_coords]
    
    # Sort strictly by timestamp
    df = df.sort_values("timestamp")
    
    # Drop any duplicate timestamps
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    
    return df.reset_index(drop=True)


def compute_kinematics(
    x: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    large_gap_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Computes per-point kinematics in metric coordinates:
    - dt: time delta to previous point in seconds (0 for first point)
    - step_dist: Euclidean distance in metres from previous point (0 for first point)
    - speed: instantaneous speed in m/s (step_dist / dt, 0 if dt == 0)
    - heading: compass bearing in degrees [0, 360) clockwise from True North.
      For stationary points or the first point, heading is imputed from forward or nearest motion.
    - is_large_gap: boolean flag indicating if dt > large_gap_threshold
    """
    n = len(x)
    if n == 0:
        return pd.DataFrame(columns=["dt", "step_dist", "speed", "heading", "is_large_gap"])
        
    dt = np.zeros(n, dtype=np.float64)
    step_dist = np.zeros(n, dtype=np.float64)
    speed = np.zeros(n, dtype=np.float64)
    heading = np.zeros(n, dtype=np.float64)
    is_large_gap = np.zeros(n, dtype=bool)
    
    if n > 1:
        dt[1:] = np.diff(timestamps)
        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.sqrt(dx * dx + dy * dy)
        step_dist[1:] = dist
        
        # Valid dt mask to avoid division by zero
        valid_dt = dt > 1e-4
        speed[valid_dt] = step_dist[valid_dt] / dt[valid_dt]
        
        # Heading computation:
        # standard math angle: theta = arctan2(dy, dx) where 0 is East (+x), 90 is North (+y)
        # compass heading: 0 is North (+y), 90 is East (+x)
        # heading = (450 - theta_deg) % 360 = arctan2(dx, dy) in degrees converted to [0, 360)
        angles_rad = np.arctan2(dx, dy)
        headings_deg = (np.degrees(angles_rad) + 360.0) % 360.0
        heading[1:] = headings_deg
        
        # First point heading imputed from 2nd point
        heading[0] = heading[1]
        
        # For intermediate points where vehicle was stationary (dx==0, dy==0), carry forward last valid heading
        stationary_mask = (step_dist[1:] < 0.1)
        for i in range(1, n):
            if step_dist[i] < 0.1:
                heading[i] = heading[i - 1]
                
        is_large_gap = dt > large_gap_threshold
        # First point is not a gap
        is_large_gap[0] = False
        
    return pd.DataFrame({
        "dt": dt,
        "step_dist": step_dist,
        "speed": speed,
        "heading": heading,
        "is_large_gap": is_large_gap
    })


def preprocess_trip(trip: TripData, large_gap_threshold: float = 5.0) -> Tuple[pd.DataFrame, TripSpatialIndex, np.ndarray]:
    """
    Complete Step 2 preprocessing pipeline for a single trip:
    1. Determines optimal local metric UTM projection.
    2. Cleans track points.
    3. Projects GPS points (lon, lat) -> (x, y) in metres.
    4. Computes kinematic motion attributes.
    5. Builds projected arc geometries, arc lengths, and STRtree spatial index.
    
    Returns:
      processed_track_df: DataFrame containing [lon, lat, timestamp, x, y, dt, step_dist, speed, heading, is_large_gap]
      spatial_index: TripSpatialIndex for candidate generation
      arc_lengths: 1D array of arc lengths (m)
    """
    # 1. Clean raw track
    cleaned_track = clean_track(trip.track)
    
    # 2. Determine local UTM CRS
    mean_lon = cleaned_track["lon"].mean()
    mean_lat = cleaned_track["lat"].mean()
    local_crs = determine_local_utm_crs(mean_lon, mean_lat)
    
    # 3. Project GPS points
    x_gps, y_gps = project_coordinates(
        cleaned_track["lon"].values,
        cleaned_track["lat"].values,
        target_crs=local_crs
    )
    
    # 4. Compute kinematics
    kinematics_df = compute_kinematics(
        x=x_gps,
        y=y_gps,
        timestamps=cleaned_track["timestamp"].values,
        large_gap_threshold=large_gap_threshold
    )
    
    processed_track = cleaned_track.copy()
    processed_track["x"] = x_gps
    processed_track["y"] = y_gps
    processed_track["dt"] = kinematics_df["dt"]
    processed_track["step_dist"] = kinematics_df["step_dist"]
    processed_track["speed"] = kinematics_df["speed"]
    processed_track["heading"] = kinematics_df["heading"]
    processed_track["is_large_gap"] = kinematics_df["is_large_gap"]
    processed_track["crs"] = local_crs
    
    # 5. Build arc geometries & spatial index
    arc_geoms, arc_lengths = build_arc_geometries(trip.nodes, trip.arcs, target_crs=local_crs)
    spatial_index = TripSpatialIndex(arc_geoms, target_crs=local_crs)
    
    return processed_track, spatial_index, arc_lengths
