# PathSnap: Data Quality Notes & Decisions (Step 1 EDA)

## 1. Dataset Inventory & Partitions
- **Partitions Available**: `train` (80 trips) and `student_test` (10 trips).
- **Academic Integrity Check**:
  - `FORBIDDEN_PARTITIONS = {"evaluator_test", "evaluator", "test_evaluator"}` is strictly enforced in `src/data_io.py`. Attempting to load `evaluator_test` raises a `PermissionError`.
  - Trips are strictly partitioned by folder/trip ID. No cross-partition point mixing occurs.
- **Trip Point Counts**:
  - Total GPS points: 227,577 (Train: 208,417, Student Test: 19,160).
  - Minimum points in a trip: 262.
  - Median points per trip: 1,496.5.
  - Maximum points in a trip: 19,313.
  - Mean points per trip: 2,528.6.

## 2. Road Network Properties (Per-Trip vs. Shared)
- **Finding**: Each trip contains its own unique `.nodes` and `.arcs` file with varying sizes (e.g., node counts range from 898 to 88,809).
- **Conclusion**: The network is **per-trip** (a sub-graph/bounding extract around the trip's trajectory), not a single global monolithic graph.
- **Isolated nodes**: A small percentage of nodes in each graph do not appear in arcs; these are retained without affecting arc-based spatial index generation.

## 3. Timestamp Integrity & 1 Hz Verification
- **Ordering**: All trips are strictly chronological ($t_{i+1} \ge t_i$). No sorting corrections or timestamp reversals were found.
- **Duplicates**: 0 duplicate timestamps across all 90 trips.
- **1-Second Gaps**:
  - Total intervals: 227,487.
  - Exactly 1-second intervals: 219,939 (**96.68%**, closely matching the expected ~97%).
  - Gaps > 1s: 7,548 intervals (3.32%). Most non-1s intervals are small tunnel/signal loss gaps (2s–10s), with rare long stops.
  - **Decision**: In Step 2, we will flag large time gaps (e.g. $\Delta t > 5$ s or user-defined threshold) in features rather than dropping rows, preserving sequence continuity.

## 4. Geographic Extent & Critical CRS Finding
- **Initial Assumption**: The problem description mentioned "data looks like the Netherlands".
- **Empirical Finding**: 
  - Longitude spans `[-100.42°, 137.24°]`
  - Latitude spans `[-34.64°, +61.70°]`
  - Real trips originate from diverse locations worldwide (e.g., Europe, North America, East Asia, South Africa, South America).
- **Impact on CRS Choice**:
  - A single regional CRS like EPSG:28992 (Amersfoort / RD New, valid only for the Netherlands) would completely corrupt coordinates for trips in North America or Asia.
  - Even a single UTM zone (e.g., UTM 31N) cannot span trips worldwide.
  - **Decision / Proposal**: Dynamically project each trip into its local optimal UTM projection (or compute per-trip UTM CRS via `pyproj.database.query_utm_crs_info` / `trip_lon_mean, trip_lat_mean`), ensuring metre-accurate metric distances everywhere without distortion.

## 5. Ground-Truth `.route` Integrity
- **Arc ID Validity**: 100% of arc IDs in `.route` reference valid 0-based indices in the trip's corresponding `.arcs` table.
- **Consecutive Connectivity**:
  - Across all 90 trips, the consecutive connectivity (where $\text{to\_node}(arc_i) == \text{from\_node}(arc_{i+1})$) is **99.94%**.
  - 79 out of 90 trips are 100% strictly contiguous paths.
  - 11 trips have minor transitions where short connector arcs or U-turns may be omitted in the raw trace.
  - **Decision**: The `.route` is reference information providing the true traversed corridor. In Step 3, we will use a robust projection & monotonic progression matcher along the route sequence to reliably assign reference arc labels to GPS points.
