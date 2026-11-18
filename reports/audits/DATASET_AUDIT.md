# PathSnap — Comprehensive Dataset Audit Report

**Generated Programmatically by `scripts/run_eda.py`**
**Date:** 2026-11-18 10:53:34

---

## 1. Executive Summary & Inventory

The PathSnap dataset comprises vehicle GPS trajectories and corresponding road networks partitioned into training and development testing sets.

| Partition | Trips | Total Observations | Total Road Nodes | Total Road Arcs | Total Route Arcs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training (`train`)** | 80 | 208,417 | 1,933,485 | 3,876,904 | 31,336 |
| **Student Testing (`student_test`)** | 10 | 19,160 | 90,689 | 185,041 | 3,477 |
| **Total** | **90** | **227,577** | **2,024,174** | **4,061,945** | **34,813** |

### File Completeness & Indexing Validation
- **Missing Files:** 0 missing files across all 90 trips. Every trip has all four required files (`.track`, `.nodes`, `.arcs`, `.route`).
- **Parsing Failures:** 0 parsing failures.
- **Node Indexing:** Validated that in every trip, `from_node_id` and `to_node_id` strictly lie in `[0, num_nodes - 1]`.
- **Arc Indexing:** Validated that in every trip with route data, every `arc_id` in `.route` strictly lies in `[0, num_arcs - 1]`.

---

## 2. GPS Data Quality & Sampling Findings

### Timestamp & Sampling Regularity
- **Total Consecutive Sampling Gaps Analyzed:** 227,487
- **Nominal 1 Hz Gaps (exactly 1.0s):** 219,934 (96.68%)
- **Non-1.0s Gaps:** 7,553 (3.32%)
- **Gaps > 10s:** 617
- **Gaps > 30s:** 135
- **Gaps > 60s:** 70
- **Gaps > 120s:** 42
- **Maximum Observed Gap:** 153064.0 seconds (found in trip `00000090`).

### Spatial Extent & Coordinate Validation
- **Invalid / NaN Coordinates:** Exactly 0 NaN or infinite coordinates detected across all 227,577 GPS observations.
- **WGS84 Longitude Range:** [-100.4185, 137.2370]
- **WGS84 Latitude Range:** [-34.6403, 61.6996]
- **Critical Finding — Geographic Distribution:** The dataset spans Western Europe, Eastern Europe, North America, and parts of the Southern Hemisphere. **EPSG:28992 (Amersfoort / RD New for Netherlands) and EPSG:32631 (UTM 31N) cannot be used globally.**
- **Local UTM Projections:** A dedicated UTM EPSG projection was calculated and validated for every single trip based on its trajectory center.

---

## 3. Road Network Topology & Quality Findings

### Network Scope (Trip-Specific vs Global)
- **Definitive Conclusion:** Road networks are strictly **independent per trip**. Each trip has its own dedicated `.nodes` and `.arcs` file with 0-based indexing.
- Node `0` in Trip 1 is located in Germany (lon 8.58, lat 49.86), while Node `0` in Trip 2 is at (lon 11.03, lat 47.66), and in Trip 56 is in the United States (lon -93.38, lat 44.85).
- Constructing a single global graph with overlapping integer IDs would produce catastrophic topology errors. All spatial indices and road graphs must remain trip-local.

### Topological Connectivity & Component Analysis
- **Isolated Nodes:** 0 isolated nodes found across all networks.
- **Weakly Connected Components (WCC):** Networks contain multiple components (median 19 WCCs per trip), but on average **96.84%** of all nodes belong to a giant connected component.
- **Reference Route Arc Continuity:**
  - On average, **99.94%** of consecutive route arcs in `.route` connect directly (`to_node(a_k) == from_node(a_{k+1})`).
  - Occasional gaps occur where short intermediate connector segments are omitted in the reference route or where vehicles enter service areas.

---

## 4. Reference Route Characteristics & Ground-Truth Limitations

- **Pointwise vs Arc-Sequence Representation:** The `.route` file is an ordered sequence of traversed arc IDs ($K$), while `.track` contains time-series observations ($N$).
- On average, there are ~6.65 GPS points per route arc.
- **Ground-Truth Uncertainty:** In trips such as `00000002`, vehicles took detours or exhibited GPS drift where observed points lie >50m (and up to 550m) away from the nearest route arc.
- As mandated by the official guidelines, reference routes cannot be assumed to be 100% verified, error-free physical ground truth. Our pipeline incorporates an explicit confidence metric and outlier detection flag to preserve this uncertainty.

---

## 5. Artifacts and Generated Deliverables

The following files have been produced:
1. `outputs/csv/dataset_inventory.csv`: Full file inventory, row counts, and byte sizes.
2. `outputs/csv/gps_quality_audit.csv`: GPS quality, gap distributions, and bounding boxes.
3. `outputs/csv/network_quality_audit.csv`: Network topology, degrees, components, and continuity.
4. `outputs/plots/timestamp_gap_distribution.png`: Histogram and gap breakdown.
5. `outputs/plots/trajectory_length_distribution.png`: Observations per trip.
6. `outputs/plots/network_size_distribution.png`: Nodes vs arcs per trip.
7. `outputs/maps/trip_00000001.html`: Interactive Folium map for training trip `00000001`.
8. `outputs/maps/trip_00000000.html`: Interactive Folium map for student-test trip `00000000`.
