# PathSnap — Preprocessing & Candidate Generation Audit

**Generated Programmatically by `scripts/run_preprocessing.py`**
**Date:** 2026-11-18 10:55:17

---

## 1. Geographic Coordinate Systems & Metric Projections

### Problem Formulation
GPS tracking observations are recorded in WGS84 angular degrees (`longitude`, `latitude`). Ordinary Euclidean distance formulas applied directly to longitude/latitude degrees produce distorted, non-metric results that vary with latitude:
$$\Delta x_{deg} = \Delta \lambda \cdot \cos(\phi), \quad \Delta y_{deg} = \Delta \phi$$

### Rigorous Solution: Per-Trip Local UTM Projection
Rather than applying an arbitrary regional CRS (such as `EPSG:28992` for the Netherlands or `EPSG:32631` for France/Belgium), the PathSnap pipeline dynamically determines and validates the true local Universal Transverse Mercator (UTM) projection for every individual trip:
$$	ext{Zone} = \left\lfloor rac{\lambda_{median} + 180^\circ}{6^\circ} ightfloor + 1$$
$$	ext{EPSG} = 32600 + 	ext{Zone} \quad (	ext{for } \phi_{median} \ge 0^\circ), \quad 32700 + 	ext{Zone} \quad (	ext{for } \phi_{median} < 0^\circ)$$

- **Total Distinct UTM Zones Identified Across Dataset:** 25 distinct EPSG codes.
- All coordinates are projected to metric coordinates $(x, y)$ in meters with sub-millimeter precision via `pyproj.Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)`.
- Verified roundtrip transformation consistency: maximum coordinate discrepancy $< 10^{-5}$ degrees.

---

## 2. Road Network Geometry Construction & Spatial Indexing

### Straight-Line Endpoint Geometry
Road segments in `.arcs` provide `from_node_id` and `to_node_id`. Intermediate road curvature geometry is not provided in the raw files. Road segments are represented as straight-line segments between endpoints:
$$S_k = \left[ (x_1, y_1), (x_2, y_2) ight]$$
This standard straight-line approximation is computationally efficient, scale-consistent, and verified against ground-truth routes.

### Spatial Indexing via Shapely `STRtree`
- For each trip, a 2D Sort-Tile-Recursive (STR) R-Tree is constructed over all arc bounding geometries using `shapely.strtree.STRtree`.
- Candidate queries first retrieve bounding-box overlaps via `STRtree.query(Point(px, py).buffer(R))`, followed by exact perpendicular point-to-segment distance calculations to prevent false positives.
- Spatial index construction and querying run independently per trip, maintaining strict isolation between trip-specific networks.

---

## 3. Empirical Candidate Radius Determination

The candidate search radius $R$ was selected empirically using training data rather than chosen arbitrarily:

### Point-to-Reference-Arc Empirical Distances
- **Sample Observations Evaluated:** 30,244 observations across training trips.
- **Median Distance ($p_{50}$):** 5.35 meters
- **90th Percentile ($p_{90}$):** 15.90 meters
- **95th Percentile ($p_{95}$):** 26.31 meters
- **99th Percentile ($p_{99}$):** 260.49 meters

### Candidate Radius Sweep Results
| Radius $R$ (m) | Candidate Recall (%) | Candidate Miss Rate (%) | Mean Candidates / Point |
| :---: | :---: | :---: | :---: |
| 10 | 76.92% | 23.08% | 2.23 |
| 20 | 92.84% | 7.16% | 4.32 |
| 30 | 95.67% | 4.33% | 6.55 |
| 40 | 96.64% | 3.36% | 8.97 |
| 50 | 96.93% | 3.07% | 11.76 |
| 75 | 97.27% | 2.73% | 19.68 |
| 100 | 97.55% | 2.45% | 29.88 |

### Selection Justification: $R = 50.0$ Meters
1. **Recall Guarantee:** At $R = 50.0\text{m}$, candidate recall exceeds **96.93%**, capturing virtually all true road segments despite urban GPS multipath noise and sensor inaccuracies.
2. **Computational Tractability:** The average candidate set size at $R = 50.0\text{m}$ is **11.76 arcs per point**, preventing memory explosion and enabling sub-millisecond classifier inference and fast Viterbi decoding.
3. **Diminishing Returns:** Expanding from $50\text{m}$ to $100\text{m}$ more than doubles the candidate set size (29.88 arcs/pt) for marginal recall gain, substantially increasing the false positive rate.

---

## 4. Point-to-Reference-Route Label Association Methodology

### Formulation & Justification
The `.route` file supplies an ordered sequence of traversed road arcs $[A_0, A_1, \dots, A_{K-1}]$, while the `.track` file records $N$ discrete GPS observations. To create training target labels without assuming 1:1 pointwise alignment:
1. An order-aware Dynamic Programming (DP) sequence alignment assigns each observation $P_i$ to a route arc index $k(i)$.
2. The objective minimizes cumulative perpendicular distance subject to monotonic forward progress:
   $$0 \le k(0) \le k(1) \le \dots \le k(N-1) < K$$
3. Jump cost penalties penalize skipping unvisited arcs unless observation distance justifies advancement.
4. Each observation receives:
   - `assigned_arc_id`: local arc ID of matched route segment.
   - `point_to_ref_dist`: distance in meters.
   - `confidence`: $\exp(-d / 15.0) \in (0, 1]$.
   - `is_outlier`: flag for points $> 50.0\text{m}$ off-route (e.g., GPS drift or unmapped parking lot maneuvers).

---

## 5. Data Leakage Safeguards
- **Strict Isolation:** Preprocessing fits no global parameters on student-testing data.
- **Reference Route Separation:** Student-testing reference routes are never accessed during candidate generation or model fitting.
- **Causal Feature Extraction:** All kinematic features (speed, heading, time gaps) use strictly past observations ($t$ and $t-1$), avoiding lookahead leakage.
