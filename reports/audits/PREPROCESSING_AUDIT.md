# Preprocessing Audit

Status: **implemented and executed** for 80 training and 10 student-test trips.

## Projection and geometry

- WGS84 longitude/latitude is validated before projection.
- Each trip uses the UTM zone selected from its median combined GPS/network coordinate; exact EPSG metadata is in `outputs/csv/trip_crs.csv`.
- `pyproj.Transformer(..., always_xy=True)` preserves longitude/latitude order.
- Every arc is represented by a straight metric line between its from/to nodes because intermediate geometry is unavailable.
- Each trip has its own arc array and STRtree; arc IDs never cross trip boundaries.

## Reference-route labelling

- A monotonic dynamic program minimizes projected point-to-route-arc distance.
- Staying on an arc is free; skipping a route position costs 5 m; soft start/end omission costs 2 m per position.
- Outputs retain route position, assigned distance, distance margin, confidence, and an uncertainty flag.
- These labels are reference-derived and are not claimed as independently verified physical ground truth.

## Candidate radius

- All-training assigned distances: p95=142.096442 m; p99=205.822684 m.
- Trips with extreme reference mismatch are detected from training only when their median distance exceeds Q3 + 5*IQR (11.861302 m). Excluded from radius calibration only: ['00000033', '00000054'].
- Robust calibration p95=18.842315 m; p99=69.828885 m across 78 trips.
- Selected radius=70.0 m (robust training p99 rounded upward to 5 m).
- Calibration recall=99.002510%; miss rate=0.997490%. Across all 80 trips, recall=89.595858%; miss rate=10.404142%.
- Excluded trips are still processed and included in coverage reporting; this rule only prevents anomalous reference geometry from inflating every candidate set.
- Student-test data and labels were not used to choose the radius.

## Leakage safeguards

Train and student-test labels are stored in separate directories. Radius fitting reads training labels only. Prediction functions do not accept reference-route data.

## Executed command

`python scripts/run_preprocessing.py --data-root "../Student folder"`
