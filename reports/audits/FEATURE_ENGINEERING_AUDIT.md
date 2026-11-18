# Feature Engineering Audit

Status: **implemented and executed**. Candidate radius: 70.0 m.

## Row and label definition

One row represents one trip-local `(GPS observation, candidate arc)` pair. `target=1` exactly when candidate `arc_id` equals the monotonic reference assignment; all other candidates are 0. Observations whose assigned arc is absent remain explicit misses and create no artificial positive.

Training candidate rows: 2,817,917. Student-test candidate rows: 324,291.
Training reference miss rate: 10.404142%. Student-test reference miss rate: 0.109603%.
Training no-candidate rate: 7.334335%. Student-test no-candidate rate: 0.000000%.

## Features (causal; no future observations)

| Feature | Definition | Unit / missing handling |
|---|---|---|
| point_arc_distance_m | perpendicular Shapely point-to-line distance | m |
| from_node_distance_m | Euclidean metric distance to from-node | m |
| to_node_distance_m | Euclidean metric distance to to-node | m |
| projection_position | line projection distance / arc length, clipped [0,1] | ratio; zero for zero-length arc |
| vehicle_heading_deg | `atan2(delta_easting, delta_northing)` from previous GPS point | degrees; first point NaN |
| arc_bearing_deg | same bearing formula from arc endpoints | degrees |
| heading_difference_deg | `abs((vehicle-arc+180) mod 360 - 180)` | degrees; first point NaN |
| reverse_direction | 1 when heading difference > 90 | binary; first point 0 after comparison |
| vehicle_speed_mps | previous-to-current metric displacement / positive time gap | m/s; first/non-positive gap NaN |
| time_gap_s | current timestamp minus previous timestamp | s; first point NaN |
| arc_length_m | straight endpoint geometry length | m |
| candidate_rank | ascending point-to-arc distance, deterministic arc-ID tie break | ordinal |
| candidate_count | number of candidates for the observation | count |
| nearest_alternative_distance_m | second-nearest distance for rank 1; nearest distance otherwise | m; NaN when only one candidate |

Missing numeric values are retained in Parquet and imputed from training-fold medians inside model pipelines. No scaler or imputer is fit on student-test data.

## Executed command

`python scripts/build_features.py --data-root "../Student folder"`
