# Dataset Audit

Status: **executed** on both supplied partitions. Input: `/Users/apoorva_kogta/Documents/ML-Hackathon-Codex/Student folder`.

## Inventory

- Trips: {'student_test': 10, 'train': 80}.
- GPS rows: {'student_test': 19160, 'train': 208417}.
- Node rows: {'student_test': 90689, 'train': 1933485}.
- Arc rows: {'student_test': 185041, 'train': 3876904}.
- Reference-route rows: {'student_test': 3477, 'train': 31336}.
- Missing/parsing/validation failures: 0.
- Exact network-file hashes: 90 unique across 90 trips, establishing trip-local independent networks.

## GPS quality

- Exactly-one-second gaps: 219,934 / 227,487 (96.6798%).
- Gaps above 10/30/60/120 seconds: 617 / 135 / 70 / 42.
- Maximum gap: 153,064.0 s.
- NaNs, duplicates, ordering, coordinate extents, and per-trip anomalies are recorded in `outputs/csv/dataset_inventory.csv`.

## Geographic and network findings

- Per-trip UTM EPSG codes observed: [32614, 32615, 32616, 32617, 32622, 32629, 32630, 32631, 32632, 32633, 32634, 32635, 32636, 32637, 32638, 32639, 32640, 32641, 32643, 32648, 32651, 32653, 32721, 32722, 32735].
- Arc IDs are row indices and are never shared across trip contexts.
- Invalid arc/node references and invalid route IDs: 0 and 0.
- Degree, isolated-node, weak/strong component, and consecutive-route connectivity fields are in the inventory CSV.

## Reference-route interpretation

The `.route` rows are chronological arc IDs, not pointwise GPS labels. Supplied routes are treated as uncertain reference information, consistent with the official documentation. Point labels are created later with monotonic dynamic programming.

## Executed command

`python scripts/run_eda.py --data-root ../Student\ folder`
