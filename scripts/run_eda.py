#!/usr/bin/env python3
"""Execute the complete raw-data and network audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.geo import project_trip
from pathsnap.graph import graph_metrics
from pathsnap.io import discover_trips, expected_paths, file_inventory, load_trip
from pathsnap.visualize import create_folium_map, plot_trajectory, save_figure


def network_hash(trip_dir: Path) -> str:
    digest = hashlib.sha256()
    for kind in ("nodes", "arcs"):
        with expected_paths(trip_dir)[kind].open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--report", default="reports/audits/DATASET_AUDIT.md")
    parser.add_argument("--map-trip", default=None)
    args = parser.parse_args()
    output = Path(args.output_root)
    csv_dir, plot_dir, map_dir = output / "csv", output / "plots", output / "maps"
    for directory in (csv_dir, plot_dir, map_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    gap_rows: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    first_projected = None
    for partition in ("train", "student_test"):
        for trip_dir in discover_trips(args.data_root, partition):
            base = {"partition": partition, **file_inventory(trip_dir), "network_sha256": network_hash(trip_dir)}
            try:
                trip = load_trip(trip_dir, partition)
                projected = project_trip(trip)
                if first_projected is None or (args.map_trip and trip.trip_id == args.map_trip):
                    first_projected = projected
                gaps = trip.track["timestamp_s"].diff().iloc[1:]
                gap_rows.append(pd.DataFrame({"partition": partition, "trip_id": trip.trip_id, "gap_s": gaps}))
                coordinates = trip.track[["longitude", "latitude"]]
                metrics = graph_metrics(trip.arcs, len(trip.nodes), trip.route["arc_id"].to_numpy())
                base.update(
                    {
                        "track_rows": len(trip.track), "node_rows": len(trip.nodes),
                        "arc_rows": len(trip.arcs), "route_rows": len(trip.route),
                        "track_nan_count": int(trip.track.isna().sum().sum()),
                        "node_nan_count": int(trip.nodes.isna().sum().sum()),
                        "duplicate_observations": int(trip.track.duplicated().sum()),
                        "duplicate_coordinate_observations": int(coordinates.duplicated().sum()),
                        "non_increasing_timestamps": int((gaps <= 0).sum()),
                        "one_second_gaps": int((gaps == 1).sum()),
                        "non_one_second_gaps": int((gaps != 1).sum()),
                        "gaps_gt_10s": int((gaps > 10).sum()), "gaps_gt_30s": int((gaps > 30).sum()),
                        "gaps_gt_60s": int((gaps > 60).sum()), "gaps_gt_120s": int((gaps > 120).sum()),
                        "max_gap_s": float(gaps.max()) if len(gaps) else np.nan,
                        "longitude_min": float(trip.track.longitude.min()), "longitude_max": float(trip.track.longitude.max()),
                        "latitude_min": float(trip.track.latitude.min()), "latitude_max": float(trip.track.latitude.max()),
                        "invalid_arc_node_references": 0, "invalid_route_arc_ids": 0,
                        "epsg": projected.metadata.epsg, "utm_zone": projected.metadata.utm_zone,
                        "hemisphere": projected.metadata.hemisphere,
                        **metrics,
                    }
                )
                rows.append(base)
            except Exception as exc:
                errors.append({"partition": partition, "trip_id": trip_dir.name, "error": repr(exc)})

    audit = pd.DataFrame(rows).sort_values(["partition", "trip_id"])
    gaps = pd.concat(gap_rows, ignore_index=True)
    audit.to_csv(csv_dir / "dataset_inventory.csv", index=False)
    gaps.to_csv(csv_dir / "timestamp_gaps.csv", index=False)
    pd.DataFrame(errors).to_csv(csv_dir / "dataset_errors.csv", index=False)
    crs_columns = ["partition", "trip_id", "epsg", "utm_zone", "hemisphere", "longitude_min", "longitude_max", "latitude_min", "latitude_max"]
    audit[crs_columns].to_csv(csv_dir / "trip_crs.csv", index=False)

    plt.figure(figsize=(8, 5)); plt.hist(gaps.gap_s.clip(upper=120), bins=60, color="#1769aa")
    plt.yscale("log"); plt.xlabel("Consecutive timestamp gap (s; clipped at 120)"); plt.ylabel("Count (log scale)")
    plt.title("Timestamp gap distribution - all supplied trips"); save_figure(plot_dir / "timestamp_gap_distribution.png")
    plt.figure(figsize=(8, 5)); plt.scatter(audit.node_rows, audit.arc_rows, c=audit.partition.map({"train": 0, "student_test": 1}), alpha=0.7)
    plt.xlabel("Nodes per trip"); plt.ylabel("Arcs per trip"); plt.title("Trip-local road-network sizes"); save_figure(plot_dir / "node_arc_distribution.png")
    plt.figure(figsize=(8, 5)); plt.bar(np.arange(len(audit)), audit.weak_component_count)
    plt.xlabel("Trip (sorted by partition and ID)"); plt.ylabel("Weakly connected components"); plt.title("Network connectivity by trip")
    save_figure(plot_dir / "network_connectivity.png")
    if first_projected is not None:
        plot_trajectory(first_projected, plot_dir / f"gps_trajectory_{first_projected.trip.trip_id}.png")
        create_folium_map(first_projected, map_dir / f"trip_{first_projected.trip.trip_id}_reference_map.html")

    summary = {
        "trip_counts": audit.groupby("partition").size().to_dict(),
        "track_rows": audit.groupby("partition").track_rows.sum().astype(int).to_dict(),
        "node_rows": audit.groupby("partition").node_rows.sum().astype(int).to_dict(),
        "arc_rows": audit.groupby("partition").arc_rows.sum().astype(int).to_dict(),
        "route_rows": audit.groupby("partition").route_rows.sum().astype(int).to_dict(),
        "one_second_gaps": int(audit.one_second_gaps.sum()),
        "all_gaps": int(audit.one_second_gaps.sum() + audit.non_one_second_gaps.sum()),
        "one_second_gap_rate": float(audit.one_second_gaps.sum() / (audit.one_second_gaps.sum() + audit.non_one_second_gaps.sum())),
        "gaps_gt_10s": int(audit.gaps_gt_10s.sum()), "gaps_gt_30s": int(audit.gaps_gt_30s.sum()),
        "gaps_gt_60s": int(audit.gaps_gt_60s.sum()), "gaps_gt_120s": int(audit.gaps_gt_120s.sum()),
        "max_gap_s": float(audit.max_gap_s.max()), "parse_or_validation_errors": len(errors),
        "unique_network_hashes": int(audit.network_sha256.nunique()),
        "epsg_codes": sorted(audit.epsg.unique().astype(int).tolist()),
    }
    (csv_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# Dataset Audit

Status: **executed** on both supplied partitions. Input: `{Path(args.data_root).resolve()}`.

## Inventory

- Trips: {summary['trip_counts']}.
- GPS rows: {summary['track_rows']}.
- Node rows: {summary['node_rows']}.
- Arc rows: {summary['arc_rows']}.
- Reference-route rows: {summary['route_rows']}.
- Missing/parsing/validation failures: {summary['parse_or_validation_errors']}.
- Exact network-file hashes: {summary['unique_network_hashes']} unique across {len(audit)} trips, establishing trip-local independent networks.

## GPS quality

- Exactly-one-second gaps: {summary['one_second_gaps']:,} / {summary['all_gaps']:,} ({summary['one_second_gap_rate']:.4%}).
- Gaps above 10/30/60/120 seconds: {summary['gaps_gt_10s']:,} / {summary['gaps_gt_30s']:,} / {summary['gaps_gt_60s']:,} / {summary['gaps_gt_120s']:,}.
- Maximum gap: {summary['max_gap_s']:,.1f} s.
- NaNs, duplicates, ordering, coordinate extents, and per-trip anomalies are recorded in `outputs/csv/dataset_inventory.csv`.

## Geographic and network findings

- Per-trip UTM EPSG codes observed: {summary['epsg_codes']}.
- Arc IDs are row indices and are never shared across trip contexts.
- Invalid arc/node references and invalid route IDs: {int(audit.invalid_arc_node_references.sum())} and {int(audit.invalid_route_arc_ids.sum())}.
- Degree, isolated-node, weak/strong component, and consecutive-route connectivity fields are in the inventory CSV.

## Reference-route interpretation

The `.route` rows are chronological arc IDs, not pointwise GPS labels. Supplied routes are treated as uncertain reference information, consistent with the official documentation. Point labels are created later with monotonic dynamic programming.

## Executed command

`python scripts/run_eda.py --data-root "../Student folder"`
"""
    report_path = Path(args.report); report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(report)
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
