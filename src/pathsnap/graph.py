"""Road-network quality and reference-route connectivity metrics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(arcs: pd.DataFrame, n_nodes: int) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(
        (int(row.from_node_id), int(row.to_node_id), {"arc_id": int(arc_id)})
        for arc_id, row in arcs.iterrows()
    )
    return graph


def route_connectivity(arcs: pd.DataFrame, route_arc_ids: np.ndarray) -> dict[str, int | float]:
    route_arc_ids = np.asarray(route_arc_ids, dtype=np.int64)
    if len(route_arc_ids) < 2:
        return {"route_transitions": 0, "connected_route_transitions": 0, "route_connectivity_rate": 1.0}
    selected = arcs.iloc[route_arc_ids]
    connected = (
        selected["to_node_id"].to_numpy()[:-1]
        == selected["from_node_id"].to_numpy()[1:]
    )
    return {
        "route_transitions": int(len(connected)),
        "connected_route_transitions": int(connected.sum()),
        "route_connectivity_rate": float(connected.mean()),
    }


def graph_metrics(arcs: pd.DataFrame, n_nodes: int, route_arc_ids: np.ndarray) -> dict[str, int | float]:
    graph = build_graph(arcs, n_nodes)
    in_degree = np.fromiter((degree for _, degree in graph.in_degree()), dtype=np.int64)
    out_degree = np.fromiter((degree for _, degree in graph.out_degree()), dtype=np.int64)
    weak_sizes = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    strong_sizes = sorted((len(c) for c in nx.strongly_connected_components(graph)), reverse=True)
    result: dict[str, int | float] = {
        "node_count": int(n_nodes),
        "arc_count": int(len(arcs)),
        "isolated_nodes": int(((in_degree + out_degree) == 0).sum()),
        "weak_component_count": int(len(weak_sizes)),
        "largest_weak_component_nodes": int(weak_sizes[0] if weak_sizes else 0),
        "strong_component_count": int(len(strong_sizes)),
        "largest_strong_component_nodes": int(strong_sizes[0] if strong_sizes else 0),
        "mean_in_degree": float(in_degree.mean()) if n_nodes else 0.0,
        "max_in_degree": int(in_degree.max()) if n_nodes else 0,
        "mean_out_degree": float(out_degree.mean()) if n_nodes else 0.0,
        "max_out_degree": int(out_degree.max()) if n_nodes else 0,
    }
    result.update(route_connectivity(arcs, route_arc_ids))
    return result
