import numpy as np

from pathsnap.geo import build_arc_geometries, project_trip, select_local_utm
from pathsnap.graph import graph_metrics, route_connectivity


def test_utm_selection_northern_and_southern():
    assert select_local_utm(np.array([8.0]), np.array([50.0])).epsg == 32632
    assert select_local_utm(np.array([18.0]), np.array([-34.0])).epsg == 32734


def test_projection_and_geometry_creation(toy_trip):
    projected = project_trip(toy_trip)
    assert projected.metadata.epsg == 32632
    assert projected.track_xy.shape == (3, 2)
    assert len(projected.arc_geometries) == 4
    assert 30 < projected.arc_lengths_m[0] < 40
    assert np.isfinite(projected.arc_bearings_deg).all()


def test_invalid_geometry_reference_rejected():
    import pandas as pd
    import pytest
    with pytest.raises(ValueError, match="missing node"):
        build_arc_geometries(np.array([[0.0, 0.0]]), pd.DataFrame({"from_node_id": [0], "to_node_id": [1]}))


def test_graph_and_reference_connectivity(toy_trip):
    result = graph_metrics(toy_trip.arcs, len(toy_trip.nodes), toy_trip.route.arc_id.to_numpy())
    assert result["weak_component_count"] == 1
    assert result["route_connectivity_rate"] == 1.0
    disconnected = route_connectivity(toy_trip.arcs, np.array([0, 2]))
    assert disconnected["route_connectivity_rate"] == 0.0
