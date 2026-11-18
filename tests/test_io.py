from pathlib import Path

import pytest

from pathsnap.io import DataValidationError, load_trip


def test_file_parsing_and_row_indices(toy_trip):
    assert list(toy_trip.track.columns) == ["longitude", "latitude", "timestamp_s"]
    assert toy_trip.track.index.name == "track_row_id"
    assert toy_trip.arcs.index.tolist() == [0, 1, 2, 3]
    assert toy_trip.route.arc_id.tolist() == [0, 1, 2]


def test_missing_file_is_rejected(tmp_path: Path):
    trip = tmp_path / "x"; trip.mkdir()
    with pytest.raises(FileNotFoundError):
        load_trip(trip)


def test_wrong_column_count_is_rejected(tmp_path: Path):
    trip = tmp_path / "00000001"; trip.mkdir()
    (trip / "00000001.track").write_text("8 50\n")
    (trip / "00000001.nodes").write_text("8 50\n")
    (trip / "00000001.arcs").write_text("0 0\n")
    (trip / "00000001.route").write_text("0\n")
    with pytest.raises(DataValidationError, match="columns"):
        load_trip(trip)


def test_invalid_node_reference_is_rejected(tmp_path: Path):
    trip = tmp_path / "00000001"; trip.mkdir()
    (trip / "00000001.track").write_text("8 50 1\n")
    (trip / "00000001.nodes").write_text("8 50\n")
    (trip / "00000001.arcs").write_text("0 2\n")
    (trip / "00000001.route").write_text("0\n")
    with pytest.raises(DataValidationError, match="arc-to-node"):
        load_trip(trip)


def test_invalid_route_arc_is_rejected(tmp_path: Path):
    trip = tmp_path / "00000001"; trip.mkdir()
    (trip / "00000001.track").write_text("8 50 1\n")
    (trip / "00000001.nodes").write_text("8 50\n")
    (trip / "00000001.arcs").write_text("0 0\n")
    (trip / "00000001.route").write_text("3\n")
    with pytest.raises(DataValidationError, match="route arc"):
        load_trip(trip)
