from pathlib import Path

import pytest

from pathsnap.io import load_trip


@pytest.fixture
def toy_trip(tmp_path: Path):
    trip_dir = tmp_path / "train" / "00000001"
    trip_dir.mkdir(parents=True)
    (trip_dir / "00000001.track").write_text("8.0000 50.0000 1\n8.0005 50.0000 2\n8.0010 50.0000 3\n")
    (trip_dir / "00000001.nodes").write_text("8.0000 50.0000\n8.0005 50.0000\n8.0010 50.0000\n8.0015 50.0000\n")
    (trip_dir / "00000001.arcs").write_text("0 1\n1 2\n2 3\n1 0\n")
    (trip_dir / "00000001.route").write_text("0\n1\n2\n")
    return load_trip(trip_dir, "train")
