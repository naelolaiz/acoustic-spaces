"""The BRAS rooms in validation/bras load as closed rooms with every position inside."""

import json
from pathlib import Path

import pytest

from acoustic_spaces import Room

BRAS = Path(__file__).resolve().parent.parent / "validation" / "bras"

# Wheel tests run on a copy of tests/ and examples/ only.
pytestmark = pytest.mark.skipif(not BRAS.is_dir(), reason="needs the source checkout")


# The sizes BRAS gives for its models (3 Surface descriptions/_descr/mat_CRx.txt). The
# conversion merges a 1 cm strip in CR3, which moves it by 0.2 m2 and 0.15 m3. For CR1
# BRAS gives each room's volume (121.4 and 102.3 m3; the doorway between them adds
# 1.7 m3) and the area of each material, which add up to 378.84 m2.
@pytest.mark.parametrize(
    ("name", "volume_m3", "area_m2", "pairs"),
    [
        ("CR1_DoorAngle1", 225.4, 378.84, 4),
        ("CR2", 146.1, 202.53, 10),
        ("CR3", 3330.6, 2763.0, 10),
        ("CR4", 8656.5, 5851.3, 10),
    ],
)
def test_bras_rooms_are_closed_and_match_their_published_size(
    name: str, volume_m3: float, area_m2: float, pairs: int
) -> None:
    room = Room.from_yaml(BRAS / name / "scene.yaml")
    assert room.report.closed, room.report.summary()
    assert room.volume_m3 == pytest.approx(volume_m3, abs=0.2)
    assert room.report.surface_area_m2 == pytest.approx(area_m2, abs=0.3)
    assert room.problems() == []
    measured = json.loads((BRAS / name / "measured.json").read_text(encoding="utf-8"))
    assert {s.name for s in room.scene.sources} == set(measured["sources"])
    assert {li.name for li in room.scene.listeners} == set(measured["listeners"])
    assert len(measured["pairs"]) == pairs
