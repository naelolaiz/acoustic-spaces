"""SketchUp files, read through openskp (the sketchup extra)."""

import sys
from pathlib import Path

import pytest

from acoustic_spaces.geometry import load_mesh
from acoustic_spaces.geometry.checks import check_mesh, clean_mesh

INCH = 0.0254


def _box(path: Path, back_of_floor: str | None = None) -> None:
    """A 4 x 3 x 2.5 m box: carpet on the floor, plaster on the walls, the ceiling
    unpainted. SketchUp takes coordinates in inches."""
    openskp = pytest.importorskip("openskp")
    builder = openskp.create()
    carpet = builder.add_material("carpet", (120, 60, 40))
    plaster = builder.add_material("plaster", (230, 230, 220))
    glass = builder.add_material("glass", (150, 200, 230))
    x, y, z = (v / INCH for v in (4.0, 3.0, 2.5))
    corners = [(0.0, 0.0), (x, 0.0), (x, y), (0.0, y)]
    floor = [(a, b, 0.0) for a, b in corners]
    ceiling = [(a, b, z) for a, b in corners]
    builder.add_face(floor, material=carpet, back_material=glass if back_of_floor else None)
    builder.add_face(ceiling[::-1])
    for i in range(4):
        j = (i + 1) % 4
        builder.add_face([floor[i], floor[j], ceiling[j], ceiling[i]], material=plaster)
    builder.save(str(path))


def test_a_sketchup_box_keeps_its_size_and_materials(tmp_path: Path) -> None:
    _box(tmp_path / "box.skp")
    mesh = load_mesh(tmp_path / "box.skp")
    assert sorted(mesh.labels) == [("carpet",), ("default",), ("plaster",)]
    report = check_mesh(*clean_mesh(mesh))
    assert report.closed
    assert report.volume_m3 == pytest.approx(30.0, rel=1e-6)
    assert report.surface_area_m2 == pytest.approx(2 * 12 + 2 * 2.5 * 7, rel=1e-6)
    floor = [mesh.labels.index(("carpet",)) == label for label in mesh.face_labels]
    assert mesh.vertices[mesh.faces[floor]][..., 2] == pytest.approx(0.0)


def test_a_face_painted_on_both_sides_keeps_one_material(tmp_path: Path) -> None:
    _box(tmp_path / "box.skp", back_of_floor="glass")
    mesh = load_mesh(tmp_path / "box.skp")
    assert len(mesh.faces) == 12
    assert ("glass",) not in mesh.labels


def test_reading_sketchup_without_openskp_says_how_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "room.skp").write_bytes(b"")
    monkeypatch.setitem(sys.modules, "openskp", None)
    with pytest.raises(ImportError, match=r"needs openskp: .*uv sync --extra sketchup"):
        load_mesh(tmp_path / "room.skp")
