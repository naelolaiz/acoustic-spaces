from pathlib import Path

import pytest

pv = pytest.importorskip("pyvista")

from acoustic_spaces.room import Room  # noqa: E402
from acoustic_spaces.viz import plot_room, room_polydata, show_room  # noqa: E402

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


def test_polydata_carries_materials() -> None:
    room = Room.from_yaml(SHOEBOX)
    poly = room_polydata(room)
    assert poly.n_cells == 12
    assert sorted(set(poly.cell_data["material"].tolist())) == [0, 1, 2]


def test_plot_room_adds_surfaces_and_markers() -> None:
    room = Room.from_yaml(SHOEBOX)
    plotter = pv.Plotter(off_screen=True)
    plot_room(room, plotter)
    # three materials (surface + outline each), a source and a listener with its arrow
    assert len(plotter.renderer.actors) >= 3 * 2 + 3
    plotter.close()


@pytest.mark.render
def test_screenshot(tmp_path: Path) -> None:
    out = tmp_path / "room.png"
    show_room(Room.from_yaml(SHOEBOX), screenshot=out, window_size=(320, 200))
    assert out.stat().st_size > 1000
