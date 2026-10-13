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


def test_paths_are_drawn_by_order() -> None:
    from acoustic_spaces.engine import early_paths
    from acoustic_spaces.viz import paths_polydata, plot_paths

    paths = early_paths(Room.from_yaml(SHOEBOX), max_order=2)
    poly = paths_polydata(paths, 2)
    assert poly.n_lines == 18
    plotter = pv.Plotter(off_screen=True)
    legend = plot_paths(paths, plotter)
    assert [entry[0] for entry in legend] == ["direct (1)", "1st order (6)", "2nd order (18)"]
    plotter.close()


def test_response_plot(tmp_path: Path) -> None:
    from acoustic_spaces import simulate
    from acoustic_spaces.viz import plot_response

    room = Room.from_yaml(SHOEBOX)
    settings = room.scene.simulation.model_copy(update={"ir_length_s": 0.5, "seed": 1}, deep=True)
    settings.late.rays = 2000
    response = simulate(room, settings=settings)["talker", "seat-a"]
    out = tmp_path / "response.png"
    plot_response(response, out)
    assert out.stat().st_size > 10_000
    from acoustic_spaces.viz import plot_directions

    out = tmp_path / "directions.png"
    plot_directions(response, out)
    assert out.stat().st_size > 10_000


@pytest.mark.render
def test_screenshot_with_paths(tmp_path: Path) -> None:
    from acoustic_spaces.engine import early_paths

    room = Room.from_yaml(SHOEBOX)
    out = tmp_path / "paths.png"
    show_room(room, screenshot=out, window_size=(320, 200), paths=early_paths(room))
    assert out.stat().st_size > 1000


def test_impulse_response_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import numpy as np

    from acoustic_spaces.viz import plot_impulse_response

    rng = np.random.default_rng(0)
    t = np.arange(48_000) / 48_000
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.4) + 1e-3 * rng.standard_normal(len(t))
    out = tmp_path / "ir.png"
    plot_impulse_response(ir, 48_000, out, title="test")
    assert out.stat().st_size > 10_000
