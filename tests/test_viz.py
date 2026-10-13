from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from acoustic_spaces.room import Room  # noqa: E402
from acoustic_spaces.viz import (  # noqa: E402
    balloon_polydata,
    plot_room,
    room_polydata,
    show_room,
)

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


def test_directivity_balloon_points_where_the_source_faces() -> None:
    from acoustic_spaces.directivity import FirstOrder
    from acoustic_spaces.scene.model import Orientation

    turn = Orientation(yaw=90).rotation()
    balloon = balloon_polydata(FirstOrder.pattern("cardioid"), [1, 2, 3], turn, size=2.0)
    points = np.asarray(balloon.points) - [1, 2, 3]
    length = np.linalg.norm(points, axis=1)
    assert length.max() == pytest.approx(2.0, rel=1e-3)
    # The sphere has no ring exactly on the equator: the tip is a few degrees off.
    np.testing.assert_allclose(points[np.argmax(length)] / length.max(), [0, 1, 0], atol=0.06)
    assert balloon.point_data["level"].max() == 0.0


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


@pytest.mark.render
def test_particle_animation(tmp_path: Path) -> None:
    from PIL import Image

    from acoustic_spaces.cli import main
    from acoustic_spaces.engine import trace_particles
    from acoustic_spaces.viz import animate_particles

    room = Room.from_yaml(SHOEBOX)
    particles = trace_particles(room, count=300, duration=0.03, frames=3)
    out = tmp_path / "particles.gif"
    animate_particles(room, particles, out, color="order", window_size=(320, 200))
    with Image.open(out) as gif:
        assert gif.n_frames == 3 and gif.size == (320, 200)
    with pytest.raises(ValueError, match="gif or .mp4"):
        animate_particles(room, particles, tmp_path / "particles.avi")
    args = ["animate", str(SHOEBOX), "--out", str(tmp_path / "cli.gif"), "--frames", "2"]
    assert main([*args, "--particles", "200", "--duration", "20", "--band", "500"]) == 0
    assert (tmp_path / "cli.gif").stat().st_size > 1000
