from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from acoustic_spaces.examples import list_examples  # noqa: E402
from acoustic_spaces.room import Room  # noqa: E402
from acoustic_spaces.viz import (  # noqa: E402
    balloon_polydata,
    marker_sizes,
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


@pytest.mark.parametrize(
    "example",
    [e for e in list_examples() if not e.missing_extra],
    ids=lambda example: example.name,
)
def test_every_example_is_drawn(example) -> None:
    # tree_belt's lone oak: a round crown, with no edges to outline.
    plotter = pv.Plotter(off_screen=True)
    plot_room(Room.from_yaml(example.path), plotter)
    assert plotter.renderer.actors
    plotter.close()


def test_markers_shrink_where_points_crowd() -> None:
    # In a model 100 m across, markers are at least 1.2 m in radius and arrows 7.2 m
    # long, unless they crowd.
    radius, arrow = marker_sizes([[0, 0, 0], [1, 0, 0], [60, 0, 0]], 100.0)
    assert radius.tolist() == pytest.approx([0.45, 0.45, 1.2])  # a metre apart: not touching
    assert arrow.tolist() == pytest.approx([2.7, 2.7, 7.2])
    # On top of each other: a fifth of the size.
    radius, _arrow = marker_sizes([[0, 0, 0], [0, 0, 0]], 100.0)
    assert radius.tolist() == pytest.approx([0.24, 0.24])
    # Points 3 m and more apart are not shrunk; an arrow stops short of a marker ahead
    # of it, not of one beside it.
    points = [[0, 0, 0], [5, 0, 0], [0, 3, 0]]
    radius, arrow = marker_sizes(points, 100.0, [[1, 0, 0], [0, 0, 1], [1, 0, 0]])
    assert radius.tolist() == pytest.approx([1.2, 1.2, 1.2])
    assert arrow.tolist() == pytest.approx([0.9 * (5 - 1.2), 7.2, 7.2])
    # Far apart, a fifth of their spacing, up to 2.5 % of the model's size.
    radius, arrow = marker_sizes([[0, 0, 0], [10, 0, 0]], 100.0)
    assert radius.tolist() == pytest.approx([2.0, 2.0])
    radius, arrow = marker_sizes([[0, 0, 0], [50, 0, 0]], 100.0)
    assert radius.tolist() == pytest.approx([2.5, 2.5])
    assert arrow.tolist() == pytest.approx([15, 15])


def test_line_and_surface_sources_are_drawn() -> None:
    from acoustic_spaces.scene import Scene
    from acoustic_spaces.viz import extent_polydata

    scene = Scene.from_yaml(SHOEBOX)
    line = scene.sources[0].model_copy(update={"line_m": 2.0})
    surface = scene.sources[0].model_copy(update={"name": "wall", "surface_m": (2.0, 1.0)})
    assert extent_polydata(scene.sources[0], 0.05) is None
    tube = extent_polydata(line, 0.05)
    np.testing.assert_allclose(np.ptp(tube.points, axis=0), [0.1, 2.0, 0.1], atol=0.01)
    rectangle = extent_polydata(surface, 0.05)
    assert rectangle.n_cells == 1 and rectangle.area == pytest.approx(2.0)
    scene.sources = [line, surface]
    plotter = pv.Plotter(off_screen=True)
    before = len(plot_room(Room.from_yaml(SHOEBOX), pv.Plotter(off_screen=True)).renderer.actors)
    plot_room(Room.from_scene(scene), plotter)
    assert len(plotter.renderer.actors) == before + 3  # another sphere, a tube, a rectangle
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


def test_an_array_balloon_is_drawn_per_octave() -> None:
    # An array's pattern is in third octaves; band 4 is still the 1 kHz octave.
    from acoustic_spaces.directivity import array_directivity
    from acoustic_spaces.scene import ArraySpec

    column = array_directivity(ArraySpec(count=8, spacing_m=0.2))
    balloon = balloon_polydata(column, [0, 0, 0], band=4)
    octave = balloon_polydata(column.in_bands("octave"), [0, 0, 0], band=4)
    np.testing.assert_allclose(balloon.points, octave.points)
    bass = balloon_polydata(column.in_bands("octave"), [0, 0, 0], band=1)
    assert not np.allclose(balloon.points, bass.points, atol=1e-3)


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


def test_diffracted_paths_are_drawn_apart() -> None:
    from acoustic_spaces import Scene
    from acoustic_spaces.engine import early_paths
    from acoustic_spaces.viz import DIFFRACTED_COLOUR, path_groups, plot_paths

    scene = Scene.model_validate(
        {
            "geometry": {
                "floor_plan": [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]],
                "height": 3.0,
            },
            "materials": {"m": {"absorption": [0.3] * 8}},
            "default_material": "m",
            "sources": [{"name": "s", "position": [8.0, 2.0, 1.5]}],
            "listeners": [{"name": "l", "position": [2.0, 8.0, 1.5]}],
        }
    )
    paths = early_paths(Room.from_scene(scene), max_order=1)
    groups = path_groups(paths)
    assert groups[-1].label == "diffracted" and groups[-1].colour == DIFFRACTED_COLOUR
    assert groups[-1].mask.sum() == paths.diffracted.sum() > 0
    assert not any(g.mask[paths.diffracted].any() for g in groups[:-1])
    assert len(path_groups(paths, max_order=0)[-1].mask.nonzero()[0]) == np.sum(
        paths.diffracted & (paths.order == 0)
    )
    plotter = pv.Plotter(off_screen=True)
    legend = plot_paths(paths, plotter)
    assert legend[-1] == [f"diffracted ({paths.diffracted.sum()})", DIFFRACTED_COLOUR]
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


def test_leaks_are_drawn() -> None:
    from acoustic_spaces.engine.leaks import find_leaks
    from acoustic_spaces.examples import example_path
    from acoustic_spaces.viz import MAX_ESCAPES_DRAWN, plot_leaks

    room = Room.from_yaml(example_path("leaky_classroom"))
    leaks = find_leaks(room, rays=2000)
    plotter = pv.Plotter(off_screen=True)
    legend = plot_leaks(leaks, room, plotter, render=False)
    assert [label for label, _ in legend] == ["open edges (8)", f"escaping rays ({leaks.escaped})"]
    edges, rays, *markers = plotter.renderer.actors.values()
    assert edges.mapper.dataset.n_lines == 8  # round the window, and round the slit
    assert rays.mapper.dataset.n_lines <= MAX_ESCAPES_DRAWN < leaks.escaped
    assert len(markers) == 2 + 1  # a sphere on each gap, and their labels
    plotter.close()
    # A closed room has nothing to draw.
    plotter = pv.Plotter(off_screen=True)
    shoebox = Room.from_yaml(SHOEBOX)
    assert plot_leaks(find_leaks(shoebox, rays=100), shoebox, plotter, render=False) == []
    assert len(plotter.renderer.actors) == 0
    plotter.close()


@pytest.mark.render
def test_screenshot_with_leaks(tmp_path: Path) -> None:
    from acoustic_spaces.cli import main

    out = tmp_path / "leaks.png"
    scene = SHOEBOX.parent.parent / "leaky_classroom" / "scene.yaml"
    assert main(["view", str(scene), "--leaks", "--rays", "500", "--screenshot", str(out)]) == 0
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


def test_comparison_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import numpy as np

    from acoustic_spaces.viz import COMPARISON_COLOURS, plot_comparison

    rng = np.random.default_rng(0)

    def decay(t60: float, fs: int, length: float) -> np.ndarray:
        t = np.arange(int(length * fs)) / fs
        return rng.standard_normal(len(t)) * np.exp(-6.9 * t / t60)

    out = tmp_path / "compare.png"
    figure = plot_comparison(
        decay(0.5, 44_100, 1.0), 44_100, decay(0.4, 48_000, 1.0), 48_000, out, title="test"
    )
    assert out.stat().st_size > 10_000
    assert [ax.get_title() for ax in figure.axes][::4] == ["63 Hz", "1 kHz"]
    for ax in figure.axes:  # both decays in every band
        colours = {line.get_color() for line in ax.lines if len(line.get_xdata())}
        assert colours == set(COMPARISON_COLOURS)
    legend = figure.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["simulated", "measured"]
    # A response too short to draw leaves its side out, and the legend in place.
    figure = plot_comparison(decay(0.5, 48_000, 1.0), 48_000, np.zeros(10), 48_000)
    assert all(len(ax.lines) == 1 for ax in figure.axes[1:])
    assert len(figure.axes[0].get_legend().get_texts()) == 2


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


def test_walk_animation(tmp_path: Path) -> None:
    from PIL import Image

    from acoustic_spaces.viz import animate_walk
    from acoustic_spaces.walkthrough import WalkPoints, WalkResponses

    room = Room.from_yaml(SHOEBOX)
    points = WalkPoints(
        np.array([0.0, 0.5, 1.0]),
        np.array([[1.0, 1, 1.6], [2, 1, 1.6], [3, 1, 1.6]]),
        np.array([0.0, 90.0, 180.0]),  # an arrow shows the way it faces
    )
    walk = WalkResponses("visitor", "listeners", points, ("a", "b", "c"), None)
    out = tmp_path / "walk.gif"
    animate_walk(room, walk, out, fps=2, window_size=(320, 200))
    with Image.open(out) as gif:
        assert gif.n_frames == 3 and gif.size == (320, 200)
    with pytest.raises(ValueError, match="gif or .mp4"):
        animate_walk(room, walk, tmp_path / "walk.avi")

    from acoustic_spaces.audio import write_wav
    from acoustic_spaces.cli import main
    from acoustic_spaces.scene import Walk

    seat = room.scene.listeners[0]
    walking = seat.model_copy(update={"walk": Walk(to=[(6.0, 4.0, 1.2)], speed_m_s=1.0)})
    room.scene.model_copy(update={"listeners": [walking]}).to_yaml(tmp_path / "scene.yaml")
    write_wav(tmp_path / "dry.wav", np.zeros((800, 1)), 8000)
    args = ["walkthrough", str(tmp_path / "scene.yaml"), f"talker={tmp_path / 'dry.wav'}"]
    args += ["--out", str(tmp_path / "walk.wav"), "--rays", "200", "--length", "0.2"]
    assert main([*args, "--step", "0.5", "--video", str(tmp_path / "cli.gif"), "--fps", "4"]) == 0
    with Image.open(tmp_path / "cli.gif") as gif:
        assert gif.n_frames == 5


def test_walk_video_carries_the_sound(tmp_path: Path, capsys, monkeypatch) -> None:
    import importlib.util
    import subprocess

    from acoustic_spaces.audio import write_wav
    from acoustic_spaces.cli import main
    from acoustic_spaces.scene import Walk
    from acoustic_spaces.viz import animate_walk, check_video
    from acoustic_spaces.walkthrough import WalkPoints, WalkResponses

    real = importlib.util.find_spec
    gone = {"imageio_ffmpeg"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n in gone else real(n))
    with pytest.raises(ImportError, match="uv sync --extra viz"):
        check_video(tmp_path / "walk.mp4")
    gone.clear()
    ffmpeg = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()

    room = Room.from_yaml(SHOEBOX)
    seat = room.scene.listeners[0]
    walking = seat.model_copy(update={"walk": Walk(to=[(6.0, 4.0, 1.2)], speed_m_s=1.0)})
    room.scene.model_copy(update={"listeners": [walking]}).to_yaml(tmp_path / "scene.yaml")
    write_wav(tmp_path / "dry.wav", np.zeros((16000, 1)), 8000)
    args = ["walkthrough", str(tmp_path / "scene.yaml"), f"talker={tmp_path / 'dry.wav'}"]
    args += ["--out", str(tmp_path / "walk.wav"), "--rays", "200", "--length", "0.2"]
    args += ["--step", "0.5", "--video", str(tmp_path / "walk.mp4"), "--fps", "4"]
    assert main(args) == 0
    # As long as the sound (2.2 s), not only the walk.
    assert "walk.mp4: 2.2 s at 4 frames per second, with the sound" in capsys.readouterr().out
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(tmp_path / "walk.mp4")], capture_output=True, text=True
    )
    assert "Video: h264" in probe.stderr and "Audio: aac" in probe.stderr

    points = WalkPoints(np.array([0.0, 1.0]), np.array([[1.0, 1, 1.6], [2, 1, 1.6]]), None)
    walk = WalkResponses("visitor", "listeners", points, ("a", "b"), None)
    write_wav(tmp_path / "three.wav", np.zeros((800, 3)), 8000)
    with pytest.raises(ValueError, match="one or two channels, not 3"):
        animate_walk(room, walk, tmp_path / "three.mp4", sound=tmp_path / "three.wav")


def _map():
    from acoustic_spaces.engine import coverage_map
    from acoustic_spaces.scene import Coverage

    room = Room.from_yaml(SHOEBOX)
    settings = room.scene.simulation.model_copy(update={"ir_length_s": 1.0, "seed": 1}, deep=True)
    settings.late.method = "statistical"
    result = coverage_map(room, settings=Coverage(spacing=2.0), simulation=settings)
    return room, result.get()


def test_map_tiles_carry_the_values() -> None:
    from acoustic_spaces.viz import map_polydata, map_values, plot_coverage

    room, coverage = _map()
    poly = map_polydata(coverage, "ts", band=4)
    assert poly.n_cells == len(coverage.grid) == 12
    # Centre time shows in milliseconds; tiles sit at listening height.
    np.testing.assert_allclose(poly.cell_data["value"], coverage.value("ts", 4) * 1e3)
    np.testing.assert_allclose(np.asarray(poly.points)[:, 2], 1.2)
    values, title = map_values(coverage, "sti")
    assert title == "STI" and values.shape == (12,)
    assert map_values(coverage, "c80", band=2)[1] == "C80 (dB), 250 Hz"
    with pytest.raises(ValueError, match="unknown parameter"):
        map_values(coverage, "rasti")
    plotter = pv.Plotter(off_screen=True)
    actor = plot_coverage(coverage, plotter, parameter="spl_a")
    np.testing.assert_allclose(actor.bounds, (0.0, 8.0, 0.0, 6.0, 1.2, 1.2))
    assert actor.mapper.scalar_range == pytest.approx(
        (coverage.value("spl_a").min(), coverage.value("spl_a").max())
    )
    # See-through: the sources and listeners half under it stay in sight.
    assert 0.0 < actor.prop.opacity < 1.0
    plotter.close()


def test_map_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from acoustic_spaces.viz import plot_map

    room, coverage = _map()
    out = tmp_path / "map.png"
    figure = plot_map(coverage, out, parameter="c50", band=3, room=room)
    assert out.stat().st_size > 10_000
    # Every point labelled with its value; the source drawn and named.
    texts = [t.get_text() for t in figure.axes[0].texts]
    assert len(texts) == 12 + 1 and "talker" in texts
    assert "C50 (dB), 500 Hz" in figure.get_suptitle()


@pytest.mark.render
def test_screenshot_with_a_map(tmp_path: Path) -> None:
    room, coverage = _map()
    out = tmp_path / "map3d.png"
    show_room(room, screenshot=out, window_size=(320, 200), coverage=coverage, parameter="g")
    assert out.stat().st_size > 1000
