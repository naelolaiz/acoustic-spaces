"""The leak finder (engine.leaks)."""

from pathlib import Path

import numpy as np
import pytest
from conftest import skip_without_extra

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine.leaks import RAYS, find_leaks, open_edges
from acoustic_spaces.examples import example_path, list_examples

#: The corners of a 6 x 5 x 3 m box and its faces, each its own group.
CORNERS = [(0, 0, 0), (6, 0, 0), (6, 5, 0), (0, 5, 0), (0, 0, 3), (6, 0, 3), (6, 5, 3), (0, 5, 3)]
FACES = {
    "floor": (1, 4, 3, 2),
    "ceiling": (5, 6, 7, 8),
    "south": (1, 2, 6, 5),
    "north": (3, 4, 8, 7),
    "west": (4, 1, 5, 8),
    "east": (2, 3, 7, 6),
}


def _box(tmp_path: Path, without=(), source=(2.1, 1.7, 1.4), **materials) -> Room:
    obj = tmp_path / "box.obj"
    lines = [f"v {x} {y} {z}" for x, y, z in CORNERS]
    for name, face in FACES.items():
        if name not in without:
            lines += [f"g {name}", "f " + " ".join(map(str, face))]
    obj.write_text("\n".join(lines) + "\n")
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"m": {"absorption": [0.1] * 8}, **materials},
                "default_material": "m",
                "assignments": {name: name for name in materials},
                "sources": [{"name": "s", "position": list(source)}],
                "listeners": [{"name": "l", "position": [4.0, 3.0, 1.2]}],
            }
        )
    )


def test_the_command_line_reports_leaks(capsys: pytest.CaptureFixture[str]) -> None:
    from acoustic_spaces import cli

    assert cli.LEAK_RAYS == RAYS
    scene = str(example_path("leaky_classroom"))
    assert cli.main(["validate", scene, "--leaks", "--rays", "300"]) == 0
    out, err = capsys.readouterr()
    assert "leaks: 300 rays, up to 100 reflections each:" in out
    assert "by east_wall\n     at (9.00," in out
    assert "validate --leaks" in err  # the open model's warning says how to find them
    assert cli.main(["validate", scene, "--leaks", "--rays", "0"]) == 1
    assert "rays and max_reflections must be positive" in capsys.readouterr().err


def test_a_closed_model_keeps_every_ray(tmp_path: Path) -> None:
    leaks = find_leaks(_box(tmp_path), rays=500)
    assert leaks.escaped == 0 and leaks.gaps == []
    assert leaks.reflections == 500 * leaks.max_reflections
    assert "none escaped" in leaks.summary()


def test_a_missing_face_is_one_gap_of_its_size(tmp_path: Path) -> None:
    room = _box(tmp_path, without=("ceiling",))
    corners, faces = open_edges(room)
    assert len(corners) == 4 and len(faces) == 4
    leaks = find_leaks(room, rays=500)
    assert leaks.escaped == leaks.leaked == 500  # nothing keeps them in
    (gap,) = leaks.gaps
    assert gap.kind == "gap" and gap.rays == 500
    assert gap.area_m2 == pytest.approx(30.0)
    np.testing.assert_allclose(gap.size_m, (6.0, 5.0))
    assert set(gap.surfaces) <= {"south", "north", "west", "east"} and len(gap.surfaces) == 3
    # The rays leave where the ceiling was.
    np.testing.assert_allclose(leaks.points[:, 2], 3.0, atol=1e-9)
    assert np.all(leaks.gap == 0)
    assert "through a gap of about 30 m2 (6.00 x 5.00 m) by " in leaks.summary()


def test_the_example_classroom_has_its_two_holes_found() -> None:
    # The window's glass is missing (3 x 1.2 m) and the ceiling stops 3 cm short of the
    # north wall (9 m long): each is one gap, the window the one most rays find.
    leaks = find_leaks(Room.from_yaml(example_path("leaky_classroom")), rays=2000)
    window, slit = leaks.gaps
    assert window.surfaces == ("east_wall",)
    assert window.area_m2 == pytest.approx(3.6)
    np.testing.assert_allclose(window.size_m, (3.0, 1.2), atol=1e-9)
    np.testing.assert_allclose(window.centre, [9.0, 3.5, 1.6], atol=0.15)
    assert slit.surfaces[:2] in (("ceiling", "north_wall"), ("north_wall", "ceiling"))
    assert slit.area_m2 == pytest.approx(0.27)
    np.testing.assert_allclose(slit.size_m, (9.0, 0.03), atol=1e-9)
    np.testing.assert_allclose(slit.centre[1:], [6.985, 3.0], atol=0.02)
    assert window.rays > 5 * slit.rays
    # Every ray left through one of them, where it crossed the hole.
    assert leaks.escaped == window.rays + slit.rays
    x, y, z = leaks.points[leaks.gap == 0].T
    np.testing.assert_allclose(x, 9.0, atol=1e-9)
    assert np.all((y > 2.0 - 1e-9) & (y < 5.0 + 1e-9) & (z > 1.0 - 1e-9) & (z < 2.2 + 1e-9))
    x, y, z = leaks.points[leaks.gap == 1].T
    np.testing.assert_allclose(z, 3.0, atol=1e-9)
    assert np.all((y > 6.97 - 1e-9) & (y < 7.0 + 1e-9))


def test_an_opening_to_the_outside_is_not_a_leak(tmp_path: Path) -> None:
    room = _box(tmp_path, ceiling={"transmission": 1.0})
    leaks = find_leaks(room, rays=500)
    (gap,) = leaks.gaps
    assert gap.kind == "surface" and gap.surfaces == ("ceiling",)
    assert leaks.escaped == 500 and leaks.leaked == 0
    np.testing.assert_allclose(leaks.points[:, 2], 3.0, atol=1e-9)
    assert "left through ceiling, which lets sound through" in leaks.summary()


def test_a_source_outside_a_closed_model(tmp_path: Path) -> None:
    leaks = find_leaks(_box(tmp_path, source=(9.0, 2.0, 1.5)), rays=500)
    outside = leaks.gaps[0]
    assert outside.kind == "outside" and outside.surfaces == ("s",)
    assert "left source s without meeting a surface: it is outside" in leaks.summary()
    with pytest.raises(ValueError):
        find_leaks(_box(tmp_path), rays=0)


@pytest.mark.parametrize("name", [e.name for e in list_examples() if e.name != "leaky_classroom"])
def test_the_other_examples_do_not_leak(name: str) -> None:
    skip_without_extra(name)
    assert find_leaks(Room.from_yaml(example_path(name)), rays=300).escaped == 0
