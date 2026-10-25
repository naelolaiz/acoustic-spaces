"""The edge fade: every specular path takes the terms at the edges it passes, so that it
fades in and out at them, as exact solutions do."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.diffraction import EdgeDiffraction, find_edges
from acoustic_spaces.engine.edge_fade import _FACE, EdgeFade
from acoustic_spaces.engine.model import TraceModel

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
#: The surfaces' extent (m): their far edges arrive too late to count.
EXTENT = 300.0
BANDS = 8


def _ground() -> list:
    a = EXTENT
    return [[(-a, -a, 0.0), (a, -a, 0.0), (a, a, 0.0), (-a, a, 0.0)]]


def _screen(y0: float, y1: float) -> list:
    """A thin screen in the plane x = 0 from y0 to y1, through the ground and far up."""
    a = EXTENT
    return [[(0.0, y0, -a), (0.0, y1, -a), (0.0, y1, a), (0.0, y0, a)]]


def _corner() -> list:
    """A building filling x > 0, y > 0, through the ground and far up."""
    a = EXTENT
    return [
        [(0.0, 0.0, -a), (0.0, 0.0, a), (0.0, a, a), (0.0, a, -a)],
        [(0.0, 0.0, -a), (a, 0.0, -a), (a, 0.0, a), (0.0, 0.0, a)],
    ]


MODELS = {
    "screen end": lambda: _ground() + _screen(0.0, EXTENT),
    "gap": lambda: _ground() + _screen(2.0, EXTENT) + _screen(-EXTENT, -2.0),
    "corner": lambda: _ground() + _corner(),
    "strip": lambda: _ground() + _screen(-1.0, 1.0),
}


class _Engine:
    """The early paths from ``source`` with the edge diffraction in ``mode``, every
    band added in pressure, with or without the fade."""

    def __init__(self, folder: Path, name: str, source, mode: str = "direct") -> None:
        lines, faces = [], []
        for quad in MODELS[name]():
            base = len(lines)
            lines += [f"v {x} {y} {z}" for x, y, z in quad]
            faces.append("f " + " ".join(str(base + i + 1) for i in range(4)))
        obj = folder / "fade.obj"
        obj.write_text("\n".join(lines + ["g surfaces"] + faces) + "\n")
        self.source = np.asarray(source, dtype=float)
        room = Room.from_scene(
            Scene.model_validate(
                {
                    "geometry": {"file": str(obj)},
                    "materials": {"hard": {"absorption": [0.0] * BANDS, "scattering": 0.0}},
                    "default_material": "hard",
                    "sources": [{"name": "s", "position": self.source.tolist()}],
                    "listeners": [{"name": "l", "position": (self.source + 1.0).tolist()}],
                }
            )
        )
        self.model = TraceModel.from_room(room)
        self.c = room.speed_of_sound()
        self.air = np.zeros(BANDS)
        self.candidates = early_stage.exhaustive_candidates(self.model, self.source, 2)
        self.diffraction = EdgeDiffraction(
            self.model,
            self.source,
            speed_of_sound=self.c,
            air=self.air,
            mode=mode,
            edges=find_edges(self.model),
            coherent=BANDS,
        )
        self.fade = EdgeFade(self.diffraction, self.candidates)

    def level(self, receiver, fade: bool = True) -> np.ndarray:
        """The level in each band, dB re the direct sound in free field."""
        receiver = np.asarray(receiver, dtype=float)
        paths = early_stage.find_paths(
            self.model,
            self.source,
            receiver,
            self.candidates,
            speed_of_sound=self.c,
            air=self.air,
            max_delay_s=0.5,
            coherent=BANDS,
        )
        terms = self.fade.terms(receiver, paths, 0.5) if fade else None
        paths = self.diffraction.apply(receiver, paths, 0.5)
        if fade:
            paths = self.fade.join(paths, terms)
        squared = float(np.sum((receiver - self.source) ** 2))
        return 10.0 * np.log10(total_energy(paths) * 4.0 * np.pi * squared)


# The level (dB re the direct sound in free field) over a rigid ground, by images: the
# geometric paths from the source and its image in the ground, and each edge's exact
# term (Biot-Tolstoy-Medwin, validation/edge_diffraction.py), averaged over 96
# frequencies in each band (validation/edge_fade.py). The screen end's shadow boundary
# is at y = 1, the gap's at y = +-4, the corner's reflection boundary at x = -4.5 and its
# shadow boundary at x = 4.5; the reflection off the ground crosses each with the
# direct sound, and only the fade bends it with diffraction: direct. Off the strip 2 m
# wide, the reflection's boundary is at y = 2.875, and the edge diffraction bends only
# the reflection, not its copy off the ground.
EXACT = {
    ("screen end", (-4.0, -1.0, 0.5), (4.0, 0.95, 1.5)): [
        1.61, 0.99, 0.02, -3.03, -8.33, -1.38, -2.43, -2.04,
    ],
    ("screen end", (-4.0, -1.0, 0.5), (4.0, 1.05, 1.5)): [
        1.47, 0.79, -0.27, -3.43, -9.1, -2.28, -3.85, -4.0,
    ],
    ("screen end", (-4.0, -1.0, 0.5), (4.0, 2.5, 1.5)): [
        -0.26, -1.76, -3.9, -8.13, -17.77, -11.58, -16.87, -18.47,
    ],
    ("gap", (-4.0, 0.0, 0.5), (4.0, 3.9, 1.5)): [
        2.11, 1.03, 0.35, -2.27, -9.42, -0.53, -2.63, -0.92,
    ],
    ("gap", (-4.0, 0.0, 0.5), (4.0, 4.1, 1.5)): [
        1.87, 0.7, -0.17, -2.95, -10.72, -2.16, -5.08, -4.29,
    ],
    ("corner", (-3.0, 2.0, 0.5), (-4.6, -3.0, 1.5)): [
        5.69, 5.59, 4.16, -1.91, 4.04, 3.49, 2.8, 3.3,
    ],
    ("corner", (-3.0, 2.0, 0.5), (4.6, -3.0, 1.5)): [
        3.89, 2.75, 1.34, -1.65, -9.79, -1.19, -4.11, -3.1,
    ],
    ("corner", (-3.0, 2.0, 0.5), (6.5, -3.0, 1.5)): [
        3.33, 1.91, 0.11, -2.97, -11.73, -6.76, -10.15, -13.23,
    ],
    ("strip", (-4.0, -1.5, 0.5), (-3.0, 2.8031, 1.5)): [
        4.49, 5.67, 3.73, -3.48, 4.35, 2.29, 3.5, 3.01,
    ],
}  # fmt: skip


@pytest.mark.parametrize("case", list(EXACT), ids=lambda c: f"{c[0]} at {c[2]}")
def test_the_exact_solution(tmp_path: Path, case) -> None:
    name, source, receiver = case
    engine = _Engine(tmp_path, name, source)
    exact = np.array(EXACT[case])
    assert np.abs(engine.level(receiver) - exact).max() < 0.2
    # Without the fade the reflection off the ground steps at full strength: over 1 dB off.
    assert np.abs(engine.level(receiver, fade=False) - exact).max() > 1.0


def test_no_step_at_a_boundary(tmp_path: Path) -> None:
    """Across the screen end's shadow boundary, 1 cm at a time: the level changes no
    faster than the exact solution's, about 0.1 dB a centimetre at 1 kHz."""
    engine = _Engine(tmp_path, "screen end", (-4.0, -1.0, 0.5))
    ys = np.arange(0.9, 1.1, 0.01) + 0.0031  # none exactly on the boundary
    faded = np.array([engine.level((4.0, y, 1.5)) for y in ys])
    plain = np.array([engine.level((4.0, y, 1.5), fade=False) for y in ys])
    assert np.abs(np.diff(faded, axis=0)).max() < 0.25
    assert np.abs(np.diff(plain, axis=0)).max() > 5.0


def test_nothing_twice_where_the_edge_diffraction_bends_it(tmp_path: Path) -> None:
    """With first_order the edge diffraction bends the reflections off the corner's
    walls and off the ground; the fade adds nothing to what it does there."""
    engine = _Engine(tmp_path, "corner", (-3.0, 2.0, 0.5), mode="first_order")
    for x in (-4.6, 4.6, 6.5):
        faded, plain = engine.level((x, -3.0, 1.5)), engine.level((x, -3.0, 1.5), False)
        assert np.abs(faded - plain).max() < 0.01


def test_the_street_garden_has_no_step() -> None:
    """The passing car in the street example, heard in the garden: where its reflection
    off the offices and the road comes through the gap between the houses and the
    barrier, the level stepped by 11.9 dB(A) in 0.5 m of road."""
    scene = Scene.from_yaml(EXAMPLES / "street" / "scene.yaml")
    car = next(s for s in scene.sources if s.name == "car")
    weights = 10 ** ((np.array(car.level_db) + [-26.2, -16.1, -8.6, -3.2, 0, 1.2, 1, -1.1]) / 10)
    levels = []
    for x in (-42.5, -42.0):
        moved = car.model_copy(update={"position": (x, -2.0, 0.05), "walk": None})
        moved = moved.model_copy(update={"enabled": True})
        paths = early_paths(scene.model_copy(update={"sources": [moved]}), "car", "garden")
        levels.append(10 * np.log10(np.sum(total_energy(paths) * weights)))
    assert abs(levels[1] - levels[0]) < 1.5


@pytest.fixture(scope="module")
def gallery() -> Room:
    return Room.from_yaml(EXAMPLES / "gallery" / "scene.yaml")


@pytest.mark.parametrize(
    ("x", "y", "band"),
    [
        # The terms' cross terms with their paths (in the bands added in energy) went to
        # other paths: the edge diffraction hands its paths back sorted by arrival.
        (4.0, 14.66, 7),
        # Of the two orders of a reflection off the floor and off wall_1 (perpendicular
        # planes, one image), the one that cannot be was kept, and that one never fades.
        (5.5, 10.49, 0),
        # The floor bounce of a path off the floor and wall_6 lies just behind wall_3,
        # where the corner stands on the floor: the leg blocked by the corner's face
        # next to it stops the path, and the path over the corner reflects off the floor
        # in front of wall_3, which is what has to be clear.
        (3.0, 12.0, 0),
        # The same for the last bounce of ceiling, floor, ceiling, where the corner
        # meets the ceiling.
        (5.5, 8.92, 0),
        # Into the plane of wall_3 the size scattering weakens a third reflection off it
        # (as the early stage has it), and the wave of the corner that takes over from
        # it behind the plane was not weakened alike.
        (3.0, 7.99, 0),
        # At 7.20 a path missed reflects where the planes of wall_3 and wall_6 meet,
        # and was left out.
        (4.0, 7.19, 0),
        # At 12.00 the paths over the corner reach it at its top, where it meets the
        # ceiling, and neither the leg before the reflection off the ceiling nor the leg
        # after it took them.
        (4.0, 11.99, 2),
    ],
)
def test_no_step_round_the_gallery_corner(gallery: Room, x: float, y: float, band: int) -> None:
    """Round the gallery's corner (8, 8), with reflections of up to third order: 1 cm
    further on, the level changes by less than 0.1 dB; with each of these bugs it
    stepped by 0.2 to 2.2 dB there."""
    scene = gallery.scene
    visitor = scene.listeners[1]
    levels = []
    for at in (y, y + 0.01):
        moved = visitor.model_copy(update={"position": (x, at, 1.6)})
        room = Room.from_scene(scene.model_copy(update={"listeners": [moved]}), reuse=gallery)
        levels.append(10 * np.log10(total_energy(early_paths(room, max_order=3))[band]))
    assert abs(levels[1] - levels[0]) < 0.1


def test_a_block_of_reflections_alone(tmp_path: Path) -> None:
    """A block of events with no leg in it still gives the edges of its reflections'
    planes to test, and an empty one gives none (in a city of 1,600 buildings a block
    had no leg, and the fade failed)."""
    fade = _Engine(tmp_path, "corner", (-3.0, 2.0, 0.5)).fade
    table = {"kind": np.array([_FACE]), "plane": np.array([1]), "ends": np.full((1, 2, 1), -1)}
    slot, edge = fade._pairs(table, np.array([0]))
    assert len(slot) == len(edge) > 0
    assert np.all(slot == 0)
    slot, edge = fade._pairs(table, np.array([], dtype=int))
    assert len(slot) == len(edge) == 0


@pytest.mark.parametrize(("order", "mode", "y"), [(0, "direct", 8.5), (1, "first_order", 16.3)])
def test_nothing_where_the_edge_diffraction_has_every_path(
    gallery: Room, monkeypatch, order: int, mode: str, y: float
) -> None:
    """Where the edge diffraction bends every path the early stage follows, the fade
    changes nothing. It took away the terms of reflections off faces whose planes the
    listener is behind, though the early stage follows them nowhere at that order: at 63
    Hz the level stepped by 2.9 dB where the listener crossed wall_3's plane (y = 8) and
    was up to 4.1 dB low beyond it."""
    scene = gallery.scene
    moved = scene.listeners[1].model_copy(update={"position": (4.0, y, 1.6)})
    room = Room.from_scene(scene.model_copy(update={"listeners": [moved]}), reuse=gallery)
    faded = total_energy(early_paths(room, max_order=order, diffraction=mode))
    monkeypatch.setattr(EdgeFade, "terms", lambda self, *args: None)
    plain = total_energy(early_paths(room, max_order=order, diffraction=mode))
    assert np.abs(10 * np.log10(faded / plain)).max() < 0.05


def _outdoors(listener, *, barrier_m: float = 3.0, shed: bool = False, **early) -> Room:
    """A source 1 m over a ground, a barrier ``barrier_m`` high 10 m away along y and, with
    ``shed``, a shed as high as the barrier is out of the way."""
    geometry = {
        "ground": {"extent": [[-100, -100], [100, 100]], "sky": 40},
        "barriers": [{"name": "wall", "path": [[10, -30], [10, 30]], "height": barrier_m}],
    }
    if shed:
        footprint = [[20, 20], [26, 20], [26, 26], [20, 26]]
        geometry["buildings"] = [{"name": "shed", "footprint": footprint, "height": barrier_m}]
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": geometry,
                "materials": {
                    "ground": {"absorption": [1.0 if shed else 0.0] * 8, "scattering": 0.0},
                    "hard": {"absorption": [0.05] * 8, "scattering": 0.0},
                },
                "default_material": "ground",
                "assignments": {"wall": "hard", "shed": "hard"},
                "sources": [{"name": "s", "position": [0.0, 0.0, 1.0]}],
                "listeners": [{"name": "l", "position": list(listener)}],
                "simulation": {
                    "ir_length_s": 0.3,
                    "early": {"method": "image_sources", **early},
                },
            }
        )
    )


@pytest.mark.parametrize("height", [1.5, 8.0])
def test_a_barrier_top_is_not_on_the_leg_before_the_ground(monkeypatch, height: float) -> None:
    """Behind a barrier on a rigid ground the edge diffraction bends the direct sound and
    the reflection off the ground before the barrier and after it: the fade adds
    nothing. The leg before the reflection had taken the top's term too, where the line
    to the listener's image passes the barrier beyond the ground, as well as the leg
    after it, which the top blocks: up to 0.8 dB less at 125 Hz."""
    room = _outdoors((20.0, 0.0, height), max_order=2, diffraction="first_order")
    faded = total_energy(early_paths(room))
    monkeypatch.setattr(EdgeFade, "terms", lambda self, *args: None)
    plain = total_energy(early_paths(room))
    assert np.abs(10 * np.log10(faded / plain)).max() < 0.01


def test_a_wall_as_high_as_a_roof_does_not_stand_on_its_plane() -> None:
    """Right on the shadow line of a wall's end, with a shed as high as the wall 40 m
    away: no path reflects off the plane of the shed's roof from below. The end of the
    wall, which lies in that plane, counted as standing on it, so the reflection off it
    past the end came as strong as the sound round the end: 2.8 to 6.3 dB louder, and at
    1 kHz, where the two cancelled, 3.1 dB softer."""
    room = _outdoors((12.0, -36.0, 1.5), shed=True, max_order=1, diffraction="direct")
    model = TraceModel.from_room(room)
    plane = model.face_plane
    roof = np.flatnonzero(
        (np.abs(np.abs(model.plane_normal[plane, 2]) - 1.0) < 1e-9)
        & (np.abs(np.abs(model.plane_offset[plane]) - 3.0) < 1e-9)
    )
    assert len(roof)
    paths = early_paths(room)
    assert paths.diffracted.any()
    assert not any(np.isin(faces, roof).any() for faces in paths.faces)
