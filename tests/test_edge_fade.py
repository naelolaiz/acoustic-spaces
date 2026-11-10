"""The edge fade: every specular path takes the terms at the edges it passes, so that it
fades in and out at them, as exact solutions do."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.clusters import Clusters
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.diffraction import EdgeDiffraction, find_edges, sheets
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
        bent = self.diffraction.apply(receiver, paths, 0.5)
        if fade:
            bent = self.fade.join(bent, self.fade.terms(receiver, paths, 0.5, bent))
        paths = bent
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
        1.61, 0.99, 0.03, -3.02, -8.34, -1.37, -2.43, -2.05,
    ],
    ("screen end", (-4.0, -1.0, 0.5), (4.0, 1.05, 1.5)): [
        1.47, 0.79, -0.27, -3.42, -9.11, -2.27, -3.85, -4.0,
    ],
    ("screen end", (-4.0, -1.0, 0.5), (4.0, 2.5, 1.5)): [
        -0.26, -1.75, -3.9, -8.12, -17.77, -11.58, -16.87, -18.47,
    ],
    ("gap", (-4.0, 0.0, 0.5), (4.0, 3.9, 1.5)): [
        2.11, 1.03, 0.35, -2.27, -9.43, -0.53, -2.63, -0.91,
    ],
    ("gap", (-4.0, 0.0, 0.5), (4.0, 4.1, 1.5)): [
        1.87, 0.7, -0.17, -2.94, -10.73, -2.16, -5.08, -4.28,
    ],
    ("corner", (-3.0, 2.0, 0.5), (-4.6, -3.0, 1.5)): [
        5.68, 5.59, 4.16, -1.9, 4.03, 3.49, 2.81, 3.3,
    ],
    ("corner", (-3.0, 2.0, 0.5), (4.6, -3.0, 1.5)): [
        3.89, 2.75, 1.34, -1.65, -9.79, -1.19, -4.11, -3.09,
    ],
    ("corner", (-3.0, 2.0, 0.5), (6.5, -3.0, 1.5)): [
        3.34, 1.91, 0.12, -2.97, -11.72, -6.76, -10.14, -13.22,
    ],
    ("strip", (-4.0, -1.5, 0.5), (-3.0, 2.8031, 1.5)): [
        4.48, 5.67, 3.74, -3.49, 4.35, 2.29, 3.5, 3.02,
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


@pytest.mark.parametrize("x", [-41.61, 30.40])
def test_the_street_garden_goes_on_where_a_second_event_stops_a_path(x: float) -> None:
    """Where the reflection off the offices of a path that the barrier or the houses
    block reaches an end of the façade, two events stop the path instead of one: 1 cm
    further on, the level in the garden changes in every band as it did over the
    centimetre before, to 0.15 dB. Such a path was left out with its terms (up to 1.9 dB
    at 500 Hz at once); then the products of its terms took the delay of one of them,
    so that the band's frequencies stopped interfering with them unlike on the other
    side of the step (0.28 dB at 125 Hz). What is left (0.11 dB at 125 Hz) is the paths
    over the edges below the edge diffraction's least energy, which are left out, as
    they change from one side to the other."""
    scene = Scene.from_yaml(EXAMPLES / "street" / "scene.yaml")
    car = next(s for s in scene.sources if s.name == "car")
    levels = []
    for at in (x - 0.01, x, x + 0.01):
        moved = car.model_copy(update={"position": (at, -2.0, 0.05), "walk": None})
        moved = moved.model_copy(update={"enabled": True})
        paths = early_paths(scene.model_copy(update={"sources": [moved]}), "car", "garden")
        levels.append(10 * np.log10(total_energy(paths)))
    before, after = np.diff(levels, axis=0)
    assert np.abs(after - before).max() < 0.15


@pytest.mark.parametrize(
    ("listener", "x", "most"),
    [
        ("garden", 0.0, 0.5),
        ("garden", 60.01, 0.5),
        ("pavement", 50.01, 0.5),
        ("garden", 10.01, 1.0),
    ],
)
def test_the_street_does_not_step_where_the_barrier_s_top_fades(
    listener: str, x: float, most: float
) -> None:
    """Paths over the barrier's top fade out past its ends. Where the edge diffraction's
    term of a path fades, the fade must leave it out whether its path is found or not
    (in the garden at x = 0.00 and 60.01 m it counted it again, by up to 2.1 dB at 63
    and 500 Hz in 1 cm), and must count it on the lit side where the path's reflection
    lies on another face in the same plane, the houses' façade (on the pavement at
    x = 50.01 m, 0.95 dB at 63 Hz). Where the car crosses the plane of the houses' end
    wall, at x = 10.01 m, the fade's path off the plane over the top was taken for the
    edge diffraction's over the top's image in the plane, which is as long, and taken
    off as if it were there (1.7 dB at 125 Hz); the sound over the houses' corner dips
    there anyway, by 0.6 dB at 250 Hz in that centimetre before the fade."""
    scene = Scene.from_yaml(EXAMPLES / "street" / "scene.yaml")
    car = next(s for s in scene.sources if s.name == "car")
    levels = []
    for at in (x - 0.01, x):
        moved = car.model_copy(update={"position": (at, -2.0, 0.05), "walk": None})
        moved = moved.model_copy(update={"enabled": True})
        paths = early_paths(scene.model_copy(update={"sources": [moved]}), "car", listener)
        levels.append(10 * np.log10(total_energy(paths)))
    # But 8 kHz, where a centimetre is a quarter of a wavelength.
    assert np.abs(levels[1] - levels[0])[:-1].max() < most


@pytest.mark.parametrize("y", [6.135, 7.325])
def test_the_terms_of_a_narrow_face_cancel_in_energy_too(y: float) -> None:
    """Behind the archway's wall, where the reflection off the arch's jamb (40 cm deep)
    appears: the terms of the jamb's two edges nearly cancel in pressure, and added in
    energy (from 250 Hz up) they made the level step by up to 4.4 dB in 1 cm; with the
    cross terms between the terms of one event, by 1.4 dB. As pieces of the waves they
    join (the jamb's diffracted wave and the reflection's), with the events a path gets
    past split alike, by 0.03 dB or less up to 1 kHz and 0.12 dB at 8 kHz, as it does
    with every band in pressure."""
    scene = Scene.from_yaml(EXAMPLES / "archway" / "scene.yaml")
    room = Room.from_scene(scene)
    listener = next(item for item in scene.listeners if item.name == "behind the wall")
    levels = []
    for at in (y - 0.005, y + 0.005):
        moved = listener.model_copy(update={"position": (7.6, at, 1.2)})
        moved_room = Room.from_scene(scene.model_copy(update={"listeners": [moved]}), reuse=room)
        levels.append(10 * np.log10(total_energy(early_paths(moved_room, "singer", moved.name))))
    assert np.abs(levels[1] - levels[0])[2:].max() < 0.15


def test_a_term_goes_on_past_the_foot_of_its_edge_as_the_edge_s_image() -> None:
    """Behind the archway's wall at y = 6.32 m, a third-order reflection that the jamb
    cuts off (off the bar's side wall, the floor and the dining room's far wall) takes
    the term of the jamb's corner on the leg the jamb blocks, where the path over the
    corner touches it just above the floor. A centimetre on, that point lies below the
    corner's foot: the corner goes on as its image in the floor, the path over the
    image is the path over the corner on the leg after the floor bounce, and its term
    is the blocked leg's. The term ended there instead, and the level stepped by
    0.63 dB at 125 Hz and 0.11 dB from 250 Hz to 1 kHz; now it changes as it does over
    the centimetres either side, by 0.04 dB or less."""
    scene = Scene.from_yaml(EXAMPLES / "archway" / "scene.yaml")
    room = Room.from_scene(scene)
    listener = next(item for item in scene.listeners if item.name == "behind the wall")
    levels = []
    for at in (6.32, 6.33):
        moved = listener.model_copy(update={"position": (7.6, at, 1.2)})
        moved_room = Room.from_scene(scene.model_copy(update={"listeners": [moved]}), reuse=room)
        levels.append(10 * np.log10(total_energy(early_paths(moved_room, "singer", moved.name))))
    assert np.abs(levels[1] - levels[0])[:5].max() < 0.04


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
        # A path off the ceiling, the floor and the ceiling again, whose last reflection
        # lies past the ceiling's face where the corner blocks the leg after it: two
        # events stop it, and it was left out with its terms.
        (5.5, 11.99, 0),
        # Off wall_6, wall_7 and wall_2 (wall_7 perpendicular to the others): before
        # 16.40 the path reflects off wall_2 before wall_7, an order the rays had not
        # found, and the fade looked at no other.
        (3.0, 16.39, 0),
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


def test_a_path_missed_at_the_foot_of_the_gallery_s_wall_4(gallery: Room) -> None:
    """For the listener round the gallery's corner, as anywhere on the line x = 4 m 1.6 m
    high, a path off the floor and wall_6 that the early stage does not find reflects
    off the floor (as the image method places it) right at the foot of wall_4, where the
    floor's face stops. That face goes on only inside, so the leg arriving there is the
    one that crosses wall_4's plane, as for a listener 1 um farther from wall_6, where
    the reflection lies on the face; the level lies within 0.01 dB of the levels 1 um
    either side in every band. Taking the leg leaving it, as where the reflection lies
    past the face, gave 1.1 dB less at 63 Hz."""
    scene = gallery.scene
    listener = next(item for item in scene.listeners if item.name == "round the corner")
    levels = []
    for x in (4.0 - 1e-6, 4.0, 4.0 + 1e-6):
        moved = listener.model_copy(update={"position": (x, 15.0, 1.6)})
        room = Room.from_scene(scene.model_copy(update={"listeners": [moved]}), reuse=gallery)
        levels.append(10 * np.log10(total_energy(early_paths(room))))
    assert np.abs(levels[1] - levels[0]).max() < 0.01
    assert np.abs(levels[1] - levels[2]).max() < 0.01


def test_the_waves_keep_the_edge_fade_s_pieces_as_they_come(gallery: Room, monkeypatch) -> None:
    """The edge fade leaves out the pieces with nothing in them before it keeps their
    pressure, so that the waves keep the others as they come: they left those out by
    copying the rest, while the pages they came from stayed for the pieces next to them."""
    kept, left_out = [], []
    add, heard = Clusters.add, EdgeFade._heard

    def counted_add(self, rows, keys, pressure, distance, anchor=False, own=None):
        if np.asarray(pressure).dtype == np.complex64:
            energy = np.mean(np.abs(pressure) ** 2, axis=2)
            kept.append(energy.max(axis=1, initial=0.0) > 0.0)
        return add(self, rows, keys, pressure, distance, anchor, own)

    def counted_heard(self, pressure):
        out = heard(self, pressure)
        left_out.append(int(np.sum(~out["heard"])))
        return out

    monkeypatch.setattr(Clusters, "add", counted_add)
    monkeypatch.setattr(EdgeFade, "_heard", counted_heard)
    early_paths(gallery, max_order=3)
    assert sum(left_out) > 0 and kept and all(some.all() for some in kept)


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


def _outdoors(
    listener,
    *,
    barrier_m: float = 3.0,
    shed: bool = False,
    source=(0.0, 0.0, 1.0),
    path=((10, -30), (10, 30)),
    behind=None,
    **early,
) -> Room:
    """A ``source`` (by default 1 m over a ground), a barrier ``barrier_m`` high along
    ``path`` (by default 10 m away, along y), with ``behind`` another as high along that
    path, and, with ``shed``, a shed as high as the barrier out of the way."""
    geometry = {
        "ground": {"extent": [[-100, -100], [100, 100]], "sky": 40},
        "barriers": [
            {"name": name, "path": [list(p) for p in line], "height": barrier_m}
            for name, line in (("wall", path), ("wall_2", behind))
            if line is not None
        ],
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
                "assignments": {"wall": "hard", "wall_2": "hard", "shed": "hard"},
                "sources": [{"name": "s", "position": list(source)}],
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


#: Behind a barrier 20 m long on a rigid ground, from a source 1.5 m up 2 m in from its
#: end: the listeners either side of the heights where the reflection off the ground
#: crosses the barrier's foot (1.5 m) and where the paths over the end and the top that
#: reflect off the ground before them pass the ground there, and on the plane of the
#: top with the source there too; and behind the barrier bent 5 m from that end, where
#: the face of its far end lies in the other part's plane, and where the reflection lies
#: in front of the bent part's foot, past the plane of the near end's face; so behind a
#: curved one (six parts 4 m long, each turned 8 degrees from the last), too slightly
#: bent anywhere for an edge there; and behind barriers bent by 30 and 45 degrees away
#: from the source and by 18 degrees towards it, whose bends diffract.
_STRAIGHT, _BENT = ((10, -10), (10, 10)), ((10, -10), (10, -5), (13, 10))
_BENT_30, _BENT_45 = ((10, -10), (10, -5), (18.66, 10)), ((10, -10), (10, -5), (25, 10))
_BENT_IN = ((10, -10), (10, -5), (5, 10))
_TURNS = np.radians(90.0 - 8.0 * np.arange(6))
_CURVED = tuple(
    map(
        tuple,
        np.cumsum(
            [[10.0, -10.0], *(4.0 * np.column_stack([np.cos(_TURNS), np.sin(_TURNS)]))], axis=0
        ).tolist(),
    )
)
_FOOT_LISTENERS = (
    [(_STRAIGHT, (0.0, -8.0, 1.5), (20.0, -9.9, z)) for z in (1.47, 1.4725, 1.49, 1.4999, 1.5001)]
    + [
        (_STRAIGHT, (0.0, -8.0, 1.5), (20.0, -4.0, z))
        for z in (1.37, 1.39, 1.49, 1.51, 1.65, 1.67, 1.73)
    ]
    + [(_STRAIGHT, (0.0, -10.0, 3.0), (20.0, y, 3.0)) for y in (-9.9, -9.5)]
    + [(_BENT, (0.0, -8.0, 1.5), (20.0, -4.0, z)) for z in (1.49, 1.51)]
    + [(_BENT, (0.0, 0.0, 2.5), (20.0, 0.0, 2.2)), (_CURVED, (0.0, -8.0, 1.5), (20.0, 4.0, 1.4))]
    + [
        (_BENT_30, (0.0, 0.0, 2.5), (25.0, 0.0, 2.38)),
        (_BENT_45, (0.0, 0.0, 2.5), (25.0, 0.0, 2.2)),
    ]
    + [(_BENT_IN, (0.0, -8.0, 1.5), (20.0, -4.0, 1.6))]
)


@pytest.mark.parametrize(("path", "source", "listener"), _FOOT_LISTENERS)
def test_a_reflection_off_the_ground_behind_a_barrier_s_foot_is_bent_once(
    monkeypatch, path, source, listener
) -> None:
    """The reflection off the ground just behind a barrier's foot, which the barrier stops
    on the leg before it (or just in front, on the leg after it): the edge diffraction
    bends it over the top and round the ends, before the reflection and after it, and the
    fade adds nothing. The leg that gets past the barrier took terms of the one it stops,
    as one more event: of an end, whose path over it reflects off the ground on that
    leg's side (the leg it stops took it only where no other edge explained the stop and
    a face of that end blocked it), and of the top, where the line from the source's
    image to the listener came nearest to it above the ground though the barrier's face
    crosses it below; behind a bent or curved barrier, of its near end, whose face's
    plane crosses the leg that gets past in front of the part that stops the other (the
    leg it stops took it only where a face of that end blocked it, not where the face it
    bends into does). Their products with the stop's terms replaced the edge
    diffraction's own paths: the level stepped by up to 4.4 dB at 125 Hz between
    listeners 2.5 mm apart, and 5.3 dB 2 cm apart; they took 3.1 dB off the level at
    63 Hz behind the bent barrier, 1.1 dB at 250 Hz behind the curved one, and 3.6, 1.4
    and 6.3 dB in a band behind the bends of 30, 45 and 18 degrees. Where the bend
    itself diffracts, its two parts must still count as one obstacle, not two: else
    1.4 dB at 125 Hz is lost behind the bends of 30 and 45 degrees."""
    room = _outdoors(listener, source=source, path=path, max_order=2, diffraction="first_order")
    faded = total_energy(early_paths(room))
    monkeypatch.setattr(EdgeFade, "terms", lambda self, *args: None)
    plain = total_energy(early_paths(room))
    assert np.abs(10 * np.log10(faded / plain)).max() < 0.01


def test_a_barrier_bent_or_not_is_one_sheet_and_a_room_s_walls_are_not() -> None:
    """A barrier is one obstacle whether its bends diffract or not, and a building is
    one, but the walls of a room, which meet at inside corners, are six."""
    for path in (_STRAIGHT, _BENT, _BENT_30, _BENT_45, _BENT_IN, _CURVED):
        model = TraceModel.from_room(_outdoors((20.0, 0.0, 1.5), path=path))
        barrier = model.plane_two_sided[model.face_plane]
        assert len(np.unique(sheets(model)[barrier])) == 1
    model = TraceModel.from_room(_outdoors((20.0, 0.0, 1.5), shed=True))
    corners = np.asarray(model.room.mesh.vertices)[model.room.mesh.faces]
    shed = (corners[:, :, :2].min(axis=(1, 2)) >= 20.0) & (
        corners[:, :, :2].max(axis=(1, 2)) <= 26.0
    )
    assert shed.sum() == 10 and len(np.unique(sheets(model)[shed])) == 1
    room = Room.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")
    labels = sheets(TraceModel.from_room(room))
    assert np.array_equal(np.unique(labels, return_counts=True)[1], [2] * 6)


def test_in_a_model_not_oriented_only_the_faces_of_a_plane_join() -> None:
    """Where the model says not which side of its faces is solid (an open model, such as
    the leaky classroom's), an inside corner cannot be told from a bend that diffracts:
    the six surfaces of the room are six sheets, not one."""
    model = TraceModel.from_room(Room.from_yaml(EXAMPLES / "leaky_classroom" / "scene.yaml"))
    assert not model.oriented
    labels = sheets(model)
    assert len(np.unique(labels)) == model.plane_count
    assert all(len(np.unique(model.face_plane[labels == label])) == 1 for label in labels)


#: Two barriers 4 m apart, one shaped like a U and one like a Z, from the source at
#: (0, -8, 1.5) 2 m in from their ends: listeners either side of the height (1.715 m)
#: where the path from the source round the first barrier's end to the listener's image
#: in the ground passes the end's foot.
_BEHIND = {
    "two barriers": {"path": _STRAIGHT, "behind": ((14, -10), (14, 10))},
    "U": {"path": ((10, -10), (10, 10), (14, 10), (14, -10))},
    "Z": {"path": ((10, -10), (10, 0), (14, 0), (14, -10))},
}


@pytest.mark.parametrize("mode", ["first_order", "second_order"])
@pytest.mark.parametrize("shape", list(_BEHIND))
def test_a_path_over_an_edge_is_clear_past_the_ground(shape: str, mode: str) -> None:
    """Behind two barriers the paths over an edge that reflect off the ground are clear
    on every leg, not only up to the ground: the level does not step between 1.70 and
    1.72 m (under 0.15 dB from 63 to 500 Hz). Below, the paths of the missed reflection
    off the ground round the first barrier's end, on to the listener, and round the
    second's, back from the ground past the first's foot to the source, went through the
    other barrier: the level stepped by up to 15.7 dB at 63 Hz."""
    levels = [
        10
        * np.log10(
            total_energy(
                early_paths(
                    _outdoors(
                        (20.0, -4.0, z),
                        source=(0.0, -8.0, 1.5),
                        max_order=2,
                        diffraction=mode,
                        **_BEHIND[shape],
                    )
                )
            )
        )
        for z in (1.70, 1.72)
    ]
    assert np.abs(levels[1] - levels[0])[:4].max() < 0.15


# The level (dB re the direct sound in free field) behind that barrier, from the source
# at (0, -8, 1.5): the exact solution by images in the ground (the barrier and its image,
# a screen 6 m high, each of its four edges' BTM terms from the source and its image,
# validation/edge_diffraction.py's btm_first), averaged over 96 frequencies in each band.
_FOOT_EXACT = {
    (20.0, -9.9, 1.49): [2.46, -1.59, -3.11, -4.7, -10.88, -13.28, -14.0, -16.93],
    (20.0, -4.0, 1.45): [2.83, -5.2, -10.41, -9.81, -11.64, -14.98, -18.97, -22.25],
    (20.0, -4.0, 1.6): [2.61, -5.75, -9.4, -10.34, -11.56, -15.87, -19.32, -21.27],
}


#: Behind the curved barrier, from the source at (0, -8, 1.5), as _FOOT_EXACT has it (the
#: barrier and its image, each part's top and its image, the two ends) from 63 to 500 Hz.
_CURVED_EXACT = {(20.0, 4.0, 1.4): [-1.06, -4.09, -11.22, -7.1]}


@pytest.mark.parametrize("listener", list(_FOOT_EXACT))
def test_behind_a_barrier_s_foot_as_the_exact_solution(listener) -> None:
    """With both_sides the edge diffraction has the four paths over the top and the two
    round each end, and with the fade the level is the exact one, to 0.2 dB from 63 Hz
    to 1 kHz (the exact solution leaves out the air's absorption, 0.1 dB at 1 kHz over
    these 20 m, and the barrier's 5 %). Where the reflection off the ground lies just
    behind the barrier's foot or just in front of it, it was up to 7.1 dB too low."""
    source = (0.0, -8.0, 1.5)
    room = _outdoors(
        listener,
        source=source,
        path=_STRAIGHT,
        max_order=2,
        diffraction="first_order",
        diffraction_reflections="both_sides",
    )
    squared = float(np.sum((np.asarray(listener) - np.asarray(source)) ** 2))
    level = 10 * np.log10(total_energy(early_paths(room)) * 4 * np.pi * squared)
    assert np.abs(level - _FOOT_EXACT[listener])[:5].max() < 0.2


@pytest.mark.parametrize("listener", list(_CURVED_EXACT))
def test_behind_a_curved_barrier_as_the_exact_solution(listener) -> None:
    """Behind the curved barrier, where the reflection off the ground lies in front of a
    part's foot and the plane of the near end's face crosses the leg before it, with
    both_sides the level is the exact one to 0.1 dB from 63 to 500 Hz. The fade gave the
    leg before the reflection the near end's term as one more event: 0.6 dB too loud at
    63 Hz, 1.2 and 1.5 dB too soft at 125 and 250 Hz."""
    source = (0.0, -8.0, 1.5)
    room = _outdoors(
        listener,
        source=source,
        path=_CURVED,
        max_order=2,
        diffraction="first_order",
        diffraction_reflections="both_sides",
    )
    squared = float(np.sum((np.asarray(listener) - np.asarray(source)) ** 2))
    level = 10 * np.log10(total_energy(early_paths(room)) * 4 * np.pi * squared)
    assert np.abs(level[:4] - _CURVED_EXACT[listener]).max() < 0.1


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


@pytest.mark.parametrize("x", [-95.333, -78.0])
def test_no_path_reflects_off_a_roof_s_plane_from_below(x: float) -> None:
    """The train horn in the railway example, heard 25 m behind the barrier, where its
    reflection off the house's front wall and then off the plane of the roof, from below
    and 10 m in front of the house, has its point on the wall at an end of the wall.
    The house's corners, whose tops touch only the roof's corners, stood on the roof's
    plane, and the roof's front edge explained a reflection off the back of the roof:
    the level stepped by up to 2.0 dB at 2 kHz there, and such terms made it up to
    1.6 dB too loud at 250 Hz over 50 m of track. Now no path reflects off that plane
    outside the roof, and the level does not step."""
    scene = Scene.from_yaml(EXAMPLES / "railway" / "scene.yaml")
    horn = next(s for s in scene.sources if s.name == "train_horn")
    levels = []
    for at in (x - 0.01, x + 0.01):
        moved = horn.model_copy(update={"position": (at, 0.0, 2.2), "walk": None, "enabled": True})
        paths = early_paths(
            scene.model_copy(update={"sources": [moved]}), "train_horn", "behind_barrier_25m"
        )
        levels.append(10 * np.log10(total_energy(paths)))
        roof = [p for points in paths.points for p in points[1:-1] if abs(p[2] - 7.0) < 1e-6]
        assert all(-66.0 - 1e-6 <= p[0] <= -54.0 + 1e-6 and p[1] >= 20.0 - 1e-6 for p in roof)
    # But 8 kHz, where 2 cm is half a wavelength.
    assert np.abs(levels[1] - levels[0])[:-1].max() < 0.1
