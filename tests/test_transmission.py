"""Sound through surfaces: rooms found from the sides of the faces, partitions and
openings, against diffuse-field theory and against one room with the opening left out."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from scipy.linalg import expm

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.cli import main
from acoustic_spaces.engine import TraceModel, coverage_map
from acoustic_spaces.engine.diffraction import find_edges
from acoustic_spaces.engine.particles import trace_particles
from acoustic_spaces.engine.simulate import early_paths
from acoustic_spaces.parameters import decay_curve
from acoustic_spaces.scene import Coverage, Material

SOURCE = [1.3, 2.2, 1.6]


def _box(x0, x1, width, height, skip=()):
    """The quads of a box from x0 to x1, with the walls at x0 and x1 in ``skip`` left
    out."""
    quads = [
        ((x0, 0, 0), (x0, width, 0), (x1, width, 0), (x1, 0, 0)),
        ((x0, 0, height), (x1, 0, height), (x1, width, height), (x0, width, height)),
        ((x0, 0, 0), (x1, 0, 0), (x1, 0, height), (x0, 0, height)),
        ((x0, width, 0), (x0, width, height), (x1, width, height), (x1, width, 0)),
    ]
    if "x0" not in skip:
        quads.append(((x0, 0, 0), (x0, 0, height), (x0, width, height), (x0, width, 0)))
    if "x1" not in skip:
        quads.append(((x1, 0, 0), (x1, width, 0), (x1, width, height), (x1, 0, height)))
    return quads


def _coupled(path: Path, first, second, width, height, opening=None) -> Path:
    """Rooms from 0 to ``first`` and on to ``first + second`` along x, sharing a wall:
    a partition with an opening (y0, y1, z0, z1) in it, all opening (``"all"``), or all
    partition (None)."""
    x = first
    groups = {
        "room1": _box(0, first, width, height, skip=("x1",)),
        "room2": _box(first, first + second, width, height, skip=("x0",)),
    }
    if opening == "all":
        groups["opening"] = [((x, 0, 0), (x, width, 0), (x, width, height), (x, 0, height))]
    elif opening is None:
        groups["partition"] = [((x, 0, 0), (x, width, 0), (x, width, height), (x, 0, height))]
    else:
        y0, y1, z0, z1 = opening
        groups["opening"] = [((x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1))]
        groups["partition"] = [
            ((x, 0, 0), (x, y0, 0), (x, y0, height), (x, 0, height)),
            ((x, y1, 0), (x, width, 0), (x, width, height), (x, y1, height)),
            ((x, y0, z1), (x, y1, z1), (x, y1, height), (x, y0, height)),
        ]
    return _write(path, groups)


def _row(path: Path, rooms: int, length, width, height) -> Path:
    """``rooms`` rooms ``length`` long in a row along x, a partition between each two."""
    groups = {"walls": [], "partition": []}
    for k in range(rooms):
        skip = ("x0",) * (k > 0) + ("x1",) * (k < rooms - 1)
        groups["walls"] += _box(k * length, (k + 1) * length, width, height, skip=skip)
        if k > 0:
            x = k * length
            groups["partition"].append(
                ((x, 0, 0), (x, width, 0), (x, width, height), (x, 0, height))
            )
    return _write(path, groups)


def _write(path: Path, groups) -> Path:
    """An OBJ file of the quads of each group."""
    lines, n = [], 0
    for name, quads in groups.items():
        lines.append(f"g {name}")
        for quad in quads:
            lines += ["v " + " ".join(f"{c:g}" for c in p) for p in quad]
            lines.append(f"f {n + 1} {n + 2} {n + 3} {n + 4}")
            n += 4
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _scene(obj: Path, materials, assignments, listeners, source=SOURCE, **simulation) -> Scene:
    return Scene.model_validate(
        {
            "geometry": {"file": str(obj)},
            "materials": materials,
            "assignments": assignments,
            "sources": [{"name": "s", "position": source}],
            "listeners": [{"name": n, "position": p} for n, p in listeners.items()],
            "simulation": {"seed": 3, **simulation},
        }
    )


def _flat(alpha: float, **more) -> dict:
    return {"absorption": [alpha] * 8, "scattering": 1.0, **more}


def test_materials_let_sound_through() -> None:
    wall = Material(transmission_loss=[20, 25, 30, 35, 40, 45, 50, 55])
    passing = wall.transmission_coefficients()
    np.testing.assert_allclose(passing, 10.0 ** (-np.arange(20, 60, 5) / 10))
    # Without absorption values the absorption is what passes.
    np.testing.assert_allclose(wall.coefficients()[0], passing)
    opening = Material(transmission=1)
    assert opening.coefficients()[0] == [1.0] * 8
    assert opening.transmission_coefficients() == [1.0] * 8
    assert Material(absorption=[0.1] * 8).transmission_coefficients() is None
    with pytest.raises(ValidationError, match="more than the absorption"):
        Material(absorption=[0.1] * 8, transmission=0.2).coefficients()
    with pytest.raises(ValidationError, match="not both"):
        Material(transmission=0.1, transmission_loss=10)
    with pytest.raises(ValidationError):
        Material(transmission_loss=-3)


def test_rooms_come_from_the_sides_of_the_faces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    obj = _coupled(tmp_path / "rooms.obj", 4, 6, 4, 3, opening=(1.5, 2.5, 0.0, 2.0))
    scene = _scene(
        obj,
        {"wall": _flat(0.1), "door": {"transmission": 1}, "thin": _flat(0.1, transmission=0.01)},
        {"room1": "wall", "room2": "wall", "partition": "thin", "opening": "door"},
        {"one": [3, 1, 1], "two": [8, 3, 1]},
    )
    room = Room.from_scene(scene)
    np.testing.assert_allclose(sorted(room.room_volumes), [48.0, 72.0])
    assert room.volume_m3 == pytest.approx(120.0)
    assert room.problems() == [] and room.warnings() == []
    one, two, outside = room.room_of([[3, 1, 1], [8, 3, 1], [12, 3, 1]])
    assert one != two and one >= 0 and two >= 0 and outside == -1
    # The partition and the opening have a room on both sides, the outer walls on one.
    sides = (room.face_room >= 0).sum(axis=1)
    between = np.isin(room.mesh.face_labels, [room.mesh.labels.index(("partition",))])
    between |= np.isin(room.mesh.face_labels, [room.mesh.labels.index(("opening",))])
    assert np.all(sides[between] == 2) and np.all(sides[~between] == 1)

    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(scene.model_dump(mode="json")), encoding="utf-8")
    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "closed, 2 rooms of 48.00, 72.00 m3" in out
    rows = [line.split() for line in out.splitlines() if line.startswith("  R (dB)")]
    assert rows == [
        ["R", "(dB)", *["0.0"] * 8, "sound", "reduction"],
        ["R", "(dB)", *["20.0"] * 8, "sound", "reduction"],
    ]


def test_an_opening_is_not_there(tmp_path: Path) -> None:
    # Two 5 x 4 x 3 m rooms joined by an opening the size of the wall between them
    # sound like one 10 x 4 x 3 m room.
    alpha = {"absorption": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3], "scattering": 1.0}
    listeners = {"near": [3.9, 1.7, 1.2], "far": [8.3, 1.1, 2.1]}
    settings = {"ir_length_s": 1.5, "late": {"rays": 4000}}
    joined = _scene(
        _coupled(tmp_path / "joined.obj", 5, 5, 4, 3, opening="all"),
        {"wall": alpha, "opening": {"transmission": 1}},
        {"room1": "wall", "room2": "wall", "opening": "opening"},
        listeners,
        **settings,
    )
    single = Scene.model_validate(
        {
            **joined.model_dump(mode="json"),
            "geometry": {"box": [10, 4, 3]},
            "assignments": {},
            "default_material": "wall",
        }
    )
    a, b = simulate(Room.from_scene(joined)), simulate(Room.from_scene(single))
    for name in listeners:
        p, q = a["s", name].parameters(), b["s", name].parameters()
        np.testing.assert_allclose(p.t30[1:7], q.t30[1:7], rtol=0.04)
        np.testing.assert_allclose(p.g[1:7], q.g[1:7], atol=0.3)
        np.testing.assert_allclose(p.c80[1:7], q.c80[1:7], atol=0.5)


@pytest.mark.parametrize(("transmission", "volume"), [(0.01, 120.0), (0.0, 48.0)])
def test_walls_drawn_as_a_shell_join_the_rooms_sound_passes_between(
    tmp_path: Path, transmission: float, volume: float
) -> None:
    # Two rooms of 48 and 72 m3 inside a closed shell, as walls drawn as solids are, the
    # source and the listener in the first: the second holds air when the wall between
    # them lets some sound through, and not with transmission 0, which joined them too.
    low, high = (-1, -1, -1), (11, 5, 4)
    (x0, y0, z0), (x1, y1, z1) = low, high
    shell = [
        ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),
        ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),
        ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)),
        ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)),
        ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)),
    ]
    groups = {
        "room1": _box(0, 4, 4, 3, skip=("x1",)),
        "room2": _box(4, 10, 4, 3, skip=("x0",)),
        "partition": [((4, 0, 0), (4, 4, 0), (4, 4, 3), (4, 0, 3))],
        "shell": shell,
    }
    scene = _scene(
        _write(tmp_path / "shell.obj", groups),
        {"wall": _flat(0.2), "partition": _flat(0.2, transmission=transmission)},
        {"room1": "wall", "room2": "wall", "shell": "wall", "partition": "partition"},
        {"l": [2.5, 2.0, 1.5]},
    )
    assert Room.from_scene(scene).volume_m3 == pytest.approx(volume)


@pytest.mark.parametrize("loss_db", [30.0, 50.0])
def test_level_difference_across_a_partition(tmp_path: Path, loss_db: float) -> None:
    # Two 5 x 4 x 3 m rooms, 20 % absorbing everywhere, with a 12 m2 partition between
    # them. In diffuse fields the level behind it is G = 10 log10(1600 pi S tau / A^2),
    # A the absorption area of each room: the rooms' level difference R + 10 log10(A / S).
    # The source stands in the middle of its room: nearer the partition, its direct
    # sound would send more through than a diffuse field does.
    room = Room.from_scene(
        _scene(
            _coupled(tmp_path / "parted.obj", 5, 5, 4, 3),
            {"wall": _flat(0.2), "partition": _flat(0.2, transmission_loss=loss_db)},
            {"room1": "wall", "room2": "wall", "partition": "partition"},
            {"one": [3.9, 1.0, 1.0], "two": [8.0, 2.5, 1.5]},
            source=[2.1, 2.3, 1.7],
            ir_length_s=1.0,
            late={"rays": 3000},
        )
    )
    result = simulate(room)
    area = -2 * (5 * 4 + 5 * 3 + 4 * 3) * math.log(1 - 0.2)  # Eyring
    tau = 10 ** (-loss_db / 10)
    behind = 10 * math.log10(1600 * math.pi * 12 * tau / area**2)
    one, two = result["s", "one"].parameters().g, result["s", "two"].parameters().g
    np.testing.assert_allclose(two[:5], behind, atol=0.4)  # little air absorption here
    np.testing.assert_allclose((one - two)[:5], loss_db + 10 * math.log10(area / 12), atol=0.5)
    # A map over the second room finds what the listener there finds, away from the
    # partition (the rows 1.5 m and more from it are 0.2 to 0.3 dB lower, being farther;
    # the rows nearer it are higher, as the partition radiates into the room). So does
    # the statistical method. Its field reaches each point from the direct sound on, as
    # Barron has it: here 0.2 dB below what the rays find, 0.3 dB below the steady state.
    late = coverage_map(room, settings=Coverage(surfaces=["room2"], spacing=1.0)).get()
    away = late.grid.points[:, 0] > 6.5
    np.testing.assert_allclose(late.values["g"][away, :5].mean(axis=0), two[:5], atol=0.5)
    near = late.values["g"][~away, :5].mean(axis=0)
    assert np.all(near > late.values["g"][away, :5].mean(axis=0) + 0.5)
    room.scene.simulation.late.method = "statistical"
    statistical = simulate(room)["s", "two"].parameters().g
    np.testing.assert_allclose(statistical, two, atol=0.4)


def _double(path: Path, listeners, **simulation) -> tuple[Room, np.ndarray]:
    """A dry room (96 m3) opening through a 2 m2 doorway into a live one (192 m3), and
    the two-room diffuse-field model's decay rates (1/s) at 1 kHz: d(w1, w2)/dt =
    rates @ (w1, w2)."""
    first, second, width, height = 4, 8, 6, 4
    dry, live = 0.25, 0.03
    room = Room.from_scene(
        _scene(
            _coupled(path, first, second, width, height, (2.5, 3.5, 0, 2)),
            {"dry": _flat(dry), "live": _flat(live), "opening": {"transmission": 1}},
            {"room1": "dry", "room2": "live", "partition": "live", "opening": "opening"},
            listeners,
            ir_length_s=2.5,
            **simulation,
        )
    )
    c, m = room.speed_of_sound(), room.air_attenuation()[4]
    v1, v2, door = first * width * height, second * width * height, 2.0
    s1 = 2 * (first * width + first * height + width * height)
    s2 = 2 * (second * width + second * height + width * height)
    wall = width * height  # the partition is live, with the doorway in it
    a1 = -s1 * math.log(1 - ((s1 - wall) * dry + (wall - door) * live + door) / s1)
    a2 = -s2 * math.log(1 - ((s2 - door) * live + door) / s2)
    a1, a2 = a1 + 4 * m * v1, a2 + 4 * m * v2
    return room, (c / 4) * np.array([[-a1 / v1, door / v1], [door / v2, -a2 / v2]])


def test_coupled_rooms_decay_in_two_slopes(tmp_path: Path) -> None:
    # The dry room's decay falls fast, then follows the live room's, as the two-room
    # diffuse-field model has it.
    listeners = {"one": [2.9, 1.1, 1.2], "two": [9.0, 4.5, 1.5]}
    room, rates = _double(tmp_path / "double.obj", listeners, late={"rays": 4000})
    result = simulate(room)
    band = 4  # 1 kHz
    slow = 6 * math.log(10) / -np.linalg.eigvals(rates).real.max()

    response = result["s", "one"]
    step = response.time_step
    edc = decay_curve(response.energy()[band], step)
    times = np.arange(len(edc)) * step
    model = decay_curve(np.array([expm(rates * t)[0, 0] for t in times]) * step, step)
    start = np.argmax(edc < -5)  # past the direct sound, where the field is not diffuse
    for level in (10, 20, 30):
        k = np.argmax(edc < -level)
        assert edc[k] == pytest.approx(model[k] - model[start] + edc[start], abs=1.5)
    late = (edc < -25) & (edc > -40)
    late_t = -60 / np.polyfit(times[late], edc[late], 1)[0]
    early_t = -60 / np.polyfit(times[: np.argmax(edc < -10)], edc[: np.argmax(edc < -10)], 1)[0]
    assert late_t == pytest.approx(slow, rel=0.12)
    assert early_t < 0.5 * late_t


def test_the_statistical_method_and_maps_cross_rooms(tmp_path: Path) -> None:
    # The statistical method follows the two-room model in each room, and a map over
    # both rooms finds at its points what listeners there find (listeners where the map
    # has points: 1 m apart from the rooms' corner, 1.2 m up).
    listeners = {"one": [2.5, 1.5, 1.2], "two": [9.5, 4.5, 1.2]}
    room, rates = _double(tmp_path / "double.obj", listeners, late={"method": "statistical"})
    result = simulate(room)
    band = 4
    for k, name in enumerate(listeners):
        response = result["s", name]
        step = response.time_step
        edc = decay_curve(response.energy()[band], step)
        times = np.arange(len(edc)) * step
        model = decay_curve(np.array([expm(rates * t)[k, 0] for t in times]) * step, step)
        start = np.argmax(edc < -10)  # past the early paths
        for level in (20, 30, 40):
            at = np.argmax(edc < -level)
            assert edc[at] == pytest.approx(model[at] - model[start] + edc[start], abs=0.3)
    found = coverage_map(room, settings=Coverage(surfaces=["room1", "room2"], spacing=1.0))
    late = found.get()
    points = late.grid.points
    assert np.any(points[:, 0] < 4.0) and np.any(points[:, 0] > 4.0)
    for name, position in listeners.items():
        k = int(np.argmin(np.linalg.norm(points - position, axis=1)))
        np.testing.assert_allclose(points[k], position)
        values = result["s", name].parameters()
        for key in ("g", "t30", "c80"):
            np.testing.assert_allclose(late.values[key][k], getattr(values, key), atol=0.01)


def test_the_mixing_time_comes_from_the_listeners_room(tmp_path: Path) -> None:
    # The statistical method's mixing time and the density of the synthesized tail come
    # from the volume of the listener's room, not from both rooms' 288 m3. The
    # detector's radius (Lehnert's) stays that of all the rooms, which the rays fill.
    listeners = {"one": [2.0, 3.0, 2.0], "two": [8.0, 3.0, 2.0]}
    room, _ = _double(tmp_path / "double.obj", listeners, late={"rays": 1000})
    volumes = {"one": 4 * 6 * 4, "two": 8 * 6 * 4}
    outside = [20.0, 3.0, 2.0]
    np.testing.assert_allclose(room.volume_at([*listeners.values(), outside]), [96, 192, 288])
    radius = math.log10(room.speed_of_sound() * 2.5) * math.sqrt(288 / (math.pi * 1000))
    for name, volume in volumes.items():
        result = simulate(room, listeners=[name])
        assert result.details["s"]["receiver_radius_m"] == pytest.approx(radius)
        assert result["s", name].volume_m3 == pytest.approx(volume)
    room.scene.simulation.late.method = "statistical"
    result = simulate(room)
    for name, volume in volumes.items():
        response = result["s", name]
        mixing = response.transition_s - response.direct_delay_s
        assert mixing == pytest.approx(1e-3 * math.sqrt(volume))
        assert response.volume_m3 == pytest.approx(volume)


def _four_rooms(path: Path, **simulation) -> Room:
    """Four rooms 4 m long in a row, the source in the first and a listener in each,
    behind partitions that let 30 % of the sound through."""
    return Room.from_scene(
        _scene(
            _row(path, 4, 4.0, 3.5, 3.0),
            {
                "wall": _flat(0.15, scattering=0.3),
                "partition": _flat(0.35, scattering=0.3, transmission=0.3),
            },
            {"walls": "wall", "partition": "partition"},
            {f"r{k + 1}": [4.0 * k + 2.4, 2.1, 1.7] for k in range(4)},
            source=[1.3, 1.6, 1.4],
            **simulation,
        )
    )


def _arrivals(paths) -> set[tuple[int, int]]:
    """Each path's delay (us) and order."""
    return set(
        zip(np.round(paths.delay_s * 1e6).astype(int).tolist(), paths.order.tolist(), strict=True)
    )


def test_early_paths_pass_as_many_partitions_as_the_late_tracer_leaves_out(
    tmp_path: Path,
) -> None:
    # The stochastic late tracer counts every reflected path through more than two
    # partitions, so the early stage keeps none of them: the 4th room's early paths are
    # the direct sound alone. Ray tracing, which follows two partitions at most, then
    # finds what image sources find in every room.
    room = _four_rooms(tmp_path / "row.obj", early={"max_order": 3})
    for k in range(4):
        found = {
            method: early_paths(room, "s", f"r{k + 1}", method=method, diffraction="none")
            for method in ("ray_tracing", "image_sources")
        }
        assert _arrivals(found["ray_tracing"]) == _arrivals(found["image_sources"])
        assert np.any(found["image_sources"].order > 0) == (k < 3)
    # The statistical method takes the early paths' energy out of its own, whatever
    # paths they are: the early stage keeps those through three partitions.
    room.scene.simulation.late.method = "statistical"
    paths = early_paths(room, "s", "r4", method="image_sources", diffraction="none")
    assert np.any(paths.order > 0)


def test_paths_through_three_partitions_are_not_lost(tmp_path: Path) -> None:
    # Where the early stage keeps no reflected path, the late tracer counts them all:
    # the 4th room's response is the same whatever the early stage's order.
    responses = [
        simulate(
            _four_rooms(
                tmp_path / f"order{order}.obj",
                early={"max_order": order},
                late={"rays": 2000},
                ir_length_s=0.5,
            ),
            listeners=["r4"],
        )["s", "r4"]
        for order in (0, 3)
    ]
    np.testing.assert_allclose(responses[1].energy(), responses[0].energy(), rtol=1e-9)


def test_diffraction_bends_round_a_doorway(tmp_path: Path) -> None:
    # The opening is a hole for diffraction: its sides and lintel are free edges of the
    # partition, and sound bends round them to a listener out of sight in the next room.
    obj = _coupled(tmp_path / "door.obj", 4, 6, 4, 3, opening=(1.5, 2.5, 0.0, 2.0))
    scene = _scene(
        obj,
        {"wall": _flat(0.1), "door": {"transmission": 1}},
        {"room1": "wall", "room2": "wall", "partition": "wall", "opening": "door"},
        {"hidden": [6.0, 0.5, 1.0]},
        early={"max_order": 0, "diffraction": "direct"},
    )
    room = Room.from_scene(scene)
    edges = find_edges(TraceModel.from_room(room))
    free = edges.wedge == 2.0
    np.testing.assert_allclose(sorted(edges.length[free]), [1.0, 2.0, 2.0])
    np.testing.assert_allclose(edges.start[free][:, 0], 4.0)
    paths = early_paths(room)
    assert len(paths) and np.all(paths.diffracted)
    # A door that lets nothing through is a wall: no edges there, no sound.
    closed = scene.model_copy(deep=True)
    closed.materials["door"] = Material(**_flat(0.1))
    room = Room.from_scene(closed)
    assert not np.any(find_edges(TraceModel.from_room(room)).wedge == 2.0)
    assert len(early_paths(room)) == 0


def test_particles_and_the_wave_solver_pass_openings(tmp_path: Path) -> None:
    obj = _coupled(tmp_path / "door.obj", 4, 6, 4, 3, opening=(1.5, 2.5, 0.0, 2.0))
    assignments = {"room1": "wall", "room2": "wall", "partition": "wall", "opening": "door"}
    settings = {"ir_length_s": 0.3, "late": {"rays": 300}}
    wave = {"enabled": True, "max_band_hz": 63}

    def room(door, partition="wall", **more):
        materials = {"wall": _flat(0.1), "door": door, "thin": _flat(0.1, transmission_loss=20)}
        listeners = {"next": [7.0, 2.0, 1.5]}
        scene = _scene(obj, materials, {**assignments, "partition": partition}, listeners, **more)
        return Room.from_scene(scene)

    for door, through in (({"transmission": 1}, True), (_flat(0.1), False)):
        particles = trace_particles(room(door, **settings), count=2000, duration=0.05, frames=5)
        # Through the opening some particles reach the next room straight, as it
        # reflects nothing; behind a closed door none do.
        soon = particles.positions[1][:, 0] > 4.0  # after 12.5 ms
        assert np.any(soon & (particles.order[1] == 0)) == through
        assert np.any(particles.positions[-1][:, 0] > 4.0) == through
    # The wave solver's air flows through the opening, and not through a closed door.
    response = simulate(room({"transmission": 1}, wave=wave, **settings))["s", "next"]
    assert response.wave is not None and response.energy()[0].sum() > 0
    with pytest.raises(ValueError, match="listener 'next' is not in the air"):
        simulate(room(_flat(0.1), wave=wave, **settings))
    # A partition that lets some sound through is beyond the wave solver.
    with pytest.raises(ValueError, match="cannot pass sound through a surface; 'thin'"):
        simulate(room({"transmission": 1}, partition="thin", wave=wave, **settings))
