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
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.diffraction import find_edges
from acoustic_spaces.engine.edge_fade import _nearest_face
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
            lines += ["v " + " ".join(f"{c:.15g}" for c in p) for p in quad]
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


def _joined_and_single(tmp_path: Path, scattering: float, late: dict) -> list:
    """The parameters at two listeners of two 5 x 4 x 3 m rooms joined by an opening the
    size of the wall between them, and of one 10 x 4 x 3 m room, walls scattering
    `scattering`: [(joined, single) per listener]."""
    alpha = {
        "absorption": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3],
        "scattering": scattering,
    }
    listeners = {"near": [3.9, 1.7, 1.2], "far": [8.3, 1.1, 2.1]}
    joined = _scene(
        _coupled(tmp_path / "joined.obj", 5, 5, 4, 3, opening="all"),
        {"wall": alpha, "opening": {"transmission": 1}},
        {"room1": "wall", "room2": "wall", "opening": "opening"},
        listeners,
        ir_length_s=1.5,
        late=late,
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
    return [(a["s", name].parameters(), b["s", name].parameters()) for name in listeners]


def test_an_opening_is_not_there(tmp_path: Path) -> None:
    # The joined rooms sound like the one room.
    for p, q in _joined_and_single(tmp_path, 1.0, {"rays": 4000}):
        np.testing.assert_allclose(p.t30[1:7], q.t30[1:7], rtol=0.04)
        np.testing.assert_allclose(p.g[1:7], q.g[1:7], atol=0.3)
        np.testing.assert_allclose(p.c80[1:7], q.c80[1:7], atol=0.5)


def test_an_opening_leaves_the_walls_size_scattering_as_it_was(tmp_path: Path) -> None:
    # Walls that scatter little, and more as their size has it (the default): the
    # distance the sound came from to a wall runs through the opening. Without
    # splitting (which follows each room's level) the rays are the same in both models,
    # an opening drawing no random number, so they agree closely. (In the lowest bands
    # the size scattering, and so a vector reflection's direction, follows the rounding
    # of the distance summed past the opening, which the box's reflections grow.)
    for p, q in _joined_and_single(tmp_path, 0.05, {"rays": 2000, "splitting": False}):
        np.testing.assert_allclose(p.g, q.g, atol=0.005)
        np.testing.assert_allclose(p.t30[1:], q.t30[1:], rtol=0.005)
        np.testing.assert_allclose(p.t30[0], q.t30[0], rtol=0.01)


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


def test_sound_let_out_of_the_rooms_does_not_come_back_in(tmp_path: Path) -> None:
    # Two houses across a 1 m lane, their walls on the lane letting sound through. What
    # a wall lets out of a house goes into no room, and counts as absorbed: neither the
    # rays nor the rain of their reflections, which would go out through one wall and in
    # through the other, reach the second house. Through a wall the houses share, both
    # do.
    def houses(lane: float) -> dict:
        far = 4 + lane
        return {
            "house1": _box(0, 4, 4, 3, skip=("x1",)),
            "house2": _box(far, far + 4, 4, 3, skip=("x0",)),
            "lane": [((x, 0, 0), (x, 4, 0), (x, 4, 3), (x, 0, 3)) for x in {4, far}],
        }

    late = []
    for lane in (1.0, 0.0):
        room = Room.from_scene(
            _scene(
                _write(tmp_path / f"houses{lane:g}.obj", houses(lane)),
                {"wall": _flat(0.2), "thin": _flat(0.2, transmission_loss=10)},
                {"house1": "wall", "house2": "wall", "lane": "thin"},
                {"other": [4.0 + lane + 2.0, 2.1, 1.4]},
                source=[2.1, 1.9, 1.6],
                ir_length_s=0.3,
                late={"rays": 2000},
            )
        )
        assert len(TraceModel.from_room(room).room_volumes) == 2
        late.append(simulate(room)["s", "other"].late)
    apart, shared = late
    assert not apart.any()
    assert shared[:5].sum(axis=1).min() > 0.0


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


def _partitioned(obj: Path, source, listener, sized: bool = True, **early) -> Room:
    """The rooms ``obj`` (groups room1 and room2) with a 30 dB partition between them
    and any opening in it (group opening), all their faces absorbing 10 % and scattering
    nothing, nor for their size unless ``sized``."""
    materials = {
        "wall": _flat(0.1, scattering=0.0),
        "partition": _flat(0.1, scattering=0.0, transmission_loss=30),
        "door": {"transmission": 1},
    }
    assignments = {"room1": "wall", "room2": "wall", "partition": "partition", "opening": "door"}
    scene = _scene(
        obj,
        materials,
        assignments,
        {"l": list(listener)},
        source=source,
        early=early,
        scattering={"from_size": sized},
    )
    return Room.from_scene(scene)


@pytest.mark.parametrize("method", ["image_sources", "ray_tracing"])
def test_a_reflection_at_the_foot_of_a_partition_passes_through_it(
    tmp_path: Path, method: str
) -> None:
    """Listeners 3 m behind a 30 dB partition, as in the apartment example's bedroom,
    where the reflection off the floor lies on the partition's foot (x = 8), or less
    than the tolerances away: the reflection passes the partition once, as 1 mm either
    side, and the early paths' energy is theirs. At x = 8 it had passed the foot whole,
    and with image sources a second time as off the partition, the floor and the
    partition at that point: 20 to 24 dB more."""
    obj = _coupled(tmp_path / "rooms.obj", 5, 5, 4, 3)
    # The floor reflection at x = 2.5 + (x - 2.5) / 2.2. Just short of x = 8 the leg to
    # the listener meets the partition within the early stage's tolerance of the floor,
    # where the partition counts as the floor; just past it the leg to the source meets
    # the partition nearer than the intersection test's least distance, then farther.
    xs = (7.999, 8.0 - 1e-6, 8.0, 8.0 + 1e-8, 8.0 + 3e-7, 8.0 + 1e-6, 8.001)
    energy, floor, orders = [], [], []
    for x in xs:
        paths = early_paths(
            _partitioned(
                obj, [2.5, 1.0, 1.0], [x, 2.0, 1.2], max_order=3, method=method, diffraction="none"
            )
        )
        energy.append(paths.energy.sum(axis=0))
        orders.append(np.bincount(paths.order).tolist())
        (on_floor,) = [
            i for i in np.flatnonzero(paths.order == 1) if abs(paths.points[i][1][2]) < 1e-5
        ]
        assert paths.crossings[on_floor] == 1
        floor.append(paths.energy[on_floor])
    assert all(counts == orders[0] for counts in orders)
    for found in (energy, floor):
        ends = (found[0] + found[-1]) / 2
        np.testing.assert_allclose(found, np.broadcast_to(ends, (len(xs), 8)), rtol=1e-3)


def test_a_partition_drawn_as_one_face_for_each_room_is_one_partition(tmp_path: Path) -> None:
    """The partition above drawn as two coincident faces, one facing each room: the early
    paths are those of the partition drawn as one face, with the same crossings and
    energy, for listeners behind it (where the reflection off the floor lies on its
    foot, or 1 um either side, or away from it) and in the source's room. A path that
    reflected off both faces at one point had gone through the partition whole, as the
    direct sound, or, from the third order on, come back as the reflection off it once
    more, and the reflection at the foot had passed whole: at 1 kHz 22 to 26 dB more
    behind the partition and 0.05 to 0.11 dB more in the source's room. Counting both
    faces at the foot, the reflection there would pass the partition twice. No surface
    scatters for its size here: of the paths with one image (as off the partition on
    either side and through it) the two drawings keep different ones, whose surfaces
    scatter differently for their size."""
    quad = ((5, 0, 0), (5, 4, 0), (5, 4, 3), (5, 0, 3))
    one = _coupled(tmp_path / "one.obj", 5, 5, 4, 3)
    groups = {
        "room1": _box(0, 5, 4, 3, skip=("x1",)),
        "room2": _box(5, 10, 4, 3, skip=("x0",)),
        "partition": [quad, quad[::-1]],
    }
    two = _write(tmp_path / "two.obj", groups)
    for listener in (
        (8.0 - 1e-6, 2.0, 1.2),
        (8.0, 2.0, 1.2),
        (8.0 + 1e-6, 2.0, 1.2),
        (7.3, 3.1, 1.7),
        (3.5, 3.0, 1.5),
        (1.0, 3.0, 1.6),
    ):
        found = []
        for obj in (one, two):
            room = _partitioned(
                obj,
                [2.5, 1.0, 1.0],
                listener,
                sized=False,
                max_order=3,
                method="image_sources",
                diffraction="none",
            )
            paths = early_paths(room)
            order = np.lexsort((paths.energy[:, 4], paths.distance_m))
            found.append((paths.order[order], paths.crossings[order], paths.energy[order]))
        (orders, crossings, energy), (orders_two, crossings_two, energy_two) = found
        np.testing.assert_array_equal(orders_two, orders)
        np.testing.assert_array_equal(crossings_two, crossings)
        np.testing.assert_allclose(energy_two, energy, rtol=1e-9)


def test_a_reflection_at_the_foot_of_a_doorway_s_side_passes_the_doorway(
    tmp_path: Path,
) -> None:
    """Listeners behind the partition above with a doorway in it, from y = 1.5 to 2.5 m,
    where the reflection off the floor lies on the foot of either side of the doorway,
    or 1 um or 1 mm aside: on the partition's foot the reflection passes the partition
    (once), in the doorway it passes whole, and right at the doorway's side, past the
    end of the partition's face, it passes whole as well. On the partition's foot it had
    passed whole too."""
    obj = _coupled(tmp_path / "door.obj", 5, 5, 4, 3, opening=(1.5, 2.5, 0.0, 2.0))
    for side, into in ((1.5, 1.0), (2.5, -1.0)):
        floor = {}
        for offset in (-1e-3, -1e-6, 0.0, 1e-6, 1e-3):
            # The floor reflection at y = 2 + (y - 2) / 2.2.
            listener = (8.0, 2.0 + 2.2 * (side + into * offset - 2.0), 1.2)
            room = _partitioned(
                obj,
                [2.5, 2.0, 1.0],
                listener,
                max_order=1,
                method="image_sources",
                diffraction="none",
            )
            paths = early_paths(room)
            (on_floor,) = [
                i for i in np.flatnonzero(paths.order == 1) if abs(paths.points[i][1][2]) < 1e-5
            ]
            floor[offset] = (paths.crossings[on_floor], paths.energy[on_floor])
        for offset, (crossings, energy) in floor.items():
            beside = floor[-1e-3 if offset < 0 else 1e-3]
            assert crossings == beside[0] == (1 if offset < 0 else 0)
            np.testing.assert_allclose(energy, beside[1], rtol=1e-3)


# The middle listener of the rooms above (x = 7.5), 1 to 3 um from it along their middle
# line or off it, and 1 mm either side.
_IN_CORNERS = (
    (7.499, 2.0, 1.2),
    *((7.5 + offset, 2.0, 1.2) for offset in (-3e-6, -2e-6, -1.5e-6, -1e-6, 0.0)),
    *((7.5 + offset, 2.0, 1.2) for offset in (1.5e-6, 2e-6, 3e-6)),
    (7.5, 2.0, 1.2 - 1.4e-6),
    (7.5, 2.0, 1.2 + 1.8e-6),
    (7.5, 2.0 + 1.5e-6, 1.2 - 1.5e-6),
    (7.501, 2.0, 1.2),
)


@pytest.mark.parametrize("method", ["image_sources", "ray_tracing"])
def test_a_reflection_in_a_corner_at_a_partition_s_foot_passes_through_it(
    tmp_path: Path, method: str
) -> None:
    """The rooms above with the source and the listeners on their middle line, all as
    high: for the middle listener (x = 7.5), or 1 to 3 um from it along the line or off
    it, the reflections off the floor or the ceiling and a side wall lie in their corner
    right where the partition stands in it, or less than 3 um from it, and pass the
    partition there as 1 mm either side, so that the early level lies within 0.01 dB of
    the mean of theirs in every band. They had passed it whole: 14 to 28 dB more at
    1 kHz."""
    obj = _coupled(tmp_path / "rooms.obj", 5, 5, 4, 3)
    levels = []
    for listener in _IN_CORNERS:
        room = _partitioned(
            obj, [2.5, 2.0, 1.2], listener, max_order=3, method=method, diffraction="first_order"
        )
        levels.append(10 * np.log10(total_energy(early_paths(room))))
    beside = (levels[0] + levels[-1]) / 2
    assert np.abs(np.array(levels[1:-1]) - beside).max() < 0.01


@pytest.mark.parametrize("faces", [1, 2], ids=["one-face", "two-faces"])
@pytest.mark.parametrize("bend", [1e-6, 4e-6], ids=["1um", "4um"])
def test_a_reflection_at_the_foot_of_a_bent_partition_passes_it_as_a_flat_one(
    tmp_path: Path, bend: float, faces: int
) -> None:
    """The rooms and listeners above, with the partition's corner (5, 4, 3) moved ``bend``
    along x, the partition drawn as one face or as two (one facing each room, with the same
    triangles): each face's two triangles make one plane, the first's, which the second
    leaves by up to ``bend``. Each listener's paths pass the partition as with it flat:
    the same orders and crossings, and the early level within 0.01 dB in every band.
    Heights taken over the first triangle's plane alone had let reflected paths pass the
    partition whole for 9 of the 13 listeners with the corner moved 1 um and for 11 with
    it moved 4 um: up to 24 dB more."""

    def rooms(name: str, moved: float) -> Path:
        side = ((5, 0, 0), (5, 4, 0), (5 + moved, 4, 3), (5, 0, 3))
        partition = [side, (side[0], side[3], side[2], side[1])][:faces]
        groups = {
            "room1": _box(0, 5, 4, 3, skip=("x1",)),
            "room2": _box(5, 10, 4, 3, skip=("x0",)),
            "partition": partition,
        }
        return _write(tmp_path / name, groups)

    found = []
    for obj in (rooms("flat.obj", 0.0), rooms("bent.obj", bend)):
        found.append([])
        for listener in _IN_CORNERS:
            room = _partitioned(
                obj,
                [2.5, 2.0, 1.2],
                listener,
                max_order=3,
                method="image_sources",
                diffraction="first_order",
            )
            paths = early_paths(room)
            kinds = sorted(zip(paths.order.tolist(), paths.crossings.tolist(), strict=True))
            found[-1].append((kinds, 10 * np.log10(total_energy(paths))))
    for (flat_kinds, flat_level), (kinds, level) in zip(*found, strict=True):
        assert kinds == flat_kinds
        np.testing.assert_allclose(level, flat_level, rtol=0, atol=0.01)


# Cross-sections (y, z) of rooms along x: with side walls leaning in at 56 degrees to
# the floor and out at 108; leaning at 60 degrees, parallel; upright.
_LEANING = ((0, 0), (4, 0), (5, 3), (2, 3))
_SLANTED = ((0, 0), (4, 0), (4 + 3**0.5, 3), (3**0.5, 3))
_UPRIGHT = ((0, 0), (4, 0), (4, 3), (0, 3))


def _turned(y: float, z: float) -> float:
    """Where a partition turned and leaning stands across rooms along x."""
    return 5 - 0.3 * y - 0.25 * z


def _along_x(path: Path, section, partition) -> Path:
    """Two rooms along x, 10 m in all, with the cross-section ``section``, its floor from
    the first corner to the second, and between them the partition at x =
    ``partition``(y, z)."""

    def across(x):
        return [(x(y, z), y, z) for y, z in section]

    def sides(a, b):
        return [(a[i - 1], b[i - 1], b[i], a[i]) for i in range(len(section))]

    end, middle, other_end = across(lambda y, z: 0.0), across(partition), across(lambda y, z: 10.0)
    groups = {
        "room1": [*sides(end, middle), end],
        "room2": [*sides(middle, other_end), other_end],
        "partition": [middle],
    }
    return _write(path, groups)


@pytest.mark.parametrize(
    ("section", "partition", "source", "sides", "corner", "distance"),
    [
        (_LEANING, _turned, (0.6, 3.5, 1.6), (0, 3, 0), 0, 2.5),
        (_LEANING, _turned, (1.8, 3.7, 1.0), (3, 0, 3), 0, 2.5),
        (_LEANING, _turned, (1.2, 2.0, 1.1), (1,), 1, 2.5),
        (_LEANING, _turned, (2.8, 3.9, 2.0), (0,), 1, 2.5),
        (_SLANTED, lambda y, z: 5.0, (2.4, 3.246, 1.152), (3, 0, 3), 0, 0.75),
        (_SLANTED, lambda y, z: 5.0, (3.571, 3.599, 2.098), (1, 2, 1), 2, 0.75),
        (_UPRIGHT, lambda y, z: 5.0, (4.535, 2.951, 2.267), (2, 1), 2, 0.75),
    ],
    ids=[
        "floor-in-floor",
        "in-floor-in",
        "out",
        "floor",
        "wall-floor-wall",
        "wall-ceiling-wall",
        "upright",
    ],
)
def test_a_reflection_in_a_leaning_corner_at_a_partition_s_foot_passes_through_it(
    tmp_path: Path, section, partition, source, sides, corner, distance
) -> None:
    """Listeners ``distance`` from a corner of the partition's foot or top (``corner``
    of ``section``) in rooms with leaning walls or upright ones, on the line from the
    source's image in the planes of its ``sides`` through it, where the reflection off
    them lies right in the corner, or 1 to 3 um aside: the reflection passes the
    partition there as 1 mm either side, so that the early energy lies within 0.01 dB of
    the range of theirs in every band. Off the floor and the wall leaning in, in their
    56 degree corner, a path that reflects off one of them twice had passed it whole, as
    had one reflection off a single surface in the 108 degree corner: 13 to 16 dB more
    at 1 kHz. In rooms whose side walls lean at 60 degrees, paths that reflect off a
    wall, the floor or the ceiling and the wall again in their 60 degree corner, and in
    the upright rooms one off the ceiling and a wall, had passed it there and 1 to 3 um
    aside: 17 to 28 dB more at 1 kHz."""
    image = np.array(source, dtype=float)
    for side in sides:
        (y0, z0), (y1, z1) = section[side], section[(side + 1) % len(section)]
        normal = np.array([0.0, z0 - z1, y1 - y0]) / np.hypot(z0 - z1, y1 - y0)
        image -= 2 * ((image - (0.0, y0, z0)) @ normal) * normal
    y, z = section[corner]
    corner = np.array([partition(y, z), y, z])
    way = (corner - image) / np.linalg.norm(corner - image)
    aside = np.cross(way, [1.0, 0.3, 0.25])
    aside /= np.linalg.norm(aside)
    obj = _along_x(tmp_path / "rooms.obj", section, partition)
    levels = []
    for offset in (-1e-3, -3e-6, -2e-6, -1.5e-6, -1e-6, 0.0, 1e-6, 1.5e-6, 2e-6, 3e-6, 1e-3):
        listener = corner + distance * way + offset * aside
        room = _partitioned(
            obj, list(source), listener, max_order=3, method="image_sources", diffraction="none"
        )
        levels.append(10 * np.log10(early_paths(room).energy.sum(axis=0)))
    low, high = np.minimum(levels[0], levels[-1]), np.maximum(levels[0], levels[-1])
    assert np.all(np.array(levels[1:-1]) > low - 0.01)
    assert np.all(np.array(levels[1:-1]) < high + 0.01)


def _across_x(x: float, y0: float, y1: float) -> tuple:
    """A wall 3 m high across x at ``x``, from ``y0`` to ``y1``."""
    return ((x, y0, 0), (x, y1, 0), (x, y1, 3), (x, y0, 3))


def _across_y(y: float, x0: float, x1: float) -> tuple:
    """A wall 3 m high across y at ``y``, from ``x0`` to ``x1``."""
    return ((x0, y, 0), (x1, y, 0), (x1, y, 3), (x0, y, 3))


def _parted(path: Path, partitions) -> Path:
    """A 9.5 x 8.5 x 3 m space parted at x = 5 and y = 4, each part with its own floor and
    ceiling (group room1) and walls (room2), and the walls ``partitions`` between them."""
    halves = ((0, 5), (5, 9.5)), ((0, 4), (4, 8.5))
    floors = [
        ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))
        for x0, x1 in halves[0]
        for y0, y1 in halves[1]
        for z in (0, 3)
    ]
    walls = [_across_x(x, y0, y1) for x in (0, 9.5) for y0, y1 in halves[1]]
    walls += [_across_y(y, x0, x1) for y in (0, 8.5) for x0, x1 in halves[0]]
    return _write(path, {"room1": floors, "room2": walls, "partition": partitions})


@pytest.mark.parametrize(
    ("partitions", "order"),
    [
        ([_across_x(5, 0, 4), _across_y(4, 5, 9.5)], 3),
        ([_across_x(5, 0, 4), _across_x(5, 4, 8.5), _across_y(4, 5, 9.5)], 3),
        ([_across_x(5, 0, 4), _across_x(5, 4, 8.5), _across_y(4, 0, 5), _across_y(4, 5, 9.5)], 1),
    ],
    ids=["L", "T", "cross"],
)
def test_a_reflection_where_partitions_meet_passes_through_them(
    tmp_path: Path, partitions, order: int
) -> None:
    """A source at (2, 6, 1.5) and listeners at (8, 2, 1.5) in the parts of a space across
    partitions meeting at x = 5, y = 4, where the reflections off the floor and the
    ceiling lie right on the line where they meet, or, for listeners 1 to 3 um aside,
    0.5 to 1.5 um aside: each reflected path passes the partitions there as it does for
    listeners 1 mm to one side, with the same crossings and energy, and those found for
    listeners 1 mm to both sides are found. The listener's
    part is a bedroom in the corner of an L-shaped room, or one beside a wall going on
    (a T, met first by the leg from the listener), or one of four around a cross, for which
    only the reflections of the first order are taken: the legs of later ones pass the
    line where the partitions meet, as the direct sound does, and a leg there counts one
    of the partitions only. The space is 9.5 x 8.5 m, not 10 x 8, about whose middle the
    source and the listener lie: there the legs between opposite walls pass the line as
    well. The reflections had passed both walls of the bedroom whole, at the line and
    for listeners 1 to 3 um aside, and the T at the line and to the side where it counts
    one wall, and one of its two walls for listeners 1 to 3 um to the other side; at the
    cross they had passed one wall for listeners 1 to 3 um aside and both at the line."""
    source, listener = (2.0, 6.0, 1.5), np.array([8.0, 2.0, 1.5])
    aside = np.array([2.0, 3.0, 0.0]) / 13**0.5  # level, across the path
    obj = _parted(tmp_path / "parted.obj", partitions)
    found = []
    for offset in (-1e-3, -3e-6, -1e-6, 0.0, 1e-6, 3e-6, 1e-3):
        room = _partitioned(
            obj,
            list(source),
            listener + offset * aside,
            max_order=order,
            method="image_sources",
            diffraction="none",
        )
        planes = TraceModel.from_room(room).face_plane
        paths = early_paths(room)
        found.append(
            {
                tuple(planes[paths.faces[i]].tolist()): (paths.crossings[i], paths.energy[i])
                for i in np.flatnonzero(paths.order > 0)
            }
        )
    one_side, other_side = found[0], found[-1]
    for here in found[1:-1]:
        assert set(one_side) & set(other_side) <= set(here)
        for key, (crossings, energy) in here.items():
            assert any(
                key in side
                and side[key][0] == crossings
                and np.allclose(side[key][1], energy, rtol=1e-3, atol=0)
                for side in (one_side, other_side)
            ), key


_STEM_END = 5 + 4 / math.tan(math.radians(75))  # where the stem below meets the wall y = 0


def _place(groups, moved):
    """``groups`` with each corner (x, y, z) in ``moved`` moved by the vector (um) given
    for it, in every face it is a corner of."""

    def place(corner):
        for at, by in moved.items():
            if np.allclose(corner, at, rtol=0, atol=1e-9):
                return tuple(c + 1e-6 * b for c, b in zip(corner, by, strict=True))
        return corner

    return {name: [tuple(map(place, quad)) for quad in quads] for name, quads in groups.items()}


def _tee(path: Path, moved) -> Path:
    """A 10 x 8 x 3 m room parted at y = 4 by a partition in two faces (x 0 to 5 and 5 to
    10) and, where y < 4, by a stem from (5, 4) at 75 degrees to it to the wall y = 0 at
    x = ``_STEM_END``; the floor and the ceiling each in three pieces (y > 4, and either
    side of the stem), all but the partitions in group room1. The corners in ``moved``
    are moved (see ``_place``)."""
    x = _STEM_END
    floors = []
    for z in (0, 3):
        floors += [
            ((0, 4, z), (10, 4, z), (10, 8, z), (0, 8, z)),
            ((0, 0, z), (x, 0, z), (5, 4, z), (0, 4, z)),
            ((x, 0, z), (10, 0, z), (10, 4, z), (5, 4, z)),
        ]
    walls = [_across_x(0, 0, 4), _across_x(0, 4, 8), _across_x(10, 0, 4), _across_x(10, 4, 8)]
    walls += [_across_y(0, 0, x), _across_y(0, x, 10), _across_y(8, 0, 10)]
    stem = ((5, 4, 0), (x, 0, 0), (x, 0, 3), (5, 4, 3))
    groups = {
        "room1": floors + walls,
        "partition": [_across_y(4, 0, 5), _across_y(4, 5, 10), stem],
    }
    return _write(path, _place(groups, moved))


def _lean_to(path: Path, moved) -> Path:
    """A 10 x 6 m room under a roof rising from 2 m high at y = 0 to 5 m at y = 6, parted
    at x = 5 by a partition up to the roof; the floor, the roof and the walls y = 0 and
    y = 6 each in two pieces (x 0 to 5 and 5 to 10), all but the partition in group
    room1. The corners in ``moved`` are moved (see ``_place``)."""
    ends = (
        ((0, 0, 2), (0, 6, 5), (0, 6, 0), (0, 0, 0)),
        ((10, 0, 0), (10, 6, 0), (10, 6, 5), (10, 0, 2)),
    )
    walls = []
    for (x0, x1), end in zip(((0, 5), (5, 10)), ends, strict=True):
        walls += [
            ((x0, 6, 0), (x1, 6, 0), (x1, 0, 0), (x0, 0, 0)),
            ((x0, 0, 2), (x1, 0, 2), (x1, 6, 5), (x0, 6, 5)),
            ((x0, 0, 0), (x1, 0, 0), (x1, 0, 2), (x0, 0, 2)),
            ((x0, 6, 5), (x1, 6, 5), (x1, 6, 0), (x0, 6, 0)),
            end,
        ]
    groups = {"room1": walls, "partition": [((5, 0, 0), (5, 6, 0), (5, 6, 5), (5, 0, 2))]}
    return _write(path, _place(groups, moved))


@pytest.mark.parametrize(
    ("room", "moved", "corner", "source", "listener"),
    [
        (
            "tee",
            {
                (5, 4, 0): (-0.8, -0.4, 0.4),
                (5, 4, 3): (-0.1, -1.0, 0.0),
                (_STEM_END, 0, 0): (-0.6, -0.4, 0.7),
                (_STEM_END, 0, 3): (0.5, 0.3, 0.8),
                (10, 4, 0): (-0.9, -0.2, -0.4),
            },
            (_STEM_END, 0, 0),
            (4.59618, 2.261071, 1.654931),
            (5.955446, 2.128288, 1.306465),
        ),
        (
            "tee",
            {
                (5, 4, 3): (-0.5, -4.0, 0.1),
                (10, 4, 3): (1.6, 2.2, 3.0),
                (10, 8, 0): (1.0, 3.7, 1.0),
                (10, 8, 3): (3.7, -1.3, 1.0),
            },
            (5, 4, 3),
            (9.814034, 6.789543, 2.720039),
            (1.948614, 0.697341, 2.822548),
        ),
        (
            "tee",
            {
                (0, 8, 0): (-0.7156, -0.3065, 0.6277),
                (5, 4, 0): (-0.8082, -0.3831, 0.4472),
                (10, 4, 0): (-0.9067, -0.1676, -0.3871),
                (10, 8, 0): (0.2496, 0.9343, 0.2545),
            },
            (5, 4, 0),
            (4.813290556, 3.12104762, 0.438841995),
            (3.285374497, 5.904772596, 1.251639105),
        ),
        (
            "lean-to",
            {
                (0, 0, 2): (-0.3856, 0.7937, -0.4705),
                (0, 6, 0): (-0.2397, -0.7002, -0.6725),
                (0, 6, 5): (-0.5257, 0.4974, 0.6901),
                (5, 0, 2): (0.8505, -0.2658, -0.4539),
                (5, 6, 5): (-0.8844, 0.0179, -0.4665),
            },
            (5, 6, 5),
            (6.383165524, 5.021858042, 0.831056316),
            (3.579157776, 3.176864985, 1.626674097),
        ),
    ],
    ids=["stem-foot-1um", "ceiling-4um", "floor-1um", "lean-to-1um"],
)
def test_a_reflection_where_moved_partitions_meet_passes_them(
    tmp_path: Path, room, moved, corner, source, listener
) -> None:
    """The room of ``_tee`` or ``_lean_to`` with a few corners moved about 1 um, or about
    4 um, so that faces lie up to 1.8 um (stem-foot), 5.0 um (ceiling), 2.1 um (floor)
    or 3.1 um (lean-to) off the planes they are merged into, and a source and a listener
    either side of a partition. For the listener and for listeners 0.01 to 3 um to
    either side of it on two lines across the way to a corner, the early level in each
    band is no more than 3 dB above the higher of those for listeners 0.2 mm to either
    side (here at most 1.9 dB above it). For some of them a path reflects off the stem,
    the wall y = 0 and the floor within 2.1 um of the corner where they meet
    (stem-foot); off the stem and the floor within 3.3 um of the corner where they meet
    the partition (floor); off the wall y = 6, the roof and the wall again within 1.2 um
    of the corner where they meet the partition (lean-to). For two of them (ceiling),
    the leg to the ceiling of a path off the wall y = 8 and the ceiling meets the
    partition 6.7 and 6.9 um from the corner where the partition and the stem meet the
    ceiling.
    Counting a reflection point as on a plane the path reflects off wherever it lay
    within the spread of that plane's faces had let the stem-foot path pass the stem
    whole; taking that hit on the partition as the reflection off the ceiling (within
    1 um of the plane of a face of the ceiling, more than 1 um below the ceiling's plane)
    had let the ceiling path pass the partition whole; counting the floor among the
    surfaces met where the floor path reflects off the stem (on the partition, 1.1 um
    above the floor's plane and 0.45 um above the plane of the floor's faces there) only
    where that lay within 1 um of the floor's plane had let it pass the partition whole;
    and taking the lean-to path's reflection point on the wall as on the roof as well
    only where it lay within 1 um of the roof's plane (it lies 1.3 um from it, and
    0.03 um from the plane of the roof's face there) had let it pass the partition
    whole: 12, 11, 19 and 12 dB more."""
    obj = {"tee": _tee, "lean-to": _lean_to}[room](tmp_path / "room.obj", moved)
    listener = np.array(listener)
    toward = (listener - corner) / np.linalg.norm(listener - corner)
    across = np.cross(toward, (0.3, 0.2, 1.0))
    across /= np.linalg.norm(across)
    offsets = [-3, -2, -1, -0.5, -0.1, -0.01, 0, 0.01, 0.1, 0.5, 1, 2, 3]
    for aside in (across, np.cross(toward, across)):
        levels = []
        for offset in (-2e-4, *(1e-6 * k for k in offsets), 2e-4):
            room_ = _partitioned(
                obj,
                list(source),
                listener + offset * aside,
                max_order=3,
                method="image_sources",
                diffraction="none",
            )
            levels.append(10 * np.log10(total_energy(early_paths(room_))))
        levels = np.array(levels)
        assert np.all(levels[1:-1] <= np.maximum(levels[0], levels[-1]) + 3.0)


def _door_in_corner(path: Path, moved) -> Path:
    """Rooms from x = 0 to 5 and on to 10 (4 x 3 m), each with its own floor, ceiling and
    walls y = 0 and y = 4 (group room1); the partition at x = 5 between them with a
    doorway 1 m wide and 2 m high (group opening) in its corner with the wall y = 0. The
    corners in ``moved`` are moved (see ``_place``)."""
    walls = []
    for x0, x1 in ((0, 5), (5, 10)):
        walls += [
            ((x0, 0, 0), (x1, 0, 0), (x1, 4, 0), (x0, 4, 0)),
            ((x0, 0, 3), (x1, 0, 3), (x1, 4, 3), (x0, 4, 3)),
            _across_y(0, x0, x1),
            _across_y(4, x0, x1),
        ]
    walls += [_across_x(0, 0, 4), _across_x(10, 0, 4)]
    groups = {
        "room1": walls,
        "partition": [_across_x(5, 1, 4), ((5, 0, 2), (5, 1, 2), (5, 1, 3), (5, 0, 3))],
        "opening": [((5, 0, 0), (5, 1, 0), (5, 1, 2), (5, 0, 2))],
    }
    return _write(path, _place(groups, moved))


def test_a_reflection_where_a_moved_doorway_meets_a_wall_is_kept(tmp_path: Path) -> None:
    """The rooms of ``_door_in_corner`` with seven corners moved about 1 um, so that faces
    lie up to 1.8 um off the planes they are merged into. A path from the source off the
    wall y = 4, the floor and the wall y = 0 reflects off the wall y = 0 1.008 m up, within
    2.2 um of the side of the doorway (39 to 140 um for the listeners 0.2 mm aside), and
    goes on through the doorway. For a listener there and for listeners 0.01 to 3 um to
    either side of it on two lines across the way to (5, 0, 2.5), and 0.2 mm to either
    side, the early level in each band lies within 3 dB of that for the same listener
    with no corners moved (here within 0.6 dB). For one of them the leg from the
    listener meets the doorway 0.54 um from its side, 0.03 um behind the wall's plane,
    which makes it the reflection off the wall, and 0.54 um in front of the plane of the
    wall's face there. Taking the nearer of those two planes for the side the point lies
    on had lost the path for that listener (8.7 dB less), and taking the wall's plane
    alone had lost it for all of them (15 dB less)."""
    moved = {
        (0, 0, 0): (0.9692, 0.0684, -0.2366),
        (0, 0, 3): (0.3723, 0.2731, 0.887),
        (0, 4, 0): (0.4942, 0.5483, 0.6746),
        (0, 4, 3): (-0.9031, 0.4279, -0.036),
        (5, 0, 0): (-0.6068, 0.7266, -0.3223),
        (5, 0, 2): (0.8505, -0.2658, -0.4539),
        (5, 4, 0): (-0.8082, -0.3831, 0.4472),
    }
    source, listener = (
        (4.209597275, 0.287821272, 2.674231899),
        (5.305782243, 2.983614199, 2.432594281),
    )
    objs = [
        _door_in_corner(tmp_path / name, by)
        for name, by in (("moved.obj", moved), ("flat.obj", {}))
    ]
    listener = np.array(listener)
    toward = (listener - (5, 0, 2.5)) / np.linalg.norm(listener - (5, 0, 2.5))
    across = np.cross(toward, (0.3, 0.2, 1.0))
    across /= np.linalg.norm(across)
    offsets = [-3, -2, -1, -0.5, -0.1, -0.01, 0, 0.01, 0.1, 0.5, 1, 2, 3]
    for aside in (across, np.cross(toward, across)):
        for offset in (-2e-4, *(1e-6 * k for k in offsets), 2e-4):
            levels = []
            for obj in objs:
                room = _partitioned(
                    obj,
                    list(source),
                    listener + offset * aside,
                    max_order=3,
                    method="image_sources",
                    diffraction="none",
                )
                levels.append(10 * np.log10(total_energy(early_paths(room))))
            np.testing.assert_allclose(levels[0], levels[1], rtol=0, atol=3.0)


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


@pytest.mark.parametrize("method", ["image_sources", "ray_tracing"])
def test_no_path_reflects_off_an_opening(tmp_path: Path, method: str) -> None:
    """An opening reflects nothing: where the reflection off the wall would lie in the
    doorway, no path reflects there. It had been found with no energy."""
    obj = _coupled(tmp_path / "door.obj", 4, 6, 4, 3, opening=(1.5, 2.5, 0.0, 2.0))
    scene = _scene(
        obj,
        {"wall": _flat(0.1, scattering=0.0), "door": {"transmission": 1}},
        {"room1": "wall", "room2": "wall", "partition": "wall", "opening": "door"},
        {"l": [2.0, 2.0, 1.0]},  # the reflection off x = 4 at y = 2.09, z = 1.26
        early={"max_order": 2, "method": method, "diffraction": "none"},
    )
    room = Room.from_scene(scene)
    transparent = TraceModel.from_room(room).transparent
    paths = early_paths(room)
    assert np.any(paths.order > 0)
    assert not any(transparent[faces].any() for faces in paths.faces)


def _doorway(path: Path) -> Path:
    """A 10 x 8 x 3 m room with a doorway from x = 4 to 6, as high as the room, in its
    wall y = 0; the floor and the ceiling in three pieces that meet the doorway's
    sides."""
    xs = [0.0, 4.0, 6.0, 10.0]
    width, height = 8.0, 3.0
    groups: dict[str, list] = {"walls": [], "door": [], "floor": [], "ceiling": []}
    for x0, x1, name in zip(xs[:-1], xs[1:], ("walls", "door", "walls"), strict=True):
        groups[name].append(((x0, 0, 0), (x1, 0, 0), (x1, 0, height), (x0, 0, height)))
        groups["floor"].append(((x0, 0, 0), (x0, width, 0), (x1, width, 0), (x1, 0, 0)))
        groups["ceiling"].append(
            ((x0, 0, height), (x1, 0, height), (x1, width, height), (x0, width, height))
        )
    groups["walls"] += [
        ((0, width, 0), (0, width, height), (10, width, height), (10, width, 0)),
        ((0, 0, 0), (0, 0, height), (0, width, height), (0, width, 0)),
        ((10, 0, 0), (10, width, 0), (10, width, height), (10, 0, height)),
    ]
    return _write(path, groups)


def test_no_step_where_a_reflection_crosses_a_doorway_s_side(tmp_path: Path) -> None:
    """Where the reflection off the wall crosses the side of a doorway in it, from the
    doorway onto the wall, the level for listeners 2 mm apart (their reflections 1.2 mm
    apart) steps by less than 0.1 dB in every band: the edge diffraction bends the
    reflection round the doorway's side on both sides of it. The reflection off the
    doorway, found with no energy, had taken the place of that term on the doorway's
    side: 2.0 dB at 63 Hz and 0.1 to 0.6 dB above."""
    obj = _doorway(tmp_path / "doorway.obj")
    levels = []
    for x in (7.999, 8.001):  # the reflection at x = 3 + (x - 3) * 0.6
        scene = _scene(
            obj,
            {"hard": _flat(0.05, scattering=0.05), "open": {"transmission": 1}},
            {"walls": "hard", "floor": "hard", "ceiling": "hard", "door": "open"},
            {"l": [x, 2.0, 1.5]},
            source=[3.0, 3.0, 1.5],
            early={"max_order": 3, "diffraction": "first_order"},
        )
        levels.append(10 * np.log10(total_energy(early_paths(Room.from_scene(scene)))))
    assert np.abs(levels[1] - levels[0]).max() < 0.1


def test_the_face_nearest_to_a_point_in_a_doorway_is_the_wall_beside_it(
    tmp_path: Path,
) -> None:
    """The edge fade gives a reflection off its faces, next to the leg that stops its
    path, the face of its plane nearest to the point. In a doorway that is the wall
    beside it, as an opening reflects nothing; it had been the doorway, which the point
    lies in and which absorbs all sound."""
    scene = _scene(
        _doorway(tmp_path / "doorway.obj"),
        {"hard": _flat(0.05), "open": {"transmission": 1}},
        {"walls": "hard", "floor": "hard", "ceiling": "hard", "door": "open"},
        {"l": [5.0, 4.0, 1.5]},
    )
    model = TraceModel.from_room(Room.from_scene(scene))
    point = np.array([5.5, 0.0, 1.5])
    plane = model.face_plane[np.flatnonzero(model.transparent)[0]]
    face = _nearest_face(model, plane, point)
    assert not model.transparent[face]
    assert model.room.mesh.triangles[face][:, 0].min() == 6.0  # the wall from x = 6 on


def test_particles_reflect_past_an_opening_as_if_it_were_not_there(
    tmp_path: Path, monkeypatch
) -> None:
    # Each particle's first reflection in the joined rooms and in the one room: the same
    # face met along the same first leg from the source, so the same distance (through
    # the opening) for the size scattering and the same count of reflections.
    from acoustic_spaces.engine import particles

    reflection = particles._reflection

    def first_reflections(scene: Scene) -> tuple:
        distance, order = np.full(300, np.nan), np.zeros(300, dtype=np.int64)

        def recorded(model, face, d, dist, alive, now, *rest):
            first = alive & ~model.transparent[face] & np.isnan(distance)
            distance[first] = dist[first]
            order[first] = np.broadcast_to(now, first.shape)[first]
            return reflection(model, face, d, dist, alive, now, *rest)

        monkeypatch.setattr(particles, "_reflection", recorded)
        trace_particles(Room.from_scene(scene), count=300, duration=0.04, frames=2)
        return distance, order

    joined = _scene(
        _coupled(tmp_path / "joined.obj", 5, 5, 4, 3, opening="all"),
        {"wall": {"absorption": [0.1] * 8, "scattering": 0.05}, "opening": {"transmission": 1}},
        {"room1": "wall", "room2": "wall", "opening": "opening"},
        {"l": [8.0, 2.0, 1.5]},
    )
    single = Scene.model_validate(
        {
            **joined.model_dump(mode="json"),
            "geometry": {"box": [10, 4, 3]},
            "assignments": {},
            "default_material": "wall",
        }
    )
    (a, order_a), (b, order_b) = first_reflections(joined), first_reflections(single)
    assert not np.isnan(a).any() and np.any(a > 5.0)  # some through the opening first
    np.testing.assert_allclose(a, b, rtol=1e-9)
    assert (order_a == 1).all() and (order_b == 1).all()


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
