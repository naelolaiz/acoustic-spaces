"""The scattered sound of the reflections straight from the source, shared between the
rays' rain and a gather from each listener (``_native.gather_first_scattering``, with
``late.FIRST_GATHER`` directions in a run)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_transmission import _coupled, _flat, _scene

from acoustic_spaces import Room, _native
from acoustic_spaces.engine import late
from acoustic_spaces.engine.model import TraceModel

ALPHA, SCATTERING = 0.2, 0.5
#: The gather's arguments, as the tracer's.
GATHERED = (
    "bvh source directivity_a orientation directivity_table absorption scattering air "
    "surface_size band_hz size_from_order transmission face_room source_room receivers "
    "directions max_time time_step dir_time_step speed_of_sound receiver_radius threads "
    "face_group groups moment_time_step gather_ratio"
).split()
PLANTS = ("plants", "plant_outline", "plant_absorption", "plant_scattering")
PLANTS += ("plant_asymmetry", "plant_upright")


def _plane(source, receivers, rays: int, seed: int, ratio: float, **more) -> dict:
    """The tracer's arguments for ground 400 m across, absorbing ALPHA and scattering
    SCATTERING in one band, in silent air: with the specular reflections left to the
    early stage, only the rain reaches the listeners (0.5 m spheres)."""
    vertices = np.array([[-200, -200, 0], [200, -200, 0], [200, 200, 0], [-200, 200, 0]], float)
    receivers = np.asarray(receivers, float)
    args = dict(
        bvh=_native.Bvh(vertices, np.array([[0, 1, 2], [0, 2, 3]], np.int64)),
        source=np.asarray(source, float),
        directivity_a=np.zeros(0),
        orientation=np.eye(3),
        directivity_table=np.zeros((0, 0, 1)),
        absorption=np.full((2, 1), ALPHA),
        scattering=np.full((2, 1), SCATTERING),
        air=np.zeros(1),
        surface_size=np.zeros((0, 2)),
        band_hz=np.zeros(0),
        vector_scattering=False,
        size_from_order=1,
        transmission=np.zeros((0, 0)),
        face_room=np.zeros((0, 2), np.int64),
        source_room=-1,
        receivers=receivers,
        directions=np.zeros((0, 3)),
        rays=rays,
        max_time=1.0,
        time_step=1e-3,
        dir_time_step=1e-2,
        speed_of_sound=343.0,
        receiver_radius=np.full(len(receivers), 0.5),
        diffuse_rain=True,
        specular_rain=False,
        covered_order=1,
        covered_time=np.zeros(0),
        seed=seed,
        threads=0,
        window_time_step=0.01,
        first_ray=0,
        split_target=np.zeros(0),
        split_ratio=4.0,
        max_split=8,
        max_family=256,
        room_level=np.zeros(0),
        face_group=np.zeros(0, np.int64),
        groups=0,
        moment_time_step=0.01,
        gather_ratio=ratio,
    )
    args.update(more)
    return args


def _both(args: dict, samples: int) -> tuple:
    """The tracer's (energy, directional, moments) with the gather's added."""
    traced = _native.trace_stochastic(**args)
    if not samples:
        return traced[:3]
    chosen = {k: args[k] for k in (*GATHERED, *PLANTS) if k in args}
    turn = late.gather_turn(args["seed"])
    gathered = _native.gather_first_scattering(**chosen, samples=samples, turn=turn)
    return tuple(a + b for a, b in zip(traced[:3], gathered, strict=True))


def _rain(source, receivers, seeds: int, samples: int, rays: int = 4000, **more) -> np.ndarray:
    """The energy at each listener (seeds, listeners): the rays' rain alone (samples 0)
    or shared with a gather in that many directions."""
    ratio = samples / rays if samples else 0.0
    return np.array(
        [
            _both(_plane(source, receivers, rays, seed, ratio, **more), samples)[0].sum((1, 2))
            for seed in range(seeds)
        ]
    )


def _polar(centre, f) -> float:
    """The integral of f(x, y) over the plane, on a polar grid round ``centre``."""
    rho = np.geomspace(1e-5, 4000.0, 3000)
    phi = (np.arange(1440) + 0.5) * (2 * np.pi / 1440)
    r, p = np.meshgrid(rho, phi, indexing="ij")
    x, y = centre[0] + r * np.cos(p), centre[1] + r * np.sin(p)
    return float(np.sum(f(x, y) * r * np.gradient(rho)[:, None]) * (2 * np.pi / 1440))


def _scattered(source, listener, radius: float = 0.5, max_time: float = 1.0) -> float:
    """The rain of the plane's reflections at a listener, integrated: what lands on
    each patch of it, cos_in / (4 pi r_in^2), times the scattered share and Lambert's
    cos / (pi d^2) to the listener (d no shorter than the sphere's radius), in time.
    Its peaks over the source's and the listener's feet go on grids of their own."""
    s, q = np.asarray(source, float), np.asarray(listener, float)

    def f(x, y):
        r2 = (x - s[0]) ** 2 + (y - s[1]) ** 2 + s[2] ** 2
        d2 = np.maximum((x - q[0]) ** 2 + (y - q[1]) ** 2 + q[2] ** 2, radius**2)
        kept = (np.abs(x) <= 200) & (np.abs(y) <= 200)
        kept &= np.sqrt(r2) + np.sqrt(d2) < 343.0 * max_time
        lands = s[2] / np.sqrt(r2) / (4 * np.pi * r2)
        return lands * (1 - ALPHA) * SCATTERING * q[2] / np.sqrt(d2) / (np.pi * d2) * kept

    def near_source(x, y):
        a = (x - s[0]) ** 2 + (y - s[1]) ** 2 + 1e-30
        b = (x - q[0]) ** 2 + (y - q[1]) ** 2 + 1e-30
        return b / (a + b)

    return _polar(s, lambda x, y: f(x, y) * near_source(x, y)) + _polar(
        q, lambda x, y: f(x, y) * (1 - near_source(x, y))
    )


def test_a_jet_overhead_lands_no_burst_of_scattered_sound() -> None:
    # The aircraft landing demo's case: a source 305 m up, 300 m off, and a listener
    # 1.2 m above grass. A ray that lands near the listener's foot brings it most of the
    # scattered sound at once; the gather finds it there every time.
    source, listener, more = (-300.0, 0.0, 305.0), [(0.0, 0.0, 1.2)], {"max_time": 3.0}
    rays = _rain(source, listener, 24, 0, **more)[:, 0]
    shared = _rain(source, listener, 24, late.FIRST_GATHER, **more)[:, 0]
    exact = _scattered(source, listener[0], max_time=3.0)
    assert np.std(rays) / exact > 0.5  # 0.86 when written
    assert np.std(shared) / exact < 0.005  # 0.0008
    assert np.mean(shared) == pytest.approx(exact, rel=0.002)


@pytest.mark.parametrize(
    ("source", "spread"),
    [((0.0, 0.0, 30.0), 0.01), ((0.0, 0.0, 0.05), 0.03)],
    ids=["high", "near the ground"],
)
def test_rain_and_gather_add_up_to_the_scattered_sound(source, spread: float) -> None:
    # Directly under a source 30 m up and 15 and 60 m off; and 15 and 60 m from a source
    # just above the ground, whose rays land densely at its foot (where the rain keeps
    # them) and seldom near the listeners (where the gather takes over).
    listeners = [(0.0, 0.0, 1.2), (15.0, 0.0, 1.2), (60.0, 0.0, 1.2)]
    rays = _rain(source, listeners, 32, 0)
    shared = _rain(source, listeners, 32, late.FIRST_GATHER)
    exact = np.array([_scattered(source, listener) for listener in listeners])
    mean, error = shared.mean(0), shared.std(0, ddof=1) / np.sqrt(len(shared))
    assert np.all(np.abs(mean - exact) < 4 * error + 1e-3 * exact), (mean / exact, error / exact)
    assert np.all(shared.std(0) / exact < spread)
    assert np.all(shared.std(0) <= rays.std(0) * 1.05)


def test_the_gather_alone_misses_a_source_near_the_ground() -> None:
    # Why the two share it: had the gather taken all of it (gather_ratio infinite), a
    # source 5 cm up would land most of its sound in a patch a few centimetres across
    # that few of the listener's directions meet. A tiny ratio leaves it to the rays.
    source, listeners = (0.0, 0.0, 0.05), [(15.0, 0.0, 1.2)]
    shared = _rain(source, listeners, 16, late.FIRST_GATHER)[:, 0]
    alone = np.array(
        [_both(_plane(source, listeners, 4000, seed, np.inf), 16384)[0].sum() for seed in range(16)]
    )
    assert np.std(alone) > 10 * np.std(shared)
    rays = _rain(source, listeners, 4, 0)
    tiny = np.array(
        [_both(_plane(source, listeners, 4000, seed, 1e-12), 16384)[0].sum() for seed in range(4)]
    )
    np.testing.assert_allclose(tiny, rays[:, 0], rtol=1e-6)


def test_plants_and_directivity_share_alike() -> None:
    # A slab of foliage over part of the ground between the source and the listeners,
    # and a source turned to the ground: the same rays with and without the gather, so
    # what the two give apart is the rain's share (paired over seeds).
    plants = {
        "plants": np.array([[0, 0, 0, 20, 0, 4, 0, 0]], float),
        "plant_outline": np.array([[5, -200], [12, -200], [12, 200], [5, 200]], float),
        "plant_absorption": np.array([[0.02]]),
        "plant_scattering": np.array([[0.03]]),
        "plant_asymmetry": np.array([[0.3]]),
        "plant_upright": np.array([0], np.uint8),
    }
    turned = {
        "directivity_a": np.array([0.5]),
        "orientation": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
    }
    listeners = [(20.0, 0.0, 1.2), (8.0, 3.0, 1.5), (-10.0, 0.0, 1.2)]
    source = (0.0, 0.0, 8.0)
    rays = _rain(source, listeners, 24, 0, **plants, **turned)
    shared = _rain(source, listeners, 24, late.FIRST_GATHER, **plants, **turned)
    difference = shared - rays
    error = difference.std(0, ddof=1) / np.sqrt(len(difference))
    assert np.all(np.abs(difference.mean(0)) < 4 * error), (difference.mean(0), error)
    # And not by a share the same everywhere: the plants and the pattern take theirs.
    assert np.all(shared.std(0) < rays.std(0))


def test_the_gather_counts_groups_and_directions_as_the_rain() -> None:
    # Two halves of the ground in two groups, 6 arrival directions: the gather's
    # moments hold its energy once (term 1) and per group, and its directions add up
    # to its energy; the shared run's agree with the rays' alone within the noise.
    directions = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float
    )
    more = {"face_group": np.array([0, 1], np.int64), "groups": 2, "directions": directions}
    source, listeners = (0.0, 0.0, 30.0), [(10.0, 5.0, 1.2)]
    args = _plane(source, listeners, 4000, 3, late.FIRST_GATHER / 4000, **more)
    chosen = {k: args[k] for k in GATHERED}
    energy, directional, moments = _native.gather_first_scattering(
        **chosen, samples=late.FIRST_GATHER, turn=late.gather_turn(3)
    )
    assert moments.shape == (1, 1 + 2 + 3, 1, 100)
    np.testing.assert_allclose(moments[:, 0].sum(-1), energy.sum(-1), rtol=1e-12)
    np.testing.assert_allclose(moments[:, 1:3].sum((1, -1)), energy.sum(-1), rtol=1e-12)
    np.testing.assert_allclose(moments[:, 1], moments[:, 3])  # n_0 n_0 = n_0 for one hit
    np.testing.assert_allclose(directional.sum((1, 3)), energy.sum(-1), rtol=1e-12)
    assert directional[0, 4].sum() == 0.0  # all from the ground, none from above
    pairs = []
    for seed in range(16):
        old = _both(_plane(source, listeners, 4000, seed, 0.0, **more), 0)
        new = _both(
            _plane(source, listeners, 4000, seed, late.FIRST_GATHER / 4000, **more),
            late.FIRST_GATHER,
        )
        pairs.append([np.r_[a.sum(-1).ravel()] for a in (*old[1:], *new[1:])])
    old_d, old_m, new_d, new_m = (np.array(x) for x in zip(*pairs, strict=True))
    for a, b in ((old_d, new_d), (old_m, new_m)):
        difference = b - a
        error = difference.std(0, ddof=1) / 4
        assert np.all(np.abs(difference.mean(0)) <= 4 * error + 1e-12 * np.abs(a).max())


def test_the_gather_is_the_same_on_any_threads_and_for_any_listeners() -> None:
    listeners = [(0.0, 0.0, 1.2), (15.0, 0.0, 1.2), (60.0, 0.0, 1.2)]
    args = _plane((0.0, 0.0, 30.0), listeners, 100, 5, 1.0)
    chosen = {k: args[k] for k in GATHERED}
    turn = late.gather_turn(5)
    one = _native.gather_first_scattering(**(chosen | {"threads": 1}), samples=4096, turn=turn)
    four = _native.gather_first_scattering(**(chosen | {"threads": 4}), samples=4096, turn=turn)
    last = _native.gather_first_scattering(
        **(chosen | {"receivers": args["receivers"][2:], "receiver_radius": np.full(1, 0.5)}),
        samples=4096,
        turn=turn,
    )
    for a, b in zip(one, four, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(last[0][0], one[0][2])
    np.testing.assert_array_equal(late.gather_turn(5), turn)
    np.testing.assert_allclose(turn @ turn.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(turn) == pytest.approx(1.0)


def test_the_gather_adds_into_arrays_a_share_of_its_directions_at_a_time() -> None:
    # Into the tracer's arrays, in three calls of every third direction each: the same
    # as one call into zeros added to them, and each call a part of it.
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, -1]], float)
    more = {"face_group": np.array([0, 1], np.int64), "groups": 2, "directions": directions}
    args = _plane((0.0, 0.0, 30.0), [(0.0, 0.0, 1.2), (15.0, 0.0, 1.2)], 500, 7, 1.0, **more)
    chosen = {k: args[k] for k in GATHERED}
    turn = late.gather_turn(7)
    traced = _native.trace_stochastic(**args)[:3]
    whole = _native.gather_first_scattering(**chosen, samples=4096, turn=turn)
    into = [a.copy() for a in traced]
    parts = []
    for first in range(3):
        before = into[0].copy()
        found = _native.gather_first_scattering(
            **chosen,
            samples=4096,
            turn=turn,
            energy=into[0],
            directional=into[1],
            moments=into[2],
            first=first,
            stride=3,
        )
        assert found is None
        parts.append(into[0] - before)
    for a, t, w in zip(into, traced, whole, strict=True):
        np.testing.assert_allclose(a, t + w, rtol=1e-12, atol=1e-12 * np.abs(t + w).max())
    assert all(part.sum() > 0.1 * whole[0].sum() for part in parts)
    short = np.zeros((*into[0].shape[:-1], into[0].shape[-1] - 1))
    wrong = {"energy": short, "directional": into[1], "moments": into[2]}
    with pytest.raises(ValueError, match="energy must have shape"):
        _native.gather_first_scattering(**chosen, samples=64, turn=turn, **wrong)
    # Arrays it would have to copy (and so leave as they were) are refused.
    read_only = into[0].copy()
    read_only.flags.writeable = False
    shape = into[0].shape
    for other in (
        read_only,
        np.asfortranarray(into[0]),
        into[0].astype(np.float32),
        np.zeros((*shape[:-1], 2 * shape[-1]))[..., ::2],
    ):
        with pytest.raises(TypeError):
            _native.gather_first_scattering(
                **chosen, samples=64, turn=turn, energy=other, directional=into[1], moments=into[2]
            )
    # A stride past the lattice's end takes `first` alone.
    alone = [np.zeros_like(a) for a in traced]
    _native.gather_first_scattering(
        **chosen,
        samples=64,
        turn=turn,
        energy=alone[0],
        directional=alone[1],
        moments=alone[2],
        first=5,
        stride=2**64 - 1,
    )
    single = _native.gather_first_scattering(**chosen, samples=64, turn=turn, first=5, stride=64)
    np.testing.assert_array_equal(alone[0], single[0])
    with pytest.raises(ValueError, match="or none"):
        _native.gather_first_scattering(**chosen, samples=64, turn=turn, energy=into[0])
    with pytest.raises(ValueError, match="first below stride"):
        _native.gather_first_scattering(**chosen, samples=64, turn=turn, first=3, stride=3)


def test_the_gather_needs_diffuse_rain_alone_in_still_air() -> None:
    args = _plane((0.0, 0.0, 30.0), [(0.0, 0.0, 1.2)], 100, 1, 1.0)
    for wrong in ({"specular_rain": True}, {"diffuse_rain": False}):
        with pytest.raises(ValueError, match="diffuse rain alone"):
            _native.trace_stochastic(**(args | wrong))
    still = {
        "atmosphere": np.array([[0.0, 343.0, 0.0], [10.0, 343.0, 2.0]]),
        "wind_direction": np.array([1.0, 0.0]),
    }
    with pytest.raises(ValueError, match="in still air"):
        _native.trace_stochastic(**(args | still))
    with pytest.raises(ValueError, match="bad settings"):
        _native.trace_stochastic(**(args | {"gather_ratio": -1.0}))
    chosen = {k: args[k] for k in GATHERED}
    with pytest.raises(ValueError, match="gather_ratio above 0"):
        _native.gather_first_scattering(
            **(chosen | {"gather_ratio": 0.0}), samples=64, turn=np.eye(3)
        )
    with pytest.raises(ValueError, match="turn must be 3 x 3"):
        _native.gather_first_scattering(**chosen, samples=64, turn=np.eye(2))


def _rooms(path: Path, seed: int) -> Room:
    """A room 4 m long opening through a doorway into one 8 m long, the wall between them
    a partition letting 10 dB through, every face scattering half and as its size has
    it; the source in the first room, listeners in both."""
    obj = _coupled(path / "rooms.obj", 4, 8, 6, 4, (2.5, 3.5, 0, 2))
    scattering = {"scattering": 0.5}
    materials = {
        "wall": _flat(0.15, **scattering),
        "partition": _flat(0.1, transmission_loss=10, **scattering),
        "door": {"transmission": 1},
    }
    assignments = {"room1": "wall", "room2": "wall", "partition": "partition", "opening": "door"}
    listeners = {"near": [2.8, 4.1, 1.2], "beyond": [9.0, 2.0, 1.5]}
    return Room.from_scene(_scene(obj, materials, assignments, listeners, seed=seed))


def test_rooms_through_a_partition_keep_their_energy(tmp_path: Path) -> None:
    # Rays that pass the partition at random (the first batch) or both ways (the rest),
    # and the walls' size scattering from the first reflection on: the run with the
    # gather against the same rays without, per listener and band.
    room = _rooms(tmp_path, 0)
    model = TraceModel.from_room(room, room.scene.simulation)
    source = np.array(room.scene.sources[0].position, float)
    listeners = np.array([listener.position for listener in room.scene.listeners], float)
    common = {
        "rays": 3000,
        "max_time": 0.4,
        "time_step": 1e-3,
        "speed_of_sound": room.speed_of_sound(),
        "air": room.air_attenuation(model.bands),
        "radius": 0.5,
        "covered_order": 2,
    }
    differences, totals = [], []
    for seed in range(12):
        old = late.stochastic_energy(model, source, listeners, **common, seed=seed, gather=0)
        new = late.stochastic_energy(model, source, listeners, **common, seed=seed)
        differences.append(new.energy.sum(-1) - old.energy.sum(-1))
        totals.append(old.energy.sum(-1))
    difference, total = np.array(differences), np.mean(totals, axis=0)
    error = difference.std(0, ddof=1) / np.sqrt(len(difference))
    assert np.all(total > 0)
    assert np.all(np.abs(difference.mean(0)) < 4 * error + 1e-3 * total)
    # The rays' rain beyond the partition: the gather found it there too.
    assert np.all(np.abs(difference[:, 1]).max(0) > 0)


def test_each_batch_carries_a_share_of_the_gather(tmp_path: Path, monkeypatch) -> None:
    # In batches, each batch's estimate takes its own share of the gather's directions
    # (every fourth of them), so the batches differ by the gather's error as well as the
    # rays'; their mean is still the energy.
    room = _rooms(tmp_path, 0)
    model = TraceModel.from_room(room, room.scene.simulation)
    source = np.array(room.scene.sources[0].position, float)
    listeners = np.array([listener.position for listener in room.scene.listeners], float)
    common = {
        "rays": 2000,
        "max_time": 0.3,
        "time_step": 1e-3,
        "speed_of_sound": room.speed_of_sound(),
        "air": room.air_attenuation(model.bands),
        "radius": 0.5,
        "covered_order": 2,
        "seed": 4,
        "batches": 4,
    }
    gather, calls = _native.gather_first_scattering, []

    def record(**arguments):
        calls.append(arguments)

    # The same rays, sharing the rain, with a gather that adds nothing.
    monkeypatch.setattr(_native, "gather_first_scattering", record)
    bare = late.stochastic_energy(model, source, listeners, **common)
    monkeypatch.undo()
    new = late.stochastic_energy(model, source, listeners, **common)
    assert [(c["first"], c["stride"]) for c in calls] == [(k, 4) for k in range(4)]
    coarse = round(new.batch_step / new.time_step)
    slots = np.arange(0, new.energy.shape[-1], coarse)
    found = []
    for k, call in enumerate(calls):
        alone = {a: v for a, v in call.items() if a not in ("energy", "directional", "moments")}
        found.append(np.add.reduceat(gather(**alone)[0], slots, axis=-1) * 4)
        np.testing.assert_allclose(
            new.batches[:, k] - bare.batches[:, k],
            found[-1],
            rtol=1e-9,
            atol=1e-12 * bare.batches.max(),
        )
    totals = np.array(found).sum((2, 3))
    assert np.all(totals > 0) and np.all(totals.std(0) > 1e-6 * totals.mean(0))
    summed = np.add.reduceat(new.energy, slots, axis=-1)
    np.testing.assert_allclose(new.batches.mean(1), summed, rtol=1e-9, atol=1e-12 * summed.max())
