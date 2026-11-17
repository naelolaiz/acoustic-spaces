"""Where the late stage traces its rays: the choice of device, ``auto``'s trial and the
CPU and the GPU sharing the rays. None of these needs a GPU: test_gpu.py traces on one."""

import itertools
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_geometry import _to_triangle
from test_tracer_reference import REFERENCE, late_results, reference_arguments

from acoustic_spaces import Room, Scene, _native
from acoustic_spaces.engine import gpu, gpu_rays, gpu_tracer, ray_device, tuning

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    with np.load(REFERENCE) as stored:
        return dict(stored)


def _arguments(reference, name: str) -> dict:
    return reference_arguments(_native, reference, name)


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A computer with a hardware GPU (as far as ray_device can tell), a cache of its
    own (conftest's device_choices) and no trial run yet."""
    monkeypatch.setattr(gpu, "_adapter", lambda: object())
    monkeypatch.setattr(gpu, "hardware_gpu", lambda: True)
    monkeypatch.setattr(gpu, "gpu_name", lambda: "Test GPU (Vulkan, DiscreteGPU)")
    monkeypatch.setattr(ray_device, "_rates", {})
    monkeypatch.setattr(ray_device, "_trial", True)
    return tmp_path


def test_the_cpu_traces_as_before(reference) -> None:
    values = _arguments(reference, "vector")
    plan = ray_device.choose("cpu", values)
    assert plan == ray_device.Plan("cpu")
    found, followed = ray_device.trace(plan, values)
    assert followed == plan
    expected = late_results(_native.trace_stochastic(**values))
    for key, value in expected.items():
        np.testing.assert_array_equal(late_results(found)[key], value, err_msg=key)


def test_gpu_settings_need_a_gpu(reference, monkeypatch) -> None:
    monkeypatch.setattr(gpu, "_adapter", lambda: None)
    values = _arguments(reference, "vector")
    for setting in ("gpu", "both"):
        with pytest.raises(ValueError, match=f"device is {setting}"):
            ray_device.choose(setting, values)
    with pytest.raises(ValueError, match="cpu, gpu, both, auto"):
        ray_device.choose("tpu", values)
    # auto never fails: without a GPU it takes the CPU.
    monkeypatch.setattr(gpu, "hardware_gpu", lambda: False)
    assert ray_device.choose("auto", values) == ray_device.Plan("cpu", reason="no hardware GPU")


def test_what_only_the_cpu_traces_stays_there(reference, machine) -> None:
    rooms = _arguments(reference, "rooms_first")
    for setting in ("gpu", "both", "auto"):
        plan = ray_device.choose(setting, rooms)
        assert plan.device == "cpu" and "partitions" in plan.reason
    # Face groups (the spread from the absorption's uncertainty) are traced on the GPU.
    plan = ray_device.choose("gpu", _arguments(reference, "vector") | {"groups": 2})
    assert plan == ray_device.Plan("gpu")


def test_direct3d_12_leaves_bent_rays_through_vegetation_to_the_cpu(
    reference, machine, monkeypatch
) -> None:
    """Direct3D 12's shader compiler takes more than an hour over the GPU tracer with
    both rays bent by the air and vegetation, so a run with both goes to the CPU there,
    saying why; a run with either alone, and other backends, keep the GPU."""
    values = _arguments(reference, "vector")
    bent = {"atmosphere": np.array([[0.0, 343.0, 0.0], [10.0, 344.0, 1.0]])}
    plants = {"plants": np.zeros((1, 8)), "plant_upright": np.zeros(1, np.uint8)}
    for backend, declined in [("Vulkan", False), ("D3D12", True)]:
        adapter = SimpleNamespace(info={"backend_type": backend})
        monkeypatch.setattr(gpu, "_adapter", lambda adapter=adapter: adapter)
        for extra in ({}, bent, plants):
            assert ray_device.choose("gpu", values | extra) == ray_device.Plan("gpu")
        plan = ray_device.choose("gpu", values | bent | plants)
        assert (plan.device == "cpu" and "Direct3D 12" in plan.reason) == declined


def test_the_open_sky_lets_sound_out_on_any_device() -> None:
    """An outdoor scene's sky lets all sound through into no room, which keeps every
    tracer's rays the same; a face into another room, or sound passing from a source
    in no room, only the CPU follows."""
    passes = gpu_tracer.passes_sound
    through = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]])
    assert not passes(np.zeros(0), np.zeros((0, 2)), -1)
    assert not passes(through, np.array([[0, -1], [0, -1], [-1, 0]]), 0)
    assert passes(through, np.array([[0, -1], [0, 1], [-1, 0]]), 0)
    assert passes(through, np.array([[0, 1], [0, -1], [-1, 0]]), 0) is False
    assert passes(through, np.array([[0, -1], [0, -1], [0, 0]]), 0)
    assert passes(through, np.array([[0, -1], [0, -1], [-1, 0]]), -1)
    assert passes(through, np.zeros((0, 2)), 0)
    values = {
        "atmosphere": None,
        "transmission": through,
        "face_room": np.array([[0, -1], [0, -1], [-1, 0]]),
        "source_room": 0,
        "groups": 0,
    }
    assert ray_device._unsupported(values) == ""
    values["face_room"] = np.array([[0, -1], [0, 1], [-1, 0]])
    assert "partitions" in ray_device._unsupported(values)


#: Where a listener with a sphere of nine tenths of its distance from a face reads 0.1 dB
#: low on the GPU, in distances at which the GPU starts its reflections off the face.
_LOW = 4.6889


def _near(who: str, what: str, millionths: float = 10) -> str:
    """The reason the GPU gives for leaving a run to the CPU (``gpu_tracer.too_near``)."""
    if (who, what) == ("listener", "surface"):
        return (
            "sound to a listener whose sphere reaches where the GPU's reflections start, "
            f"{millionths:.3g} millionths of the model's size off a surface, is traced on the CPU"
        )
    who = {"source": "a source", "listener": "sound to a listener"}[who]
    what = {"surface": "a surface", "edge": "two surfaces that meet"}[what]
    return (
        f"{who} nearer {what} than {millionths:.3g} millionths of the model's size is traced "
        "on the CPU"
    )


def _too_near(bvh, source, listeners) -> str:
    """``gpu_tracer.too_near`` with the spheres the late stage gives listeners between
    its limits: nine tenths of each one's distance from the nearest face, worked out
    here."""
    layout = np.asarray(bvh.layout()["triangles"], dtype=float).reshape(-1, 3, 3)
    faces = np.stack([layout[:, 0], layout[:, 0] + layout[:, 1], layout[:, 0] + layout[:, 2]], 1)
    listeners = np.asarray(listeners, dtype=float).reshape(-1, 3)
    spheres = [0.9 * min(_to_triangle(place, face) for face in faces) for place in listeners]
    return gpu_tracer.too_near(bvh, source, listeners, np.array(spheres))


@pytest.fixture(params=["at once", "a row at a time"])
def tables(request, monkeypatch) -> None:
    """The table of pairs of ``gpu_tracer._meeting``, filled at once and row by row."""
    if request.param == "a row at a time":
        monkeypatch.setattr(gpu_tracer, "_ROWS", 1)


def test_the_gpu_leaves_a_source_or_a_listener_right_by_a_surface_to_the_cpu() -> None:
    """The GPU's rays meet no face nearer their start than a millionth of the model's
    size, so a source nearer a face than that sends rays through it (up to 3.2 dB of its
    late part over a floor). And the reflections off a face start a hundred-thousandth
    of the size off it: a listener with the late stage's sphere within 4.7 times that
    reads low (2 to 6 dB from 12 down to 3 millionths over the floor of a room 20 m long
    in a model of 20 km). The GPU declines a run with its source nearer a face than
    twice the first distance, or such a listener: nearer the triangle itself, not its
    plane or its box."""
    at = np.array([1.0, 2.0, 3.0])  # no corner at the origin, where a corner and an edge are one
    corners = at + np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    bvh = _native.Bvh(corners, np.array([[0, 1, 2]], dtype=np.int64))
    assert bvh.scale == pytest.approx(10.0)
    assert gpu_tracer.CLEAR * gpu_rays.T_MIN == pytest.approx(2e-6)
    assert gpu_tracer._OFFSET == pytest.approx(1e-5)
    far = at + np.array([[1.0, 1.0, 3.0], [4.0, 2.0, -1.0], [2.0, 5.0, 2.0]])
    for who, room in (("source", 2e-5), ("listener", _LOW * 1e-4)):
        for near, clear in [
            ([2.0, 2.0, 0.9 * room], [2.0, 2.0, 1.1 * room]),  # over the face
            ([2.0, 2.0, -0.9 * room], [2.0, 2.0, -1.1 * room]),  # under it
            ([2.0, -0.9 * room, 0.0], [2.0, -1.1 * room, 0.0]),  # in its plane, past an edge
            ([-0.6 * room, -0.6 * room, 0.4 * room], [-0.7 * room, -0.7 * room, 0.4 * room]),
            ([3.0, 4.0, 0.9 * room], [3.0 + 0.8 * room, 4.0 + 0.6 * room, 0.5 * room]),
        ]:
            near, clear = at + near, at + clear
            if who == "source":
                assert _too_near(bvh, near, far) == _near("source", "surface", 2)
                assert _too_near(bvh, clear, far) == ""
                continue
            # The near listener first, among the others and last.
            for place in range(len(far) + 1):
                listeners = np.insert(far, place, near, axis=0)
                assert _too_near(bvh, far[0], listeners) == _near("listener", "surface")
                assert _too_near(bvh, far[0], np.insert(far, place, clear, axis=0)) == ""
    # In the triangle's box and right by its plane, but metres past its long edge.
    assert _too_near(bvh, at + [5.0, 7.0, 1e-5], far) == ""
    # A source that near a face is not a listener that near it.
    assert _too_near(bvh, at + [2.0, 2.0, 1e-4], far) == ""
    assert _too_near(bvh, far[0], np.empty((0, 3))) == ""


def test_a_listener_is_the_gpus_while_the_start_of_its_reflections_costs_it_little() -> None:
    """A face lit evenly sends a listener at the distance c, with a sphere of the radius
    r, the share x (3 - x^2) / 2 of its scattered sound, x = c / r up to 1: the shader's
    own sum (the cosine and the square of the distance, with the distance held to the
    radius) over the plane. The GPU works it out from a hundred-thousandth of the
    model's size nearer. It leaves the listeners that read more than 0.1 dB low for
    that to the CPU, and those no farther from a face than that distance: against the
    two shares worked out here, for spheres from a tenth of that distance to 3,000 times
    it and listeners out to 1,600 times it."""
    at = np.array([1.0, 2.0, 3.0])
    corners = at + np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    bvh = _native.Bvh(corners, np.array([[0, 1, 2]], dtype=np.int64))
    start = gpu_tracer._OFFSET * bvh.scale

    def low(distance: float, radius: float) -> float:
        """How low the GPU reads (dB): the share heard from ``start`` nearer over the
        share heard."""

        def share(x: float) -> float:
            x = min(max(x, 0.0), 1.0)
            return x * (3.0 - x * x) / 2.0

        return 10.0 * math.log10(share((distance - start) / radius) / share(distance / radius))

    assert gpu_tracer.LOSS == 0.1
    assert low(0.999 * _LOW * start, 0.9 * 0.999 * _LOW * start) < -0.1
    assert low(1.001 * _LOW * start, 0.9 * 1.001 * _LOW * start) > -0.1
    rng = np.random.default_rng(3)
    distance = start * 10.0 ** rng.uniform(0.01, 3.2, 400)
    radius = start * 10.0 ** rng.uniform(-1.0, 3.5, 400)
    lows = np.array([low(c, r) for c, r in zip(distance, radius, strict=True)])
    decided = np.abs(lows + 0.1) > 1e-4  # not at the limit itself
    places = at + [2.0, 2.0, 0.0] + distance[:, None] * [0.0, 0.0, 1.0]
    found = gpu_tracer.listeners_too_near(bvh, places, radius)
    assert 100 < (lows < -0.1).sum() < 300
    np.testing.assert_array_equal(found[decided], (lows < -0.1)[decided])
    # A sphere through the face reads lower still on the CPU: what counts is the GPU from it.
    through = radius > distance
    assert (found & through).sum() > 100 and (~found & through).sum() > 4
    # No farther from the face than where the reflections start: however small the sphere.
    by = at + np.array([[2.0, 2.0, 0.9 * start], [2.0, 2.0, -start], [2.0, 2.0, 1.1 * start]])
    assert gpu_tracer.listeners_too_near(bvh, by, 0.01 * start).tolist() == [True, True, False]
    # On the face itself, where neither share is any: with a sphere and without one.
    on = at + np.array([[2.0, 2.0, 0.0]])
    for sphere in (30.0 * start, 0.0):
        assert gpu_tracer.listeners_too_near(bvh, on, sphere).tolist() == [True]
    assert gpu_tracer.listeners_too_near(bvh, by[2:], 0.0).tolist() == [False]
    # One radius for all, and each listener's own answer among the others.
    mixed = gpu_tracer.listeners_too_near(bvh, places[:50], 30.0 * start)
    assert mixed.tolist() == [low(c, 30.0 * start) < -0.1 for c in distance[:50]]
    assert 5 < mixed.sum() < 45
    assert gpu_tracer.listeners_too_near(bvh, np.empty((0, 3)), 0.5).tolist() == []


def _wedge(angle_deg: float):
    """Two faces that meet along the line x = z = 0 from y = 0 to 10 m at ``angle_deg``
    degrees (a floor towards +x and a wall), their tree, and the place a distance from
    both of them inside the angle (0) or outside both (1)."""
    angle = math.radians(angle_deg)
    lean = np.array([math.cos(angle), 0.0, math.sin(angle)])
    ends = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    corners = np.vstack([ends, [[10.0, 5.0, 0.0]], 10.0 * lean + [0.0, 5.0, 0.0]])
    bvh = _native.Bvh(corners, np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int64))
    middle = np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2)]) / math.sin(angle / 2)

    def place(distance: float, outside: bool = False) -> np.ndarray:
        return (-distance if outside else distance) * middle + [0.0, 5.0, 0.0]

    return bvh, place


@pytest.mark.parametrize("angle", [30.0, 60.0, 90.0, 93.0])
def test_the_gpu_leaves_a_point_by_an_inner_edge_to_the_cpu(angle: float, tables) -> None:
    """Beside a second face that faces the one it left, the GPU loses a reflected ray: it
    goes through that face within a millionth of the model's size of it, and where the
    two meet at less than a right angle it starts behind it within the share of a
    hundred-thousandth that points to it. A source 2.05 millionths from both sides of an
    edge, traced on the GPU, is 6 to 7.5 dB low at 30 degrees and 0.15 dB at 90, and 0.4
    to 0.5 dB low in a corner of three faces at right angles. The GPU declines a run with
    its source or a listener nearer both faces than 50 times that reach."""
    bvh, place = _wedge(angle)
    size = float(bvh.scale)
    millionths = 50 * (1 + 10 * math.cos(math.radians(angle)))
    far = np.array([[4.0, 5.0, 1.0], [3.0, 2.0, 0.5]])
    near, clear = place(0.95e-6 * millionths * size), place(1.05e-6 * millionths * size)
    assert _too_near(bvh, near, far) == _near("source", "edge", millionths)
    assert _too_near(bvh, clear, far) == ""
    # Right by both faces, the source is told of the surface, not of the edge.
    assert _too_near(bvh, place(1e-6 * size), far) == _near("source", "surface", 2)
    # A listener is too near either face alone within 46.9 millionths of it.
    reason = _near("listener", "edge", millionths)
    if millionths < 10 * _LOW:
        near, reason = place(0.95e-5 * _LOW * size), _near("listener", "surface")
        clear = place(1.05e-5 * _LOW * size)
    assert _too_near(bvh, far[0], np.vstack([far, near])) == reason
    assert _too_near(bvh, far[0], np.vstack([clear, far])) == ""
    # After a listener that the two faces are near and not too near.
    assert _too_near(bvh, far[0], np.vstack([clear, near, far])) == reason
    # Nearer one face than the other, the farther one counts: 3 millionths over the floor
    # and past the reach from the wall, then within it.
    turn = math.radians(angle)
    for out, reason in ((1.05, ""), (0.95, _near("source", "edge", millionths))):
        along = (out * 1e-6 * millionths + 3e-6 * math.cos(turn)) * size / math.sin(turn)
        assert _too_near(bvh, [along, 5.0, 3e-6 * size], far) == reason


def test_the_gpu_traces_a_point_by_an_edge_that_loses_no_ray(tables) -> None:
    """Where two faces meet at more than about 96 degrees, a reflection off one starts
    more than a millionth of the model's size from the other, and beside an outer edge
    no ray goes from one face to the other: the GPU takes a source 2.1 millionths from
    both. Nor do two faces in one plane make an edge."""
    far = np.array([[4.0, 5.0, 1.0]])
    for angle in (96.0, 120.0, 150.0, 179.0):
        bvh, place = _wedge(angle)
        assert _too_near(bvh, place(2.1e-6 * bvh.scale), far) == ""
    for angle in (30.0, 90.0):
        bvh, place = _wedge(angle)
        size = float(bvh.scale)
        assert _too_near(bvh, place(2.1e-6 * size), far) != ""
        # Outside both faces; and by the edge behind the wall, over the floor's plane: the
        # wall is ahead of the floor there, the floor is not ahead of the wall.
        assert _too_near(bvh, place(2.1e-6 * size, outside=True), far) == ""
        lean = np.array([math.cos(math.radians(angle)), 0.0, math.sin(math.radians(angle))])
        behind = np.array([0.0, 5.0, 0.0]) + 3e-6 * size * (lean + np.cross(lean, [0, 1, 0]))
        assert _too_near(bvh, behind, far) == ""
    # Under the floor by its end, on the side of the wall's plane where the floor lies:
    # the floor is ahead of the wall there, the wall is not ahead of the floor.
    assert _too_near(bvh, [2.1e-6 * size, 5.0, -2.1e-6 * size], far) == ""
    flat = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 6.0, 0.0], [0.0, 6.0, 0.0]])
    bvh = _native.Bvh(flat, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
    assert _too_near(bvh, [4.0, 3.0, 2.1e-6 * bvh.scale], far) == ""


@pytest.mark.parametrize("floor", [[0, 1, 2], [0, 2, 1]], ids=["up", "down"])
def test_a_point_in_the_plane_of_a_face_is_on_both_sides_of_it(floor: list[int], tables) -> None:
    """A source in the plane of a floor, 28 millionths of the model's size past its
    corner and 20 from the wall that the floor ends on, which goes on past the corner:
    the floor has no side there, so the two make an edge for it whichever way the
    floor's corners are listed, and as well half the least distance of a ray to either
    side of the plane."""
    corners = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 12.0, 0.0], [0.0, 6.0, 5.0]]
    )
    bvh = _native.Bvh(corners, np.array([floor, [0, 3, 4]], dtype=np.int64))
    by = 2e-5 * bvh.scale
    far = np.array([[4.0, 5.0, 1.0]])
    for off in (0.0, 0.5 * bvh.min_distance, -0.5 * bvh.min_distance):
        assert _too_near(bvh, [by, 8.0 + by, off], far) == _near("source", "edge", 50)
    assert _too_near(bvh, [3 * by, 8.0 + 3 * by, 0.0], far) == ""
    # A wall that leans over the floor at 60 degrees: the angle between the two has no sign
    # for a point in the floor's plane, and the edge holds it out to its 300 millionths.
    corners[4] = [5.0 * math.cos(math.radians(60.0)), 6.0, 5.0 * math.sin(math.radians(60.0))]
    bvh = _native.Bvh(corners, np.array([floor, [0, 3, 4]], dtype=np.int64))
    by = 2e-5 * bvh.scale
    assert _too_near(bvh, [by, 8.0 + by, 0.0], far) == _near("source", "edge", 300)


def test_two_faces_that_face_each_other_make_an_edge_only_where_they_meet(tables) -> None:
    """A floor and a ceiling face each other like the two sides of an edge of no angle,
    but a reflection off one is lost beside the other only where the two come within 11
    millionths of the model's size of each other (a millionth and a hundred-thousandth).
    A source midway between two such faces 200 or 12 millionths apart is the GPU's; in a
    slot 10 millionths wide it is the CPU's."""
    far = np.array([[4.0, 5.0, 1.0]])

    def slot(width: float):
        low = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
        corners = np.vstack([low, low + [0.0, 0.0, width]])
        return _native.Bvh(corners, np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64))

    for millionths, reason in ((200.0, ""), (12.0, ""), (10.0, _near("source", "edge", 550))):
        bvh = slot(millionths * 1e-5)
        assert bvh.scale == pytest.approx(10.0)
        midway = [2.0, 2.0, millionths * 1e-5 / 2]
        assert _too_near(bvh, midway, far) == reason


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_two_faces_that_do_not_meet_take_nothing_from_an_edge_beside_them(order, tables) -> None:
    """A source 30 millionths of the model's size from a floor, from a wall at a right
    angle to it and from a ceiling 60 millionths over the floor. The floor and the
    ceiling face each other and would hold it out to 550 millionths, but they do not
    meet (they come no nearer each other than 60 millionths, not 11): the edges with the
    wall hold it, at their 50 millionths, whichever order the faces come in."""
    size = math.sqrt(6.0**2 + 8.0**2 + 1.0**2)
    floor = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    wall = np.array([[0.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 4.0, 1.0]])
    corners = np.vstack([floor, floor + [0.0, 0.0, 6e-5 * size], wall])
    faces = np.arange(9, dtype=np.int64).reshape(3, 3)
    bvh = _native.Bvh(corners, faces[list(order)])
    assert bvh.scale == pytest.approx(size)
    far = np.array([[4.0, 5.0, 1.0]])
    assert _too_near(bvh, [3e-5 * size, 2.0, 3e-5 * size], far) == _near("source", "edge", 50)
    # Out of the wall's reach, between the floor and the ceiling still: the GPU's.
    assert _too_near(bvh, [6e-5 * size, 2.0, 3e-5 * size], far) == ""


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]])
def test_a_point_by_edges_of_two_angles_is_held_to_the_wider_reach(order, tables) -> None:
    """A source 40 millionths of the model's size from a floor, from a wall at a right
    angle to it and from one that leans over it at 30 degrees: nearer than the reach of
    both edges with the floor (50 and 483 millionths). The reason names the wider one,
    whichever order the faces come in."""
    lean = [0.0, 10.0 * math.cos(math.radians(30.0)), 10.0 * math.sin(math.radians(30.0))]
    corners = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], lean]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]], dtype=np.int64)  # floor, upright, leaning
    bvh = _native.Bvh(corners, faces[order])
    by = 4e-5 * bvh.scale
    place = [by, by / math.tan(math.radians(15.0)), by]
    reach = 50 * (1 + 10 * math.cos(math.radians(30.0)))
    assert _too_near(bvh, place, np.array([[4.0, 5.0, 1.0]])) == _near("source", "edge", reach)


def test_a_run_right_by_a_surface_is_traced_on_the_cpu(reference, machine, monkeypatch) -> None:
    values = _arguments(reference, "vector")
    assert ray_device.choose("gpu", values) == ray_device.Plan("gpu")
    bvh = values["bvh"]
    layout = np.asarray(bvh.layout()["triangles"], dtype=float).reshape(-1, 3, 3)
    # A face with no corner at the origin, where a corner and an edge are the same numbers.
    corner, one, other = next(row for row in layout if np.abs(row[0]).min() > 0.5)
    normal = np.cross(one, other) / np.linalg.norm(np.cross(one, other))
    middle = corner + (one + other) / 3
    by_source = middle + 1e-6 * bvh.scale * normal
    listeners = np.array(values["receivers"], dtype=float).reshape(-1, 3)
    assert len(listeners) > 1
    listeners[0] = middle + 5e-5 * bvh.scale * normal
    near_source, near_listener = _near("source", "surface", 2), _near("listener", "surface")
    for setting in ("gpu", "both", "auto", ray_device.Plan("gpu"), ray_device.Plan("both", 1, 2)):
        plan = ray_device.choose(setting, values | {"source": by_source})
        assert plan == ray_device.Plan("cpu", reason=near_source)
        plan = ray_device.choose(setting, values | {"receivers": listeners})
        assert plan == ray_device.Plan("cpu", reason=near_listener)
    # The GPU's tracer declines such a run itself, before it takes it.
    with pytest.raises(gpu_tracer.Unsupported, match="a source nearer a surface"):
        gpu_tracer.trace_stochastic(**values | {"source": by_source})
    with pytest.raises(gpu_tracer.Unsupported, match="sound to a listener whose sphere"):
        gpu_tracer.trace_stochastic(**values | {"receivers": listeners})
    # The listeners' spheres count: twenty times the start of a face's reflections from it,
    # a sphere through the face reads 0.2 dB low, and one that ends short of it nothing.
    start = gpu_tracer._OFFSET * bvh.scale
    listeners[0] = middle + 20.0 * start * normal
    wide = values | {"receivers": listeners, "receiver_radius": 1000.0 * start}
    small = wide | {"receiver_radius": 10.0 * start}
    assert ray_device.choose("gpu", wide) == ray_device.Plan("cpu", reason=near_listener)
    assert ray_device.choose("gpu", small) == ray_device.Plan("gpu")
    with pytest.raises(gpu_tracer.Unsupported, match="sound to a listener whose sphere"):
        gpu_tracer.trace_stochastic(**wide)
    assert gpu_tracer.too_near(bvh, values["source"], listeners, 10.0 * start) == ""
    # auto looks at the places only once it would take the GPU: no trial of such a run,
    monkeypatch.setattr(ray_device, "trial", lambda values: pytest.fail("a trial"))
    near = values | {"source": by_source}
    assert ray_device.choose("auto", near) == ray_device.Plan("cpu", reason=near_source)
    # with speeds from a run before, the CPU for this one only,
    monkeypatch.setattr(ray_device, "_rates", {ray_device._key(values): (1.0, 100.0)})
    assert ray_device.choose("auto", values).device == "gpu"
    assert ray_device.choose("auto", near) == ray_device.Plan("cpu", reason=near_source)
    # (and the trial's own reason where its speeds give the run to the CPU, without a look),
    monkeypatch.setattr(ray_device, "_rates", {ray_device._key(values): (100.0, 1.0)})
    with monkeypatch.context() as patch:
        patch.setattr(gpu_tracer, "too_near", lambda *places: pytest.fail("looked"))
        slow = ray_device.choose("auto", near)
    assert slow == ray_device.Plan("cpu", 100.0, 1.0, "the trial")
    # as with the GPU chosen by tune, or kept from a trial before with the trial off,
    for origin in ("tune", "trial"):
        ray_device.set_trial(origin == "tune")
        choice = {"device": "gpu", "cpu_rate": 1.0, "gpu_rate": 9.0, "from": origin}
        tuning.keep_late(values["max_time"], choice)
        assert ray_device.choose("auto", values).device == "gpu"
        assert ray_device.choose("auto", near) == ray_device.Plan("cpu", reason=near_source)
    # and the reason it would have had anyway where it would not take the GPU.
    monkeypatch.setattr(gpu_tracer, "too_near", lambda *places: pytest.fail("looked"))
    ray_device.set_trial(False)
    (machine / "devices.json").unlink()
    plan = ray_device.choose("auto", near)
    assert plan == ray_device.Plan("cpu", reason="no trial or tune on this computer yet")
    monkeypatch.setattr(gpu, "hardware_gpu", lambda: False)
    assert ray_device.choose("auto", near) == ray_device.Plan("cpu", reason="no hardware GPU")


@pytest.mark.parametrize(
    "source, listener, reason",
    [
        ([2.5, 2.0, 1e-5], [7.0, 4.5, 1.7], _near("source", "surface", 2)),
        ([5e-4, 5e-4, 5e-4], [7.0, 4.5, 1.7], _near("source", "edge", 50)),
        ([2.5, 2.0, 1.2], [7.0, 4.5, 5e-4], _near("listener", "surface")),
    ],
    ids=["source over the floor", "source in a corner", "listener over the floor"],
)
def test_simulate_traces_a_run_right_by_a_surface_on_the_cpu(
    machine, source, listener, reason
) -> None:
    """In a room 10 m long with the late stage set to the GPU, a source 10 µm over the
    floor, one half a millimetre from the three faces of a corner, and a listener half a
    millimetre over the floor: the run says that the CPU traced it, and gives the CPU's
    late part."""
    from acoustic_spaces.engine.simulate import simulate

    def room(device: str) -> Room:
        scene = {
            "geometry": {"box": [10.0, 7.0, 4.0]},
            "materials": {"wall": {"absorption": [0.2] * 8, "scattering": 0.3}},
            "default_material": "wall",
            "sources": [{"name": "s", "position": source}],
            "listeners": [{"name": "l", "position": listener}],
            "simulation": {
                "ir_length_s": 0.3,
                "seed": 7,
                "late": {"device": device, "rays": 300},
            },
        }
        return Room.from_scene(Scene.model_validate(scene))

    messages: list[str] = []
    found = simulate(room("gpu"), progress=messages.append)
    where = [m for m in messages if "late rays traced on" in m]
    assert where == [f"s: late rays traced on the CPU ({reason})"]
    np.testing.assert_array_equal(found["s", "l"].late, simulate(room("cpu"))["s", "l"].late)


@pytest.mark.parametrize(
    "cpu_rate, gpu_rate, threads, device",
    [
        (100.0, 10.0, 16, "cpu"),  # the GPU adds too little
        (100.0, 50.0, 16, "both"),  # 1.4 times the CPU
        (10.0, 100.0, 16, "gpu"),  # the CPU would add under a tenth
        (100.0, 110.0, 1, "cpu"),  # a close call: the safe choice
        (100.0, 130.0, 1, "gpu"),
    ],
)
def test_auto_keeps_the_fastest_and_the_cpu_on_a_close_call(
    cpu_rate, gpu_rate, threads, device
) -> None:
    assert ray_device._decide(cpu_rate, gpu_rate, threads) == device


def test_auto_runs_a_trial_once_and_saves_its_choice(reference, machine, monkeypatch) -> None:
    trials = []

    def trial(values):
        trials.append(values["rays"])
        return 100.0, 80.0

    monkeypatch.setattr(ray_device, "trial", trial)
    values = _arguments(reference, "vector") | {"threads": 8}
    plan = ray_device.choose("auto", values)
    assert plan == ray_device.Plan("both", 100.0, 80.0, "the trial")
    assert ray_device.choose("auto", values) == plan
    assert len(trials) == 1  # once per model and range of lengths in a session
    saved = tuning.late_choice(values["max_time"])
    assert saved == {"device": "both", "cpu_rate": 100.0, "gpu_rate": 80.0, "from": "trial"}
    assert "Test GPU" in (machine / "devices.json").read_text()
    assert list(tuning.late_choices()) == [0.5]  # the run is 0.4 s long
    # Without the trial, auto takes the saved choice; "both" shares by its speeds.
    ray_device.set_trial(False)
    assert not ray_device.trial_enabled()
    assert ray_device.choose("auto", values) == ray_device.Plan(
        "both", 100.0, 80.0, "the saved choice"
    )
    monkeypatch.setattr(ray_device, "_rates", {})
    assert ray_device.choose("both", values) == ray_device.Plan("both", 100.0, 80.0)
    (machine / "devices.json").unlink()
    plan = ray_device.choose("auto", values)
    assert plan == ray_device.Plan("cpu", reason="no trial or tune on this computer yet")
    assert len(trials) == 1


def test_choices_are_kept_by_range_of_response_lengths(machine) -> None:
    keys = [tuning.length_key(t) for t in (0.1, 0.3, 0.5, 0.51, 1.0, 3.5, 7.0)]
    assert keys == ["0.125", "0.5", "0.5", "1", "1", "4", "8"]
    assert tuning.late_choice(1.0) is None and tuning.late_choices() == {}
    for length, device in [(0.5, "cpu"), (2.0, "gpu")]:
        tuning.keep_late(length, {"device": device, "cpu_rate": 1, "gpu_rate": 2, "from": "tune"})
    # The shortest range at least as long, else the longest.
    picked = [tuning.late_choice(t)["device"] for t in (0.2, 0.5, 1.0, 2.0, 6.0)]
    assert picked == ["cpu", "cpu", "gpu", "gpu", "gpu"]
    assert tuning.late_entry(1.0) is None and tuning.late_entry(1.5)["device"] == "gpu"


def test_auto_takes_what_tune_kept_without_a_trial(reference, machine, monkeypatch) -> None:
    trials = []

    def trial(values):
        trials.append(values["max_time"])
        return 100.0, 10.0

    monkeypatch.setattr(ray_device, "trial", trial)
    values = _arguments(reference, "vector")  # 0.4 s
    tuning.keep_late(0.5, {"device": "gpu", "cpu_rate": 1.0, "gpu_rate": 3.0, "from": "tune"})
    tuned = ray_device.Plan("gpu", 1.0, 3.0, "acoustic-spaces tune")
    assert ray_device.choose("auto", values) == tuned and not trials
    assert ray_device.choose("both", values) == ray_device.Plan("both", 1.0, 3.0)
    # A length tune did not time: the trial, for the run's own length, kept for its range.
    longer = values | {"max_time": 1.5}
    assert ray_device.choose("auto", longer) == ray_device.Plan("cpu", 100.0, 10.0, "the trial")
    assert trials == [1.5] and tuning.late_entry(1.5)["from"] == "trial"
    # Without the trial, the nearest kept choice.
    ray_device.set_trial(False)
    kept = ray_device.choose("auto", values | {"max_time": 3.0})
    assert kept == ray_device.Plan("cpu", 100.0, 10.0, "the saved choice")
    assert ray_device.choose("auto", values) == tuned and trials == [1.5]


def test_a_plan_given_is_followed(reference, machine, monkeypatch) -> None:
    values = _arguments(reference, "vector")
    plan = ray_device.Plan("both", 2.0, 1.0)
    assert ray_device.choose(plan, values) == plan
    rooms = ray_device.choose(ray_device.Plan("gpu"), _arguments(reference, "rooms_first"))
    assert rooms.device == "cpu" and "partitions" in rooms.reason
    monkeypatch.setattr(gpu, "_adapter", lambda: None)
    with pytest.raises(ValueError, match="device is gpu"):
        ray_device.choose(ray_device.Plan("gpu"), values)
    assert ray_device.choose(ray_device.Plan("cpu"), values) == ray_device.Plan("cpu")


def test_the_wave_solver_takes_the_faster_for_its_step_count(machine, monkeypatch) -> None:
    assert tuning.wave_costs() is None and tuning.wave_prefers_gpu(100) is None
    assert gpu.use_gpu("auto", 100)  # a hardware GPU, nothing timed
    # The GPU costs more to set up and less per step: the GPU from 124 steps on.
    tuning.keep_wave({"cpu": [0.0, 1e-8], "gpu": [1e-6, 1e-9], "from": "tune"})
    assert tuning.wave_pick(tuning.wave_costs()) == "the GPU from 124 steps on, else the CPU"
    assert not gpu.use_gpu("auto", 100) and gpu.use_gpu("auto", 200)
    assert gpu.use_gpu("gpu", 100) and not gpu.use_gpu("cpu", 200)
    monkeypatch.setattr(gpu, "hardware_gpu", lambda: False)
    assert not gpu.use_gpu("auto", 200)  # never a software GPU
    for cpu, gpu_costs, pick in [
        ([0.0, 1e-8], [0.0, 1e-9], "the GPU"),
        ([0.0, 1e-9], [1e-6, 1e-8], "the CPU"),
        ([1e-6, 1e-9], [0.0, 3e-9], "the GPU up to 434 steps, else the CPU"),
    ]:
        assert tuning.wave_pick({"cpu": cpu, "gpu": gpu_costs}) == pick


def test_a_damaged_file_keeps_nothing(machine) -> None:
    (machine / "devices.json").write_text("not json")
    assert tuning.saved() == {} and tuning.late_choice(1.0) is None
    tuning.keep_late(1.0, {"device": "tpu", "cpu_rate": 1, "gpu_rate": 1})
    tuning.keep_late(2.0, {"device": "gpu", "cpu_rate": "fast"})
    tuning.keep_wave({"cpu": [1.0]})
    assert tuning.late_choices() == {} and tuning.wave_costs() is None


def test_the_trial_times_both_devices(reference, monkeypatch) -> None:
    calls = []

    def fake(device):
        def trace(**values):
            calls.append((device, values["rays"], values["max_time"], values["first_ray"]))

        return trace

    monkeypatch.setattr(_native, "trace_stochastic", fake("cpu"))
    monkeypatch.setattr(gpu_tracer, "trace_stochastic", fake("gpu"))
    cpu_rate, gpu_rate = ray_device.trial(_arguments(reference, "vector_split"))
    assert cpu_rate > 0 and gpu_rate > 0
    # A warm-up on the GPU, then the same run on each: a few rays, past the run's own,
    # for the run's own length.
    assert [c[0] for c in calls] == ["gpu", "gpu", "cpu"]
    assert calls[1][1:] == calls[2][1:] == (ray_device.TRIAL_RAYS, 0.4, 1 << 40)


def test_a_trial_the_gpu_declines_lets_go_of_its_rays_in_flight(reference, monkeypatch) -> None:
    """The warm-up keeps the GPU's rays in flight for the timed call; the trial lets go
    of them whether or not that call declines, as the run then goes to the CPU."""
    calls = []

    def on_gpu(**values):
        calls.append(values["keep"])
        if len(calls) > 1:
            raise gpu_tracer.Unsupported("the GPU's buffers cannot hold this run's results")

    monkeypatch.setattr(gpu_tracer, "trace_stochastic", on_gpu)
    monkeypatch.setattr(gpu_tracer, "release", lambda: calls.append("released"))
    with pytest.raises(gpu_tracer.Unsupported):
        ray_device.trial(_arguments(reference, "vector"))
    assert calls == [True, True, "released"]


@pytest.mark.parametrize("name", ["vector", "vector_split", "secondary"])
def test_shared_rays_add_up_to_one_run(reference, monkeypatch, name) -> None:
    """With every share traced on the CPU (the GPU declining them), sharing gives what
    one run on the CPU does: each share's energy, ray energies and weight window
    scaled to the whole run."""

    def decline(**_):
        raise gpu_tracer.Unsupported("not here")

    monkeypatch.setattr(gpu_tracer, "trace_stochastic", decline)
    values = _arguments(reference, name)
    plan = ray_device.Plan("both", cpu_rate=1.0, gpu_rate=3.0)
    found, followed = ray_device.trace(plan, values)
    # The CPU, as the GPU declined, unless the CPU took every part before it asked.
    assert followed in (ray_device.Plan("cpu", reason="not here"), plan)
    expected = late_results(_native.trace_stochastic(**values))
    for key, value in expected.items():
        np.testing.assert_allclose(
            late_results(found)[key], value, rtol=1e-9, atol=1e-300, err_msg=key
        )


@pytest.mark.parametrize("name", ["vector", "vector_split"])
def test_shared_rays_on_both_devices_add_up_to_one_run(reference, monkeypatch, name) -> None:
    """With parts on the GPU too (the CPU's tracer standing in for it, each part
    carrying its share of the energy as the GPU's do), the sums kept as the parts come
    in give what one run on the CPU does."""

    taken: list[int] = []

    def on_gpu(*, total, keep=False, **values):
        taken.append(values["rays"])
        share = values["rays"] / total
        target = np.asarray(values["split_target"]) * (total / values["rays"])
        part = list(_native.trace_stochastic(**(values | {"split_target": target})))
        for i in (0, 1, 2, 4):
            part[i] = part[i] * share
        return tuple(part)

    monkeypatch.setattr(gpu_tracer, "trace_stochastic", on_gpu)
    # Enough rays for several parts on each.
    values = _arguments(reference, name) | {"rays": 20_000}
    plan = ray_device.Plan("both", cpu_rate=1.0, gpu_rate=1.0)
    found, followed = ray_device.trace(plan, values)
    assert followed == plan and 0 < sum(taken) < values["rays"]
    expected = late_results(_native.trace_stochastic(**values))
    # A part's weight window takes the run's target times the run's rays over its own,
    # which rounds, and a ray at the edge of the window can go the other way.
    for key, value in expected.items():
        np.testing.assert_allclose(
            late_results(found)[key], value, rtol=1e-6, atol=1e-300, err_msg=key
        )


@pytest.mark.parametrize("count", [500, 4_500, 45_000])
def test_each_device_takes_its_share_of_every_call(reference, monkeypatch, count) -> None:
    """A call of few rays, as each batch of the late part makes, is shared by the
    devices' rates too: the GPU takes no more than its share at a time, so a CPU as fast
    traces the other half however few the rays."""
    monkeypatch.setattr(os, "cpu_count", lambda: 5)
    taken: dict[str, list[int]] = {"cpu": [], "gpu": []}

    def tracer(device: str, seconds: float):
        def trace(*, total=None, **values):
            taken[device].append(values["rays"])
            time.sleep(seconds)
            arrays = np.zeros(3), np.zeros(0), np.zeros(0)
            return *arrays, 0, np.zeros(2), np.zeros(2), values["rays"]

        return trace

    monkeypatch.setattr(_native, "trace_stochastic", tracer("cpu", 0.0))
    monkeypatch.setattr(gpu_tracer, "trace_stochastic", tracer("gpu", 0.05))
    values = _arguments(reference, "vector") | {"rays": count, "threads": 5}
    # As fast on the four CPU threads beside the one that keeps the GPU busy.
    plan = ray_device.Plan("both", cpu_rate=5.0, gpu_rate=4.0)
    found, followed = ray_device.trace(plan, values)
    assert followed == plan and found[-1] == sum(taken["cpu"]) + sum(taken["gpu"]) == count
    assert max(taken["gpu"], default=0) <= math.ceil(count / 2) <= sum(taken["cpu"]) + 1


def test_a_run_the_gpu_declines_says_where_it_went(reference, machine, monkeypatch) -> None:
    def decline(**_):
        raise gpu_tracer.Unsupported("the GPU's buffers cannot hold this run's results")

    monkeypatch.setattr(gpu_tracer, "trace_stochastic", decline)
    values = _arguments(reference, "vector")
    cpu = ray_device.Plan("cpu", reason="the GPU's buffers cannot hold this run's results")
    found, followed = ray_device.trace(ray_device.Plan("gpu"), values)
    assert followed == cpu
    np.testing.assert_array_equal(found[0], _native.trace_stochastic(**values)[0])
    # auto's trial declined alike takes the CPU.
    assert ray_device.choose("auto", values) == cpu

    from acoustic_spaces.engine.simulate import simulate

    room = Room.from_yaml(SHOEBOX)
    sim = room.scene.simulation
    late = sim.late.model_copy(update={"rays": 300, "device": "gpu"})
    messages: list[str] = []
    settings = sim.model_copy(update={"ir_length_s": 0.3, "late": late})
    simulate(room, settings=settings, progress=messages.append)
    where = [m for m in messages if "late rays traced on" in m]
    assert where == [f"talker: late rays traced on the CPU ({cpu.reason})"]


def test_simulate_says_where_the_late_rays_went(monkeypatch) -> None:
    from acoustic_spaces.engine.simulate import simulate

    monkeypatch.setattr(gpu, "hardware_gpu", lambda: False)
    room = Room.from_yaml(SHOEBOX)
    sim = room.scene.simulation
    auto = "the CPU (no hardware GPU)"
    for default, device, said in [
        ("cpu", "cpu", None),
        ("cpu", "auto", auto),
        ("cpu", None, None),
        ("auto", None, auto),  # a scene naming no device follows this computer's default
        ("auto", "cpu", None),
    ]:
        tuning.set_default(default)
        late = sim.late.model_copy(update={"rays": 300, "device": device})
        messages: list[str] = []
        settings = sim.model_copy(update={"ir_length_s": 0.3, "late": late})
        simulate(room, settings=settings, progress=messages.append)
        where = [m for m in messages if "late rays traced on" in m]
        assert where == ([f"talker: late rays traced on {said}"] if said else [])


def test_scenes_naming_no_device_follow_this_computers_default(machine, monkeypatch) -> None:
    assert tuning.default_device() == "cpu" and tuning.resolve_device(None) == "cpu"
    tuning.set_default("auto")
    assert tuning.resolve_device(None) == "auto" and tuning.resolve_device("gpu") == "gpu"
    with pytest.raises(ValueError, match="cpu or auto"):
        tuning.set_default("both")
    # Kept beside the timings, for this computer only.
    tuning.keep_late(1.0, {"device": "gpu", "cpu_rate": 1, "gpu_rate": 2, "from": "tune"})
    assert tuning.default_device() == "auto" and tuning.late_choice(1.0)["device"] == "gpu"
    monkeypatch.setattr(gpu, "gpu_name", lambda: "Another GPU")
    assert tuning.default_device() == "cpu"
