"""The scattered sound the ground mirrors to a listener: the rain on the listener's image
in the ground (``mirror_plane`` and ``face_mirror`` of ``_native.trace_stochastic``,
``late.ground_mirrors`` in a run), in place of the detector spheres' rare hits."""

from __future__ import annotations

import numpy as np
import pytest
from test_outdoor import OPEN, SHOEBOX, outdoor

from acoustic_spaces import Room, _native
from acoustic_spaces.engine import late, ray_device
from acoustic_spaces.engine.model import TraceModel

#: A wall 30 m wide and 10 m tall standing on ground 400 m across, 20 m in front of a
#: source 2 m up: the wall scatters all it reflects, the ground none.
WALL_X, WALL_Y, WALL_Z = 20.0, 15.0, 10.0
WALL_ALPHA, GROUND_ALPHA = 0.1, 0.3
SOURCE = np.array([0.0, 0.0, 2.0])
LISTENERS = np.array([[5.0, 8.0, 1.5], [12.0, -6.0, 1.2]])
#: The ground's mirror plane: the normal up and the offset (z = 0).
MIRROR = np.array([[0.0, 0.0, 1.0, 0.0]])


def _args(rays: int, seed: int, radius: float = 0.5, vector: bool = True, **more) -> dict:
    """The tracer's arguments for the wall on the ground, in one band in silent air, with
    the ground's mirror plane; the paths that only reflect specularly are the early
    stage's (order 1), and the wall's specular share is nil, so all the listeners hear of
    the rays is the rain."""
    ground = [[-200, -200, 0], [200, -200, 0], [200, 200, 0], [-200, 200, 0]]
    wall = [[WALL_X, -WALL_Y, 0], [WALL_X, WALL_Y, 0], [WALL_X, WALL_Y, WALL_Z]]
    wall.append([WALL_X, -WALL_Y, WALL_Z])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.int64)
    args = dict(
        bvh=_native.Bvh(np.array(ground + wall, float), faces),
        source=SOURCE,
        directivity_a=np.zeros(0),
        orientation=np.eye(3),
        directivity_table=np.zeros((0, 0, 1)),
        absorption=np.array([[GROUND_ALPHA], [GROUND_ALPHA], [WALL_ALPHA], [WALL_ALPHA]]),
        scattering=np.array([[0.0], [0.0], [1.0], [1.0]]),
        air=np.zeros(1),
        surface_size=np.zeros((0, 2)),
        band_hz=np.zeros(0),
        vector_scattering=vector,
        size_from_order=1,
        transmission=np.zeros((0, 0)),
        face_room=np.zeros((0, 2), np.int64),
        source_room=-1,
        receivers=LISTENERS,
        directions=np.zeros((0, 3)),
        rays=rays,
        max_time=0.5,
        time_step=1e-3,
        dir_time_step=1e-2,
        speed_of_sound=343.0,
        receiver_radius=np.full(len(LISTENERS), radius),
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
        mirror_plane=MIRROR,
        face_mirror=np.array([0, 0, -1, -1], np.int64),
    )
    args.update(more)
    return args


def _scattered(listener: np.ndarray) -> float:
    """What the wall scatters to a listener, integrated over the wall (midpoints of 3 cm
    squares): what lands on each patch straight from the source and off the ground (the
    source's image), times Lambert's cos / (pi d^2) to the listener and, off the
    ground, to its image."""
    y = -WALL_Y + (np.arange(1000) + 0.5) * (2 * WALL_Y / 1000)
    z = (np.arange(334) + 0.5) * (WALL_Z / 334)
    y, z = np.meshgrid(y, z, indexing="ij")
    area = (2 * WALL_Y / 1000) * (WALL_Z / 334)

    def lambert(point, power):
        r2 = (WALL_X - point[0]) ** 2 + (y - point[1]) ** 2 + (z - point[2]) ** 2
        return power * (WALL_X - point[0]) / np.sqrt(r2) / r2

    def mirrored(point):
        return point * np.array([1.0, 1.0, -1.0])

    lands = (lambert(SOURCE, 1.0) + lambert(mirrored(SOURCE), 1 - GROUND_ALPHA)) / (4 * np.pi)
    sent = (lambert(listener, 1.0) + lambert(mirrored(listener), 1 - GROUND_ALPHA)) / np.pi
    return float(np.sum(lands * (1 - WALL_ALPHA) * sent) * area)


def _energy(seeds: int, rays: int, **more) -> np.ndarray:
    """The energy at each listener (seeds, listeners)."""
    return np.array(
        [
            _native.trace_stochastic(**_args(rays, seed, **more))[0].sum((1, 2))
            for seed in range(seeds)
        ]
    )


@pytest.mark.parametrize("vector", [True, False], ids=["vector", "random"])
def test_the_ground_mirrors_a_wall_s_scattered_sound_by_the_rain(vector: bool) -> None:
    # All the listeners hear of the rays is what the wall scatters, straight or off the
    # ground: the rain on them and on their images brings it with its exact energy. The
    # detector spheres counted what came off the ground (no mirror plane), a few rays
    # each bringing a burst: the same energy, far less steadily.
    exact = np.array([_scattered(listener) for listener in LISTENERS])
    rained = _energy(16, 4000, vector=vector)
    mean, error = rained.mean(0), rained.std(0, ddof=1) / np.sqrt(len(rained))
    assert np.all(np.abs(mean - exact) < 4 * error + 1e-3 * exact), (mean / exact, error / exact)
    caught = _energy(16, 4000, vector=vector, mirror_plane=None, face_mirror=None)
    mean, error = caught.mean(0), caught.std(0, ddof=1) / np.sqrt(len(caught))
    assert np.all(np.abs(mean - exact) < 4 * error), (mean / exact, error / exact)
    # 6.6 and 15 times (vector), 7.2 and 25 times (random) when written.
    assert np.all(caught.std(0) > 5 * rained.std(0)), (caught.std(0), rained.std(0))


@pytest.mark.parametrize("vector", [True, False], ids=["vector", "random"])
def test_the_detector_leaves_the_ground_s_mirror_of_scattered_sound_to_the_rain(
    vector: bool,
) -> None:
    # The detector counts none of it any more: the response is the same whatever the
    # spheres' size (no reflection lies within 2 m of a listener but the ground's,
    # which scatters nothing). Before, each ray off the wall and the ground through a
    # sphere counted, as many more as the sphere was larger.
    small = _native.trace_stochastic(**_args(2000, 5, radius=0.5, vector=vector))[0]
    large = _native.trace_stochastic(**_args(2000, 5, radius=2.0, vector=vector))[0]
    assert small.sum() > 0.0
    np.testing.assert_allclose(large, small, rtol=1e-12, atol=0.0)


def test_mirror_planes_need_the_diffuse_rain_alone() -> None:
    with pytest.raises(ValueError, match="diffuse rain alone"):
        _native.trace_stochastic(**_args(10, 0, specular_rain=True))
    with pytest.raises(ValueError, match="face_mirror"):
        _native.trace_stochastic(**_args(10, 0, face_mirror=np.array([0, 1, -1, -1], np.int64)))
    with pytest.raises(ValueError, match="unit normal"):
        _native.trace_stochastic(**_args(10, 0, mirror_plane=np.array([[0.0, 0.0, 2.0, 0.0]])))
    assert late.MAX_MIRRORS == _native.MAX_MIRRORS


def test_the_ground_s_planes_are_its_mirrors() -> None:
    # The ground and a zone of it in one plane, a barrier standing on it: one mirror
    # plane, z = 0 facing up, for the ground's faces only. Indoors: none.
    bed = {"name": "bed", "polygon": [[-50, -2], [50, -2], [50, 2], [-50, 2]]}
    barrier = {"name": "wall", "path": [[-20, -6], [20, -6]], "height": 3}
    geometry = {"ground": OPEN["ground"] | {"zones": [bed]}, "barriers": [barrier]}
    model = TraceModel.from_room(Room.from_scene(outdoor(geometry)))
    found = late.ground_mirrors(model)
    np.testing.assert_allclose(found["mirror_plane"], MIRROR, atol=1e-12)
    labels = np.array(["ground" in label for label in model.room.mesh.labels])
    ground = labels[model.room.mesh.face_labels]
    assert ground.sum() > 2
    np.testing.assert_array_equal(found["face_mirror"], np.where(ground, 0, -1))
    assert late.ground_mirrors(TraceModel.from_room(Room.from_yaml(SHOEBOX))) == {}


def test_the_gpu_leaves_the_mirrors_to_the_cpu() -> None:
    values = {"mirror_plane": MIRROR, "transmission": np.zeros((0, 0))}
    assert "images in the ground" in ray_device._unsupported(values)
