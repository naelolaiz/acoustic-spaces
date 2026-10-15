"""Scattering: vector-based and random reflection, and the scattering of a surface's size."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine import late as late_stage
from acoustic_spaces.engine.early import exhaustive_candidates, find_paths
from acoustic_spaces.engine.model import TraceModel, size_scattering, surface_sizes
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.scene.model import Scattering

SIZE = [8.0, 6.0, 3.0]
C = 343.0


def _box(coefficient: float, absorption=0.2, **settings) -> Room:
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": SIZE},
                "materials": {"m": {"absorption": [absorption] * 8, "scattering": coefficient}},
                "default_material": "m",
                "sources": [{"name": "s", "position": [2.0, 3.0, 1.5]}],
                "listeners": [
                    {"name": "a", "position": [6.0, 3.5, 1.2]},
                    {"name": "b", "position": [5.0, 1.5, 2.1]},
                ],
                "simulation": {"seed": 3, **settings},
            }
        )
    )


def test_defaults_and_settings() -> None:
    assert Scattering() == Scattering(method="vector", from_size=True, by_band=True)
    model = TraceModel.from_room(_box(0.1))
    assert model.vector_scattering and model.surface_size is not None
    assert model.size_from_order == 3  # first-order diffraction covers orders 1 and 2
    old = {"method": "random", "from_size": False}
    plain = TraceModel.from_room(_box(0.1, scattering=old, early={"diffraction": "none"}))
    assert not plain.vector_scattering and plain.surface_size is None
    assert plain.size_from_order == 1
    with pytest.raises(ValidationError):
        Scattering(method="lambert")


def test_surface_sizes() -> None:
    model = TraceModel.from_room(_box(0.1))
    sizes = {tuple(np.round(s, 6)) for s in model.surface_size}
    assert sizes == {(3.0, 8.0), (3.0, 6.0), (6.0, 8.0)}  # walls, end walls, floor and ceiling
    # One plane, two materials, joined at an edge: one surface; an L of three squares;
    # a regular octagon, rounder than a square, counts as a square of its area.
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    vertices = np.vstack([square, square + [1, 0, 0], square + [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11]])
    # Shared corners are separate vertices: the L's squares meet at edges by position.
    _, index = np.unique(np.round(vertices, 9), axis=0, return_inverse=True)
    vertices, faces = np.unique(np.round(vertices, 9), axis=0), index.reshape(-1)[faces]
    width, length = surface_sizes(vertices, faces, np.zeros(6, dtype=np.int64)).T
    # Area 3 and perimeter 8: the rectangle (2 - 1) x (2 + 1).
    np.testing.assert_allclose(width, 1.0)
    np.testing.assert_allclose(length, 3.0)
    angle = np.arange(8) * math.pi / 4
    octagon = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(8)])
    vertices = np.vstack([[0.0, 0.0, 0.0], octagon])
    faces = np.array([[0, 1 + k, 1 + (k + 1) % 8] for k in range(8)])
    area = 8 * 0.5 * math.sin(math.pi / 4)
    sizes = surface_sizes(vertices, faces, np.zeros(8, dtype=np.int64))
    np.testing.assert_allclose(sizes, math.sqrt(area))


def test_size_scattering_follows_christensen_and_rindel() -> None:
    f = np.asarray(OCTAVE_BANDS_HZ, float)
    own = np.full((1, 8), 0.1)

    def scattering(width, length, distance, cosine):
        values = (np.array([v], float) for v in (width, length, distance, cosine))
        return size_scattering(*values, C, own)

    # A 10 x 20 m surface 5 m away, met head on: specular from f_w = c d / (2 w^2) up.
    big = scattering(10.0, 20.0, 5.0, 1.0)
    f_width = C * 5.0 / (2 * 10.0**2)  # 8.6 Hz: the whole range
    assert f_width < f[0]
    np.testing.assert_allclose(big, own)
    # A 1 x 2 m panel 10 m away: f_w = 1715 Hz, f_l = 429 Hz.
    small = scattering(1.0, 2.0, 10.0, 1.0)
    specular = np.minimum(1, f / 1715.0) * np.minimum(1, f / 428.75)
    np.testing.assert_allclose(small[0], 1 - 0.9 * specular)
    # Met at 60 degrees, the panel looks half as wide: f_w is four times as high.
    oblique = scattering(1.0, 2.0, 10.0, 0.5)
    assert np.all(oblique >= small) and oblique[0, 5] > small[0, 5]
    # Right where the sound comes from (a corner): specular.
    np.testing.assert_allclose(scattering(1.0, 2.0, 0.0, 1.0), own)


def test_early_reflections_carry_the_scattering_of_the_surfaces_size() -> None:
    room = _box(0.1, early={"diffraction": "none"})
    model = TraceModel.from_room(room)
    source = np.array(room.scene.sources[0].position)
    listener = np.array(room.scene.listeners[0].position)
    paths = find_paths(
        model,
        source,
        listener,
        exhaustive_candidates(model, source, 1),
        speed_of_sound=room.speed_of_sound(),
        air=np.zeros(8),
        max_delay_s=1.0,
    )
    # The reflection off the side wall y = 0: a surface 3 m high and 8 m long, met at
    # cos = 3 / d after d metres.
    wall = next(k for k in range(len(paths)) if abs(paths.points[k][1][1]) < 1e-9)
    incoming = paths.points[wall][1] - source
    d = np.linalg.norm(incoming)
    f = np.asarray(OCTAVE_BANDS_HZ, float)
    f_width = room.speed_of_sound() * d / (2 * (3.0 * 3.0 / d) ** 2)
    f_length = room.speed_of_sound() * d / (2 * 8.0**2)
    specular = np.minimum(1, f / f_width) * np.minimum(1, f / f_length)
    assert specular[0] < 0.8 and specular[-1] == 1.0
    distance = paths.distance_m[wall]
    # (1 - absorption) (1 - scattering), the scattering raised from 0.1 by the size.
    expected = 0.8 * 0.9 * specular / (4 * math.pi * distance**2)
    np.testing.assert_allclose(paths.energy[wall], expected, rtol=1e-9)


def _late(room: Room, covered_order: int, rays: int, **kwargs) -> np.ndarray:
    model = TraceModel.from_room(room)
    for key, value in kwargs.items():
        setattr(model, key, value)
    return late_stage.stochastic_energy(
        model,
        np.array(room.scene.sources[0].position),
        np.array([listener.position for listener in room.scene.listeners]),
        rays=rays,
        max_time=0.3,
        time_step=1e-3,
        speed_of_sound=room.speed_of_sound(),
        air=np.zeros(8),
        covered_order=covered_order,
        radius=0.3,
        seed=2,
    ).energy


@pytest.mark.parametrize(("method", "rays"), [("vector", 25_000), ("random", 100_000)])
def test_early_and_late_add_up_to_the_whole(method, rays) -> None:
    # Left out of the late part, the specular-only share of the paths up to order 3 is
    # what the early stage brings: together they hold what the tracer finds on its own.
    room = _box(0.3, scattering={"method": method}, early={"diffraction": "none"})
    model = TraceModel.from_room(room)
    source = np.array(room.scene.sources[0].position)
    whole = _late(room, -1, rays).sum(axis=2)  # per band: each traced on its own
    late = _late(room, 3, rays).sum(axis=2)
    candidates = exhaustive_candidates(model, source, 3)
    for k, listener in enumerate(room.scene.listeners):
        early = find_paths(
            model,
            source,
            np.array(listener.position),
            candidates,
            speed_of_sound=room.speed_of_sound(),
            air=np.zeros(8),
            max_delay_s=0.3,
        )
        np.testing.assert_allclose(late[k] + early.energy.sum(axis=0), whole[k], rtol=0.05)


def test_small_surfaces_scatter_a_specular_box_into_a_diffuse_one() -> None:
    # Smooth walls (scattering 0): the box keeps rays that never meet the floor or the
    # ceiling and decays slowly. Made of tiny surfaces, every reflection scatters fully,
    # and the decay is Eyring's.
    room = _box(0.0, absorption=0.1, ir_length_s=1.0)

    def decay_rate(energy: np.ndarray) -> np.ndarray:
        t = np.arange(energy.shape[-1]) * 1e-3
        inside = (t > 0.05) & (t < 0.3)
        level = 10 * np.log10(energy[..., inside].sum(axis=0) + 1e-300)
        return -np.polyfit(t[inside], level.T, 1)[0]

    tiny = np.full((len(room.mesh.faces), 2), 0.01)
    diffuse = decay_rate(_late(room, -1, 20_000, surface_size=tiny))
    specular = decay_rate(_late(room, -1, 20_000, surface_size=None))
    eyring = 60.0 / room.reverberation_time("eyring")
    np.testing.assert_allclose(diffuse, eyring, rtol=0.1)
    assert np.all(specular < 0.9 * eyring)


def test_each_band_follows_its_own_scattering() -> None:
    # Diffuse in the bass, specular in the treble: traced with one direction for all
    # bands, every band would decay alike. Each band has rays of its own instead.
    split = [1.0] * 4 + [0.0] * 4
    room = _box(0.0, absorption=0.1, ir_length_s=1.0, scattering={"from_size": False})
    room.scene.materials["m"].scattering = split
    room = Room.from_scene(room.scene)
    model = TraceModel.from_room(room)
    assert late_stage.band_groups(model) == [slice(b, b + 1) for b in range(8)]
    eyring = 60.0 / room.reverberation_time("eyring")
    t = np.arange(300) * 1e-3
    inside = (t > 0.05) & (t < 0.3)
    level = 10 * np.log10(_late(room, -1, 20_000).sum(axis=0)[:, inside] + 1e-300)
    rate = -np.polyfit(t[inside], level.T, 1)[0]
    np.testing.assert_allclose(rate[:4], eyring[:4], rtol=0.1)
    assert np.all(rate[4:] < 0.9 * eyring[4:])


def test_one_ray_carries_all_bands_where_they_scatter_alike() -> None:
    def groups(**settings) -> int:
        return len(late_stage.band_groups(TraceModel.from_room(_box(**settings))))

    assert groups(coefficient=0.1) == 8  # the size makes the bands differ
    assert groups(coefficient=1.0) == 1  # fully diffuse at any size
    assert groups(coefficient=0.1, scattering={"from_size": False}) == 1
    assert groups(coefficient=0.1, scattering={"method": "random"}) == 1
    assert groups(coefficient=0.1, scattering={"by_band": False}) == 1
