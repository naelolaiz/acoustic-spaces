"""The compiled core's ray tracing computes what it computed before.

``reference/tracer.npz`` holds fixed inputs (two small rooms, rays, sources, listeners,
materials) and what the core computed from them before the ray tree was rebuilt: the
nearest face along each ray, the faces specular rays meet, and the stochastic tracer's
energy (every reception mode and scattering method, directivity, splitting, sound
through a partition, face groups and a coverage map). These tests feed the same inputs
to the code under test, with each kind of tree, and check it computes the same, to
rounding: a faster tree, a new traversal or another tracer must not change results
unnoticed.

The reference came from main before the SAH tree (commit 5a725b2). To write it again,
only for a change that is meant to alter what the tracer computes (and say so in its
pull request), run this file with that code first on the path::

    PYTHONPATH=src python tests/test_tracer_reference.py --write
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REFERENCE = Path(__file__).parent / "reference" / "tracer.npz"
#: Threads of every run: the sum over threads is then the same sum each time.
THREADS = 2
#: Kinds of surface, their absorption and scattering per band (500, 1000, 2000 Hz), and
#: what the partition lets through.
BAND_HZ = (500.0, 1000.0, 2000.0)
FLOOR, SEATS, WALL, CEILING, PILLAR, PANEL, PARTITION = range(7)
ABSORPTION = (
    (0.02, 0.03, 0.04),
    (0.45, 0.6, 0.7),
    (0.05, 0.06, 0.07),
    (0.15, 0.2, 0.25),
    (0.03, 0.03, 0.04),
    (0.1, 0.08, 0.06),
    (0.1, 0.1, 0.1),
)
SCATTERING = (
    (0.1, 0.15, 0.2),
    (0.4, 0.6, 0.7),
    (0.05, 0.1, 0.15),
    (0.1, 0.2, 0.3),
    (0.2, 0.3, 0.4),
    (0.05, 0.05, 0.1),
    (0.05, 0.05, 0.05),
)
PARTITION_TRANSMISSION = (0.05, 0.02, 0.01)
AIR = (0.0006, 0.001, 0.0025)


class _Mesh:
    """Quads, two triangles each, with a kind of surface and a width and length."""

    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []
        self.kinds: list[int] = []
        self.sizes: list[tuple[float, float]] = []

    def quad(self, corners, kind: int) -> None:
        a, b, _, d = (np.array(c, float) for c in corners)
        base = len(self.vertices)
        self.vertices += [tuple(map(float, c)) for c in corners]
        self.faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        self.kinds += [kind, kind]
        width, length = sorted((np.linalg.norm(b - a), np.linalg.norm(d - a)))
        self.sizes += [(width, length)] * 2

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "vertices": np.array(self.vertices, float),
            "faces": np.array(self.faces, np.int64),
            "kinds": np.array(self.kinds, np.int64),
            "sizes": np.array(self.sizes, float),
        }


def _hall() -> dict[str, np.ndarray]:
    """A hall 16 by 10 by 6 m: a flat floor, four rows of raked seats, a pillar and
    two tilted reflectors."""
    m = _Mesh()
    length, width, height = 16.0, 10.0, 6.0
    m.quad([(0, 0, 0), (8, 0, 0), (8, width, 0), (0, width, 0)], FLOOR)
    edges, rises = (8.0, 10.0, 12.0, 14.0, 16.0), (0.4, 0.8, 1.2, 1.6)
    below = 0.0
    for i, rise in enumerate(rises):
        x0, x1 = edges[i], edges[i + 1]
        m.quad([(x0, 0, below), (x0, width, below), (x0, width, rise), (x0, 0, rise)], SEATS)
        m.quad([(x0, 0, rise), (x0, width, rise), (x1, width, rise), (x1, 0, rise)], SEATS)
        below = rise
    m.quad([(0, 0, 0), (0, 0, height), (0, width, height), (0, width, 0)], WALL)
    top = rises[-1]
    m.quad(
        [(length, 0, top), (length, width, top), (length, width, height), (length, 0, height)], WALL
    )
    for y in (0.0, width):
        m.quad([(0, y, 0), (length, y, 0), (length, y, height), (0, y, height)], WALL)
    m.quad(
        [(0, 0, height), (length, 0, height), (length, width, height), (0, width, height)], CEILING
    )
    lo, hi = 4.75, 5.25
    for a, b in (
        ((lo, lo), (hi, lo)),
        ((hi, lo), (hi, hi)),
        ((hi, hi), (lo, hi)),
        ((lo, hi), (lo, lo)),
    ):
        m.quad([(*a, 0), (*b, 0), (*b, height), (*a, height)], PILLAR)
    m.quad([(1, 2, 4.5), (4, 2, 4.5), (4, 8, 5.1), (1, 8, 5.1)], PANEL)
    m.quad([(9, 2, 5.2), (13, 2, 4.6), (13, 8, 4.6), (9, 8, 5.2)], PANEL)
    return m.arrays()


def _two_rooms() -> dict[str, np.ndarray]:
    """A box 12 by 6 by 4 m that a partition at x = 6 parts into two rooms, with the room
    on the front and the back of each face (-1: outside)."""
    m = _Mesh()
    length, width, height, cut = 12.0, 6.0, 4.0, 6.0
    for x0, x1 in ((0.0, cut), (cut, length)):
        m.quad([(x0, 0, 0), (x1, 0, 0), (x1, width, 0), (x0, width, 0)], FLOOR)
        m.quad(
            [(x0, 0, height), (x0, width, height), (x1, width, height), (x1, 0, height)], CEILING
        )
        for y in (0.0, width):
            m.quad([(x0, y, 0), (x0, y, height), (x1, y, height), (x1, y, 0)], WALL)
    for x in (0.0, length):
        m.quad([(x, 0, 0), (x, width, 0), (x, width, height), (x, 0, height)], WALL)
    m.quad([(cut, 0, 0), (cut, 0, height), (cut, width, height), (cut, width, 0)], PARTITION)
    out = m.arrays()
    corners = out["vertices"][out["faces"]]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    centre = corners.mean(axis=1)

    def room(p: np.ndarray) -> np.ndarray:
        inside = np.all((p > 0) & (p < (length, width, height)), axis=1)
        return np.where(inside, (p[:, 0] > cut).astype(np.int64), -1)

    out["face_room"] = np.stack([room(centre + 0.01 * normal), room(centre - 0.01 * normal)], 1)
    return out


def _directions(rng: np.random.Generator, n: int) -> np.ndarray:
    d = rng.standard_normal((n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _split_target(ray_energy: np.ndarray, reflections: np.ndarray) -> np.ndarray:
    """The weight window's expected ray energy from a first run, as the late stage
    makes it (never rising)."""
    mean = np.divide(ray_energy, reflections, out=np.zeros_like(ray_energy), where=reflections > 0)
    return np.minimum.accumulate(np.where(mean > 0, mean, np.inf)).clip(max=mean.max())


def make_inputs() -> dict[str, np.ndarray]:
    """The inputs: geometry, rays and every setting of every run."""
    rng = np.random.default_rng(20261015)
    data = {f"hall_{k}": v for k, v in _hall().items()}
    data |= {f"rooms_{k}": v for k, v in _two_rooms().items()}
    n = 600
    # In single precision, which the file stores them in.
    data["rays_origins"] = rng.uniform((0.2, 0.2, 0.2), (15.8, 9.8, 5.8), (n, 3)).astype(np.float32)
    data["rays_directions"] = _directions(rng, n).astype(np.float32)
    data["rays_t_max"] = np.where(np.arange(n) < n // 2, np.inf, rng.uniform(0.5, 12.0, n))
    faces = len(data["hall_faces"])
    data["rays_ignore"] = np.where(np.arange(n) % 7 == 0, rng.integers(0, faces, n), -1)
    data["directions"] = _directions(rng, 8)
    turn = np.radians(30.0)
    data["orientation"] = np.array(
        [[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]]
    )
    data["table"] = rng.uniform(0.3, 1.7, (3, 4, len(BAND_HZ)))
    return data


def _late_runs() -> dict[str, dict]:
    """The stochastic tracer's runs: their room and settings (arrays by name)."""
    hall = {
        "room": "hall",
        "source": (2.0, 5.0, 1.5),
        "listeners": ((9.0, 3.0, 1.6), (13.0, 6.0, 2.6), (4.0, 5.0, 1.2)),
        "radius": (0.5, 0.6, 0.5),
    }
    vector = hall | {
        "directivity_a": (0.5, 0.4, 0.3),
        "sizes": True,
        "vector": True,
        "size_from_order": 1,
        "directions": True,
        "diffuse_rain": True,
        "specular_rain": False,
        "covered_order": 2,
        "covered_time": (0.03, 0.045, 0.03),
        "seed": 11,
    }
    return {
        "vector": vector | {"rays": 2000, "first_ray": 0},
        "vector_split": vector | {"rays": 3000, "first_ray": 2000, "target": "vector"},
        "random_detector": hall
        | {
            "table": True,
            "sizes": False,
            "vector": False,
            "diffuse_rain": False,
            "specular_rain": False,
            "covered_order": -1,
            "rays": 3000,
            "seed": 12,
        },
        "secondary": hall
        | {
            "sizes": True,
            "vector": True,
            "size_from_order": 2,
            "directions": True,
            "diffuse_rain": True,
            "specular_rain": True,
            "covered_order": 2,
            "rays": 2000,
            "seed": 13,
        },
        "rooms": {
            "room": "rooms",
            "source": (2.0, 3.0, 1.5),
            "listeners": ((4.0, 2.0, 1.2), (9.0, 4.0, 1.2)),
            "radius": (0.5, 0.5),
            "sizes": True,
            "vector": True,
            "size_from_order": 1,
            "diffuse_rain": True,
            "specular_rain": False,
            "covered_order": 1,
            "transmission": True,
            "room_level": (1.0, 0.25),
            "groups": True,
            "rays": 2000,
            "first_ray": 500,
            "target": "rooms_first",
            "seed": 14,
        },
        "rooms_first": {
            "room": "rooms",
            "source": (2.0, 3.0, 1.5),
            "listeners": ((4.0, 2.0, 1.2), (9.0, 4.0, 1.2)),
            "radius": (0.5, 0.5),
            "vector": True,
            "diffuse_rain": True,
            "specular_rain": False,
            "transmission": True,
            "rays": 500,
            "seed": 14,
        },
    }


def _bvh(native, data, room: str, split: str | None):
    vertices = np.ascontiguousarray(data[f"{room}_vertices"])
    faces = np.ascontiguousarray(data[f"{room}_faces"])
    if split is None:
        return native.Bvh(vertices, faces)
    return native.Bvh(vertices, faces, split=split)


def _coefficients(data, room: str) -> tuple[np.ndarray, ...]:
    kinds = data[f"{room}_kinds"]
    absorption = np.ascontiguousarray(np.array(ABSORPTION)[kinds])
    scattering = np.ascontiguousarray(np.array(SCATTERING)[kinds])
    transmission = np.zeros_like(absorption)
    transmission[kinds == PARTITION] = PARTITION_TRANSMISSION
    return absorption, scattering, transmission


def _late(native, data, name: str, run: dict, split: str | None, out: dict) -> dict:
    room = run["room"]
    bvh = _bvh(native, data, room, split)
    absorption, scattering, transmission = _coefficients(data, room)
    bands = len(BAND_HZ)
    listeners = np.array(run["listeners"], float)
    if run.get("table"):
        first_order, table = np.ones(bands), np.ascontiguousarray(data["table"])
    else:
        first_order, table = np.array(run.get("directivity_a", ()), float), np.zeros((0, 0, bands))
    if run.get("sizes"):
        sizes, band_hz = np.ascontiguousarray(data[f"{room}_sizes"]), np.array(BAND_HZ)
    else:
        sizes, band_hz = np.zeros((0, 2)), np.zeros(0)
    if run.get("transmission"):
        passing, face_room = transmission, np.ascontiguousarray(data[f"{room}_face_room"])
        source_room = 0
    else:
        passing, face_room, source_room = np.zeros((0, 0)), np.zeros((0, 2), np.int64), -1
    target = np.zeros(0)
    if "target" in run:
        earlier = out[run["target"]]
        scale = _late_runs()[run["target"]]["rays"] / run["rays"]
        target = _split_target(earlier["ray_energy"], earlier["reflections"]) * scale
    if run.get("groups"):
        kinds = data[f"{room}_kinds"]
        face_group = np.select([kinds == FLOOR, kinds == CEILING, kinds == WALL], [0, 1, 2], -1)
        groups = 3
    else:
        face_group, groups = np.zeros(0, np.int64), 0
    directions = data["directions"] if run.get("directions") else np.zeros((0, 3))
    energy, directional, moments, escaped, ray_energy, reflections, traced = (
        native.trace_stochastic(
            bvh,
            np.array(run["source"], float),
            np.ascontiguousarray(first_order),
            np.ascontiguousarray(data["orientation"]),
            table,
            absorption,
            scattering,
            np.array(AIR),
            sizes,
            band_hz,
            bool(run.get("vector")),
            int(run.get("size_from_order", 1)),
            passing,
            face_room,
            source_room,
            listeners,
            np.ascontiguousarray(directions),
            int(run["rays"]),
            0.4,
            1e-3,
            1e-2,
            343.0,
            np.array(run["radius"], float),
            bool(run["diffuse_rain"]),
            bool(run["specular_rain"]),
            int(run.get("covered_order", -1)),
            np.array(run.get("covered_time", ()), float),
            int(run["seed"]),
            THREADS,
            0.05,
            int(run.get("first_ray", 0)),
            np.ascontiguousarray(target),
            4.0,
            8,
            256,
            np.array(run.get("room_level", ()), float),
            np.ascontiguousarray(face_group, dtype=np.int64),
            groups,
            0.02,
        )
    )
    return {
        "energy": energy,
        "directional": directional,
        "moments": moments,
        "ray_energy": ray_energy,
        "reflections": reflections,
        "counts": np.array([escaped, traced]),
    }


def _map(native, data, split: str | None) -> dict:
    bvh = _bvh(native, data, "hall", split)
    absorption, scattering, _ = _coefficients(data, "hall")
    source = np.array([2.0, 5.0, 1.5])
    nx, ny, cell = 7, 4, 2.0
    x = 1.0 + cell * (np.arange(nx) + 0.5)
    y = 1.0 + cell * (np.arange(ny) + 0.5)
    floor = np.select([x < 8, x < 10, x < 12, x < 14], [0.0, 0.4, 0.8, 1.2], 1.6)
    point = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx, 1)
    point[1, 2, 0] = -1  # round the pillar
    low = np.tile(floor + 0.8, ny)
    centres = np.stack([np.tile(x, ny), np.repeat(y, nx), low + 0.4], 1)
    onset = np.linalg.norm(centres - source, axis=1) / 343.0
    energy, escaped, ray_energy, reflections, traced = native.trace_map(
        bvh,
        source,
        np.array([0.5, 0.4, 0.3]),
        np.eye(3),
        np.zeros((0, 0, 3)),
        absorption,
        scattering,
        np.array(AIR),
        np.ascontiguousarray(data["hall_sizes"]),
        np.array(BAND_HZ),
        True,
        1,
        np.zeros((0, 0)),
        np.zeros((0, 2), np.int64),
        -1,
        np.array([1.0, 1.0]),
        cell,
        point,
        low,
        low + 0.8,
        np.full(nx * ny, cell * cell * 0.8),
        onset,
        onset + 0.02,
        1e-3,
        50,
        1e-2,
        1500,
        0.3,
        343.0,
        2,
        3,
        THREADS,
        0.05,
        0,
        np.zeros(0),
        4.0,
        8,
        256,
        np.zeros(0),
    )
    return {
        "energy": energy,
        "ray_energy": ray_energy,
        "reflections": reflections,
        "counts": np.array([escaped, traced]),
    }


def compute(native, data: dict[str, np.ndarray], split: str | None = None) -> dict:
    """Everything the reference holds, computed by ``native`` (the compiled core) from
    ``data``'s inputs, with a tree of ``split`` (None: the core's default)."""
    out: dict[str, np.ndarray] = {}
    bvh = _bvh(native, data, "hall", split)
    t, face = bvh.intersect(
        np.ascontiguousarray(data["rays_origins"], dtype=np.float64),
        np.ascontiguousarray(data["rays_directions"], dtype=np.float64),
        np.ascontiguousarray(data["rays_t_max"]),
        np.ascontiguousarray(data["rays_ignore"], dtype=np.int64),
    )
    out["rays_t"], out["rays_face"] = t, face
    source = np.array([2.0, 5.0, 1.5])
    out["specular_hall"] = native.specular_sequences(
        bvh, source, 300, 5, THREADS, np.zeros(0, np.uint8)
    )
    rooms = _bvh(native, data, "rooms", split)
    passing = (data["rooms_kinds"] == PARTITION).astype(np.uint8)
    out["specular_rooms"] = native.specular_sequences(
        rooms, np.array([2.0, 3.0, 1.5]), 300, 5, THREADS, passing
    )
    late: dict[str, dict] = {}
    # Runs whose split target another one gives come after it.
    for name, run in sorted(_late_runs().items(), key=lambda item: "target" in item[1]):
        late[name] = _late(native, data, name, run, split, late)
    late["map"] = _map(native, data, split)
    for name, values in late.items():
        out |= {f"late_{name}_{key}": value for key, value in values.items()}
    return out


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    with np.load(REFERENCE) as stored:
        return dict(stored)


@pytest.fixture(scope="module", params=["sah", "median"])
def computed(request, reference) -> dict[str, np.ndarray]:
    from acoustic_spaces import _native

    return compute(_native, reference, request.param)


def test_the_reference_holds_every_result(reference, computed) -> None:
    assert sorted(computed) == sorted(json.loads(str(reference["outputs"])))


def test_nearest_faces_are_the_reference_ones(reference, computed) -> None:
    np.testing.assert_array_equal(computed["rays_face"], reference["rays_face"])
    np.testing.assert_allclose(computed["rays_t"], reference["rays_t"], rtol=1e-12)


def test_specular_rays_meet_the_reference_faces(reference, computed) -> None:
    for key in ("specular_hall", "specular_rooms"):
        np.testing.assert_array_equal(computed[key], reference[key], err_msg=key)


@pytest.mark.parametrize("run", [*_late_runs(), "map"])
def test_stochastic_tracer_computes_the_reference_energy(reference, computed, run) -> None:
    names = ("energy", "directional", "moments", "ray_energy", "reflections", "counts")
    for key in (f"late_{run}_{name}" for name in names if f"late_{run}_{name}" in computed):
        actual, expected = computed[key], reference[key]
        assert actual.shape == expected.shape, key
        if key.endswith("_counts"):
            np.testing.assert_array_equal(actual, expected, err_msg=key)
        else:
            # Stored in single precision; double precision sums differ only by rounding.
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-35, err_msg=key)


def _write() -> None:
    from acoustic_spaces import _native

    data = make_inputs()
    computed = compute(_native, data)
    for key, value in computed.items():
        if value.dtype == np.float64 and not key.startswith("rays_"):
            value = value.astype(np.float32)
        data[key] = value
    data["outputs"] = np.array(json.dumps(sorted(computed)))
    REFERENCE.parent.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE, **data)
    print(f"wrote {REFERENCE} ({REFERENCE.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit(__doc__.split("::")[-1].strip())
    _write()
