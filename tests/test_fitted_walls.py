"""Fitted walls: the fit (CutFit) against references made from the geometry, apart from the fit.

Every reference here is computed from the model as given: volumes by the divergence theorem in
exact rationals over the given doubles, areas from the given triangles, cell counts from the grid's
shape, and the default path's stencil from the default solver. The fit measures every cell between
its own planes on its integer lattice, whose quantum lies near 2^-58 of the model's size, so its
totals agree with these references far below the tolerances used (1e-12 relative, 1e-11 for a
tessellated sphere).

Units: h is the grid's cell in metres; "cells" are lengths in units of h, "cell faces" areas in h^2.
A cell next to the grid's edge is one that shares a face with an edge cell (a cell of the grid's
outermost shell, whose pressure is always 0).
"""

import functools
import itertools
import math
import re
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from test_wave import (
    _STEPS,
    C,
    _discrete_reflection,
    _drive,
    _solver,
    _stepped_from_the_layout,
)

from acoustic_spaces import Room, Scene, _native
from acoustic_spaces.engine.impedance import Admittance
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.wave import COURANT, source_signal, wave_responses

# The grid's origin is moved off the model's round numbers by these fractions of a cell.
OFFSET = np.array([0.0137, 0.0219, 0.0291])
SIDES = ("-x", "+x", "-y", "+y", "-z", "+z")

# A box's corners are indexed x + 2 y + 4 z; its faces, normals outward.
BOX_FACES = np.array(
    [
        [0, 2, 1], [1, 2, 3],  # -z
        [4, 5, 6], [5, 7, 6],  # +z
        [0, 1, 5], [0, 5, 4],  # -y
        [2, 6, 7], [2, 7, 3],  # +y
        [0, 4, 6], [0, 6, 2],  # -x
        [1, 3, 7], [1, 7, 5],  # +x
    ],
    dtype=np.int64,
)  # fmt: skip
BOX_SIDE = {"-z": [0, 1], "+z": [2, 3], "-y": [4, 5], "+y": [6, 7], "-x": [8, 9], "+x": [10, 11]}


def box(lo, hi):
    """The corners (8, 3) of the box [lo, hi]."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[(hi if (c >> a) & 1 else lo)[a] for a in range(3)] for c in range(8)])


def turned(v, degrees, axis=2, about=None):
    """The points v turned by `degrees` about `axis` through `about` (their mean by default)."""
    about = v.mean(axis=0) if about is None else np.asarray(about, float)
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    a, b = (axis + 1) % 3, (axis + 2) % 3
    out = v - about
    x, y = out[:, a].copy(), out[:, b].copy()
    out[:, a], out[:, b] = c * x - s * y, s * x + c * y
    return out + about


def plate(corners):
    """Two triangles over four corners in order round the quad."""
    v = np.asarray(corners, float)
    return v, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def join(*parts):
    """One mesh from (vertices, faces) parts."""
    vs, fs, n = [], [], 0
    for v, f in parts:
        vs.append(np.asarray(v, float))
        fs.append(np.asarray(f, np.int64) + n)
        n += len(v)
    return np.vstack(vs), np.vstack(fs)


def grid_round(v, h, margin=2):
    """A grid of cell h round the points v with `margin` cells beyond them on every side (the edge
    cell included), its origin moved off round numbers: origin and shape."""
    origin = v.min(axis=0) - h * (margin + OFFSET)
    shape = np.ceil((v.max(axis=0) - origin) / h).astype(int) + margin
    return origin, shape


def fit(
    v, f, src, h, origin, shape, mat=None, materials=1, layer=0, threads=2, raise_refusal=False
):
    mat = np.zeros(len(f), np.int64) if mat is None else np.asarray(mat, np.int64)
    return _native.CutFit(
        np.ascontiguousarray(v, float),
        np.ascontiguousarray(f, np.int64),
        np.ascontiguousarray(mat),
        materials,
        np.asarray(origin, float),
        float(h),
        *(int(n) for n in shape),
        np.asarray(src, float),
        layer,
        threads,
        raise_refusal,
    )


def exact_volume(v, f):
    """|The mesh's signed volume| by the divergence theorem, exact over the given doubles (m^3)."""
    q = [[Fraction(float(x)) for x in p] for p in v]
    six = Fraction(0)
    for a, b, c in f:
        p, r, s = q[a], q[b], q[c]
        six += (
            p[0] * (r[1] * s[2] - r[2] * s[1])
            - p[1] * (r[0] * s[2] - r[2] * s[0])
            + p[2] * (r[0] * s[1] - r[1] * s[0])
        )
    return abs(six) / 6


def areas(v, f):
    """Each triangle's area (m^2)."""
    v = np.asarray(v, float)
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def walls(cut):
    """The rows' wall area per material column (the edge column left out), summed (cell faces)."""
    w = np.asarray(cut.cut_cells()["wall_area"])
    return [math.fsum(w[:, m]) for m in range(w.shape[1] - 1)]


def relative(a, b):
    return abs(float(a) - float(b)) / abs(float(b))


def every_array(cut):
    """Every array of cut_cells() and cut_air(), and every count of cut_stats but the timings."""
    out = {"cells/" + k: np.asarray(x) for k, x in cut.cut_cells().items()}
    out["air"] = np.asarray(cut.cut_air())
    stats = cut.cut_stats
    out.update({"stats/" + k: stats[k] for k in stats if k not in ("seconds", "stage_seconds")})
    return out


def assert_same_bits(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.shape == y.shape and x.tobytes() == y.tobytes(), k


def cell_index(p, shape):
    return int(p[0] + shape[0] * (p[1] + shape[1] * p[2]))


def next_to_edge(shape):
    """Every cell next to an edge cell (depth 1): its index, and its faces to edge cells."""
    nx, ny, nz = (int(n) for n in shape)
    out = {}
    for z in range(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                k = sum(c == 1 for c in (x, y, z)) + sum(
                    c == n - 2 for c, n in ((x, nx), (y, ny), (z, nz))
                )
                if k:
                    out[cell_index((x, y, z), shape)] = k
    return out


# ---- A1 to A2: air and walls -----------------------------------------------------------------

BOX_SIZE = np.array([2.13, 1.77, 1.31])


def a1_room(case):
    h = 0.1
    v = box((0, 0, 0), BOX_SIZE)
    if case == "moved":
        v = v + h * np.array([0.25, 0.5, 0.75])
    elif case == "turned 8":
        v = turned(v, 8)
    elif case == "turned 45":
        v = turned(v, 45)
    elif case == "tilted 8/5":
        v = turned(turned(v, 8), 5, axis=0)
    return v, BOX_FACES, h


@pytest.mark.parametrize("case", ["at 0", "moved", "turned 8", "turned 45", "tilted 8/5"])
def test_the_cut_of_a_box_holds_its_air_and_its_walls(case):
    v, f, h = a1_room(case)
    origin, shape = grid_round(v, h)
    cut = fit(v, f, v.mean(axis=0) + 0.0113, h, origin, shape)
    st = cut.cut_stats
    assert st["refused"] == 0 and st["leaks"] == 0
    assert relative(st["air_volume"], exact_volume(v, f) / Fraction(h) ** 3) <= 1e-12
    assert relative(walls(cut)[0], math.fsum(areas(v, f)) / h**2) <= 1e-12
    assert not cut.open


def test_a_free_slanted_plate_is_cut_with_its_area_on_both_sides():
    h = 0.1
    v, f = plate([(0.3, 0.2, 0.25), (1.1, 0.35, 0.4), (1.05, 0.9, 0.95), (0.25, 0.75, 0.8)])
    lo, hi = np.zeros(3), np.array([1.4, 1.2, 1.2])
    origin = lo - h * OFFSET
    shape = np.ceil((hi - origin) / h).astype(int)
    cut = fit(v, f, (0.15, 0.6, 0.6), h, origin, shape)
    st = cut.cut_stats
    assert st["refused"] == 0 and st["leaks"] == 0
    # The air is the grid's interior: (n - 2)^3 cells, each 1 to within 1e-15 on the lattice.
    assert relative(st["air_volume"], np.prod(shape - 2)) <= 1e-12
    assert relative(walls(cut)[0], 2 * math.fsum(areas(v, f)) / h**2) <= 1e-12


def icosphere(levels, radius, centre):
    t = (1 + 5**0.5) / 2
    p = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
        (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    f = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
        (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
        (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    pts = [np.array(x, float) / np.linalg.norm(x) for x in p]
    for _ in range(levels):
        mid = {}

        def middle(a, b, mid=mid):
            key = (min(a, b), max(a, b))
            if key not in mid:
                m = pts[a] + pts[b]
                pts.append(m / np.linalg.norm(m))
                mid[key] = len(pts) - 1
            return mid[key]

        nf = []
        for a, b, c in f:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(pts) * radius + np.asarray(centre, float), np.array(f, np.int64)


def test_a_tessellated_object_is_cut_exactly():
    h = 0.198
    room = box((0, 0, 0), (2.0, 2.0, 2.0))
    sphere = icosphere(3, 0.5, (1.0, 1.0, 1.0))
    assert len(sphere[1]) == 1280
    v, f = join((room, BOX_FACES), sphere)
    origin, shape = grid_round(room, h)
    cut = fit(v, f, (0.31, 0.27, 0.36), h, origin, shape)
    st = cut.cut_stats
    assert st["refused"] == 0 and st["leaks"] == 0
    air = (exact_volume(room, BOX_FACES) - exact_volume(*sphere)) / Fraction(h) ** 3
    assert relative(st["air_volume"], air) <= 1e-11
    assert st["largest_planes"] > 8


# ---- A3 to A5: threads, order of faces, whole-cell moves ----------------------------------------


def l_room():
    """An L-shaped room 1.2 x 0.9 m in plan, 0.6 m high, with two free plates turned 22.5 and 11
    degrees (each about its own centre, about z and then x), and its grid at h = 0.1 m."""
    h = 0.1
    plan = np.array([(0, 0), (1.2, 0), (1.2, 0.5), (0.6, 0.5), (0.6, 0.9), (0, 0.9)])
    n = len(plan)
    v = np.array([(x, y, z) for z in (0.0, 0.6) for x, y in plan])
    f = [(0, k + 1, k) for k in range(1, n - 1)]  # floor, normal -z (a fan from corner 0)
    f += [(n, n + k, n + k + 1) for k in range(1, n - 1)]  # ceiling, normal +z
    for k in range(n):
        a, b = k, (k + 1) % n
        f += [(a, b, n + b), (a, n + b, n + a)]
    room = (v, np.array(f, np.int64))
    p1 = plate([(0.2, 0.2, 0.15), (0.5, 0.2, 0.15), (0.5, 0.2, 0.45), (0.2, 0.2, 0.45)])
    p2 = plate([(0.8, 0.1, 0.1), (1.05, 0.1, 0.1), (1.05, 0.1, 0.4), (0.8, 0.1, 0.4)])
    p1 = (turned(turned(p1[0], 22.5), 11, axis=0), p1[1])
    p2 = (turned(turned(p2[0], 11), 22.5, axis=0), p2[1])
    v, f = join(room, p1, p2)
    mat = np.r_[np.zeros(len(room[1])), [1, 1, 2, 2]].astype(np.int64)
    origin, shape = grid_round(room[0], h)
    return v, f, mat, h, origin, shape, np.array([0.31, 0.62, 0.33])


def test_the_cut_is_the_same_on_any_number_of_threads():
    v, f, mat, h, origin, shape, src = l_room()
    one = fit(v, f, src, h, origin, shape, mat, 3, threads=1)
    three = fit(v, f, src, h, origin, shape, mat, 3, threads=3)
    assert one.cut_stats["refused"] == 0
    assert_same_bits(every_array(one), every_array(three))


def test_the_cut_does_not_depend_on_the_order_of_faces():
    # Faces reversed or shuffled: every array to the bit. Each face's corners rotated as well: every
    # decision to the bit (which cells are air, which are rows), and the measures within 1e-15
    # relative, since a face's area and its parts are measured from its corners in their given order
    # and may differ in the last bit.
    v, f, mat, h, origin, shape, src = l_room()
    base = fit(v, f, src, h, origin, shape, mat, 3)
    assert base.cut_stats["refused"] == 0
    a = base.cut_cells()
    order = np.random.default_rng(7).permutation(len(f))
    for faces, mats, rotated in (
        (f[::-1], mat[::-1], False),
        (f[order], mat[order], False),
        (f[:, [1, 2, 0]], mat, True),
        (f[order][:, [2, 0, 1]], mat[order], True),
    ):
        other = fit(v, faces, src, h, origin, shape, mats, 3)
        b = other.cut_cells()
        assert np.array_equal(base.cut_air(), other.cut_air())
        assert np.array_equal(a["cell"], b["cell"])
        for k in a:
            x, y = np.asarray(a[k]), np.asarray(b[k])
            if rotated:
                assert x.shape == y.shape and np.all(
                    np.abs(x - y) <= 1e-15 * np.maximum(1.0, np.abs(x))
                ), k
            else:
                assert x.tobytes() == y.tobytes(), k


@pytest.mark.parametrize("cells", [3, 7])
def test_the_cut_moves_with_the_grid_by_whole_cells(cells):
    v, f, mat, h, origin, shape, src = l_room()
    base = fit(v, f, src, h, origin, shape, mat, 3)
    moved = fit(v, f, src, h, origin - cells * h, shape + cells, mat, 3)
    a, b = base.cut_cells(), moved.cut_cells()

    def deep(rows, shp, shift):
        """Rows of cells at depth 2 or more in the first grid, by their index in it."""
        nx, ny, _ = shp
        c = np.asarray(rows["cell"])
        p = np.column_stack([c % nx, c // nx % ny, c // (nx * ny)]) - shift
        keep = np.all(p >= 2, axis=1) & np.all(p <= shape - 3, axis=1)
        return {cell_index(q, shape): k for k, q in zip(np.flatnonzero(keep), p[keep], strict=True)}

    da, db = deep(a, shape, 0), deep(b, shape + cells, cells)
    assert da.keys() == db.keys() and len(da) > 100
    ia, ib = list(da.values()), [db[c] for c in da]
    for key in ("air_volume", "open_area", "wall_area", "sliver"):
        x, y = np.asarray(a[key])[ia], np.asarray(b[key])[ib]
        assert np.max(np.abs(x - y)) <= 1e-12, key


# ---- A6: coincident faces ------------------------------------------------------------------------


def test_coincident_faces_count_once_with_each_side_s_material():
    # A free screen built twice with the same normals, material 1 given first, then material 0: the
    # face of least key (equal corners and orientation, then the lower material) gives both sides.
    h = 0.1
    v, f = plate([(0.32, 0.28, 0.31), (0.93, 0.28, 0.31), (0.93, 0.28, 0.87), (0.32, 0.28, 0.87)])
    v = turned(v, 8)
    faces = np.vstack([f, f])
    mat = np.array([1, 1, 0, 0], np.int64)
    origin = -h * OFFSET
    shape = np.ceil((np.array([1.2, 1.2, 1.2]) - origin) / h).astype(int)
    cut = fit(v, faces, (0.15, 0.6, 0.6), h, origin, shape, mat, 2)
    assert cut.cut_stats["refused"] == 0
    w = walls(cut)
    assert relative(w[0], 2 * math.fsum(areas(v, f)) / h**2) <= 1e-12
    assert w[1] == 0.0


# ---- A7 to A9: T-junctions, thin layers, a patch on a wall ---------------------------------------


def t_room(offset, degrees):
    """A box room [0.125, 0.875] x [0.125, 0.625] x [0.125, 0.625] m (dyadic, so exact on the
    lattice) whose -y face is three triangles: the lower half of its rectangle, and two triangles
    meeting at a T vertex on the middle of its diagonal, moved `offset` m off the diagonal towards
    the corner (x0, z1), opening a slit as wide as that; the room turned `degrees` about z."""
    v = box((0.125, 0.125, 0.125), (0.875, 0.625, 0.625))
    n = np.array([-0.5, 0.75]) / math.hypot(0.5, 0.75)
    t = np.array([0.5 + offset * n[0], 0.125, 0.375 + offset * n[1]])
    v = np.vstack([v, t])
    f = [x for k, x in enumerate(BOX_FACES.tolist()) if k not in BOX_SIDE["-y"]]
    # The -y side's corners: 0 (x0, z0), 1 (x1, z0), 5 (x1, z1), 4 (x0, z1); normals -y.
    f += [[0, 1, 5], [0, 8, 4], [8, 5, 4]]
    v = turned(v, degrees, about=(0.5, 0.375, 0.375))
    return v, np.array(f, np.int64)


@pytest.mark.parametrize(
    "offset, degrees, expect",
    [
        (0.0, 0, "fitted"),
        (0.0, 8, "thin"),
        (1e-6, 0, "refused"),
        (1e-3, 0, "refused"),
        (1e-3, 8, "refused"),
        (5e-3, 0, "open"),
        (5e-3, 8, "open"),
    ],
)
def test_a_t_junction_is_fitted_when_exact_and_refused_when_opened(offset, degrees, expect):
    h = 0.1
    v, f = t_room(offset, degrees)
    origin, shape = grid_round(v, h)
    cut = fit(v, f, (0.47, 0.39, 0.42), h, origin, shape)
    st = cut.cut_stats
    if expect == "thin":
        # Turned, the T vertex (its coordinates rounded to doubles) lies just across the diagonal
        # of the face's lower triangle (face 10), so the T's triangle beside it (face 11) overlaps
        # that triangle in their projection along y, within 7/4 kGap of it, with the room's air on
        # the sides that face each other.
        assert st["refused"] == 5
        assert re.match(r"fitted walls: faces 10 and 11 are nearly parallel", st["refusal"])
        return
    if expect == "refused":
        assert st["refused"] == 6
        assert st["refusal"].startswith("fitted walls: the air in the cell centred at")
        assert "Weld the faces along it" in st["refusal"]
        with pytest.raises(ValueError, match="^fitted walls: the air"):
            fit(v, f, (0.47, 0.39, 0.42), h, origin, shape, raise_refusal=True)
        return
    assert st["refused"] == 0
    if expect == "fitted":
        assert st["boundary_intervals"] == 0 and not cut.open
        assert relative(st["air_volume"], exact_volume(v, f) / Fraction(h) ** 3) <= 1e-12
    else:
        assert st["boundary_intervals"] > 0 and cut.open


def test_a_thin_layer_open_to_the_air_is_refused():
    h = 0.1
    room = box((0, 0, 0), (1.0, 0.8, 0.6))
    # A patch 1e-4 m above the floor, inside the room.
    patch = (
        np.array([(0.3, 0.3, 1e-4), (0.6, 0.3, 1e-4), (0.45, 0.55, 1e-4)]),
        np.array([[0, 1, 2]]),
    )
    # A box standing 1e-5 m above a slab and overhanging its edge, both inside the room, turned 8
    # degrees: the box's floor and the slab's top overlap 1e-4 cells apart, with air between them.
    slab = (box((0.2, 0.2, 0.1), (0.8, 0.6, 0.2)), BOX_FACES)
    block = (box((0.6, 0.3, 0.2 + 1e-5), (0.9, 0.5, 0.4)), BOX_FACES)
    cases = [
        (join((room, BOX_FACES), patch), {12}, BOX_SIDE["-z"]),
        (
            join((room, BOX_FACES), (turned(slab[0], 8, about=(0.5, 0.4, 0.3)), BOX_FACES),
                 (turned(block[0], 8, about=(0.5, 0.4, 0.3)), BOX_FACES)),
            {24 + k for k in BOX_SIDE["-z"]},
            [12 + k for k in BOX_SIDE["+z"]],
        ),
    ]  # fmt: skip
    for (v, f), first, second in cases:
        origin, shape = grid_round(room, h)
        cut = fit(v, f, (0.9, 0.7, 0.5), h, origin, shape)
        st = cut.cut_stats
        assert st["refused"] == 5
        text = st["refusal"]
        named = re.search(r"^fitted walls: faces (\d+) and (\d+) are nearly parallel", text)
        assert named, text
        pair = {int(named.group(1)), int(named.group(2))}
        assert pair & first and pair & set(second), text
        overlap = re.search(r"overlap over ([0-9.e+-]+) m\^2", text)
        assert overlap and float(overlap.group(1)) > 0, text


def patch_on_wall(on_plane):
    """A room with a triangle laid on its -y wall, the wall off or on a grid plane."""
    h = 0.198
    origin = np.array([0.0291, -0.0137, 0.0173])
    shape = np.array([20, 18, 16])
    y0 = origin[1] + (3 if on_plane else 3.1) * h
    v = box((0, 0, 0), (2.37, 1.71, 1.43)) + np.array(
        [origin[0] + 3.3 * h, y0, origin[2] + 2.9 * h]
    )
    x0, z0 = v[0, 0], v[0, 2]
    tri = np.array([(x0 + 0.3, y0, z0 + 0.2), (x0 + 1.1, y0, z0 + 0.3), (x0 + 0.5, y0, z0 + 0.9)])
    V = np.vstack([v, tri])
    F = np.vstack([BOX_FACES, [[8, 9, 10]]])
    mat = np.r_[np.arange(12) % 3, [1]]
    return V, F, mat, v, h, origin, shape


def test_a_patch_on_a_wall_on_a_grid_plane_keeps_the_wall_balance():
    overlaps = []
    for on_plane in (False, True):
        V, F, mat, room, h, origin, shape = patch_on_wall(on_plane)
        cut = fit(V, F, room.mean(axis=0) + 0.0113, h, origin, shape, mat, 3)
        st = cut.cut_stats
        assert st["refused"] == 0 and st["balance"] <= 1e-9
        assert st["membrane_faces"] == (3 if on_plane else 0)
        # The patch lies on the wall, so the walls are the room's area alone.
        assert relative(math.fsum(walls(cut)), math.fsum(areas(room, BOX_FACES)) / h**2) <= 1e-12
        overlaps.append(st["overlap_area"])
    assert overlaps[0] > 0 and relative(overlaps[1], overlaps[0]) <= 1e-12


# ---- A10, A11: the source's and the layer's refusals ----------------------------------------


def open_grid(h=0.1, size=1.2):
    origin = -h * OFFSET
    shape = np.ceil((np.full(3, size) - origin) / h).astype(int)
    return origin, shape


def test_the_source_refusals_say_why():
    h = 0.1
    origin, shape = open_grid(h)
    centre = lambda i: origin + (np.asarray(i) + 0.5) * h  # noqa: E731
    # A closed pocket 0.4 cells wide holding no cell centre, the source inside it.
    lo = centre((5, 5, 5)) + 0.05 * h
    pocket = (box(lo, lo + 0.4 * h), BOX_FACES)
    cut = fit(*pocket, lo + 0.2 * h, h, origin, shape)
    assert cut.cut_stats["refused"] == 7
    assert cut.cut_stats["refusal"].startswith(
        "fitted walls: the source lies in a closed space of the model that holds no cell centre"
    )
    room = (box((0.2, 0.2, 0.2), (1.0, 1.0, 1.0)), BOX_FACES)
    # The source outside the grid's interior.
    cut = fit(*room, (-0.5, 0.5, 0.5), h, origin, shape)
    assert cut.cut_stats["refused"] == 3
    assert cut.cut_stats["refusal"].startswith(
        "fitted walls: the source lies outside the grid's interior"
    )
    # The source within kGap of a free edge (a zone of the mesh's boundary).
    screen = plate([(0.3, 0.5, 0.3), (0.9, 0.5, 0.3), (0.9, 0.5, 0.9), (0.3, 0.5, 0.9)])
    cut = fit(*screen, (0.6, 0.5 + 1e-5, 0.9 + 1e-5), h, origin, shape)
    assert cut.cut_stats["refused"] == 4
    assert cut.cut_stats["refusal"].startswith("fitted walls: the source lies within kGap")
    # The source in a slot narrower than a cell between two plates, no centre reachable from it.
    x = centre((5, 5, 5))[0] + 0.1 * h
    d = 0.3 * h
    slot = join(
        plate([(x, 0.25, 0.25), (x, 0.95, 0.25), (x, 0.95, 0.95), (x, 0.25, 0.95)]),
        plate([(x + d, 0.25, 0.25), (x + d, 0.95, 0.25), (x + d, 0.95, 0.95), (x + d, 0.25, 0.95)]),
    )  # fmt: skip
    src = np.array([x + d / 2, 0.6, 0.6])
    cut = fit(*slot, src, h, origin, shape)
    assert cut.cut_stats["refused"] == 0
    with pytest.raises(
        ValueError, match="^fitted walls: no cell centre round the source can be reached"
    ):
        cut.cut_points(src.reshape(1, 3), np.zeros((0, 3)))
    # The source within two cells of the grid's edge.
    src = origin + np.array([1.3, 6.5, 6.5]) * h
    cut = fit(*screen, src, h, origin, shape)
    assert cut.cut_stats["refused"] == 0
    with pytest.raises(
        ValueError, match="^fitted walls: the source lies within [0-9.]+ cells of the grid's edge"
    ):
        cut.cut_points(src.reshape(1, 3), np.zeros((0, 3)))
    # The source next to the matched layer (4 cells): it would feed a layer cell.
    origin, shape = open_grid(h, 2.0)
    inside = plate([(0.8, 1.0, 0.8), (1.2, 1.0, 0.8), (1.2, 1.0, 1.2), (0.8, 1.0, 1.2)])
    src = origin + np.array([5.2, 7.5, 7.5]) * h
    cut = fit(*inside, src, h, origin, shape, layer=4)
    assert cut.cut_stats["refused"] == 0
    with pytest.raises(
        ValueError, match="^fitted walls: the source is in the matched layer: the cell centred at"
    ):
        cut.cut_points(src.reshape(1, 3), np.zeros((0, 3)))


def test_fitted_walls_refuse_a_model_in_or_next_to_the_matched_layer():
    h = 0.1
    origin, shape = open_grid(h, 1.6)
    # A solid box whose corner pokes 0.3 cells into a cube of the 4-cell layer (cells 1 to 4).
    corner = origin + (4.7 + np.array([0.0, 0.1, 0.2])) * h
    v = box(corner, corner + 0.5)
    cut = fit(v, BOX_FACES, (1.05, 1.05, 1.05), h, origin, shape, layer=4)
    st = cut.cut_stats
    assert st["refused"] == 8 and not cut.open
    assert st["refusal"].startswith("fitted walls: the model reaches into the matched layer")
    assert re.search(
        r"faces of it cut the air the source reaches in the cell centred at "
        r"\([-0-9.]+, [-0-9.]+, [-0-9.]+\) m",
        st["refusal"],
    )
    assert st["refusal"].endswith(
        "Move the model away from the grid's edge, or enlarge the margin round it."
    )
    # A solid box in the open whose -x face lies 0.5 cells into the cubes next to the layer (index
    # 5), then exactly in the plane between them and the next ones inward: either way those cubes
    # hold faces of the model and the air round it, and the layer, which reads its neighbours whole,
    # would meet a cut cell (code 12).
    origin, shape = open_grid(h, 2.0)
    src = origin + np.array([13.0, 9.5, 9.5]) * h
    for x in (5.5, 6.0):
        lo = origin + np.array([x, 7.2, 7.3]) * h
        cut = fit(
            box(lo, origin + np.array([10.3, 12.2, 12.3]) * h),
            BOX_FACES,
            src,
            h,
            origin,
            shape,
            layer=4,
        )
        st = cut.cut_stats
        assert st["refused"] == 12 and not cut.open, x
        assert st["refusal"].startswith(
            "fitted walls: faces of the model lie in the cell next to the matched layer centred "
            "at ("
        )
        assert re.search(
            r"centred at \([-0-9.]+, [-0-9.]+, [-0-9.]+\) m, which holds air the source "
            r"reaches",
            st["refusal"],
        )
        assert st["refusal"].endswith(
            "Move the model away from the grid's edge, or enlarge the margin round it."
        )
    # One cell further in, the box is clear of them and is fitted.
    cut = fit(
        box(origin + np.array([7.0, 7.2, 7.3]) * h, origin + np.array([10.3, 12.2, 12.3]) * h),
        BOX_FACES,
        src,
        h,
        origin,
        shape,
        layer=4,
    )
    assert cut.cut_stats["refused"] == 0


# ---- A12: the fitted points take the solver's stencil --------------------------------------


def test_the_fitted_points_take_the_solver_s_stencil():
    h = 0.1
    size = (1.13, 0.97, 0.81)
    src = np.array([0.41, 0.37, 0.33])
    rng = np.random.default_rng(11)
    probes = np.array([0.73, 0.52, 0.47]) + rng.uniform(-0.2, 0.2, (13, 3))
    solver, _ = _solver(
        size, lambda m: np.full(len(m.faces), 0.3), src, probes, cell=h, probes_per_listener=13
    )
    layout = solver.layout()
    start = np.asarray(layout["probe_start"])
    cells, weights = np.asarray(layout["probe_cells"]), np.asarray(layout["probe_weights"])
    v = box((0, 0, 0), size)
    origin = -h * (2.0 + OFFSET)
    shape = layout["shape"]
    cut = fit(v, BOX_FACES, src, h, origin, shape)
    pts = cut.cut_points(src.reshape(1, 3), probes, 13)
    for k in range(13):
        want = dict(
            zip(
                cells[start[k] : start[k + 1]].tolist(),
                weights[start[k] : start[k + 1]],
                strict=True,
            )
        )
        got = dict(pts["probes"][k])
        assert want.keys() == got.keys(), k
        assert max(abs(want[c] - got[c]) for c in want) <= 1e-12, k


def test_a_point_on_a_fitted_screen_or_next_to_a_far_centre_reads_its_own_side():
    h = 0.1
    origin, shape = open_grid(h)
    i = np.array([5, 6, 6])
    c = origin + (i + 0.5) * h
    # A screen through the centre of cell i, normal x; a point exactly there reads the + side.
    x = c[0]
    screen = plate([(x, 0.25, 0.25), (x, 0.95, 0.25), (x, 0.95, 0.95), (x, 0.25, 0.95)])
    cut = fit(*screen, c + np.array([-3.3, 0.21, 0.17]) * h, h, origin, shape)
    pts = cut.cut_points(np.array([c + np.array([-3.3, 0.21, 0.17]) * h]), c.reshape(1, 3))
    got = pts["probes"][0]
    assert got and all(cell % shape[0] >= i[0] for cell, _ in got)
    assert abs(math.fsum(w for _, w in got) - 1.0) <= 1e-12
    assert dict(got).get(cell_index(i, shape), 0.0) == pytest.approx(1.0, abs=1e-12)
    # A screen two cells square, 1e-12 cells below the centre of cell i, and a point 5e-10 cells
    # below that centre (and 5e-13 cells past it on y and z): nearly all of the point's weight lies
    # on cells it reaches only across the screen. The default path reads no cell there (the weights
    # of the cells it sees sum to under 1e-9); the fit reads the cells it reaches without crossing
    # the screen, its weights renormalised, cell i not among them.
    xs = x - 1e-12 * h
    p = c + np.array([-5e-10, 5e-13, 5e-13]) * h
    screen = plate([(xs, 0.5, 0.5), (xs, 0.7, 0.5), (xs, 0.7, 0.7), (xs, 0.5, 0.7)])
    src = c + np.array([-3.3, 0.21, 0.17]) * h
    cut = fit(*screen, src, h, origin, shape)
    got = cut.cut_points(src.reshape(1, 3), p.reshape(1, 3))["probes"][0]
    assert got and cell_index(i, shape) not in dict(got)
    assert abs(math.fsum(w for _, w in got) - 1.0) <= 1e-12
    solver = _native.WaveSolver(
        _native.Bvh(*screen), np.zeros(2, np.int64), np.zeros(1), np.zeros((0, 3)),
        np.zeros(0, np.int64),
        origin, h, *(int(n) for n in shape), src, p.reshape(1, 3), 1.0, 0, 0.0, 1,
    )  # fmt: skip
    assert not solver.probe_found(0)


# ---- A13, A14: far from the origin; a centre on a wall ------------------------------------------


def test_a_cut_far_from_the_origin_is_exact():
    h = 0.01
    for far, refused in ((1e7, 0), (2e13, 2)):
        v = box((0, 0, 0), (0.23, 0.19, 0.29)) + far
        origin, shape = grid_round(v, h)
        cut = fit(v, BOX_FACES, v.mean(axis=0) + 0.00113, h, origin, shape)
        st = cut.cut_stats
        assert st["refused"] == refused
        if refused:
            assert st["refusal"].startswith("fitted walls: the grid lies too far from the origin")
        else:
            assert (
                relative(st["air_volume"], exact_volume(v, BOX_FACES) / Fraction(h) ** 3) <= 1e-12
            )


def test_a_cell_centre_on_a_fitted_wall_lies_on_its_side():
    h = 0.1
    origin, shape = open_grid(h)
    i = 3
    xc = origin[0] + (i + 0.5) * h

    def at(dx):
        v = box((xc + dx * h, 0.25, 0.25), (0.95, 0.95, 0.95))
        cut = fit(v, BOX_FACES, (0.6, 0.6, 0.6), h, origin, shape)
        assert cut.cut_stats["refused"] == 0
        return cut

    on = at(0.0)
    air = np.asarray(on.cut_air()).reshape(shape[::-1])
    assert air[6, 6, i] == 1  # the wall's + side (the room) holds the centre
    assert np.array_equal(on.cut_air(), at(-1e-7).cut_air())
    for dx in (5e-11, 1e-9, -5e-11, -1e-9):
        here, there = at(dx), at(math.copysign(1e-7, dx))
        assert np.array_equal(here.cut_air(), there.cut_air()), dx
        assert np.array_equal(here.cut_cells()["cell"], there.cut_cells()["cell"]), dx


# ---- A15: the rows next to the grid's edge, and the open sides -------------------------------


def edge_case(name):
    """The meshes of A15, each with its grid (h, origin, shape) and source."""
    if name in ("open-topped", "layered"):
        # An open-topped box well inside a grid with room above it: every cell next to the edge
        # is air, so every side is open.
        h = 0.2
        origin = np.array([0.0137, 0.0219, 0.0291])
        shape = np.array([16, 16, 20])
        v = box((0.6, 0.6, 0.5), (2.4, 2.4, 2.0))
        f = np.array([x for k, x in enumerate(BOX_FACES.tolist()) if k not in BOX_SIDE["+z"]])
        src = np.array([1.53, 1.47, 1.21])
        if name == "layered":
            # One cell further in: the box's low sides cross the cubes at index 2, next to a layer
            # of one cell, which the fit refuses (code 12); at index 3 they are clear of them.
            v, src = v + h, src + h
        return v, f, h, origin, shape, src
    h = 0.125
    origin = np.zeros(3)
    shape = np.array([10, 10, 10])
    c = lambda lo, hi: box(np.array(lo) * h, np.array(hi) * h)  # noqa: E731  (in cells)
    if name == "chimney":
        # A box room whose walls pass through the +z edge: open on +z only.
        v = c((1.75, 1.75, 1.75), (8.25, 8.25, 11.0))
        f = np.array([x for k, x in enumerate(BOX_FACES.tolist()) if k not in BOX_SIDE["+z"]])
        return v, f, h, origin, shape, np.array([4.3, 4.6, 4.9]) * h
    if name == "closed":
        # A closed box room inside a second closed box 0.5 cells larger: the air between them holds
        # the centres next to the edge, and the air outside the second box (0.25 cells deep, holding
        # no centre) is a pocket open only to the grid's edge. Neither is the source's.
        return (
            *join((c((1.75,) * 3, (8.25,) * 3), BOX_FACES),
                  (c((1.25,) * 3, (8.75,) * 3), BOX_FACES)),
            h, origin, shape, np.array([4.3, 4.6, 4.9]) * h,
        )  # fmt: skip
    # "slab": a room open through the -x edge, a solid slab inside it covering every centre of the
    # cells next to that edge, 0.25 cells from the edge plane: the source's air reaches the -x edge
    # plane only in cubes whose own centres are solid.
    room = c((-1.0, 1.75, 1.75), (8.25, 8.25, 8.25))
    f = np.array([x for k, x in enumerate(BOX_FACES.tolist()) if k not in BOX_SIDE["-x"]])
    return (*join((room, f), (c((1.25, 2.125, 2.125), (1.875, 7.875, 7.875)), BOX_FACES)),
            h, origin, shape, np.array([4.3, 4.6, 4.9]) * h)  # fmt: skip


@pytest.mark.parametrize("name", ["open-topped", "chimney", "closed", "slab", "layered"])
def test_every_air_cell_next_to_the_grid_s_edge_meets_it(name):
    v, f, h, origin, shape, src = edge_case(name)
    layered = name == "layered"
    cut = fit(v, f, src, h, origin, shape, layer=int(layered))
    st = cut.cut_stats
    assert st["refused"] == 0
    if layered:
        # A layer of one cell: every cell next to the edge is a layer cell, none a row; layer cells
        # lie within the layer of every side, so every side is open.
        assert st["layer_cells"] == len(next_to_edge(shape)) == 1224
        assert st["edge_area"] == 0.0 and not set(next_to_edge(shape)) & set(
            cut.cut_cells()["cell"].tolist()
        )
        assert [bool(s) for s in cut.open_sides] == [True] * 6 and cut.open
        return
    rows = cut.cut_cells()
    cells = np.asarray(rows["cell"])
    edge_col = np.asarray(rows["wall_area"])[:, -1]
    shell = next_to_edge(shape)
    nx, ny, nz = (int(n) for n in shape)
    if name == "open-topped":
        # Every cell next to the edge is air (the box lies 1.35 cells or more inside the edge
        # planes, so it crosses no cube next to the edge).
        expect = shell
        sides = [True] * 6
    elif name == "chimney":
        # The room's air at the top layer of cells (k = 8) for i, j in 2..7.
        expect = {cell_index((i, j, nz - 2), shape): 1 for i in range(2, 8) for j in range(2, 8)}
        sides = [False, False, False, False, False, True]
    else:
        expect = {}
        sides = [False] * 6 if name == "closed" else [True] + [False] * 5
    air = np.asarray(cut.cut_air())
    assert {c for c in shell if air[c] == 1} == expect.keys()
    assert expect.keys() <= set(cells.tolist())
    faces = sum(expect.values())
    if name == "open-topped":
        assert (len(expect), faces) == (1224, 1400)
    # The open area towards the edge: in the open-topped case every face of a cell next to the
    # edge to an edge cell, whole (w_b w_c, 1 to within 4e-15 on the lattice); in the chimney and
    # the slab the room's opening on the edge plane, 6.5 x 6.5 cell faces, part of it in cubes whose
    # centres are solid and whose air a row owns.
    opening = {"open-topped": faces, "chimney": 6.5**2, "slab": 6.5**2, "closed": 0.0}[name]
    if opening:
        assert relative(st["edge_area"], opening) <= 1e-12
    else:
        assert st["edge_area"] == 0.0
    assert math.fsum(edge_col) == pytest.approx(st["edge_area"], rel=1e-12, abs=0.0)
    assert [bool(s) for s in cut.open_sides] == sides, dict(zip(SIDES, cut.open_sides, strict=True))
    assert cut.open == any(sides)
    if name == "closed":
        assert st["edge_pockets"] == 1 and st["other_components"] == 1
    if name == "open-topped":
        # Each row next to the edge holds in its edge column its faces to edge cells, whole: w_b w_c
        # of its cell's lattice widths across each face's axis, added in the order -x, +x, -y, +y,
        # -z, +z. The widths here differ from 1, and from each other, by up to 4e-15, so a face
        # counted across the wrong axis changes the sum. The only cut air that rows next to the edge
        # own is the air under the box's floor (which crosses the cubes at z index 2 below their
        # centres), owned by the rows below it next to the -z edge: every other row next to the
        # edge owns nothing cut and holds its own cell's volume, wx wy wz, and no wall.
        wx, wy, wz = cut.cut_widths()
        wid = (wx, wy, wz)
        volume = np.asarray(rows["air_volume"])
        wall = np.asarray(rows["wall_area"])[:, :-1]
        row = {c: k for k, c in enumerate(cells.tolist())}
        alone = 0
        for c in expect:
            p = (c % nx, c // nx % ny, c // (nx * ny))
            whole = 0.0
            for a in range(3):
                b, d = (a + 1) % 3, (a + 2) % 3
                for at_edge in (p[a] == 1, p[a] == shape[a] - 2):
                    if at_edge:
                        whole += wid[b][p[b]] * wid[d][p[d]]
            assert edge_col[row[c]] == whole, p
            if p[2] != 1:
                assert volume[row[c]] == wx[p[0]] * wy[p[1]] * wz[p[2]], p
                assert not wall[row[c]].any(), p
                alone += 1
        assert alone == len(expect) - 14 * 14
    if name in ("chimney", "closed", "slab"):
        # The room's air cells have indices 2 to 7 (to 8 along z in the chimney). Every cube next
        # to one of them with an index 1 or 8 along an axis is crossed by a wall (the slab's at
        # x index 1), except at index 8 along z in the chimney, whose top is open; the room's air
        # in each crossed cube holds no centre, and no other cube next to them is crossed. So the
        # rows are the air cells next to the edge and the air cells next to a crossed cube, each
        # owning the air between the wall and its own cube; every other air cell owns nothing cut,
        # does not lie next to the edge, and is no row.
        walled = [(1, 8), (1, 8), (1,) if name == "chimney" else (1, 8)]
        expect_rows = set(expect)
        for c in np.flatnonzero(air == 1).tolist():
            p = (c % nx, c // nx % ny, c // (nx * ny))
            if any(p[a] + d in walled[a] for a in range(3) for d in (-1, 1)):
                expect_rows.add(c)
        assert set(cells.tolist()) == expect_rows
        assert len(expect_rows) == {"chimney": 172, "closed": 152, "slab": 152}[name]


def tube_room(cap, side="-x"):
    """A closed room joined to the grid's edge on `side` by a square tube 0.8 cells wide that holds
    no cell centre, the tube closed by a cap at `cap` cells from the grid's low corner along the
    side's axis, counted outwards (the edge plane lies at 1): vertices, faces (normals out of the
    air), h, origin, shape and source. Written for -x, in cells from the grid's origin: the room
    spans x 2.2..10, y 2.6..9.4, z 2.6..10.4, and the tube y 4.6..5.4, z 6.6..7.4, through a hole in
    the room's -x wall. For another side the cell coordinates (x, y, z) go to the axes (a, a + 1,
    a + 2) of the side's axis a, x mirrored to the grid's length minus x for a high side (whose
    edge plane lies at that length minus 1), and the faces reversed with the mirror."""
    h = 0.1
    origin = -h * OFFSET
    s = SIDES.index(side)
    a, high = s // 2, s % 2 == 1
    axes = (a, (a + 1) % 3, (a + 2) % 3)
    shape = np.zeros(3, dtype=int)
    shape[list(axes)] = (14, 12, 13)

    def p(x, y, z):  # a point given in the cells of the -x case
        c = np.zeros(3)
        c[list(axes)] = (shape[a] - x if high else x, y, z)
        return origin + h * c

    room = (
        box(p(2.2, 2.6, 2.6), p(10, 9.4, 10.4)),
        [x for k, x in enumerate(BOX_FACES.tolist()) if k not in BOX_SIDE[SIDES[2 * a]]],
    )
    # The room's -x wall round the hole, in four rectangles (y0, z0, y1, z1), normals -x.
    rects = [
        (2.6, 2.6, 4.6, 10.4),
        (5.4, 2.6, 9.4, 10.4),
        (4.6, 2.6, 5.4, 6.6),
        (4.6, 7.4, 5.4, 10.4),
    ]
    wall = [
        plate([p(2.2, y0, z0), p(2.2, y0, z1), p(2.2, y1, z1), p(2.2, y1, z0)])
        for y0, z0, y1, z1 in rects
    ]
    c, x = cap, 2.2
    tube = [
        plate([p(c, 4.6, 6.6), p(x, 4.6, 6.6), p(x, 4.6, 7.4), p(c, 4.6, 7.4)]),  # -y
        plate([p(c, 5.4, 6.6), p(c, 5.4, 7.4), p(x, 5.4, 7.4), p(x, 5.4, 6.6)]),  # +y
        plate([p(c, 4.6, 6.6), p(c, 5.4, 6.6), p(x, 5.4, 6.6), p(x, 4.6, 6.6)]),  # -z
        plate([p(c, 4.6, 7.4), p(x, 4.6, 7.4), p(x, 5.4, 7.4), p(c, 5.4, 7.4)]),  # +z
        plate([p(c, 4.6, 6.6), p(c, 4.6, 7.4), p(c, 5.4, 7.4), p(c, 5.4, 6.6)]),  # the cap, -x
    ]
    v, f = join(room, *wall, *tube)
    if high:
        f = f[:, ::-1].copy()
    return v, f, h, origin, shape, p(6.3, 5.7, 6.2)


@pytest.mark.parametrize("cap", [1.0, -0.5], ids=["capped in the edge plane", "open"])
@pytest.mark.parametrize("side", SIDES)
def test_a_wall_in_the_edge_plane_closes_air_without_centres_there(side, cap):
    # The room meets the side's edge plane only through the tube, in the cubes next to the edge
    # cells, whose centres are solid. Capped exactly in the edge plane, the cap is a wall there: a
    # closed piece, which opens no side. Capped beyond the grid, the tube's cross-section on the
    # edge plane is open air of the source's, and that side alone is open.
    v, f, h, origin, shape, src = tube_room(cap, side)
    cut = fit(v, f, src, h, origin, shape)
    st = cut.cut_stats
    assert st["refused"] == 0 and st["leaks"] == 0
    air = np.asarray(cut.cut_air())
    assert not [c for c in next_to_edge(shape) if air[c] == 1]
    # Inside the grid the air is the room and the tube up to the edge plane in both cases: the
    # volume of the mesh capped in that plane.
    v1, f1, *_ = tube_room(1.0, side)
    assert relative(st["air_volume"], exact_volume(v1, f1) / Fraction(h) ** 3) <= 1e-12
    edge = np.asarray(cut.cut_cells()["wall_area"])[:, -1]
    cells = np.asarray(cut.cut_cells()["cell"])
    s = SIDES.index(side)
    if cap == 1.0:
        assert [bool(x) for x in cut.open_sides] == [False] * 6 and not cut.open
        assert st["edge_area"] == 0.0 and not edge.any()
        return
    assert [bool(x) for x in cut.open_sides] == [k == s for k in range(6)] and cut.open
    # The grid planes at 5 and 7 cells across the tube split its cross-section on the edge plane
    # into four squares 0.4 cells wide, each in a cube next to an edge cell whose air the cell
    # beyond it (index 2 from the edge) owns: each square is in that row's edge column.
    a, b, c = s // 2, (s // 2 + 1) % 3, (s // 2 + 2) % 3
    beyond = shape[a] - 3 if s % 2 else 2
    y = [origin[b] + h * t for t in (4.6, 5.0, 5.4)]
    z = [origin[c] + h * t for t in (6.6, 7.0, 7.4)]

    def at(j, k):
        q = [0, 0, 0]
        q[a], q[b], q[c] = beyond, j, k
        return cell_index(q, shape)

    held = {cells[r]: edge[r] for r in np.flatnonzero(edge)}
    assert sorted(held) == sorted(at(j, k) for j in (4, 5) for k in (6, 7))
    for j in (4, 5):
        for k in (6, 7):
            square = (y[j - 3] - y[j - 4]) * (z[k - 5] - z[k - 6]) / h**2
            assert relative(held[at(j, k)], square) <= 1e-12
    opening = (y[2] - y[0]) * (z[2] - z[0]) / h**2
    assert relative(st["edge_area"], opening) <= 1e-12
    assert math.fsum(edge) == pytest.approx(st["edge_area"], rel=1e-12, abs=0.0)


@pytest.mark.parametrize("side", SIDES)
def test_a_room_through_one_side_of_the_grid_is_open_on_that_side_only(side):
    # The chimney of A15 turned through each side in turn, so that each side is the only one open
    # in some case: a side given for another, or left out, is seen.
    h, shape = 0.125, np.array([10, 10, 10])
    s = SIDES.index(side)
    a = s // 2
    lo, hi = np.full(3, 1.75), np.full(3, 8.25)
    if s % 2:
        hi[a] = 11.0
    else:
        lo[a] = -1.0
    f = np.array([x for k, x in enumerate(BOX_FACES.tolist()) if k not in BOX_SIDE[side]])
    cut = fit(box(lo * h, hi * h), f, np.array([4.3, 4.6, 4.9]) * h, h, np.zeros(3), shape)
    st = cut.cut_stats
    assert st["refused"] == 0
    assert [bool(x) for x in cut.open_sides] == [k == s for k in range(6)] and cut.open
    # The air cells next to the edge are the room's 6 x 6 cells at index 1 (8) along the side's
    # axis, and the opening on the edge plane is 6.5 x 6.5 cell faces.
    expect = set()
    for i in range(2, 8):
        for j in range(2, 8):
            p = [i, j]
            p.insert(a, 8 if s % 2 else 1)
            expect.add(cell_index(p, shape))
    air = np.asarray(cut.cut_air())
    assert {c for c in next_to_edge(shape) if air[c] == 1} == expect
    assert relative(st["edge_area"], 6.5**2) <= 1e-12


# ---- Group B: the update on fitted cells (WaveSolver with fitted_walls=True) -------------------
#
# References: the geometry (volumes, areas, the analytic modes of a box at the 7-point stencil's
# own dispersion, the reflection off a duct's end), the default path's layout where walls lie on
# cell faces, and the update's rules applied in NumPy to CutFit's rows; never the fitted solver's
# own output.

G = 1.0 / 3.0  # the 7-point stencil's axial weight
MARGIN = 2.0**-18  # the share every link weight is lowered by (fdtd.hpp, kMargin)
STEPS = np.array(_STEPS)


def wave(
    v,
    f,
    h,
    origin,
    shape,
    src,
    probes=None,
    *,
    fitted=True,
    mat=None,
    beta=(0.0,),
    branches=(),
    edge=1.0,
    layer=0,
    damping=0.0,
    speed=None,
    threads=1,
    **more,
):
    """A wave solver over the mesh (v, f) on the grid (h, origin, shape) with the source `src` and
    the listeners `probes`, its walls fitted (on cell faces with fitted=False): each face's material
    `mat` (all 0 by default), each material's admittance `beta`, and `branches` a list of
    (material, rows (n, 3) of each branch's m, r and q)."""
    v = np.ascontiguousarray(v, float)
    f = np.ascontiguousarray(f, np.int64)
    mat = np.zeros(len(f), np.int64) if mat is None else np.asarray(mat, np.int64)
    rows = [np.zeros((0, 3))] + [np.asarray(r, float) for _, r in branches]
    owners = [np.zeros(0, np.int64)] + [np.full(len(r), m, np.int64) for m, r in branches]
    probes = np.zeros((0, 3)) if probes is None else np.asarray(probes, float).reshape(-1, 3)
    if fitted:
        more.update(fitted_walls=True, vertices=v, faces=f)
    return _native.WaveSolver(
        _native.Bvh(v, f),
        mat,
        np.asarray(beta, float),
        np.ascontiguousarray(np.vstack(rows)),
        np.concatenate(owners),
        np.asarray(origin, float),
        float(h),
        *(int(n) for n in shape),
        np.asarray(src, float),
        probes,
        edge_admittance=edge,
        layer=layer,
        layer_damping=damping,
        threads=threads,
        speed=speed,
        **more,
    )


def updates(lay, cells):
    """How a layout steps each of `cells` (air cells): its 18 link weights, coef_a, coef_b,
    coef_inv and source (an interior cell: the stencil's six weights, its centre, 1, 1 and 0); and
    its branch states as sorted (branch, weight) pairs."""
    at = {c: k for k, c in enumerate(lay["special"].tolist())}
    links = lay["link_start"].astype(np.int64)
    states = lay["state_start"].astype(np.int64)
    out = np.zeros((len(cells), 22))
    branch = []
    for n, c in enumerate(cells):
        k = at.get(int(c))
        if k is None:
            assert lay["types"][c] == 1, c
            out[n, :6] = lay["weights"][:6]
            out[n, 18:] = lay["centre"][0], 1.0, 1.0, 0.0
            branch.append([])
            continue
        d = lay["link_direction"][links[k] : links[k + 1]]
        out[n, d] = lay["link_weight"][links[k] : links[k + 1]]
        out[n, 18:] = lay["coef_a"][k], lay["coef_b"][k], lay["coef_inv"][k], lay["source"][k]
        s = slice(states[k], states[k + 1])
        pairs = zip(lay["state_branch"][s].tolist(), lay["state_weight"][s].tolist(), strict=True)
        branch.append(sorted(pairs))
    return out, branch


def operator(solver):
    """The rigid update of a solver without a layer, p+ = (2 - K) p - p-, over its air cells: K
    made symmetric by the cells' volumes, (D^1/2 K D^-1/2 + its transpose) / 2 with D = diag(V),
    V from cut_cells() for the special cells and 1 for the others; with the layout, the cells and
    their V."""
    lay = solver.layout()
    nx, ny = int(lay["shape"][0]), int(lay["shape"][1])
    types = np.asarray(lay["types"])
    assert not len(lay["layer_cells"])
    cells = np.flatnonzero((types == 1) | (types == 2))
    index = np.full(types.size, -1)
    index[cells] = np.arange(cells.size)
    offset = STEPS @ np.array([1, nx, nx * ny])
    inner = np.flatnonzero(types == 1)
    special = lay["special"].astype(np.int64)
    owner = np.repeat(special, np.diff(lay["link_start"].astype(np.int64)))
    rows = [inner] * 7 + [special, owner]
    cols = [inner] + [inner + offset[k] for k in range(6)]
    cols += [special, owner + offset[lay["link_direction"]]]
    vals = [np.full(inner.size, 2.0 - float(lay["centre"][0]))]
    vals += [np.full(inner.size, -float(lay["weights"][k])) for k in range(6)]
    vals += [2.0 - lay["coef_a"].astype(float), -lay["link_weight"].astype(float)]
    r, c, x = (np.concatenate(a) for a in (rows, cols, vals))
    assert np.all(index[c] >= 0)  # every link reaches air
    k = sp.csr_matrix((x, (index[r], index[c])), shape=(cells.size, cells.size))
    volume = np.ones(types.size)
    cut = solver.cut_cells()
    if cut:
        volume[cut["special"]] = cut["volume"]
    root = np.sqrt(volume[cells])
    b = sp.diags(root) @ k @ sp.diags(1.0 / root)
    return ((b + b.T) / 2).tocsc(), lay, cells, volume[cells]


def lowest_modes(b, count):
    mu, u = sla.eigsh(b, k=count, sigma=-1e-4, which="LM", tol=1e-10)
    order = np.argsort(mu)
    return mu[order], u[:, order]


def box_modes(size, count):
    """The `count` lowest modes (l, m, n) of a rigid box of `size`, by their exact frequencies."""
    lmn = [q for q in itertools.product(range(8), repeat=3) if sum(q)]
    lmn.sort(key=lambda q: sum((n / s) ** 2 for n, s in zip(q, size, strict=True)))
    return lmn[:count]


def rotation(yaw, pitch):
    """Turned by `yaw` degrees about z after `pitch` degrees about x."""
    a, b = math.radians(yaw), math.radians(pitch)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    rx = np.array([[1.0, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return rz @ rx


def matched_modes(solver, size, turn, centre, h, origin, count):
    """The `count` lowest modes of a rigid box of `size` (turned by the matrix `turn` about
    `centre`) as the solver's update rings them: for each exact mode (l, m, n), the computed mode
    whose shape is most like its own (volume-weighted), with its frequency, the exact one at the
    7-point stencil's own dispersion (the mean over the mode's eight plane waves), the computed
    mode's shape and its index."""
    b, lay, cells, volume = operator(solver)
    mu, u = lowest_modes(b, count + 8)
    nx, ny = int(lay["shape"][0]), int(lay["shape"][1])
    ijk = np.stack([cells % nx, cells // nx % ny, cells // (nx * ny)], axis=1)
    room = (origin + (ijk + 0.5) * h - centre) @ turn + centre
    rate = C / (COURANT * h)
    computed = rate * np.arcsin(np.sqrt(np.clip(mu, 0.0, None)) / 2) / math.pi
    signs = np.array(list(itertools.product((1, -1), repeat=3)))
    out = []
    for lmn in box_modes(size, count):
        shape = np.prod([np.cos(n * math.pi * room[:, a] / size[a]) for a, n in enumerate(lmn)], 0)
        shape = shape * np.sqrt(volume)
        j = int(np.argmax(np.abs(shape @ u) / np.linalg.norm(u, axis=0)))
        k = (
            signs * (np.array(lmn) * math.pi / size)
        ) @ turn.T  # the plane waves in the grid's axes
        s = np.sqrt(np.sum(np.sin(k * h / 2) ** 2, axis=1))
        exact = float(np.mean(rate * np.arcsin(COURANT * s) / math.pi))
        out.append((lmn, computed[j], exact, u[:, j], j))
    return out, cells, volume


def mode_errors(solver, size, turn, centre, h, origin, count):
    modes, _, _ = matched_modes(solver, size, turn, centre, h, origin, count)
    assert len({j for *_, j in modes}) == count  # each exact mode found its own
    return np.array([100.0 * (f / exact - 1.0) for _, f, exact, _, _ in modes])


def test_fitted_walls_on_cell_faces_are_the_cell_faces_walls():
    # A box whose walls lie exactly on cell faces (dyadic cell and origin, so that every plane is
    # exact), rigid, then absorbing: walls of admittance 0.05 and a floor of 0.1 with three
    # branches. Every air cell is stepped as the default path steps it: its links, coef_a, coef_b,
    # coef_inv, source and branch states within 1e-6. A whole cell face of wall gives the default
    # boundary weight g / (2 L), and its branches the default gains.
    h = 0.125
    v = box((0, 0, 0), (1.5, 1.25, 1.0))
    origin, shape = np.full(3, -0.25), (16, 14, 12)
    src, probe = (0.71, 0.58, 0.47), (1.1, 0.9, 0.6)
    rate = C / (COURANT * h)
    fit_ = Admittance(0.05, np.array([0.6, 0.3]), np.array([2.0, 1.5]), np.array([100.0, 250.0]))
    floor = np.zeros(12, np.int64)
    floor[BOX_SIDE["-z"]] = 1
    for beta, mat, branches in (
        ((0.0,), None, ()),
        ((0.05, fit_.beta + 0.1), floor, [(1, fit_.branches(rate))]),
    ):
        made = {
            fitted: wave(
                v,
                BOX_FACES,
                h,
                origin,
                shape,
                src,
                probe,
                fitted=fitted,
                mat=mat,
                beta=beta,
                branches=branches,
            )
            for fitted in (True, False)
        }
        lays = {k: s.layout() for k, s in made.items()}
        air = np.flatnonzero(lays[False]["types"] != 0)
        assert np.array_equal(air, np.flatnonzero(lays[True]["types"] != 0))
        (want, want_states), (got, got_states) = updates(lays[False], air), updates(lays[True], air)
        assert np.abs(got - want).max() <= 1e-6
        assert [[b for b, _ in s] for s in got_states] == [[b for b, _ in s] for s in want_states]
        for g_, w_ in zip(got_states, want_states, strict=True):
            assert all(abs(x - y) <= 1e-6 for (_, x), (_, y) in zip(g_, w_, strict=True))
        if branches:
            assert sum(map(len, got_states)) > 0 and (got[:, 19] < 1.0).sum() > 0


def l_room_solver(fitted=True, beta=(0.0, 0.0, 0.0), **more):
    v, f, mat, h, origin, shape, src = l_room()
    return wave(
        v, f, h, origin, shape, src, (0.95, 0.25, 0.3), fitted=fitted, mat=mat, beta=beta, **more
    )


def room_box(yaw):
    """The room-sized box 4.13 x 3.37 x 2.71 m, rigid, on the default 19.8 cm grid, turned `yaw`
    degrees about z: its solver, size, turn, centre, cell and origin."""
    size, h = np.array([4.13, 3.37, 2.71]), 0.19815505479559642
    turn, centre = rotation(yaw, 0), size / 2
    v = (box((0, 0, 0), size) - centre) @ turn.T + centre
    origin = v.min(axis=0) - h * (2.0 + np.array([0.271, 0.613, 0.137]))
    shape = np.ceil((v.max(axis=0) - origin) / h).astype(int) + 2
    solver = wave(v, BOX_FACES, h, origin, shape, centre + 0.1)
    return solver, size, turn, centre, h, origin


def test_fitted_links_are_symmetric_and_keep_the_bound():
    # The L-room with its turned plates: each link stands for the same area from both of its cells
    # (V_i w_ij = V_j w_ji, V = cut_cells()["volume"], 1 for a plain cell), every link has its
    # reverse, each cell's links weigh at most 2 together, and the largest eigenvalue of the update
    # made symmetric is below 4 (Gershgorin's bound, so the time step stays the stencil's). In the
    # L-room and in the room-sized box turned 22.5 degrees, each cell's coef_a and links sum to at
    # most 2 (coef_a is rounded down), so that no eigenvalue is below 0, where a uniform pressure
    # would grow. With coef_a rounded to the nearest float instead, 154 cells of that box sum to
    # more than 2 and its update has an eigenvalue of -1.7e-10.
    solver = l_room_solver()
    b, lay, cells, volume = operator(solver)
    nx, ny = int(lay["shape"][0]), int(lay["shape"][1])
    offset = STEPS @ np.array([1, nx, nx * ny])
    opposite = np.array([k ^ 1 if k < 6 else 0 for k in range(18)])
    for k in range(6, 18):  # the edge-diagonal direction with both steps reversed
        opposite[k] = int(np.flatnonzero((STEPS == -STEPS[k]).all(axis=1))[0])
    weight, _ = updates(lay, cells)
    v_of = dict(zip(cells.tolist(), volume.tolist(), strict=True))
    row_of = {c: n for n, c in enumerate(cells.tolist())}
    pairs = 0
    for n, c in enumerate(cells.tolist()):
        for k in np.flatnonzero(weight[n, :18]):
            j = c + offset[k]
            there = weight[row_of[j], opposite[k]]
            assert there > 0, (c, k)
            assert abs(v_of[c] * weight[n, k] - v_of[j] * there) <= 1e-6, (c, k)
            pairs += 1
    assert pairs > 0 and np.all(weight[:, :18].sum(axis=1) <= 2.0)
    assert (weight[:, 6:18] > 0).any()  # diagonal links (the plates are turned)
    assert np.all(weight[:, 18].astype(float) + weight[:, :18].astype(float).sum(axis=1) <= 2.0)
    top = sla.eigsh(b, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0]
    assert 3.5 < top < 4.0
    bottom = sla.eigsh(b, k=1, sigma=-1e-3, which="LM", return_eigenvectors=False, tol=1e-14)[0]
    assert bottom >= -1e-13
    b, lay, cells, _ = operator(room_box(22.5)[0])
    weight, _ = updates(lay, cells)
    assert (weight[:, 6:18] > 0).any()
    assert np.all(weight[:, 18].astype(float) + weight[:, :18].astype(float).sum(axis=1) <= 2.0)
    bottom = sla.eigsh(b, k=1, sigma=-1e-3, which="LM", return_eigenvectors=False, tol=1e-14)[0]
    assert bottom >= -1e-13


def rules_on_rows(cut, shape):
    """The volumes and areas the update's rules give, from CutFit's rows alone (a grid without a
    layer): for each row, and each plain cell a sliver link reaches, (V_air, V, its 18 open areas
    after the sliver rule, and the V a clamp before the sliver rule would give). A sliver f on a
    row's face k scales the link on the opposite face, from the row and from the cell there, by
    w = 3 l / ((l + 1/2) (l + 1)) with l = 1 + f (the product of both factors when both of its
    cells have one); a plain cell keeps its whole faces and its volume from the fit's widths; V is
    the larger of V_air and g / 2 times the sum of the scaled areas (summed in direction order,
    then multiplied once). Also, for each link the rule scales (its two cells), the slivers'
    thicknesses that scale it."""
    rows = cut.cut_cells()
    cells = np.asarray(rows["cell"]).tolist()
    air = np.asarray(cut.cut_air())
    nx, ny, nz = (int(n) for n in shape)
    offset = STEPS @ np.array([1, nx, nx * ny])
    widths = cut.cut_widths()

    def ijk(c):
        return c % nx, c // nx % ny, c // (nx * ny)

    def on_edge(c):
        p = ijk(c)
        return min(p) == 0 or p[0] == nx - 1 or p[1] == ny - 1 or p[2] == nz - 1

    area = {c: np.array(rows["open_area"][n], float) for n, c in enumerate(cells)}
    v_air = {c: float(rows["air_volume"][n]) for n, c in enumerate(cells)}
    factor, touched = {}, {}
    for n, c in enumerate(cells):
        for k in range(6):
            f = float(rows["sliver"][n][k])
            m = c + offset[k ^ 1]
            if f > 0 and not on_edge(m) and air[m] == 1:
                key = (min(c, m), max(c, m))
                factor[key] = factor.get(key, 1.0) * sliver_factor(f)
                touched.setdefault(key, []).append(f)
    for key in factor:
        for c in key:
            if c not in area:  # a plain cell: its whole faces towards the air
                p = ijk(c)
                whole = [
                    widths[(a + 1) % 3][p[(a + 1) % 3]] * widths[(a + 2) % 3][p[(a + 2) % 3]]
                    for a in range(3)
                ]
                open_ = [
                    air[c + offset[k]] in (1, 2) and not on_edge(c + offset[k]) for k in range(6)
                ]
                area[c] = np.array(
                    [whole[k // 2] if open_[k] else 0.0 for k in range(6)] + [0.0] * 12
                )
                v_air[c] = widths[0][p[0]] * widths[1][p[1]] * widths[2][p[2]]

    def clamped(c, a):
        total = 0.0
        for x in a:
            total += x
        return max(v_air[c], G / 2 * total)

    out = {}
    for c, a in area.items():
        scaled = a.copy()
        for k in range(6):
            scaled[k] *= factor.get((min(c, c + offset[k]), max(c, c + offset[k])), 1.0)
        out[c] = (v_air[c], clamped(c, scaled), scaled, clamped(c, a))
    return out, touched


def two_cells_wide():
    """A closed box from 3.7 to 6.3 cells along x, to 8.3 along y and to 7.3 along z, on a grid of
    0.125 m cells from the origin: two cells span its width along x, and each owns a sliver of 0.3
    cells beyond its outer x face, so that both cells' slivers scale the 4 x 3 links between
    them."""
    h = 0.125
    v = box(np.array([3.7, 3.7, 3.7]) * h, np.array([6.3, 8.3, 7.3]) * h)
    return v, BOX_FACES, h, np.zeros(3), np.array([10, 12, 11]), np.array([5.03, 6.1, 5.4]) * h


@pytest.mark.parametrize("model", ["open turned box", "box two cells wide"])
def test_fitted_cells_take_their_volumes_and_links_from_the_fit_s_rows(model):
    # Every special cell's V (cut_cells()) and its link weights equal the update's rules applied to
    # CutFit's rows (links lowered(g A' / V) for each open area A' > 0 after the sliver rule, coef_a
    # 2 less their sum rounded down); a row the rules leave out is a plain cell to the bit; the
    # other special cells are the source's, whose volume is their widths' product. cut_stats counts
    # the rules' raised cells (raised, most_raised) and the links the sliver rule scales
    # (sliver_links), of them those both of whose cells' slivers scale it (sliver_both) and those
    # a sliver thicker than a cell scales (sliver_thick). The open turned box (1.2 x 0.9 x 0.6 m,
    # turned 40 degrees about z after 35 about x, on 10 cm cells) has sliver links, cells whose
    # volume the clamp raises (some of them with a sliver link), and cells a clamp before the
    # sliver rule would raise; the box two cells wide has 12 links that both of their cells'
    # slivers scale, each by the product of the two factors.
    if model == "open turned box":
        size, h = np.array([1.2, 0.9, 0.6]), 0.1
        turn, centre = rotation(40, 35), size / 2
        v = (box((0, 0, 0), size) - centre) @ turn.T + centre
        f = np.delete(BOX_FACES, BOX_SIDE["+z"], axis=0)
        origin = v.min(axis=0) - h * (2.0 + OFFSET)
        shape = np.ceil((v.max(axis=0) - origin) / h).astype(int) + 2
        src = centre + 0.013
    else:
        v, f, h, origin, shape, src = two_cells_wide()
    solver = wave(v, f, h, origin, shape, src)
    rule, touched = rules_on_rows(fit(v, f, src, h, origin, shape), shape)
    lay, cut = solver.layout(), solver.cut_cells()
    volume = dict(zip(cut["special"].tolist(), cut["volume"].tolist(), strict=True))
    weight, _ = updates(lay, np.array(sorted(rule)))
    raised = order = 0
    for n, c in enumerate(sorted(rule)):
        v_air, v_rule, a, v_before = rule[c]
        want = np.where(a > 0, lowered(G * a / v_rule), np.float32(0.0))
        assert np.array_equal(weight[n, :18], want), c
        assert weight[n, 18] == rounded_down(2.0 - want.astype(float).sum()), c
        if c in volume:
            assert volume[c] == v_rule, c
        else:  # a row spared as plain
            assert np.array_equal(want[:6], lowered(np.full(6, G))) and not want[6:].any(), c
        raised += v_rule > v_air * (1 + 1e-12)
        order += v_before != v_rule
    stats = solver.cut_stats
    assert stats["raised"] == raised
    lifted = [r[1] / r[0] for r in rule.values() if r[1] > r[0] * (1 + 1e-12)]
    assert stats["most_raised"] == max(lifted, default=1.0)
    assert stats["sliver_links"] == len(touched)
    assert stats["sliver_both"] == sum(len(x) > 1 for x in touched.values())
    assert stats["sliver_thick"] == sum(max(x) > 1.0 for x in touched.values())
    others = set(volume) - set(rule)
    assert others <= set(lay["special"][lay["source"] != 0].tolist())
    widths = fit(v, f, src, h, origin, shape).cut_widths()
    nx, ny = int(shape[0]), int(shape[1])
    for c in others:
        i, j, k = c % nx, c // nx % ny, c // (nx * ny)
        assert volume[c] == widths[0][i] * widths[1][j] * widths[2][k], c
    if model == "open turned box":
        assert stats["sliver_links"] > 0 and raised > 0 and order > 0 and others
    else:
        assert stats["sliver_both"] == 12


@pytest.mark.parametrize("shift", [(0.311, 0.173, 0.457), (-0.48, -0.48, -0.48)])
def test_a_fitted_box_rings_at_its_exact_modes(shift):
    # A rigid box on 10 cm cells, its walls `shift` cells off the cell faces: the 10 lowest modes of
    # the fitted update within 0.05 % of the box's exact ones at the 7-point stencil's dispersion.
    # With walls on cell faces the box is off by up to half a cell on each side, which moves them by
    # more than 0.5 %.
    size, h = np.array([2.43, 2.07, 1.53]), 0.1
    v = box((0, 0, 0), size)
    origin = -h * (2.0 + np.asarray(shift))
    shape = np.ceil((size - origin) / h).astype(int) + 2
    errors = {
        fitted: mode_errors(
            wave(v, BOX_FACES, h, origin, shape, size / 2 + 0.013, fitted=fitted),
            size,
            np.eye(3),
            size / 2,
            h,
            origin,
            10,
        )
        for fitted in (True, False)
    }
    assert np.abs(errors[True]).max() <= 0.05
    assert np.abs(errors[False]).max() > 0.5


@pytest.mark.parametrize(("yaw", "low", "high"), [(14, -0.146, 0.452), (45, -0.146, 3.367)])
def test_a_turned_fitted_box_rings_as_its_diagonal_links_say(yaw, low, high):
    # The room-sized box 4.13 x 3.37 x 2.71 m on the default 19.8 cm grid, turned about z: its 16
    # lowest modes within -0.146..+0.452 % of the exact ones (at the stencil's own dispersion) at 14
    # degrees and within -0.146..+3.367 % at 45 degrees (a diagonal link weighs g A, which stiffens
    # steep walls to first order in the cell). A diagonal link halved or dropped moves the modes
    # below the lower bound.
    solver, size, turn, centre, h, origin = room_box(yaw)
    errors = mode_errors(solver, size, turn, centre, h, origin, 16)
    assert low <= errors.min() and errors.max() <= high, errors.round(3)


def test_fitted_walls_decay_as_the_modes_say():
    # The room-sized box on 19.8 cm cells, turned 8 degrees, every wall of admittance 0.05: each of
    # the 9 lowest modes decays at its first-order rate from the update's own terms, sigma T = sum
    # g_i psi_i^2 / sum V_i psi_i^2 (psi the rigid update's mode, g_i = V_i (1 - coef_b) the cell's
    # wall term), within 4 % of the modal theory's c beta sum_a e_a / L_a (e_a = 2 for an axis along
    # which the mode varies, else 1: the mean of its square over the walls over that inside).
    size, h, beta = np.array([4.13, 3.37, 2.71]), 0.19815505479559642, 0.05
    turn, centre = rotation(8, 0), size / 2
    v = (box((0, 0, 0), size) - centre) @ turn.T + centre
    origin = v.min(axis=0) - h * (2.0 + np.array([0.5, 0.25, 0.4]))
    shape = np.ceil((v.max(axis=0) - origin) / h).astype(int) + 2
    solver = wave(v, BOX_FACES, h, origin, shape, centre + 0.1, beta=(beta,))
    modes, cells, volume = matched_modes(solver, size, turn, centre, h, origin, 9)
    lay = solver.layout()
    loss = np.zeros(int(np.prod(lay["shape"])))
    loss[lay["special"].astype(np.int64)] = 1.0 - lay["coef_b"].astype(float)
    loss = loss[cells]  # 1 - coef_b = g / V
    period = COURANT * h / C
    for lmn, *_, psi, _ in modes:
        u = psi / np.sqrt(volume)  # the update's own mode
        sigma = np.sum(loss * volume * u**2) / np.sum(volume * u**2) / period
        theory = C * beta * sum((2 if n else 1) / s for n, s in zip(lmn, size, strict=True))
        assert sigma / theory == pytest.approx(1.0, abs=0.04), lmn


def test_rows_equal_to_plain_cells_stay_plain():
    # A box with its walls on cell faces (dyadic, so that every lattice width is exactly 1): the
    # rows next to its walls have a wall and fewer links, and the uncut rows next to those have the
    # stencil's weights to the bit. The special cells are exactly the rows whose coefficients
    # (computed here from CutFit's rows by the update's rules: links lowered(g A / V), coef_a 2 less
    # their sum rounded down, the wall term) differ from a plain cell's, and the source's cells;
    # plain_rows counts the other rows.
    h = 0.125
    v = box((0, 0, 0), (1.5, 1.25, 1.0))
    origin, shape, src = np.full(3, -0.25), (16, 14, 12), (0.71, 0.58, 0.47)
    solver = wave(v, BOX_FACES, h, origin, shape, src, beta=(0.2,))
    cut = fit(v, BOX_FACES, src, h, origin, shape)
    rows = cut.cut_cells()
    weight = lowered(G * np.asarray(rows["open_area"]) / np.asarray(rows["air_volume"])[:, None])
    weight = np.where(np.asarray(rows["open_area"]) > 0, weight, np.float32(0.0))
    centre = rounded_down(2.0 - weight.astype(float).sum(axis=1))
    plain = (
        np.all(weight[:, :6] == lowered(np.full(6, G)), axis=1)
        & np.all(weight[:, 6:] == 0, axis=1)
        & (centre == np.float32(2.0 - 6 * float(lowered(np.array([G]))[0])))
        & ~np.asarray(rows["wall_area"]).any(axis=1)
    )
    lay = solver.layout()
    source_cells = set(lay["special"][lay["source"] != 0].tolist())
    special = set(lay["special"].tolist())
    cells = np.asarray(rows["cell"])
    assert special - source_cells == set(cells[~plain].tolist()) - source_cells
    assert plain.sum() > 0 and (~plain).sum() > 0
    assert solver.cut_stats["plain_rows"] == len(set(cells[plain].tolist()) - source_cells)


def rounded_down(x):
    """Each double as the update stores coef_a: the largest float not above it."""
    x = np.asarray(x, float)
    f = x.astype(np.float32)
    return np.where(f.astype(float) > x, np.nextafter(f, np.float32(-np.inf)), f)


def lowered(x):
    """Each double as the update stores a link weight: times 1 - 2^-18, rounded down to a float."""
    x = np.asarray(x, float) * (1.0 - MARGIN)
    f = x.astype(np.float32)
    return np.where(f.astype(float) > x, np.nextafter(f, np.float32(0.0)), f)


def duct(length, far_beta=0.0, far_branches=(), src_x=10.0, at=(30.0,), seconds=0.45, fitted=True):
    """The pressure at x = each of `at` in a duct 0.3 x 0.3 m from x = 0 (admittance 1) to x =
    length (far_beta, and the branches far_branches), the source at src_x, its walls fitted (on
    cell faces with fitted=False) on 10 cm cells whose faces lie 0.0137 cells off x = 0; with the
    sample rate."""
    h, width = 0.1, 0.3
    v = box((0, 0, 0), (length, width, width))
    mat = np.zeros(12, np.int64)
    mat[BOX_SIDE["-x"]], mat[BOX_SIDE["+x"]] = 1, 2
    origin = -h * (2.0 + OFFSET)
    shape = np.ceil((np.array([length, width, width]) - origin) / h).astype(int) + 2
    rate = C / (COURANT * h)
    branches = [(2, far_branches)] if len(far_branches) else ()
    probes = [(x, width / 2, width / 2) for x in at]
    solver = wave(
        v,
        BOX_FACES,
        h,
        origin,
        shape,
        (src_x, width / 2, width / 2),
        probes,
        mat=mat,
        beta=(0.0, 1.0, far_beta),
        branches=branches,
        fitted=fitted,
    )
    return solver.run(_drive(rate, h, int(seconds * rate))), rate


def duct_end(e, cells=800):
    """The length of the duct of duct() whose far end lies `e` cells past a cell face, about
    `cells` cells long."""
    return 0.1 * (cells + e - OFFSET[0])


@pytest.mark.parametrize("e", [-0.48, -0.25, 0.25, 0.48])
def test_a_fitted_duct_end_lies_where_the_model_puts_it(e):
    # A plane wave down a duct to a rigid end `e` cells past a cell face, read at two cell centres
    # 15 and 15.5 m before the end, the source 5 m before them: each frequency's wave at 40-200 Hz
    # parted into the waves going towards the end and back (p = A e^-ikx + B e^ikx at the grid's own
    # wave number along an axis, sin(w T / 2) = L sin(k h / 2)), B / A = e^-2ikx puts the end x
    # within 0.03 cells of the model's on average (weighed by |A|^2). With walls on cell faces the
    # same measure puts it on the cell face the default path takes, half a cell or less away. The
    # run stops before the source's sound comes back off the duct's near end.
    h, end = 0.1, duct_end(e)
    origin = -h * (2.0 + OFFSET[0])
    first = int(round((end - 15.5 - origin) / h - 0.5))
    x = origin + (np.array([first, first + 5]) + 0.5) * h
    face = origin + h * round((end - origin) / h)  # the default path's wall
    for fitted, place in ((True, end), (False, face)):
        p, rate = duct(end, src_x=end - 20.5, at=x, seconds=0.36, fitted=fitted)
        n_fft = 1 << 14
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        band = (f >= 40) & (f <= 200)
        k = 2 / h * np.arcsin(np.sin(math.pi * f[band] / rate) / COURANT)
        p1, p2 = (np.fft.rfft(y, n_fft)[band] for y in p)
        det = np.exp(1j * k * (x[1] - x[0])) - np.exp(-1j * k * (x[1] - x[0]))
        a = (p1 * np.exp(1j * k * x[1]) - p2 * np.exp(1j * k * x[0])) / det
        b = (p2 * np.exp(-1j * k * x[0]) - p1 * np.exp(-1j * k * x[1])) / det
        turns = (-np.angle(b / a) / (2 * k) - place) * k / math.pi  # half wavelengths
        off = (turns - np.round(turns)) * math.pi / k / h
        mean = np.sum(off * np.abs(a) ** 2) / np.sum(np.abs(a) ** 2)
        assert abs(mean) <= 0.03, (fitted, mean)


def duct_reflection(pressure, rate, path):
    """The far end's reflection off duct()'s duct heard at 30 m, `path` metres from the source by
    the end: 40-200 Hz, the incident and reflected waves gated with Hann windows, the square root of
    their energies' ratio."""
    steps = len(pressure)
    _, delay = source_signal(rate, 0.1)
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    gain = np.clip((f - 40) / 20, 0, 1) * np.clip((200 - f) / 50, 0, 1)
    pressure = np.fft.irfft(np.fft.rfft(pressure, n_fft) * gain, n_fft)[:steps]
    t = (np.arange(steps) - delay) / rate

    def gated(length):
        x = t - length / C
        window = np.where(np.abs(x) < 0.04, 0.5 + 0.5 * np.cos(np.pi * x / 0.04), 0.0)
        return float(np.sum((pressure * window) ** 2))

    return math.sqrt(gated(path) / gated(20.0))


@pytest.mark.parametrize(
    ("beta", "e", "within"),
    [(0.5, 0.0, 0.012), (0.5, 0.25, 0.012), (0.5, -0.48, 0.012), (1.0, 0.48, 0.1)],
)
def test_a_fitted_absorbing_duct_end_reflects_by_its_admittance(beta, e, within):
    # The duct's far end of admittance beta, `e` cells past a cell face: it reflects (1 - beta) /
    # (1 + beta) at normal incidence, within 0.012 for beta 0.5; for beta 1 at 0.48 cells within
    # 0.1, the stated limit (the wall term acts on the pressure at the cell's centre, up to a cell
    # from the wall).
    end = duct_end(e)
    pressure, rate = duct(end, far_beta=beta)
    reflection = duct_reflection(pressure[0], rate, 2 * end - 40.0)
    assert abs(reflection - (1 - beta) / (1 + beta)) <= within


RESONANT = Admittance(0.05, np.array([0.6]), np.array([2.0]), np.array([100.0]))


@functools.cache
def resonant_reference():
    """The incident wave of the frequency-dependent duct: 200 m long, heard at 110 m before the
    near end's echo comes by; with the sample rate."""
    reference, rate = duct(200.0, src_x=60.0, at=(110.0,), seconds=0.6)
    return reference[0], rate


def fitted_end_reflection(f, rate, e, end):
    """The reflection at frequencies `f` off a duct's end of admittance `end` (an Admittance), the
    end `e` cells past a cell face (-0.5 < e < 0.5), as the fitted update steps it along the duct's
    axis: the end's row holds V = 1 + e cells of air (its own and, for e > 0, the sliver of
    thickness e beyond its face), its link to the cell before it is scaled by w(1 + e) on both
    sides for e > 0 (the sliver rule), and its wall term is (L/2)(beta + the sum of 1 / (2 a) over
    the branches) / V, each branch's state weighing (L/2) / V, on the pressure at the row's centre;
    every other cell steps the 7-point stencil's leapfrog along an axis. With the pressure going as
    z^n, z = e^(2 pi i f / rate), each equation is linear in the incident and reflected waves, and
    the reflection, referred to the row's centre, is solved exactly at each frequency."""
    g = 1.0 / 3.0
    half = math.sqrt(g) / 2  # L/2, a whole face's wall weight
    phi = 2 * np.pi * np.asarray(f, float) / rate
    z = np.exp(1j * phi)
    zeta = np.exp(2j * np.arcsin(np.sin(phi / 2) / math.sqrt(g)))  # one cell along the axis
    volume = 1.0 + e
    lam = 1.0 + max(e, 0.0)
    link = g * 3 * lam / ((lam + 0.5) * (lam + 1))  # the link, from the cell before (V = 1)
    wall = end.beta * (z - 1 / z)
    for m, r, q in end.branches(rate):
        a = m + 0.5 * r + 0.5 * q
        now, before = 2 * m / a, (2 * m + q) / a
        velocity = (z - 1 / z) / (2 * a) / (z - now - (1 - before) / z)  # per unit of pressure
        wall = wall + (z - 1 / z) / (2 * a) + (now - before / z) * velocity
    wall = half * wall / volume
    # The cells before the row carry zeta^-n + R zeta^n (n cells from the row). The cell before it:
    # g ((1 + R) - p1) = link (p - p1), p1 = zeta + R / zeta; the row: (z - 2 + 1 / z + wall) p +
    # link / V (p - p1) = 0.
    b1 = g * (zeta - 1) - link * zeta
    a11 = g * (1 - 1 / zeta) + link / zeta
    a12 = -link * np.ones_like(z)
    row = z - 2 + 1 / z + wall + link / volume
    a21, a22, b2 = -link / volume / zeta, row, link / volume * zeta
    return (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21)


@pytest.mark.parametrize("e", [0.25, -0.48])
def test_a_fitted_duct_end_reflects_as_its_frequency_dependent_admittance_says(e):
    # A plane wave down a duct to an end whose admittance resonates at 100 Hz, the end `e` cells
    # past a cell face: the reflected wave (the run less a run in a longer duct) over a rigid end's
    # at the same place, against the fitted update's own reflection off that end along the axis
    # (fitted_end_reflection), within 0.012. Its distance from the default path's reflection (a
    # wall half a cell past the last centre, _discrete_reflection) is the stated limit of the wall
    # term on the row's centre: at most the analysis's own distance plus 0.005 (0.0103 at 0.25
    # cells, 0.0228 at -0.48). Branch weights not divided by the row's volume, or divided twice,
    # move the reflection by 0.09 to 0.25.
    reference, rate = resonant_reference()
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    band = (f > 30) & (f < 250)
    f = f[band]
    expected = np.abs(_discrete_reflection(f, rate, np.conj(RESONANT(f, rate))))
    fitted = fitted_end_reflection(f, rate, e, RESONANT)
    analysis = np.abs(fitted / fitted_end_reflection(f, rate, e, Admittance(0.0)))
    end = duct_end(e, 1300)
    ratio = []
    for far in (RESONANT, Admittance(0.0)):
        pressure, _ = duct(
            end,
            far_beta=far.beta,
            far_branches=far.branches(rate),
            src_x=60.0,
            at=(110.0,),
            seconds=0.6,
        )
        ratio.append(np.fft.rfft(pressure[0] - reference, n_fft)[band])
    measured = np.abs(ratio[0] / ratio[1])
    np.testing.assert_allclose(measured, analysis, atol=0.012)
    limit = np.abs(analysis - expected).max()
    assert np.abs(measured - expected).max() <= limit + 0.005, limit


def l_room_points(count=24, seed=3):
    """`count` points spread through the L-room's air (its notch, x > 0.6 and y > 0.5, left out)."""
    rng = np.random.default_rng(seed)
    points = rng.uniform([0.05, 0.05, 0.05], [1.15, 0.85, 0.55], (4 * count, 3))
    return points[~((points[:, 0] > 0.6) & (points[:, 1] > 0.5))][:count]


def stable_case(case):
    """The L-room of l_room() for a long run, heard at points spread through it: rigid, every wall
    of admittance 1, or open (its ceiling left out, in a grid with a matched layer of 4 cells)."""
    v, f, mat, h, origin, shape, src = l_room()
    probes = l_room_points()
    if case == "rigid":
        return wave(v, f, h, origin, shape, src, probes, mat=mat, beta=(0.0, 0.0, 0.0)), h
    if case == "absorbing":
        return wave(v, f, h, origin, shape, src, probes, mat=mat, beta=(1.0, 1.0, 1.0)), h
    keep = np.r_[0:4, 8 : len(f)]  # the ceiling: the fan of 4 faces after the floor's 4
    origin = v.min(axis=0) - h * (6.0 + OFFSET)  # edge, layer, the cell next to it, then the room
    shape = np.ceil((v.max(axis=0) - origin) / h).astype(int) + 6
    solver = wave(
        v,
        f[keep],
        h,
        origin,
        shape,
        src,
        probes,
        mat=mat[keep],
        beta=(0.0, 0.0, 0.0),
        layer=4,
        damping=0.5,
        threads=2,
    )
    return solver, h


@pytest.mark.parametrize("case", ["rigid", "absorbing", "open"])
def test_fitted_walls_stay_stable_over_a_long_run(case):
    # 20,000 steps (3.4 s), the rms over 24 points spread through the room in tenths of the run: a
    # rigid closed room keeps its energy (the last tenth within 0.97 to 1.05 of the second, none
    # above 1.1 times it); with every wall of admittance 1, and open to a matched layer through its
    # missing ceiling, the sound decays to single precision's rounding by the half (a millionth of
    # the first tenth's rms) and does not grow again. In the rigid room each point's mean over each
    # tenth is taken out first: the update's lowest mode, the uniform pressure, rings at an
    # eigenvalue of about 3e-9 (coef_a rounded down to a float), a period of 110,000 steps, and
    # holds next to no energy (lambda u^2), but moves the mean by up to 0.6 in this small room,
    # whose cells are nearly all special.
    solver, h = stable_case(case)
    rate = C / (COURANT * h)
    pressure = solver.run(_drive(rate, h, 20_000)).reshape(24, 10, -1).astype(float)
    if case == "rigid":
        pressure = pressure - pressure.mean(axis=2, keepdims=True)
    rms = np.sqrt(np.mean(pressure**2, axis=(0, 2)))
    if case == "rigid":
        assert 0.97 <= rms[-1] / rms[1] <= 1.05 and np.all(rms[1:] <= 1.1 * rms[1])
    else:
        assert np.all(rms[5:] <= 1e-6 * rms[0]) and np.all(rms[5:] <= 1.01 * rms[5])


def test_the_source_next_to_a_fitted_wall_puts_in_all_its_sound():
    # The source 0.2 cells from a wall turned 8 degrees: it feeds the cells CutFit.cut_points()
    # gives it, each w / V of its weight w, so that the sum of source x V over them is 1
    # (finite-volume injection: the sound the source puts in does not depend on the volumes of its
    # cells).
    h = 0.1
    v = turned(box((0, 0, 0), (1.3, 1.1, 0.9)), 8)
    origin, shape = grid_round(v, h)
    turn = rotation(8, 0)
    normal = turn @ np.array([-1.0, 0.0, 0.0])
    src = v[0] + turn @ np.array([0.0, 0.47, 0.41]) - 0.2 * h * normal
    solver = wave(v, BOX_FACES, h, origin, shape, src)
    lay, cut = solver.layout(), solver.cut_cells()
    volume = dict(zip(cut["special"].tolist(), cut["volume"].tolist(), strict=True))
    fed = {
        int(c): float(s) * volume[int(c)]
        for c, s in zip(lay["special"], lay["source"], strict=True)
        if s
    }
    want = dict(
        fit(v, BOX_FACES, src, h, origin, shape).cut_points(src.reshape(1, 3), np.zeros((0, 3)))[
            "source"
        ]
    )
    assert fed.keys() == want.keys() and len(fed) > 1
    assert max(abs(fed[c] - want[c]) for c in fed) <= 1e-6
    assert math.fsum(fed.values()) == pytest.approx(1.0, abs=1e-6)
    assert any(abs(volume[c] - 1.0) > 0.05 for c in fed)  # a cut cell among them


def plate_grid(h, layer):
    """The plate of test_wave's _plate_solver (an 8 x 6 m rigid plate at z = 0 in the open) on its
    grid with 8 cells of room and `layer` cells of matched layer inside them."""
    v, f = plate([(0, 0, 0), (8, 0, 0), (8, 6, 0), (0, 6, 0)])
    src = np.array([2.0, 2.0, 1.5])
    listeners = np.array([[6.0, 4.0, 1.2], [4.0, 3.0, 3.0], [7.5, 1.0, 0.5]])
    points = np.vstack([src, listeners])
    low = np.minimum(v.min(0), points.min(0) - 2 * h)
    high = np.maximum(v.max(0), points.max(0) + 2 * h)
    origin = low - h * (2.0 + 8 + OFFSET)
    shape = np.ceil((high - origin) / h).astype(int) + 2 + 8
    return wave(v, f, h, origin, shape, src, listeners, layer=layer, damping=0.5), origin


def sliver_factor(f):
    lam = 1.0 + f
    return 3.0 * lam / ((lam + 0.5) * (lam + 1.0))


def test_a_layered_grid_fits_its_room_as_an_unlayered_one_but_at_the_layer():
    # The plate lies 0.3624 cells above the cell faces at z index 10, so each cell under it at depth
    # 9 owns the air between them, a sliver of f = 0.3624 cells through its +z face. On the same
    # grid without a layer, the link through its -z face is scaled by w(1 + f) on both sides; with
    # an 8-cell layer the cell across it is a layer cell, which reads it whole, and the link stays
    # whole. Every special cell of both grids is stepped alike within 1e-6 but on those links, which
    # cut_stats counts (layer_whole_links) with the least factor it leaves out (layer_whole_least,
    # w(1 + f) from the plate's height), and their coef_a.
    h = 0.3
    (layered, origin), (bare, _) = plate_grid(h, 8), plate_grid(h, 0)
    f = (0.0 - origin[2]) / h % 1.0  # the plate's height above its cube's lower face (cells)
    lay, ref = layered.layout(), bare.layout()
    common = np.intersect1d(lay["special"], ref["special"])
    (got, _), (want, _) = updates(lay, common), updates(ref, common)
    nx, ny = int(lay["shape"][0]), int(lay["shape"][1])
    differ = np.abs(got - want).max(axis=1) > 1e-6
    assert np.all(common[differ] // (nx * ny) == 9)  # at depth 9, under the plate
    assert np.all(lay["types"][common[differ] - nx * ny] == 3)  # a layer cell across -z
    assert differ.sum() == layered.cut_stats["layer_whole_links"] > 100
    assert bare.cut_stats["layer_whole_links"] == 0
    assert layered.cut_stats["layer_whole_least"] == pytest.approx(sliver_factor(f), rel=1e-12)
    assert bare.cut_stats["layer_whole_least"] == 1.0
    np.testing.assert_allclose(got[differ, 4] * sliver_factor(f), want[differ, 4], rtol=1e-6)
    assert got[differ, 4] == pytest.approx(0.2447, abs=1e-4)
    assert want[differ, 4] == pytest.approx(0.2273, abs=1e-4)
    rest = np.ones(22, bool)
    rest[[4, 18]] = False
    assert np.abs(got[differ][:, rest] - want[differ][:, rest]).max() <= 1e-6
    np.testing.assert_allclose(
        got[differ, 18] - want[differ, 18], want[differ, 4] - got[differ, 4], atol=1e-6
    )
    volume = {}
    for s in (layered, bare):
        cut = s.cut_cells()
        volume[s is layered] = dict(
            zip(cut["special"].tolist(), cut["volume"].tolist(), strict=True)
        )
    assert max(abs(volume[True][c] - volume[False][c]) for c in common.tolist()) <= 1e-6


def test_fitted_walls_need_the_7_point_stencil_still_air_and_the_mesh():
    # With fitted walls the solver refuses, each with a message starting "fitted walls:", the
    # 19-point stencil, the air's motion (even still air given as arrays), a call without the mesh
    # the BVH was built from, and faces whose count is not the BVH's; and a grid whose cells'
    # lattice widths stray from 1 by more than 2^-24 (code 13), while the same box nearer the
    # origin is fitted: widths that stray by between 2^-25 and 2^-24 are fitted, by between 2^-24
    # and 2^-23 refused, on each axis.
    h = 0.1
    v = box((0, 0, 0), (0.8, 0.7, 0.6))
    origin, shape = grid_round(v, h)
    src = (0.41, 0.33, 0.29)
    with pytest.raises(ValueError, match="^fitted walls: they need the 7-point stencil"):
        wave(v, BOX_FACES, h, origin, shape, src, stencil=19)
    n = int(np.prod(shape))
    still = np.zeros(n, np.float32)
    with pytest.raises(ValueError, match="^fitted walls: they are not available in moving air"):
        wave(v, BOX_FACES, h, origin, shape, src, flow_x=still, flow_y=still)
    bvh = _native.Bvh(v, BOX_FACES)
    args = (
        bvh,
        np.zeros(12, np.int64),
        np.zeros(1),
        np.zeros((0, 3)),
        np.zeros(0, np.int64),
        origin,
        h,
        *(int(k) for k in shape),
        np.asarray(src),
        np.zeros((0, 3)),
    )
    with pytest.raises(ValueError, match="^fitted walls: the solver needs the vertices and faces"):
        _native.WaveSolver(*args, fitted_walls=True)
    with pytest.raises(ValueError, match="^fitted walls: the faces given are not the BVH's"):
        _native.WaveSolver(*args, fitted_walls=True, vertices=v, faces=BOX_FACES[:10])
    # Far from the origin the lattice widths of the cells stray from 1 by ulp(x) / h: the first
    # place, in powers of ten, where they stray by more than 2^-24 is refused (code 13), a place
    # where they stay within it is fitted.
    h = 0.01
    small = box((0, 0, 0), (0.23, 0.19, 0.29))
    stray = {}
    for far in (1e5, 1e6, 1e7, 1e8):
        v = small + far
        origin, shape = grid_round(v, h)
        cut = fit(v, BOX_FACES, v.mean(axis=0) + 0.00113, h, origin, shape)
        assert cut.cut_stats["refused"] == 0
        stray[far] = max(np.abs(np.asarray(w) - 1.0).max() for w in cut.cut_widths())
    far = min(x for x in stray if stray[x] > 2.0**-24)
    assert stray[1e5] <= 2.0**-24 < stray[far]
    for at, refused in ((1e5, False), (far, True)):
        v = small + at
        origin, shape = grid_round(v, h)
        src = v.mean(axis=0) + 0.00113
        if refused:
            with pytest.raises(
                ValueError,
                match="^fitted walls: the grid lies too far from the "
                "origin of its coordinates for its cells to keep their width",
            ):
                wave(v, BOX_FACES, h, origin, shape, src)
        else:
            assert wave(v, BOX_FACES, h, origin, shape, src).cut_stats["refused"] == 0
    # The same box 1.2e7 m out along x on cells of 0.031 m: its widths stray by between 2^-25 and
    # 2^-24, and it is fitted.
    h = 0.031
    v = small + np.array([1.2e7, 0.0, 0.0])
    origin, shape = grid_round(v, h)
    src = v.mean(axis=0) + 0.00113
    cut = fit(v, BOX_FACES, src, h, origin, shape)
    assert 2.0**-25 < max(np.abs(np.asarray(w) - 1.0).max() for w in cut.cut_widths()) <= 2.0**-24
    assert wave(v, BOX_FACES, h, origin, shape, src).cut_stats["refused"] == 0
    # The same box 2^24 m out along one axis on cells of 0.031 m: that axis's widths stray by
    # between 2^-24 and 2^-23, the others' by less than 2^-24, and the message names the axis.
    for axis in range(3):
        v = small + np.where(np.arange(3) == axis, 16777216.37, 0.0)
        origin, shape = grid_round(v, h)
        src = v.mean(axis=0) + 0.00113
        cut = fit(v, BOX_FACES, src, h, origin, shape)
        stray = [np.abs(np.asarray(w) - 1.0).max() for w in cut.cut_widths()]
        assert 2.0**-24 < stray[axis] <= 2.0**-23
        assert all(stray[b] <= 2.0**-24 for b in range(3) if b != axis)
        with pytest.raises(
            ValueError,
            match="^fitted walls: the grid lies too far from the origin of its coordinates for its "
            f"cells to keep their width: the cell from {'xyz'[axis]} = ",
        ):
            wave(v, BOX_FACES, h, origin, shape, src)


def test_fitted_walls_take_one_speed_of_sound_for_the_whole_grid():
    # With fitted walls the solver refuses a speed of sound per cell, whether it changes from cell
    # to cell or is the same in every cell, given by keyword or in its place among the arguments,
    # with a message starting "fitted walls:" that says what to change. Without it (one speed of
    # sound for the whole grid) the same model is fitted, and walls on cell faces keep the field.
    h = 0.1
    v = box((0, 0, 0), (0.8, 0.7, 0.6))
    origin, shape = grid_round(v, h)
    src = (0.41, 0.33, 0.29)
    n = int(np.prod(shape))
    varying = np.linspace(0.9, 1.0, n, dtype=np.float32)
    message = (
        "^fitted walls: they take one speed of sound for the whole grid, and a speed per cell was "
        "given: leave the speed of each cell out, or take walls on cell faces$"
    )
    for speed in (varying, np.ones(n, np.float32)):
        with pytest.raises(ValueError, match=message):
            wave(v, BOX_FACES, h, origin, shape, src, speed=speed)
    args = (
        _native.Bvh(v, BOX_FACES),
        np.zeros(12, np.int64),
        np.zeros(1),
        np.zeros((0, 3)),
        np.zeros(0, np.int64),
        origin,
        h,
        *(int(k) for k in shape),
        np.asarray(src, float),
        np.zeros((0, 3)),
    )
    # edge_admittance, layer, layer_damping, threads, stencil, then the speed
    with pytest.raises(ValueError, match=message):
        _native.WaveSolver(
            *args, 1.0, 0, 0.0, 1, 7, varying, fitted_walls=True, vertices=v, faces=BOX_FACES
        )
    fitted = wave(v, BOX_FACES, h, origin, shape, src)
    assert fitted.fitted_walls and fitted.layout()["scaled"][0] == 0
    lay = wave(v, BOX_FACES, h, origin, shape, src, fitted=False, speed=varying).layout()
    assert lay["scaled"][0] == 1
    assert np.array_equal(lay["scale"], (varying.astype(float) ** 2).astype(np.float32))


@pytest.mark.parametrize("walls", ["rigid", "resonant"])
def test_the_cpu_steps_a_fitted_layout_as_its_coefficients_say(walls):
    # The L-room with its turned plates (diagonal links on the 7-point grid), rigid, then with
    # resonant walls on two of its materials: 300 steps of the solver against its layout stepped in
    # NumPy with the update's operations in its order, to the bit.
    v, f, mat, h, origin, shape, src = l_room()
    rate = C / (COURANT * h)
    fit_ = Admittance(0.05, np.array([0.6, 0.3]), np.array([2.0, 1.5]), np.array([100.0, 250.0]))
    walls_ = {"rigid": {"beta": (0.0, 0.0, 0.0)}}
    walls_["resonant"] = {
        "beta": (fit_.beta, 0.02, fit_.beta),
        "branches": [(0, fit_.branches(rate)), (2, fit_.branches(rate))],
    }
    solver = wave(v, f, h, origin, shape, src, (0.95, 0.25, 0.3), mat=mat, **walls_[walls])
    lay = solver.layout()
    assert (solver.branch_states > 0) == (walls == "resonant")
    assert (np.asarray(lay["link_direction"]) >= 6).any()
    signal = _drive(rate, h, 300)
    pressure = solver.run(signal)
    assert np.abs(pressure).max() > 0.0
    assert np.array_equal(pressure, _stepped_from_the_layout(solver, signal))


def test_the_fit_alone_places_the_source_and_the_listeners():
    # A source exactly on a free screen and a listener exactly on a wall: the solver feeds and reads
    # the cells and weights CutFit.cut_points() gives (the fit's exact way test), and says the
    # source lies on a face; where the default path's casts read otherwise, the solver still reads
    # the fit's cells; and a point the screen hides from its listener is out of its sight.
    h = 0.125
    room = box((0, 0, 0), (1.5, 1.25, 1.0))
    screen = plate([(0.6, 0.3, 0.2), (0.6, 0.9, 0.2), (0.6, 0.9, 0.8), (0.6, 0.3, 0.8)])
    v, f = join((room, BOX_FACES), screen)
    origin = -h * (2.0 + OFFSET)
    shape = np.ceil((np.array([1.5, 1.25, 1.0]) - origin) / h).astype(int) + 2
    src, probes = np.array([0.6, 0.55, 0.45]), np.array([[0.0, 0.7, 0.5], [0.3, 0.8, 0.4]])
    want = fit(v, f, src, h, origin, shape).cut_points(src.reshape(1, 3), probes)
    assert want["source_on_face"]

    def placed(solver):
        lay = solver.layout()
        volume = np.ones(len(lay["types"]))
        cut = solver.cut_cells()
        if cut:
            volume[cut["special"]] = cut["volume"]
        source = {
            int(c): float(s) * volume[c]
            for c, s in zip(lay["special"], lay["source"], strict=True)
            if s
        }
        start = lay["probe_start"].astype(np.int64)
        cells, weights = lay["probe_cells"].astype(np.int64), lay["probe_weights"]
        read = [
            dict(zip(cells[a:b].tolist(), weights[a:b].tolist(), strict=True))
            for a, b in itertools.pairwise(start)
        ]
        return source, read

    fitted = wave(v, f, h, origin, shape, src, probes)
    source, read = placed(fitted)
    assert fitted.cut_stats["source_on_face"] == 1 and fitted.source_on_face
    assert source.keys() == dict(want["source"]).keys()
    assert max(abs(source[c] - w) for c, w in want["source"]) <= 1e-6
    for got, expect in zip(read, want["probes"], strict=True):
        assert expect and got.keys() == dict(expect).keys()
        assert max(abs(got[c] - w) for c, w in expect) <= 1e-12
    # A listener 5e-10 cells across a screen from a cell's centre: the default path
    # reads no cell there, the fitted solver the cells on its own side that cut_points() gives.
    h = 0.1
    origin, shape = open_grid(h)
    c = origin + (np.array([5, 6, 6]) + 0.5) * h
    xs = c[0] - 1e-12 * h
    screen = plate([(xs, 0.5, 0.5), (xs, 0.7, 0.5), (xs, 0.7, 0.7), (xs, 0.5, 0.7)])
    src = c + np.array([-3.3, 0.21, 0.17]) * h
    p = (c + np.array([-5e-10, 5e-13, 5e-13]) * h).reshape(1, 3)
    want = fit(*screen, src, h, origin, shape).cut_points(src.reshape(1, 3), p)["probes"][0]
    _, read = placed(wave(*screen, h, origin, shape, src, p))
    assert want and read[0].keys() == dict(want).keys()
    assert max(abs(read[0][c] - w) for c, w in want) <= 1e-12
    default = wave(*screen, h, origin, shape, src, p, fitted=False)
    assert placed(default)[1] == [{}] and not default.probe_found(0)
    # The first room with the source off the screen and two listeners of two points each: the
    # first pair on one side of the screen, the second either side of it, so that the segment from
    # the second listener to its other point crosses the screen, which hides that point.
    h = 0.125
    origin = -h * (2.0 + OFFSET)
    shape = np.ceil((np.array([1.5, 1.25, 1.0]) - origin) / h).astype(int) + 2
    ears = np.array([[0.3, 0.5, 0.5], [0.4, 0.5, 0.5], [0.5, 0.6, 0.5], [0.7, 0.6, 0.5]])
    pairs = wave(v, f, h, origin, shape, (0.3, 0.55, 0.45), ears, probes_per_listener=2)
    assert [pairs.probe_in_sight(k) for k in range(4)] == [True, True, True, False]
    assert not pairs.source_on_face and pairs.cut_stats["source_on_face"] == 0


def holed_box(lo, hi, hole):
    """A closed box [lo, hi] whose +x wall has a square hole of side `hole` (metres) in its
    middle."""
    v = box(lo, hi)
    f = [x for k, x in enumerate(BOX_FACES.tolist()) if k not in BOX_SIDE["+x"]]
    x, (y0, z0), (y1, z1) = hi[0], lo[1:], hi[1:]
    yc, zc, s = (y0 + y1) / 2, (z0 + z1) / 2, hole / 2
    outer = [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]]
    inner = [[x, yc - s, zc - s], [x, yc + s, zc - s], [x, yc + s, zc + s], [x, yc - s, zc + s]]
    ring = []
    for i in range(4):
        a, b = i, (i + 1) % 4
        ring += [[a, b, 4 + b], [a, 4 + b, 4 + a]]
    return join((v, np.array(f)), (np.array(outer + inner, float), np.array(ring)))


def test_a_fitted_open_model_reports_its_open_sides():
    # A box with a hole in its +x wall, four cells of margin round it, no layer: the source's air
    # reaches the grid's edge on every side, as the default path's flood says; the closed box does
    # not reach it.
    h = 0.1
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([1.2, 0.9, 0.8])
    origin = lo - h * (2.0 + 4 + OFFSET)
    shape = np.ceil((hi - origin) / h).astype(int) + 2 + 4
    src = (0.43, 0.37, 0.31)
    v, f = holed_box(lo, hi, 0.3)
    fitted, default = (wave(v, f, h, origin, shape, src, fitted=x) for x in (True, False))
    assert fitted.open and default.open
    assert tuple(fitted.open_sides) == tuple(default.open_sides) == (True,) * 6
    closed = wave(box(lo, hi), BOX_FACES, h, origin, shape, src)
    assert not closed.open and tuple(closed.open_sides) == (False,) * 6


def test_fitted_walls_off_change_nothing():
    # fitted_walls=False with the mesh given: every array of the layout equal to the bit to the call
    # without it, and no fit.
    v, f, mat, h, origin, shape, src = l_room()
    plain = wave(
        v, f, h, origin, shape, src, (0.95, 0.25, 0.3), fitted=False, mat=mat, beta=(0.1, 0.2, 0.3)
    )
    given = wave(
        v,
        f,
        h,
        origin,
        shape,
        src,
        (0.95, 0.25, 0.3),
        fitted=False,
        mat=mat,
        beta=(0.1, 0.2, 0.3),
        fitted_walls=False,
        vertices=v,
        faces=f,
    )
    assert_same_bits(plain.layout(), given.layout())
    assert not given.fitted_walls and given.cut_cells() == {} and given.cut_stats == {}


def test_a_fitted_open_model_meets_the_grid_s_edge_as_cell_faces_do():
    # An open-topped box with its walls on cell faces (dyadic), four cells of margin, no layer, the
    # grid's edge of admittance 0.4: every cell next to the edge is stepped as the default path
    # steps it (its links, coef_a, coef_b and coef_inv within 1e-6: no link to the edge, and the
    # edge's admittance over the whole face), and the open sides are the default path's.
    h = 0.125
    lo, hi = np.full(3, 0.75), np.array([2.25, 2.0, 1.75])
    f = np.array([x for k, x in enumerate(BOX_FACES.tolist()) if k not in BOX_SIDE["+z"]])
    v = box(lo, hi)
    origin, shape = np.zeros(3), (24, 22, 20)
    src = (1.31, 1.17, 1.09)
    made = {x: wave(v, f, h, origin, shape, src, fitted=x, edge=0.4) for x in (True, False)}
    lays = {x: s.layout() for x, s in made.items()}
    shell = np.array(sorted(next_to_edge(shape)))
    assert np.all(lays[True]["types"][shell] == 2) and np.all(lays[False]["types"][shell] == 2)
    (got, _), (want, _) = updates(lays[True], shell), updates(lays[False], shell)
    assert np.abs(got[:, :21] - want[:, :21]).max() <= 1e-6
    assert np.all(got[:, 19] < 1.0)  # the edge's admittance
    assert made[True].open and tuple(made[True].open_sides) == tuple(made[False].open_sides)


def test_fitted_walls_refuse_with_the_fit_s_words():
    # With fitted walls the fit alone decides the model, and its refusals come first: a box reaching
    # 0.3 cells into a 4-cell layer, and a source in a pocket holding no cell centre, get the fit's
    # messages, not the default path's.
    h = 0.1
    origin, shape = open_grid(h, 1.6)
    corner = origin + (4.7 + np.array([0.0, 0.1, 0.2])) * h
    v = box(corner, corner + 0.5)
    src = (1.05, 1.05, 1.05)
    with pytest.raises(ValueError, match="^the model reaches into the matched layer$"):
        wave(v, BOX_FACES, h, origin, shape, src, layer=4, fitted=False)
    with pytest.raises(ValueError, match="^fitted walls: the model reaches into the matched layer"):
        wave(v, BOX_FACES, h, origin, shape, src, layer=4)
    origin, shape = open_grid(h)
    lo = origin + (np.array([5, 5, 5]) + 0.5) * h + 0.05 * h
    pocket = box(lo, lo + 0.4 * h)
    with pytest.raises(ValueError, match="^the source sees no cell of the wave grid$"):
        wave(pocket, BOX_FACES, h, origin, shape, lo + 0.2 * h, fitted=False)
    with pytest.raises(ValueError, match="^fitted walls: the source lies in a closed space"):
        wave(pocket, BOX_FACES, h, origin, shape, lo + 0.2 * h)


def gate_room(dx):
    """A room shaped as a tetrahedron with three walls along the grid and a slanted face, moved by
    dx along x, on a dyadic grid; its slanted face leaves a corner of air of legs
    a e, b e and c e in the cube above cell (10, 9, 4), e = 1 - (q - P) . (1/a, 1/b, 1/c)."""
    h, a, b, c = 0.125, 3.0, 2.6, 0.9
    origin = np.full(3, -0.3125)
    shape = (int(np.ceil((a + 0.7) / h)), int(np.ceil((b + 0.7) / h)), int(np.ceil((c + 0.7) / h)))
    p = np.array([dx, 0.0, 0.0])
    v = np.array([p, p + [a, 0, 0], p + [0, b, 0], p + [0, 0, c]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    return v, f, h, origin, shape, p + np.array([0.35, 0.3, 0.15])


def test_the_sliver_gate_moves_one_link_by_a_stated_step():
    # The room of gate_room() at two adjacent doubles of its shift: the corner of air that cell o =
    # (10, 9, 4) owns through its +z face touches o's cube over A_s = 9.999999999996e-10 cell faces
    # below and 1.0000000000027e-09 above, on either side of the gate of 1e-9 below which the fit
    # takes no sliver. Below, o's link through its -z face is whole; above, it is scaled by w(1 +
    # f), f = V_s / A_s = c e / (3 h) from the geometry, 0.99999919936: a jump of 8.0e-7 of its
    # weight, on both of its cells. sliver_near_gate counts the contact on both sides; nothing else
    # in the layout changes but those two links and the two cells' coef_a.
    below, above = -0.08332732852756568, -0.08332732852756566
    made = {}
    for dx in (below, above):
        v, f, h, origin, shape, src = gate_room(dx)
        made[dx] = wave(v, f, h, origin, shape, src)
    nx, ny = shape[0], shape[1]
    o = 10 + 9 * nx + 4 * nx * ny
    m = o - nx * ny  # o's neighbour across its -z face
    lays = {dx: s.layout() for dx, s in made.items()}
    (lo, _), (hi, _) = updates(lays[below], [o, m]), updates(lays[above], [o, m])
    a, b, c = 3.0, 2.6, 0.9
    q = origin + h * np.array([10, 9, 5])
    e = 1.0 - ((q[0] - above) / a + q[1] / b + q[2] / c)
    factor = sliver_factor(c * e / (3 * h))
    assert factor == pytest.approx(0.99999919936, abs=1e-11)
    volume = {}
    for dx, s in made.items():
        cut = s.cut_cells()
        volume[dx] = dict(zip(cut["special"].tolist(), cut["volume"].tolist(), strict=True))
    for n, (cell, k) in enumerate(((o, 4), (m, 5))):
        whole = lowered(G / volume[below].get(cell, 1.0))  # a plain cell's V is 1
        scaled = lowered(G * factor / volume[above].get(cell, 1.0))
        assert lo[n, k] == whole and abs(hi[n, k] - scaled) <= 1e-7 * scaled
        assert (lo[n, k] - hi[n, k]) / lo[n, k] == pytest.approx(8.0e-7, abs=1.5e-7)
        assert hi[n, 18] - lo[n, 18] == pytest.approx(lo[n, k] - hi[n, k], abs=1e-7)
    for s in made.values():
        assert s.cut_stats["sliver_near_gate"] == 1
    assert made[below].cut_stats["sliver_links"] + 1 == made[above].cut_stats["sliver_links"]
    # Every other coefficient of every air cell the same to the bit; m is special above only.
    air = np.flatnonzero(lays[below]["types"] != 0)
    assert np.array_equal(air, np.flatnonzero(lays[above]["types"] != 0))
    (all_lo, _), (all_hi, _) = (updates(lays[x], air) for x in (below, above))
    same = all_lo == all_hi
    for cell, k in ((o, 4), (m, 5), (o, 18), (m, 18)):
        same[np.searchsorted(air, cell), k] = True
    assert same.all()
    for key in ("probe_cells", "probe_weights", "state_weight"):
        assert np.array_equal(lays[below][key], lays[above][key]), key


def speck_room(hole):
    """A closed box 1.5 m on a side on a dyadic grid with two parallel membranes 0.3 cells apart
    across x, the first in the lattice plane between cells 6 and 7 with a square hole of side
    `hole` (metres) centred on the face between cells o = (6, 8, 8) and (7, 8, 8), the second
    without one."""
    h, size = 0.125, 1.5
    origin = np.full(3, -0.3125)
    shape = (int((size + 0.625) / h) + 1,) * 3
    x0 = origin[0] + 7 * h
    yc, zc = origin[1] + 8.5 * h, origin[2] + 8.5 * h
    lo, hi = 0.5, 1.0
    s = hole / 2
    outer = [[x0, lo, lo], [x0, hi, lo], [x0, hi, hi], [x0, lo, hi]]
    inner = [[x0, yc - s, zc - s], [x0, yc + s, zc - s], [x0, yc + s, zc + s], [x0, yc - s, zc + s]]
    ring = []
    for i in range(4):
        a, b = i, (i + 1) % 4
        ring += [[a, b, 4 + b], [a, 4 + b, 4 + a]]
    x1 = x0 + 0.3 * h
    v, f = join(
        (box((0, 0, 0), (size,) * 3), BOX_FACES),
        (np.array(outer + inner, float), np.array(ring)),
        plate([(x1, lo, lo), (x1, hi, lo), (x1, hi, hi), (x1, lo, hi)]),
    )
    return v, f, h, origin, shape, np.array([0.25, 1.2, 1.2])


def test_a_speck_nearly_closes_its_owner_s_opposite_link():
    # The chamber between the membranes in the cube of (7, 8, 8), 0.3 cells^3, holds no centre and
    # is owned by o through the hole: a sliver of thickness f = V_s / A_s = 0.3 / (hole / h)^2 cells
    # through o's +x face. As the hole shrinks the link through o's -x face is scaled, on both of
    # its cells, by w(1 + f), down to about 1e-8 just above the gate (a hole of 3.953e-6 m, A_s just
    # over 1e-9 cell faces), and is whole just below it (3.952e-6 m). cut_stats counts the links a
    # sliver thicker than a cell scales (sliver_thick) as CutFit's rows give them: o's link from a
    # hole of 0.05 m (f = 1.9) down to the gate, not from a hole of 0.08 m (f = 0.73), whose grid
    # has slivers between half a cell and a cell thick.
    for hole in (0.08, 0.05, 1e-3, 1e-5, 3.953e-6, 3.952e-6):
        v, f, h, origin, shape, src = speck_room(hole)
        solver = wave(v, f, h, origin, shape, src)
        nx = shape[0]
        o = 6 + 8 * nx + 8 * nx * nx
        m = o - 1
        _, touched = rules_on_rows(fit(v, f, src, h, origin, shape), shape)
        thick = sum(max(x) > 1.0 for x in touched.values())
        assert solver.cut_stats["sliver_thick"] == thick, hole
        area = (hole / h) ** 2
        assert (max(touched.get((m, o), [0.0])) > 1.0) == (area > 1e-9 and 0.3 / area > 1.0)
        if hole == 0.08:
            assert thick == 0 and any(0.5 < max(x) <= 1.0 for x in touched.values())
        (got, _) = updates(solver.layout(), [o, m])
        cut = solver.cut_cells()
        volume = dict(zip(cut["special"].tolist(), cut["volume"].tolist(), strict=True))
        factor = sliver_factor(0.3 / area) if area > 1e-9 else 1.0
        for n, (cell, k) in enumerate(((o, 0), (m, 1))):
            measured = got[n, k] * volume.get(cell, 1.0) / (G * (1.0 - MARGIN))
            assert measured == pytest.approx(factor, rel=2e-7), (hole, cell)
        if hole == 3.953e-6:
            assert factor < 1.1e-8
    # A third membrane 0.3 cells inside the cube of (4, 8, 8) gives m a sliver of 0.3 cells
    # beyond its -x face as well: the link between m and o is then scaled by both cells' slivers,
    # o's thicker than a cell (from a hole of 0.05 m) and m's thinner, and counts as one.
    v, f, h, origin, shape, src = speck_room(0.05)
    x2 = origin[0] + 4.7 * h
    v, f = join((v, f), plate([(x2, 0.5, 0.5), (x2, 1.0, 0.5), (x2, 1.0, 1.0), (x2, 0.5, 1.0)]))
    nx = shape[0]
    o = 6 + 8 * nx + 8 * nx * nx
    m = o - 1
    _, touched = rules_on_rows(fit(v, f, src, h, origin, shape), shape)
    assert len(touched[(m, o)]) == 2 and min(touched[(m, o)]) < 1.0 < max(touched[(m, o)])
    thick = sum(max(x) > 1.0 for x in touched.values())
    assert wave(v, f, h, origin, shape, src).cut_stats["sliver_thick"] == thick


# ---- Group D: fitted walls through wave_responses (walls="fitted") ------------------------------
#
# References: the fit's refusal words; the mesh's faces less its openings', selected here from the
# model; and the run on the model with its openings' faces taken out of its file.


def obj_room(path, v, groups, source, listener, materials, assignments):
    """A room from an OBJ file written at `path`: the vertices v (written as the doubles they are)
    and `groups`, (name, faces) in order; one source and one listener."""
    lines = [f"v {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in np.asarray(v, float)]
    for name, faces in groups:
        lines.append(f"g {name}")
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")
    data = {
        "geometry": {"file": str(path)},
        "materials": materials,
        "default_material": "m",
        "assignments": assignments,
        "sources": [{"name": "s", "position": list(source)}],
        "listeners": [{"name": "l", "position": list(listener)}],
    }
    return Room.from_scene(Scene.model_validate(data))


def fitted_responses(room, said=None):
    """wave_responses of the room from its source to its listener, 63 Hz band, 0.1 s, with
    fitted walls; and the progress lines, in `said` when given."""
    said = [] if said is None else said
    responses, info = wave_responses(
        TraceModel.from_room(room),
        np.array(room.scene.sources[0].position),
        [room.scene.listeners[0].position],
        ["l"],
        max_band_hz=63,
        length_s=0.1,
        sample_rate=48_000,
        speed_of_sound=C,
        walls="fitted",
        say=said.append,
    )
    return responses, info, said


HARD = {"m": {"absorption": [0.1] * 8, "scattering": 0.0}}


def test_wave_responses_raise_the_fit_s_refusals(tmp_path):
    # The T-junction room of A7, 8 times as large, its T vertex 1.6 mm off the diagonal: on the
    # 63 Hz band's 7-point grid (cells of 0.40 m) the slit it opens is narrower than a hundredth of
    # a cell, and the fit refuses it, before any step. Welded, the room runs.
    source, listener = np.array([0.47, 0.39, 0.42]) * 8, np.array([0.7, 0.5, 0.3]) * 8
    for offset in (0.0, 2e-4):
        v, f = t_room(offset, 0)
        room = obj_room(tmp_path / "t.obj", v * 8, [("walls", f)], source, listener, HARD, {})
        if offset == 0.0:
            _, info, _ = fitted_responses(room)
            assert info["wave_fitted_walls"] == 1.0 and "wave_open" not in info
            continue
        said = []
        with pytest.raises(ValueError, match=r"^fitted walls: the air in the cell centred at") as e:
            fitted_responses(room, said)
        assert "Weld the faces along it" in str(e.value)
        assert said == []


def test_wave_responses_give_the_fit_the_faces_of_the_bvh(tmp_path, monkeypatch):
    # A box whose +x wall has a 1 m square opening (transmission 1), its two faces written between
    # the other walls' and the +x wall's, three materials; its corners lie off the origin, at
    # coordinates single precision does not hold, so vertices rounded on the way to the fit are
    # seen. Every solver made (the closed grid tried
    # first, then the one with the layer) is given the mesh's vertices and its faces less the
    # opening's, in the mesh's order, the order the wave grid's walls are found in; and the fitted
    # run is, to the bit, the run on the box with the opening's faces left out of the file. The
    # opening is open: the run has the matched layer, and its wave_fit_s is the two fits' seconds.
    given, seconds = [], []

    class Kept(_native.WaveSolver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            given.append(kwargs)
            seconds.append(self.cut_stats["seconds"])

    monkeypatch.setattr(_native, "WaveSolver", Kept)
    lo, hi = np.array([0.1, 0.1, 0.1]), np.array([4.1, 3.1, 2.6])
    v, f = holed_box(lo, hi, 1.0)
    assert not np.array_equal(v.astype(np.float32), v)
    walls, ring = f[:10], f[10:]
    opening = np.array([[12, 13, 14], [12, 14, 15]])  # the hole's corners, vertices 12 to 15
    materials = {
        **HARD,
        "panel": {"absorption": [0.4] * 8, "scattering": 0.0},
        "gap": {"absorption": [1.0] * 8, "scattering": 0.0, "transmission": [1.0] * 8},
    }
    source, listener = (1.13, 1.37, 1.21), (3.02, 2.11, 1.43)
    runs = []
    for with_opening in (True, False):
        groups = [("walls", walls[:5]), ("hole", opening), ("walls", walls[5:]), ("front", ring)]
        if not with_opening:
            del groups[1]
        assignments = {"hole": "gap", "front": "panel"}
        room = obj_room(tmp_path / "box.obj", v, groups, source, listener, materials, assignments)
        model = TraceModel.from_room(room)
        assert model.transparent.sum() == (2 if with_opening else 0)
        given.clear()
        seconds.clear()
        runs.append(fitted_responses(room)[:2])
        assert len(given) == 2 and runs[-1][1]["wave_fit_s"] == seconds[0] + seconds[1]
        for kwargs in given:
            assert kwargs["fitted_walls"] is True
            assert np.array_equal(kwargs["vertices"], room.mesh.vertices)
            assert np.array_equal(kwargs["faces"], room.mesh.faces[~model.transparent])
    (opened, info), (left_out, reference) = runs
    assert info["wave_open"] == 1.0 and info["wave_fitted_walls"] == 1.0
    assert info["wave_cells"] == reference["wave_cells"]
    assert np.array_equal(opened[0].signals, left_out[0].signals)


# ---- The checks of validation/wave_walls.py ----------------------------------------------------


def _wave_walls_script():
    """validation/wave_walls.py, imported by its path."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "validation" / "wave_walls.py"
    spec = importlib.util.spec_from_file_location("wave_walls_checked", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fitted_walls_validation_s_checks_fail_past_the_stated_figures() -> None:
    """Each check of validation/wave_walls.py passes a figure just inside its limit and
    fails one just outside: the limits are the figures stated for fitted walls to the
    precision they are stated in (0.05 % along the grid, 0.38 % up to 14 degrees, 4.3 %
    beyond, 0.04 dB along the grid and 0.18 dB slanted), the dispersion's 1e-5, the 7-point
    relation's 1e-9 from its closed form, and the controls on cell faces (at least 1 % and
    1 dB); the box's levels within 1.82 dB; the L-shaped room's reference, and fitted walls
    no further off than cell faces; the duct's 0.012, 0.1, 0.02 and 0.0278; the decay's
    4 %; the branch weights' 1e-6; and the long runs' criteria, so that no check can pass
    whatever it is given."""
    script = _wave_walls_script()

    def outcome(checks: list[dict]) -> list[bool]:
        return [c["passed"] for c in checks]

    def faces(off: float) -> dict:
        return {
            "kind": "faces",
            "off_relation": 100 * off,
            "walls": "fitted",
            "stencil": 7,
            "grid_band_hz": 125,
        }

    assert outcome(script.dispersion_checks([faces(0.9e-5), faces(1.1e-5)])) == [True, False]
    # On the 7-point stencil the relation is also checked against its closed form.
    written = [{**faces(0.0), "relation_off_closed_form": off} for off in (0.9e-9, 1.1e-9)]
    assert outcome(script.dispersion_checks(written)) == [True, True, True, False]

    def turned(walls: str, yaw: float, pitch: float, worst: float) -> dict:
        return {
            "walls": walls,
            "grid_band_hz": 125,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "min_percent": -0.001,
            "max_percent": worst,
        }

    limits = [((0.0, 0.0), 0.055), ((14.0, 0.0), 0.385), ((22.5, 11.0), 4.35), ((45.0, 0.0), 4.35)]
    rows = [
        turned("fitted", *turn, x) for turn, limit in limits for x in (0.99 * limit, 1.01 * limit)
    ]
    assert outcome(script.turned_checks(rows)) == [True, False] * len(limits)
    for worst, passed in ((1.01, True), (0.99, False)):
        control = [turned("cell_faces", 45.0, 0.0, worst), turned("cell_faces", 0.0, 0.0, 0.5)]
        assert outcome(script.turned_checks(control)) == [passed]

    converged = [{"stencil": 7, "grid_band_hz": 125, "moves_db": x} for x in (0.0099, 0.0101)]
    assert outcome(script.levels_checks(converged)) == [True, False]

    def wall(walls: str, angle: float, share: float, error: float, stencil: int = 7) -> dict:
        return {
            "walls": walls,
            "stencil": stencil,
            "angle_deg": angle,
            "share": share,
            "rms_db": [error, 0.0, 0.0],
        }

    walls = [
        wall("fitted", 0.0, -0.48, 0.0449),
        wall("fitted", 0.0, 0.25, 0.0451),
        wall("fitted", 14.0, 0.0, 0.1849),
        wall("fitted", 8.0, 0.0, 0.1851),
        wall("fitted", 25.0, 0.0, 9.0),  # stated nothing: no check
        wall("cell_faces", 0.0, -0.48, 1.01),
        wall("cell_faces", 0.0, -0.48, 0.99),
        wall("cell_faces", 0.0, -0.48, 0.5, stencil=19),  # the 7-point stencil's alone
    ]
    assert outcome(script.wall_checks(walls)) == [True, False, True, False, True, False]

    def cost(walls: str, scheme: str, step_ms: float, fit_s) -> dict:
        return {
            "model": "box",
            "walls": walls,
            "scheme": scheme,
            "step_ms": step_ms,
            "fit_s": fit_s,
        }

    plain = cost("cell_faces", "standard", 1.0, None)
    within, over = cost("fitted", "auto", 2.39, 2.89), cost("fitted", "auto", 2.41, 2.91)
    assert outcome(script.cost_checks([plain, within])) == [True, True]
    assert outcome(script.cost_checks([plain, over])) == [False, False]

    # The turned box's worst over the scanned placements, and the box moved along the grid.
    scans = [
        {
            "grid_band_hz": 125,
            "yaw_deg": 14.0,
            "pitch_deg": 0.0,
            "placements": [0] * 3,
            "worst_percent": x,
        }
        for x in (0.384, 0.386)
    ]
    assert outcome(script.scan_checks(scans)) == [True, False]
    moved = [
        {
            "walls": walls,
            "kind": "shift",
            "case": "moved",
            "grid_band_hz": 125,
            "lowest": {"min_percent": -0.001, "max_percent": x},
        }
        for walls, x in (("fitted", 0.054), ("fitted", 0.056), ("cell_faces", 9.0))
    ]
    assert outcome(script.moved_checks(moved)) == [True, False]

    def levels(walls: str, worst: float, band: int = 63) -> dict:
        db = {b: [0.0, worst] for b in (63, 125, 250)[: (63, 125, 250).index(band) + 1]}
        return {
            "walls": walls,
            "stencil": 7,
            "grid_band_hz": band,
            "100_ms": db,
            "300_ms": {b: [0.0] for b in db},
        }

    assert outcome(script.box_levels_checks([levels("fitted", 1.82)])) == [True, True]
    assert outcome(script.box_levels_checks([levels("fitted", -1.83)])) == [False, True]
    assert outcome(script.box_levels_checks([levels("cell_faces", 1.83)])) == [True]
    assert outcome(script.box_levels_checks([levels("cell_faces", 1.82)])) == [False]

    def l_room(walls: str, f: float, shift=(0.3, 0.2, 0.4)) -> dict:
        return {
            "walls": walls,
            "grid_band_hz": 125,
            "shift_cells": list(shift),
            "lowest_hz": f,
            "plan_on_cell_hz": 62.0,
            "continuum_hz": 63.0,
        }

    faces = [l_room("cell_faces", 62.0 * (1 + 0.9e-5)), l_room("cell_faces", 62.0 * (1 + 1.1e-5))]
    assert outcome(script.lroom_checks(faces)) == [True, False]
    # Fitted walls no further from the continuum (63 Hz) than cell faces (1.586 % below),
    # on either side of it.
    for f, held in ((63.99, True), (64.01, False), (62.01, True), (61.99, False)):
        assert outcome(script.lroom_checks([faces[0], l_room("fitted", f)])) == [True, held]

    def duct(beta: float, end: float, off: float) -> dict:
        return {"walls": "fitted", "beta": beta, "end_cells": end, "off": off}

    def branch(end: float, off: float, written: float = 0.0) -> dict:
        return {
            "walls": "fitted",
            "end_cells": end,
            "off_grid_theory": off,
            "admittance_off_written": written,
        }

    def decay(ratio: float) -> dict:
        return {"walls": "fitted", "grid_band_hz": 125, "modes": [{"ratio": 1.0}, {"ratio": ratio}]}

    def weights(off: float, walls: str = "fitted") -> dict:
        return {"walls": walls, "grid_band_hz": 125, "off": off}

    found = {
        "duct": [duct(0.5, 0.25, 0.0119), duct(0.5, 0.48, -0.0121), duct(1.0, 0.0, 0.099)]
        + [duct(1.0, 0.48, 0.101)],
        "duct_branch": [branch(0.0, 0.0199, 0.9e-12), branch(0.25, 0.0201)]
        + [branch(-0.48, 0.0277), branch(-0.48, 0.0279)],
        "decay": [decay(0.961), decay(1.041)],
        "branch_weights": [weights(0.9e-6), weights(-1.1e-6), weights(0.5, "cell_faces")],
    }
    duct_ends = [True, False, True, False]
    branch_ends = [True, True, False, True, False]
    expected = [*duct_ends, *branch_ends, True, False, True, False]
    assert outcome(script.absorbing_checks(found)) == expected
    assert outcome(script.absorbing_checks({**found, "duct": []})[:2]) == [True, True]
    written = {"duct": [], "duct_branch": [branch(0.0, 0.0, 1.1e-12)], "decay": []}
    assert outcome(script.absorbing_checks(written)) == [False, True]

    def long_run(case: str, oscillating, full=None, **more) -> dict:
        full = oscillating if full is None else full
        return {
            "model": "room",
            "case": case,
            "finite": True,
            "oscillating": list(oscillating),
            "full": list(full),
            **more,
        }

    steady = [1.0] + [0.5] * 19
    bounded = {"least_coefficient": 0.0, "row_sum": 2.0, "largest_mu": 3.99}
    rigid = long_run("rigid", steady, **bounded)
    assert outcome(script.long_checks([rigid])) == [True] * 7
    over = {"least_coefficient": -1e-12, "row_sum": 2.0 + 1e-12, "largest_mu": 4.01}
    grown = long_run("rigid", [*steady[:-1], 0.5 * 1.051], **over)
    assert outcome(script.long_checks([grown])) == [True, True, False, True, False, False, False]
    swollen = long_run("rigid", [1.0, 0.5, *[0.5] * 9, 0.56, *[0.5] * 8])
    assert outcome(script.long_checks([swollen])) == [True, True, True, False]
    decaying = [1.0, 1e-3, 1e-4, 1e-5, 1e-6, *[1e-7] * 5, *[1.09e-7] * 10]
    assert outcome(script.long_checks([long_run("strong", decaying)])) == [True, True, True]
    level = [1.0, 1e-3, 1e-4, 1e-5, 1e-6, *[1e-6] * 5, *[1.05e-6] * 10]
    assert outcome(script.long_checks([long_run("strong", level)])) == [True, False, True]
    regrowing = [*decaying[:10], *[1.2e-7] * 10]
    assert outcome(script.long_checks([long_run("open", regrowing)])) == [True, True, False]
    floor = [*decaying[:10], *[1.1e-6] * 10]
    assert outcome(script.long_checks([long_run("absorbing", floor)])) == [True, False, False]
