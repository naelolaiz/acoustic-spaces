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

import math
import re
from fractions import Fraction

import numpy as np
import pytest
from test_wave import _solver

from acoustic_spaces import _native

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
    assert re.search(r"cell centred at \([-0-9.]+, [-0-9.]+, [-0-9.]+\) m", st["refusal"])
    assert st["refusal"].endswith(
        "Move the model away from the grid's edge, or enlarge the margin round it."
    )


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
        return v, f, h, origin, shape, np.array([1.53, 1.47, 1.21])
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
