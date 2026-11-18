"""Fitted walls: four small cases whose rows are expected from the geometry, in exact rationals,
apart from the fit. They see the errors that no self-check of the fit can see: a material given to
the wrong side of a face (cases A and B), a leaf's volume given to another region of its cube (A),
the direction of a sliver (A and C), the second-best owner taken for a region (C), and the
owner taken in a tie of the owner rule (D).

Units: the grid's cell is h = 0.125 m and its origin 0, so a coordinate of c cells is c h metres,
exactly. The grid has 10 cells per axis (cells 0 and 9 are the grid's edge and never air). Every
room lies in [1.25, 8.75] cells per axis. Cell (i, j, k) has index i + 10 j + 100 k.

  A  "panel": a box room [1.25, 8.75]^3 of material 0 and, inside it, two coincident faces back to
     back on the plane x = 5.25 over y, z in [3.25, 6.75]: the face whose normal is +x has material
     1, the face whose normal is -x material 2. Cells (5, j, k), j, k in {4, 5}, are split by the
     panel; their part x < 5.25 holds no centre and goes to the cell (4, j, k) across the square
     x = 5, a sliver of 0.25 cells through +x.
  B  "membrane": the same room with the two faces on the grid plane x = 5. The squares between
     cells (4, j, k) and (5, j, k) carry the walls: the low side shows material 2, the high side 1.
  C  "tilted": a prism room, x in [1.25, 8.75], y >= 1.25, x + 2 y <= 15.875, z in [1.25, 8.75].
     A cell whose centre lies beyond the tilted wall but whose corner holds air has two offers, from
     (i, j - 1, k) through the square y = j (open length d) and from (i - 1, j, k) through the
     square x = i (open length d / 2); the owner is the first, and the second-best owner differs.
  D  "ties": a prism room, x >= 1.25, y >= 1.25, x + y <= 9.5, z in [1.25, 8.75]: a wall at 45
     degrees to the grid. A cell with i + j = 9 (2 <= i <= 7) has its centre beyond the wall and a
     triangle of air in its corner, of legs 0.5; its -x neighbour (i - 1, j, k) and its -y
     neighbour (i, j - 1, k) each offer it an open length of 0.5, an exact tie, and the triangle
     goes to the lower cell index, the -y neighbour.
"""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

from acoustic_spaces import _native

H = 0.125
N = 10
LO, HI = Fr(5, 4), Fr(35, 4)


def idx(i, j, k):
    return i + N * j + N * N * k


def box_mesh(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]
    f = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    return v, f


def panel(v, f, x, a, b):
    """Two coincident faces back to back on the plane x over y, z in [a, b]: two triangles with
    normal +x, then two with normal -x, on the same four vertices."""
    n = len(v)
    v = v + [[x, a, a], [x, b, a], [x, b, b], [x, a, b]]
    plus = [[n, n + 1, n + 2], [n, n + 2, n + 3]]  # (0, 1, 0) x (0, 1, 1) = +x
    minus = [[n, n + 2, n + 1], [n, n + 3, n + 2]]
    return v, f + plus + minus


def interval(lo, hi, i):
    """The length of [lo, hi] inside cell i of one axis, exact."""
    return max(Fr(0), min(hi, Fr(i + 1)) - max(lo, Fr(i)))


def add_whole(rows, cubes, materials):
    """The rows of the uncut cells next to a cube (a cell that a face meets): the fit gives each
    such cell, unless it is on the grid's edge, a region of its whole volume, which holds its centre
    and so is its own (the rows of the room's air cells with nothing cut that touch no cube are not
    rows)."""
    steps = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for i, j, k in cubes:
        for d in steps:
            c = (i + d[0], j + d[1], k + d[2])
            if c in cubes or min(c) < 1 or max(c) > N - 2 or idx(*c) in rows:
                continue
            rows[idx(*c)] = {"air": Fr(1), "wall": [Fr(0)] * (materials + 1), "sliver": [Fr(0)] * 6}


def box_cubes():
    return {
        (i, j, k)
        for i in range(1, 9)
        for j in range(1, 9)
        for k in range(1, 9)
        if {i, j, k} & {1, 8}
    }


def room_box_expect(materials):
    """The rows of the box room alone: every cell with an index 1 or 8 on some axis, its air and its
    walls."""
    rows = {}
    for k in range(1, 9):
        for j in range(1, 9):
            for i in range(1, 9):
                if not ({i, j, k} & {1, 8}):
                    continue
                L = [interval(LO, HI, t) for t in (i, j, k)]
                wall = Fr(0)
                for a, t in enumerate((i, j, k)):
                    if t in (1, 8):
                        wall += L[(a + 1) % 3] * L[(a + 2) % 3]
                walls = [Fr(0)] * (materials + 1)
                walls[0] = wall
                rows[idx(i, j, k)] = {
                    "air": L[0] * L[1] * L[2],
                    "wall": walls,
                    "sliver": [Fr(0)] * 6,
                }
    return rows


def expect_A():
    rows = room_box_expect(3)
    P0, P1 = Fr(13, 4), Fr(27, 4)
    for k in range(3, 7):
        for j in range(3, 7):
            a = interval(P0, P1, j) * interval(P0, P1, k)
            r = {"air": Fr(1), "wall": [Fr(0)] * 4, "sliver": [Fr(0)] * 6}
            if j in (4, 5) and k in (4, 5):
                r["air"] = Fr(3, 4)  # x in [5.25, 6]: the part that holds the centre 5.5
                r["wall"][1] = Fr(
                    1
                )  # the + side of the panel's plane shows the face whose normal is +x
                o = {
                    "air": Fr(1) + Fr(1, 4),
                    "wall": [Fr(0), Fr(0), Fr(1), Fr(0)],
                    "sliver": [Fr(0)] * 6,
                }
                o["sliver"][1] = Fr(1, 4) / Fr(
                    1
                )  # 0.25 cells of (5, j, k) through the square x = 5, direction +x
                rows[idx(4, j, k)] = o
            else:
                r["wall"][1] = a  # the panel's edge lies in the cell: one region sees both sides
                r["wall"][2] = a
            rows[idx(5, j, k)] = r
    add_whole(rows, box_cubes() | {(5, j, k) for j in range(3, 7) for k in range(3, 7)}, 3)
    return rows


def expect_B():
    rows = room_box_expect(3)
    P0, P1 = Fr(13, 4), Fr(27, 4)
    for k in range(3, 7):
        for j in range(3, 7):
            a = interval(P0, P1, j) * interval(P0, P1, k)
            lo = {"air": Fr(1), "wall": [Fr(0), Fr(0), a, Fr(0)], "sliver": [Fr(0)] * 6}
            hi = {"air": Fr(1), "wall": [Fr(0), a, Fr(0), Fr(0)], "sliver": [Fr(0)] * 6}
            rows[idx(4, j, k)] = lo
            rows[idx(5, j, k)] = hi
    # the faces on the grid plane x = 5 meet the closed boxes of the cells on both sides
    add_whole(
        rows, box_cubes() | {(i, j, k) for i in (4, 5) for j in range(3, 7) for k in range(3, 7)}, 3
    )
    return rows


class Prism:
    """A prism room: x >= 1.25, x <= 8.75, y >= 1.25 and a x + b y <= c (the tilted wall, a, b > 0),
    z in [1.25, 8.75], all of material 0. No grid corner lies on the tilted wall."""

    def __init__(self, a, b, c):
        self.a, self.b, self.c = Fr(a), Fr(b), Fr(c)

    def xy(self):
        """The room's section, counterclockwise."""
        a, b, c = self.a, self.b, self.c
        if a * HI + b * LO < c:
            return [(LO, LO), (HI, LO), (HI, (c - a * HI) / b), (LO, (c - a * LO) / b)]
        return [(LO, LO), ((c - b * LO) / a, LO), (LO, (c - a * LO) / b)]

    def reaches_hi(self):
        return self.a * HI + self.b * LO < self.c

    def cell_xy(self, i, j):
        poly = [(Fr(i), Fr(j)), (Fr(i + 1), Fr(j)), (Fr(i + 1), Fr(j + 1)), (Fr(i), Fr(j + 1))]
        for a, b, c in ((-1, 0, -LO), (1, 0, HI), (0, -1, -LO), (self.a, self.b, self.c)):
            poly = clip(poly, Fr(a), Fr(b), Fr(c)) if poly else poly
        return poly

    def seg_len_x(self, i, j):
        """The open length of the square x = i over y in [j, j + 1] (the room's section), exact."""
        if not (LO < i < HI):
            return Fr(0)
        top = (self.c - self.a * i) / self.b
        return interval(LO, top, j)

    def seg_len_y(self, i, j):
        """The open length of the square y = j over x in [i, i + 1]."""
        if not (LO < j):
            return Fr(0)
        right = min(HI, (self.c - self.b * j) / self.a)
        return interval(LO, right, i)

    def on_tilt(self, p):
        return self.a * p[0] + self.b * p[1] == self.c


def clip(poly, a, b, c):
    """The part of a convex polygon (exact points) where a x + b y <= c."""
    out = []
    m = len(poly)
    for t in range(m):
        p, q = poly[t], poly[(t + 1) % m]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            s = fp / (fp - fq)
            out.append((p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])))
    return out


def area2(poly):
    s = Fr(0)
    for t in range(len(poly)):
        p, q = poly[t], poly[(t + 1) % len(poly)]
        s += p[0] * q[1] - q[0] * p[1]
    return abs(s) / 2


PRISM_C = Prism(1, 2, Fr(127, 8))  # x + 2 y <= 15.875
PRISM_D = Prism(1, 1, Fr(19, 2))  # x + y <= 9.5


def expect_prism(pr):
    """Regions: each cell's part of the room (convex, so one region). A cell whose centre is inside
    owns its own; the others take owners round by round: each round, a region without owner gets the
    owner whose offers (open area of the squares to regions owned before the round) sum to the most,
    the lowest cell index where two owners' sums are equal (a tie). Returns the rows and, for each
    region with more than one offer, its owner, the second-best owner and their sums."""
    zl = {k: interval(LO, HI, k) for k in range(N)}
    reg = {}
    for j in range(N):
        for i in range(N):
            poly = pr.cell_xy(i, j)
            if poly and area2(poly) > 0:
                reg[(i, j)] = poly
    centre_in = {}
    for i, j in reg:
        x, y = Fr(2 * i + 1, 2), Fr(2 * j + 1, 2)
        centre_in[(i, j)] = LO < x < HI and LO < y and pr.a * x + pr.b * y < pr.c

    def crossed(i, j):
        # the tilted wall passes through the cell's square (no grid corner lies on it)
        v = [pr.a * (i + di) + pr.b * (j + dj) for di in (0, 1) for dj in (0, 1)]
        return min(v) < pr.c < max(v) and i <= 8 and j >= 1

    x_wall = (1, 8) if pr.reaches_hi() else (1,)
    cubes = {
        (i, j, k)
        for i in range(1, 9)
        for j in range(1, 9)
        for k in range(1, 9)
        if (i, j) in reg and (i in x_wall or j == 1 or k in (1, 8) or crossed(i, j))
    }
    steps6 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    whole = {(i + d[0], j + d[1], k + d[2]) for (i, j, k) in cubes for d in steps6} - cubes
    whole = {
        c
        for c in whole
        if 1 <= min(c) and max(c) <= N - 2 and (c[0], c[1]) in reg and centre_in[(c[0], c[1])]
    }
    nodes = cubes | whole
    owner = {}
    for i, j, k in nodes:
        if centre_in[(i, j)]:
            owner[(i, j, k)] = idx(i, j, k)
    contested = []
    while True:
        new = {}
        for i, j, k in sorted(nodes):
            if (i, j, k) in owner:
                continue
            offers = {}
            nb = [
                ((i - 1, j, k), pr.seg_len_x(i, j) * zl[k]),
                ((i + 1, j, k), pr.seg_len_x(i + 1, j) * zl[k]),
                ((i, j - 1, k), pr.seg_len_y(i, j) * zl[k]),
                ((i, j + 1, k), pr.seg_len_y(i, j + 1) * zl[k]),
                ((i, j, k - 1), area2(reg[(i, j)]) if k > 1 else Fr(0)),
                ((i, j, k + 1), area2(reg[(i, j)]) if k < 8 else Fr(0)),
            ]
            for c, a in nb:
                if a > 0 and c in owner:
                    offers[owner[c]] = offers.get(owner[c], Fr(0)) + a
            if offers:
                best = sorted(offers.items(), key=lambda t: (-t[1], t[0]))
                new[(i, j, k)] = best[0][0]
                if len(best) > 1:
                    contested.append(
                        {
                            "cell": idx(i, j, k),
                            "owner": best[0][0],
                            "second": best[1][0],
                            "areas": [best[0][1], best[1][1]],
                        }
                    )
        if not new:
            break
        owner.update(new)
    # the tilted wall's area per unit of x in a cell of unit length in z
    tilt_per_dx = math.sqrt(float(pr.a * pr.a + pr.b * pr.b)) / float(pr.b)
    rows = {}
    for (i, j, k), o in owner.items():
        poly = reg[(i, j)]
        vol = area2(poly) * zl[k]
        # walls: the room's boundary inside the cell: x = 1.25, x = 8.75, y = 1.25 (axis walls),
        # the tilted wall (its length in the cell from its extent in x), the floor and the ceiling.
        wall_axis = Fr(0)
        tilt_dx = Fr(0)
        for t in range(len(poly)):
            p, q = poly[t], poly[(t + 1) % len(poly)]
            if p[0] == q[0] and p[0] in (LO, HI):
                wall_axis += abs(q[1] - p[1]) * zl[k]
            elif p[1] == q[1] and p[1] == LO:
                wall_axis += abs(q[0] - p[0]) * zl[k]
            elif pr.on_tilt(p) and pr.on_tilt(q) and p != q:
                tilt_dx += abs(q[0] - p[0]) * zl[k]
        if k in (1, 8):
            wall_axis += area2(poly)
        r = rows.setdefault(
            o,
            {
                "air": Fr(0),
                "wall_exact": Fr(0),
                "wall_tilt_dx": Fr(0),
                "sliver_vol": [Fr(0)] * 6,
                "sliver_area": [Fr(0)] * 6,
            },
        )
        r["air"] += vol
        r["wall_exact"] += wall_axis
        r["wall_tilt_dx"] += tilt_dx
        oi, oj, ok = o % N, (o // N) % N, o // (N * N)
        d = (i - oi, j - oj, k - ok)
        if d in steps6:
            kk = steps6.index(d)
            r["sliver_vol"][kk] += vol
            # the open area between the owner's own region and this one: the square they share
            if kk == 1:
                a = pr.seg_len_x(i, j) * zl[k]
            elif kk == 0:
                a = pr.seg_len_x(oi, oj) * zl[k]
            elif kk == 3:
                a = pr.seg_len_y(i, j) * zl[k]
            elif kk == 2:
                a = pr.seg_len_y(oi, oj) * zl[k]
            else:
                a = area2(poly)
            r["sliver_area"][kk] += a
    out = {}
    for o, r in rows.items():
        sl = [
            r["sliver_vol"][q] / r["sliver_area"][q]
            if r["sliver_area"][q] > Fr(1, 10**9)
            else Fr(0)
            for q in range(6)
        ]
        out[o] = {
            "air": r["air"],
            "wall": [float(r["wall_exact"]) + float(r["wall_tilt_dx"]) * tilt_per_dx, 0.0],
            "sliver": sl,
        }
    # rows are the owners that are air cells and own a cut region: here every owner
    return out, contested


def prism_mesh(pr):
    xy = pr.xy()
    m = len(xy)
    v = [[p[0], p[1], LO] for p in xy] + [[p[0], p[1], HI] for p in xy]
    f = [[0, t + 1, t] for t in range(1, m - 1)] + [[m, m + t, m + t + 1] for t in range(1, m - 1)]
    for t in range(m):
        u = (t + 1) % m
        f += [[t, u, u + m], [t, u + m, t + m]]
    return v, f


SOURCE = [2.3, 2.7, 2.9]


def run_fit(v, f, mats, materials):
    vm = np.ascontiguousarray(np.array([[float(c) * H for c in p] for p in v], float))
    fm = np.ascontiguousarray(np.array(f, np.int64))
    cut = _native.CutFit(
        vm,
        fm,
        np.asarray(mats, np.int64),
        materials,
        np.zeros(3),
        H,
        N,
        N,
        N,
        np.asarray(SOURCE, float) * H,
        0,
        2,
        False,
    )
    return cut


def case_mesh(name):
    """The case's mesh (vertices in cells, faces, materials, their count) and its rows."""
    v, f = box_mesh([LO] * 3, [HI] * 3)
    if name == "A":
        va, fa = panel(v, f, Fr(21, 4), Fr(13, 4), Fr(27, 4))
        return va, fa, [0] * 12 + [1, 1, 2, 2], 3, expect_A()
    if name == "B":
        vb, fb = panel(v, f, Fr(5), Fr(13, 4), Fr(27, 4))
        return vb, fb, [0] * 12 + [1, 1, 2, 2], 3, expect_B()
    if name == "C":
        vc, fc = prism_mesh(PRISM_C)
        expect, contested = expect_prism(PRISM_C)
        # The case is meant to hold regions whose second-best owner differs from the owner.
        assert sum(1 for c in contested if c["owner"] != c["second"]) == 24
        assert not any(c["areas"][0] == c["areas"][1] for c in contested)
        return vc, fc, [0] * len(fc), 1, expect
    vd, fd = prism_mesh(PRISM_D)
    expect, contested = expect_prism(PRISM_D)
    # The case is meant to hold exact ties: 6 regions per layer of cells, 8 layers, each offered
    # equal open areas by its -x and its -y neighbour and given to the -y neighbour, the lower
    # index.
    ties = [c for c in contested if c["areas"][0] == c["areas"][1]]
    assert len(ties) == len(contested) == 48
    assert all(c["owner"] == c["cell"] - N and c["second"] == c["cell"] - 1 for c in ties)
    return vd, fd, [0] * len(fd), 1, expect


@pytest.mark.parametrize("name", ["A", "B", "C", "D"])
def test_rows_match_the_geometry(name):
    v, f, mats, materials, expect = case_mesh(name)
    cut = run_fit(v, f, mats, materials)
    st = cut.cut_stats
    assert st["refused"] == 0, st["refusal"]
    if name == "D":
        assert st["ties"] == 48
    cc = cut.cut_cells()
    cells = [int(c) for c in np.asarray(cc["cell"])]
    assert sorted(cells) == sorted(expect)
    air = np.asarray(cc["air_volume"], float)
    wall = np.asarray(cc["wall_area"], float).reshape(len(cells), materials + 1)
    sliver = np.asarray(cc["sliver"], float).reshape(len(cells), 6)
    for r, c in enumerate(cells):
        e = expect[c]
        ijk = (c % N, (c // N) % N, c // (N * N))
        assert air[r] == pytest.approx(float(e["air"]), abs=1e-12), ("air", ijk)
        for m in range(materials + 1):
            assert wall[r][m] == pytest.approx(float(e["wall"][m]), abs=1e-12), ("wall", ijk, m)
        for q in range(6):
            assert sliver[r][q] == pytest.approx(float(e["sliver"][q]), abs=1e-12), (
                "sliver",
                ijk,
                q,
            )
