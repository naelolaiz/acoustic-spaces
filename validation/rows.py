"""Rows of roofs and barriers: the march over a whole row with what its rigid edges add
(:mod:`acoustic_spaces.engine.rows`) against a wave calculation written apart from the
engine.

    python validation/rows.py [--engine] [--case TEXT] [--json results.json]
    python validation/rows.py --references [SET ...]

The scenes are the same along the edges: thin rigid screens and rigid blocks with flat
tops (roofs) across the way between x = 0 and x = 10, in free space or on a rigid
ground 6 m below the tops, the source 10 m before x = 0 and the listener 5 m past
x = 10, a centimetre to 2 m above the tops' plane or below it, or 1.5 m above the
ground. In the last sets the source stands over a roof, 0.5 or 3 m before x = 10: a
roof that ends there, alone or with a parapet 0.1, 0.3 or 1.1 m high on its edge, and
a roof that goes on under the listener with a screen of those heights on it.
:data:`SETS` lists them.

The reference is a wave calculation in two dimensions, turned into a point source's.
The field of a point source at y = 0 is a sum over the wavenumber along the edges of
two-dimensional fields, each at the wavenumber kappa = sqrt(k^2 - ky^2) across them:

    G3(k) = (1/pi) [ int_0^k G2(kappa) kappa dkappa / sqrt(k^2 - kappa^2)
                     + int_0^inf G2(i gamma) gamma dgamma / sqrt(k^2 + gamma^2) ]

G2 at every real kappa is the Fourier transform of one run in the time domain, and at
i gamma its Laplace transform. The run is by finite differences (a staggered grid of
2 cm cells, rigid obstacles as whole cells, split-field matched layers on the four
sides; a block that runs into the bottom layer is endless downwards), from a line
source with a Gaussian rate of volume velocity. The same run without obstacles gives
the wavelet, which divides out the grid's dispersion over the direct distance. The
levels are stored in ``rows_wave.json`` (dB re free field in the 63 to 500 Hz octave
bands, at the engine's eight frequencies a band); ``--references`` computes them again,
3 to 6 minutes a listener's place on one core.

Each case is listed with the wave calculation's level and how far from it are the march
alone (its screens and walls reflect nothing), the march with the edges' terms (what
the module gives) and, with ``--engine``, the early stage of a simulation of the same
scene with ``second_order`` and ``first_order`` diffraction, each way round; the
simulation does not take the module's sound yet. The cases go by the angle the sound
bends by at the row's ends, up from the source to the first top or down from the last
one to the listener (:func:`bend`), and :data:`LIMITS` holds what the module must keep
to in each group, up to a bend of 45 degrees (the steepest of the cases is 42), and for
a source over a roof at any bend (up to 66 degrees, from a source on the roof half a
metre before a parapet 1.1 m high).

The reference tests itself: with one rigid screen it must be within
:data:`REFERENCE_LIMIT_DB` of the exact solution, the direct sound plus the
Biot-Tolstoy-Medwin integral over the edge.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_diffraction import (  # noqa: E402
    EXTENT,
    C,
    Edge,
    K,
    _room,
    btm_first,
    whole_early_stage,
)

from acoustic_spaces.engine import rows  # noqa: E402

#: The wave calculation's levels (:func:`references`).
STORED = Path(__file__).with_name("rows_wave.json")
#: Its cells (m) and the length of its runs (s).
GRID_M = 0.02
DURATION_S = 0.5
#: The wavenumbers of the 63 to 500 Hz octave bands, eight a band, and the frequencies.
WAVENUMBERS = K[:32]
FREQUENCIES = (WAVENUMBERS * C / (2.0 * math.pi)).reshape(4, -1)
#: Where the source and the listener stand along the way (cell centres), and the
#: height (m) of the tops over the ground in the cases that have one.
X_SOURCE, X_LISTENER = -10.0 + 0.5 * GRID_M, 15.0 + 0.5 * GRID_M
GROUND_M = 6.0
#: The rows in free space are marched this high (m) above a rigid ground in the 500 Hz
#: band, and as many wavelengths up in the others: the ground's reflection would come
#: up to the first top almost straight from below, which the module is asked to leave
#: out.
LIFT_M = 300.0
#: The most (dB) the wave calculation may be from the exact solution for one screen.
REFERENCE_LIMIT_DB = 0.15
#: The most (dB) the module's level may be from the wave calculation's in a band, by
#: group: whether the row stands on a ground, and the bend at its ends (degrees, up
#: to). On a ground the rows of screens alone are apart: there the module is lowest
#: (between rigid screens on a rigid ground sound goes to and fro, and a march one way
#: has none coming back; no run here shows that this is the cause). With the source
#: over a roof the groups go by what stands on the roof, at any bend: nothing up to the
#: roof's edge, a parapet on that edge, or a screen on a roof that goes on under the
#: listener, each lower than :data:`LOW_M` or not (the edges' terms take a screen for a
#: half-plane, and have no sound that goes on between its top and the roof).
LIMITS = {
    ("free space", 6.0): 0.4,
    ("free space", 25.0): 0.2,
    ("on a ground", 6.0): 0.7,
    ("on a ground", 25.0): 0.8,
    ("on a ground", 45.0): 1.3,
    ("screens on a ground", 6.0): 1.3,
    ("screens on a ground", 25.0): 2.6,
    ("screens on a ground", 45.0): 3.5,
    ("over a roof", 90.0): 0.2,
    ("parapet on a roof's edge", 90.0): 0.9,
    ("low parapet on a roof's edge", 90.0): 1.2,
    ("screen on a roof", 90.0): 1.7,
    ("low screen on a roof", 90.0): 3.1,
}
#: The groups' bends (degrees, up to).
BENDS = (6.0, 25.0, 45.0)
#: A screen on a roof under the source is low up to this height (m) above the roof.
LOW_M = 0.5


# -- the cases -------------------------------------------------------------------------


def _screen(x: float, top: float = 0.0) -> tuple:
    """A screen one cell thick from ``x`` on."""
    return (x, x + GRID_M, top)


def _steps(step: float, below: bool) -> tuple:
    """Two roofs against each other, the second ``step`` m higher; the listener and the
    sources by the higher one's plane."""
    high = max(0.0, step)
    name = f"roofs 0..5 and 5..10, the second {step:+.1f} m"
    row = [(0.0, 5.0, 0.0), (5.0, 10.0, step)]
    return (name, row, high + (-0.49 if below else 0.01), [high + 0.01, high + 0.51])


def _on_ground(name: str, row: list) -> list:
    """``row`` on the ground, for listeners just above the tops, half a metre below
    them and 1.5 m above the ground, each with three sources."""
    row = [(a, b, top + GROUND_M) for a, b, top in row]
    sources = [GROUND_M + 0.01, GROUND_M + 0.51, 1.51]
    return [(name, row, zl, sources) for zl in (GROUND_M + 0.01, GROUND_M - 0.49, 1.51)]


def _over_a_roof(parapet: float) -> list:
    """A roof under the source up to the cell that begins at x = 10, its last (it goes
    on behind the source without end), with a parapet ``parapet`` m high on that cell,
    or none (0): for listeners 2 m below the roof's plane and 1 m above it, each with
    the sources of :data:`_NEAR`."""
    if parapet:
        name = f"roof to 10 under the source, parapet {parapet:g} m on its edge"
        row = [(-_BEHIND_M, 10.0, 0.0), _screen(10.0, parapet)]
    else:
        name = "roof to 10 under the source"
        row = [(-_BEHIND_M, 10.0 + GRID_M, 0.0)]
    return [(name, row, zl, _NEAR) for zl in (-1.99, 1.01)]


def _low_screen(height: float) -> tuple:
    """A screen ``height`` m high at x = 10 on a roof under both ends, the listener
    1.5 m above the roof, with the sources of :data:`_NEAR`."""
    name = f"roof under both ends, screen {height:g} m high at 10"
    return (name, [(-_BEHIND_M, 2.0 * _BEHIND_M, 0.0), _screen(10.0, height)], 1.51, _NEAR)


#: How far (m) behind the source a roof under it begins: past the wave calculation's
#: frame, where it has no end.
_BEHIND_M = 30.0
#: Sources 0.5 and 3 m before the middle of the cell that begins at x = 10 (a screen's,
#: or the roof's last), a centimetre and 1.5 m above the roof: (x, z).
_NEAR = [(10.0 + 0.5 * GRID_M - before, z) for before in (0.5, 3.0) for z in (0.01, 1.51)]
_ROOF_SCREEN = [(0.0, 9.0, 0.0), _screen(10.0)]
_TWO_ROOFS = [(0.0, 4.0, 0.0), (6.0, 10.0, 0.0)]
_THREE_SCREENS = [_screen(0.0), _screen(5.0), _screen(10.0)]
_THREE_ROOFS = [(0.0, 2.0, 0.0), (4.0, 6.0, 0.0), (8.0, 10.0, 0.0)]
#: By set: (name, the row as (from x, to x, top z), the listener's height, the sources'
#: heights or, for sources that do not stand at :data:`X_SOURCE`, their (x, z)), heights
#: from the tops' plane, or from the ground in the sets on one. One run of the wave
#: calculation serves a listener and its sources.
SETS: dict[str, list] = {
    "tops": [
        (
            f"roof 0..9, screen at 10 with its top {top:+.2f} m",
            [(0.0, 9.0, 0.0), _screen(10.0, top)],
            0.01,
            [0.01],
        )
        for top in (-0.5, -0.2, -0.1, -0.04, -0.02, 0.02, 0.04, 0.1, 0.2, 0.5)
    ],
    "rows": [
        ("two roofs 0..4 and 6..10", _TWO_ROOFS, 0.01, [0.01, 0.51, 2.01]),
        ("two roofs 0..4 and 6..10", _TWO_ROOFS, -0.49, [0.01, 0.51, 2.01]),
        ("three screens at 0, 5, 10", _THREE_SCREENS, 0.01, [0.01, 0.51]),
        ("three screens at 0, 5, 10", _THREE_SCREENS, -0.49, [0.01, 0.51]),
        ("roof 0..9, screen at 10", _ROOF_SCREEN, -0.49, [0.01, 0.51, 2.01]),
        ("roof 0..9, screen at 10", _ROOF_SCREEN, 0.01, [0.51, 2.01]),
        ("three roofs 0..2, 4..6, 8..10", _THREE_ROOFS, 0.01, [0.01, 0.51]),
        ("three roofs 0..2, 4..6, 8..10", _THREE_ROOFS, -0.49, [0.01, 0.51]),
        ("two screens at 0 and 10", [_screen(0.0), _screen(10.0)], 0.01, [0.01]),
    ],
    "one_roof": [
        ("one roof 0..4", [(0.0, 4.0, 0.0)], 0.01, [0.01, 0.51]),
        ("one roof 0..9", [(0.0, 9.0, 0.0)], 0.01, [0.01, 0.51]),
        ("one roof 6..10", [(6.0, 10.0, 0.0)], 0.01, [0.01, 0.51]),
        ("one roof 0..4", [(0.0, 4.0, 0.0)], -0.49, [0.01, 0.51]),
        ("one roof 0..10", [(0.0, 10.0, 0.0)], -0.49, [0.01, 0.51]),
    ],
    "steps": [
        _steps(-1.0, False),
        _steps(-0.2, False),
        _steps(-0.2, True),
        _steps(0.2, False),
        _steps(0.2, True),
        _steps(1.0, False),
    ],
    "down": [
        ("roof 0..9, screen at 10", _ROOF_SCREEN, -1.99, [0.01, 2.01]),
        ("two roofs 0..4 and 6..10", _TWO_ROOFS, -1.99, [0.01, 2.01]),
        ("one screen at 10", [_screen(10.0)], -1.99, [0.01, 2.01]),
        ("one roof 0..10", [(0.0, 10.0, 0.0)], -1.99, [0.01, 2.01]),
    ],
    "ground_one_screen": _on_ground("one screen at 10", [_screen(10.0)]),
    "ground_one_roof": _on_ground("one roof 0..10", [(0.0, 10.0, 0.0)]),
    "ground_roof_screen": _on_ground("roof 0..9, screen at 10", _ROOF_SCREEN),
    "ground_two_roofs": _on_ground("two roofs 0..4 and 6..10", _TWO_ROOFS),
    "ground_three_screens": _on_ground("three screens at 0, 5, 10", _THREE_SCREENS),
    "roof_edge": _over_a_roof(0.0),
    "roof_parapet_low": _over_a_roof(0.1) + _over_a_roof(0.3),
    "roof_parapet": _over_a_roof(1.1),
    "roof_low_screen": [_low_screen(0.1), _low_screen(0.3), _low_screen(1.1)],
}


def _cell(height: float) -> float:
    """The centre of the wave calculation's cell at ``height``."""
    return (math.floor(height / GRID_M) + 0.5) * GRID_M


def _place(source) -> tuple[float, float]:
    """Where a set's source stands (its cell's centre): a height at :data:`X_SOURCE`,
    or (x, z)."""
    x, z = source if isinstance(source, tuple) else (X_SOURCE, source)
    return (x, _cell(z))


def cases(sets=None) -> list[dict]:
    """The cases of ``sets`` (all by default), one for each source of each listener."""
    out = []
    for key in SETS if sets is None else sets:
        for name, row, zl, sources in SETS[key]:
            for source in sources:
                out.append(
                    {
                        "set": key,
                        "name": name,
                        "ground": key.startswith("ground"),
                        "row": [list(obstacle) for obstacle in row],
                        "source": list(_place(source)),
                        "listener": [X_LISTENER, _cell(zl)],
                    }
                )
    return out


def _key(case: dict) -> str:
    """The name a case's levels are stored by."""
    where = "ground" if case["ground"] else "free"
    key = f"{case['name']} | {where} | S {case['source'][1]:+.2f} | L {case['listener'][1]:+.2f}"
    if case["source"][0] != X_SOURCE:
        key += f" | S at {case['source'][0]:+.2f}"
    return key


def obstacles_of(case: dict, lift: float = 0.0) -> list[tuple[float, float, float]]:
    """The case's row as the march takes it: (near, far, top) from the source along the
    way and above the ground, a screen (one cell thick in the wave calculation) at its
    cell's middle, and a roof that ends where a screen's cell begins up to there (the
    screen stands on its edge)."""
    middles = {a: 0.5 * (a + b) for a, b, _ in case["row"] if _thin(a, b)}
    out = []
    for a, b, top in case["row"]:
        if _thin(a, b):
            a = b = 0.5 * (a + b)
        b = middles.get(b, b)
        out.append((a - case["source"][0], b - case["source"][0], top + lift))
    return out


def _thin(a: float, b: float) -> bool:
    """Whether what stands from ``a`` to ``b`` is a screen, one cell thick."""
    return b - a <= GRID_M + 1e-9


def bend(case: dict) -> float:
    """The angle (degrees) the sound bends by at the row's ends: up from the source to
    the first top past it (none where it stands over a roof with nothing else on its
    way) or down from the last top to the listener, the larger; 0 where both look over
    them."""
    (xs, zs), last = case["source"], case["row"][-1]
    ahead = [(a, top) for a, _, top in case["row"] if a > xs]
    up = math.atan2(ahead[0][1] - zs, ahead[0][0] - xs) if ahead else 0.0
    down = math.atan2(last[2] - case["listener"][1], case["listener"][0] - last[1])
    return math.degrees(max(up, down, 0.0))


def group(case: dict) -> str:
    under = [(b, top) for a, b, top in case["row"] if a < case["source"][0] < b]
    if under:
        # The source stands over a roof: alone, with a parapet on the roof's edge, or
        # with a screen on a roof that goes on under the listener.
        end, roof = under[0]
        screens = [(a, top - roof) for a, b, top in case["row"] if _thin(a, b)]
        if not screens:
            return "over a roof"
        kind = "parapet on a roof's edge" if screens[0][0] == end else "screen on a roof"
        return f"low {kind}" if screens[0][1] < LOW_M else kind
    if not case["ground"]:
        return "free space"
    thin = all(_thin(a, b) for a, b, _ in case["row"])
    return "screens on a ground" if thin and len(case["row"]) > 1 else "on a ground"


# -- the wave calculation ----------------------------------------------------------------


def fdtd(solids, source, receivers, x_range, z_range, layer: float = 1.5, courant: float = 0.7):
    """The pressure at ``receivers`` (the cells they lie in) over time, from a line
    source at ``source`` whose rate of volume velocity is a Gaussian, among the rigid
    ``solids`` (rectangles: from x, to x, from z, to z; infinite where they leave the
    grid). Returns the traces (steps, receivers), the time step and the places of the
    source's and the receivers' cells."""
    from scipy.special import erf

    h = GRID_M
    sigma_t = 1.0 / (2.0 * math.pi * 350.0)
    x0, x1 = x_range[0] - layer, x_range[1] + layer
    z0, z1 = z_range[0] - layer, z_range[1] + layer
    nx, nz = int(round((x1 - x0) / h)), int(round((z1 - z0) / h))
    dt = courant * h / C
    steps = int(math.ceil(DURATION_S / dt))
    f32 = np.float32

    def cell(point):
        return int(math.floor((point[0] - x0) / h)), int(math.floor((point[1] - z0) / h))

    def centre(ij):
        return (x0 + (ij[0] + 0.5) * h, z0 + (ij[1] + 0.5) * h)

    solid = np.zeros((nx, nz), dtype=bool)
    for xa, xb, za, zb in solids:
        ia = 0 if xa == -math.inf else int(round((xa - x0) / h))
        ib = nx if xb == math.inf else int(round((xb - x0) / h))
        ja = 0 if za == -math.inf else int(round((za - z0) / h))
        jb = nz if zb == math.inf else int(round((zb - z0) / h))
        solid[max(ia, 0) : min(ib, nx), max(ja, 0) : min(jb, nz)] = True
    # The faces between two cells are open only where both are air.
    shut_x = np.nonzero(solid[1:, :] | solid[:-1, :])
    shut_z = np.nonzero(solid[:, 1:] | solid[:, :-1])

    order = 3
    strongest = -(order + 1) * C * math.log(1e-8) / (2.0 * layer)

    def profile(coordinate, lo, hi):
        depth = np.maximum(np.maximum(lo + layer - coordinate, coordinate - (hi - layer)), 0.0)
        return strongest * (depth / layer) ** order

    def coefficients(sigma):
        half = 0.5 * sigma * dt
        return ((1.0 - half) / (1.0 + half)).astype(f32), (1.0 / (1.0 + half)).astype(f32)

    xc, xf = x0 + (np.arange(nx) + 0.5) * h, x0 + np.arange(1, nx) * h
    zc, zf = z0 + (np.arange(nz) + 0.5) * h, z0 + np.arange(1, nz) * h
    apx, bpx = coefficients(profile(xc, x0, x1))
    avx, bvx = coefficients(profile(xf, x0, x1))
    apz, bpz = coefficients(profile(zc, z0, z1))
    avz, bvz = coefficients(profile(zf, z0, z1))
    # With a density of 1: dv/dt = -grad p, dp/dt = -c^2 div v.
    bvx, bvz = (bvx * dt / h)[:, None], (bvz * dt / h)[None, :]
    bpx, bpz = (bpx * C * C * dt / h)[:, None], (bpz * C * C * dt / h)[None, :]
    apx, avx, apz, avz = apx[:, None], avx[:, None], apz[None, :], avz[None, :]

    px, pz, p = (np.zeros((nx, nz), f32) for _ in range(3))
    vx, vz = np.zeros((nx + 1, nz), f32), np.zeros((nx, nz + 1), f32)
    vxi, vzi = vx[1:-1, :], vz[:, 1:-1]
    dx, dz = np.zeros((nx - 1, nz), f32), np.zeros((nx, nz - 1), f32)
    ex, ez = np.zeros((nx, nz), f32), np.zeros((nx, nz), f32)
    at = cell(source)
    heard = [cell(r) for r in receivers]
    if solid[at] or any(solid[r] for r in heard):
        raise ValueError("the source or a receiver lies in an obstacle")
    rix, riz = np.array([r[0] for r in heard]), np.array([r[1] for r in heard])
    traces = np.zeros((steps, len(heard)))
    t0 = 6.0 * sigma_t
    amplitude = C * C * dt / (h * h)
    for n in range(steps):
        np.subtract(p[1:, :], p[:-1, :], out=dx)
        dx *= bvx
        vxi *= avx
        vxi -= dx
        vxi[shut_x] = 0.0
        np.subtract(p[:, 1:], p[:, :-1], out=dz)
        dz *= bvz
        vzi *= avz
        vzi -= dz
        vzi[shut_z] = 0.0
        np.subtract(vx[1:, :], vx[:-1, :], out=ex)
        ex *= bpx
        px *= apx
        px -= ex
        np.subtract(vz[:, 1:], vz[:, :-1], out=ez)
        ez *= bpz
        pz *= apz
        pz -= ez
        # The volume velocity at the time of this pressure, (n + 1/2) dt.
        q = 0.5 * (1.0 + erf(((n + 0.5) * dt - t0) / (sigma_t * math.sqrt(2.0))))
        px[at] += 0.5 * amplitude * q
        pz[at] += 0.5 * amplitude * q
        np.add(px, pz, out=p)
        traces[n] = p[rix, riz]
    return traces, dt, centre(at), [centre(r) for r in heard]


def _spectra(trace, dt: float, kappa, gamma, taper: float = 0.15):
    """The trace's Fourier transforms at the wavenumbers ``kappa`` and its Laplace
    transforms at ``gamma`` (its samples are at (n + 1) dt), the last ``taper`` of it
    faded out."""
    n = len(trace)
    t = (np.arange(n) + 1.0) * dt
    window = np.ones(n)
    m = int(taper * n)
    window[n - m :] = 0.5 * (1.0 + np.cos(np.pi * (np.arange(m) + 0.5) / m))
    x = trace * window * dt
    real = np.empty(len(kappa), dtype=complex)
    for a in range(0, len(kappa), 2000):
        real[a : a + 2000] = np.exp(1j * C * np.outer(kappa[a : a + 2000], t)) @ x
    return real, np.exp(-C * np.outer(gamma, t)) @ x


def point_source(trace, free, distance: float, dt: float, k=WAVENUMBERS) -> np.ndarray:
    """The pressure of a point source at the wavenumbers ``k`` (free field exp(ikr) / r)
    from the two-dimensional ``trace`` among the obstacles and ``free`` without them,
    ``distance`` m from the source."""
    from scipy.special import hankel1, k0

    trace, free, step = trace[3::4], free[3::4], 4.0 * dt
    theta, w_theta = np.polynomial.legendre.leggauss(6000)
    theta, w_theta = 0.25 * np.pi * (theta + 1.0), 0.25 * np.pi * w_theta
    g, w_g = np.polynomial.legendre.leggauss(400)
    gamma, w_g = 6.0 / distance * (g + 1.0), 6.0 / distance * w_g
    out = np.empty(len(k), dtype=complex)
    for index, wavenumber in enumerate(k):
        kappa = wavenumber * np.sin(theta)
        real, decaying = _spectra(trace, step, kappa, gamma)
        real_free, decaying_free = _spectra(free, step, kappa, gamma)
        g2 = real / real_free * (0.25j * hankel1(0, kappa * distance))
        g2_decaying = decaying / decaying_free * (k0(gamma * distance) / (2.0 * np.pi))
        g3 = np.sum(w_theta * g2 * kappa)
        g3 += np.sum(w_g * g2_decaying * gamma / np.sqrt(wavenumber**2 + gamma**2))
        out[index] = 4.0 * g3
    return out


def band_levels(pressure: np.ndarray, distance: float) -> np.ndarray:
    """(4,) the band levels (dB re free field) of ``pressure`` at ``WAVENUMBERS``."""
    energy = (np.abs(pressure) ** 2).reshape(4, -1).mean(axis=1)
    return 10.0 * np.log10(energy * distance**2)


def references(sets) -> None:
    """Runs the wave calculation for ``sets`` and stores its levels in ``STORED``,
    beside those of the other sets."""
    stored = json.loads(STORED.read_text()) if STORED.exists() else {}
    for key in sets:
        ground = key.startswith("ground")
        z_range = (0.0, GROUND_M + 2.5) if ground else (-2.5, 2.5)
        frame = dict(x_range=(-11.0, 16.0), z_range=z_range)
        floor = [(-math.inf, math.inf, -math.inf, 0.0)] if ground else []
        free: dict = {}
        for name, row, zl, sources in SETS[key]:
            started = time.time()
            solids = floor + [(a, b, -math.inf, top) for a, b, top in row]
            listener = (X_LISTENER, _cell(zl))
            places = [_place(source) for source in sources]
            # By reciprocity one run from the listener's place serves its sources.
            traces, dt, at, heard = fdtd(solids, listener, places, **frame)
            if (zl, tuple(sources)) not in free:
                free[(zl, tuple(sources))] = fdtd([], listener, places, **frame)[0]
            without = free[(zl, tuple(sources))]
            for i, place in enumerate(heard):
                distance = math.hypot(at[0] - place[0], at[1] - place[1])
                p = point_source(traces[:, i], without[:, i], distance, dt)
                case = {"name": name, "ground": ground, "source": list(place), "listener": list(at)}
                stored[_key(case)] = [round(float(v), 3) for v in band_levels(p, distance)]
            print(
                f"{key}: {name}, listener {zl:+.2f} m ({time.time() - started:.0f} s)", flush=True
            )
            STORED.write_text(json.dumps(stored, indent=0) + "\n")


# -- the comparison ----------------------------------------------------------------------


def march(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """(4,) the band levels (dB re free field) of the march alone and of the module's
    sound for ``case``."""
    (xs, zs), (xl, zl) = case["source"], case["listener"]
    found = []
    for f in FREQUENCIES:
        lift = 0.0 if case["ground"] else LIFT_M * FREQUENCIES[-1].min() / f.min()
        standing = obstacles_of(case, lift)
        # Lifted, the ground's reflection is too steep for the march, and is left out.
        steep = "leave out" if lift else "raise"
        ends = (zs + lift, xl - xs, zl + lift)
        found.append(rows.over_row(f, C, None, *ends, standing, steep_reflections=steep))
    distance = math.hypot(xl - xs, zl - zs)
    alone = band_levels(np.concatenate([row.marched for row in found]), distance)
    return alone, band_levels(np.concatenate([row.pressure for row in found]), distance)


def simulated(case: dict, folder: Path) -> dict[str, list]:
    """The early stage's band levels (dB re free field) for ``case`` by diffraction
    mode, each way round."""
    quads = []
    if case["ground"]:
        e = EXTENT
        quads.append([(-e, -e, 0.0), (e, -e, 0.0), (e, e, 0.0), (-e, e, 0.0)])
    foot = 0.0 if case["ground"] else -EXTENT
    for near, far, top in obstacles_of(case):
        a, b, e = near + case["source"][0], far + case["source"][0], EXTENT
        if far == near:
            quads.append([(a, -e, foot), (a, e, foot), (a, e, top), (a, -e, top)])
            continue
        quads.append([(a, -e, top), (b, -e, top), (b, e, top), (a, e, top)])
        quads.append([(a, -e, foot), (a, -e, top), (a, e, top), (a, e, foot)])
        quads.append([(b, -e, foot), (b, e, foot), (b, e, top), (b, -e, top)])
        quads.append([(a, -e, foot), (b, -e, foot), (b, -e, top), (a, -e, top)])
        quads.append([(a, e, foot), (a, e, top), (b, e, top), (b, e, foot)])
    source = (case["source"][0], 0.0, case["source"][1])
    listener = (case["listener"][0], 0.0, case["listener"][1])
    out = {}
    for mode in ("second_order", "first_order"):
        early = {"diffraction": mode, "coherent_max_band_hz": 500, "method": "image_sources"}
        ways = []
        for a, b in ((source, listener), (listener, source)):
            room = _room(folder, "row", quads, a, b, {"early": early})
            with np.errstate(divide="ignore"):
                ways.append(np.asarray(whole_early_stage(room))[:4].tolist())
        out[mode] = ways
    return out


def reference_check(stored: dict) -> list[dict]:
    """The wave calculation for one screen in free space against the exact solution."""
    edge = Edge([0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
    out = []
    for case in cases(["down"]):
        if case["name"] != "one screen at 10" or _key(case) not in stored:
            continue
        x = 0.5 * (case["row"][0][0] + case["row"][0][1])
        a = np.array([case["source"][0] - x, 0.0, case["source"][1]])
        b = np.array([case["listener"][0] - x, 0.0, case["listener"][1]])
        distance = float(np.linalg.norm(b - a))
        exact = btm_first(edge, a, b, WAVENUMBERS)
        if a[2] + (b[2] - a[2]) * -a[0] / (b[0] - a[0]) > 0.0:  # in sight over the top
            exact = exact + np.exp(-1j * WAVENUMBERS * distance) / distance
        off = np.array(stored[_key(case)]) - band_levels(exact, distance)
        out.append({"source": case["source"], "listener": case["listener"], "off_db": off.tolist()})
    return out


def _bends(name: str, angle: float) -> float:
    """The bend (degrees, up to) that the summary lists a case of the group ``name``
    under: that of the group's own limit, or of :data:`BENDS`; infinite beyond them."""
    own = sorted(to for (g, to) in LIMITS if g == name)
    return next((to for to in (*own, *BENDS) if angle <= to), math.inf)


def _row(values) -> str:
    return " ".join(f"{v:+6.2f}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--references", nargs="*", metavar="SET", help="run the wave calculation")
    parser.add_argument("--engine", action="store_true", help="simulate each scene too")
    parser.add_argument("--case", help="only the cases whose name holds this text")
    parser.add_argument("--json", type=Path, help="write the results here")
    args = parser.parse_args()
    if args.references is not None:
        references(args.references or list(SETS))
        return
    stored = json.loads(STORED.read_text())
    failed = False
    checks = reference_check(stored)
    for check in checks:
        ok = max(abs(v) for v in check["off_db"]) <= REFERENCE_LIMIT_DB
        failed |= not ok
        print(f"wave calculation, one screen, less the exact solution: {_row(check['off_db'])} dB")
    results = []
    worst: dict[tuple, list] = {}
    folder = Path(tempfile.mkdtemp())
    for case in cases():
        if args.case and args.case not in case["name"]:
            continue
        if _key(case) not in stored:
            print(f"no reference for {case['name']} (run --references {case['set']})")
            failed = True
            continue
        wave = np.array(stored[_key(case)])
        alone, with_edges = march(case)
        angle = bend(case)
        limit = next((v for (g, to), v in LIMITS.items() if g == group(case) and angle <= to), None)
        entry = {
            **case,
            "group": group(case),
            "bend_deg": angle,
            "wave_db": wave.tolist(),
            "march_db": (alone - wave).tolist(),
            "module_db": (with_edges - wave).tolist(),
            "limit_db": limit,
        }
        line = f"{case['name'][:44]:44s} S {case['source'][1]:5.2f} L {case['listener'][1]:5.2f}"
        line += f" {angle:4.1f} deg  wave {_row(wave)}  march {_row(alone - wave)}"
        line += f"  module {_row(with_edges - wave)}"
        if args.engine:
            entry["simulated_db"] = {
                mode: [(np.array(way) - wave).tolist() for way in ways]
                for mode, ways in simulated(case, folder).items()
            }
            for mode, ways in entry["simulated_db"].items():
                line += f"  {mode} {' / '.join(_row(way) for way in ways)}"
        bucket = (group(case), _bends(group(case), angle))
        worst.setdefault(bucket, []).append(entry)
        if limit is not None and np.max(np.abs(with_edges - wave)) > limit:
            failed = True
            line += "  OVER THE LIMIT"
        results.append(entry)
        print(line, flush=True)
    print()
    for (name, to), entries in sorted(worst.items()):
        module = np.array([e["module_db"] for e in entries])
        alone = np.array([e["march_db"] for e in entries])
        angles = f"up to {to:g}" if math.isfinite(to) else f"more than {BENDS[-1]:g}"
        line = f"{name}, bend {angles} degrees: {len(entries)} cases, march alone"
        line += f" {alone.min():+.2f} to {alone.max():+.2f} dB, module {module.min():+.2f} to"
        line += f" {module.max():+.2f} dB"
        if args.engine:
            for mode in ("second_order", "first_order"):
                both = np.array([e["simulated_db"][mode] for e in entries], dtype=float)
                heard = both[np.isfinite(both)]
                line += f", {mode} {heard.min():+.2f} to {heard.max():+.2f} dB"
                if not np.isfinite(both).all():
                    line += " or no sound"
        print(line)
    if args.json:
        args.json.write_text(json.dumps({"reference": checks, "cases": results}, indent=1) + "\n")
    if failed:
        raise SystemExit("a case is over its limit")


if __name__ == "__main__":
    main()
