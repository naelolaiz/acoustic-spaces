"""Barriers and buildings in wind: the engine's parabolic equation with them inside,
against two references written apart from the engine.

    python validation/barrier_wind.py [--only full_wave|started|engine|convergence]
        [--case NAME] [--band HZ] [--wind M_S] [--reference ring|plane] [--quick]
        [--full-wave-json results.json] [--json results.json]

Behind a barrier or a building in wind the engine keeps each path's level in still air
and multiplies it, in each octave band, by the weather's effect G = E_wind / E_still: the
band energy of its parabolic equation (:mod:`acoustic_spaces.engine.parabolic`) with the
obstacles inside (zero-field screens, rigid roofs) in the weather, over that in still air
(:mod:`acoustic_spaces.engine.refracted`). Its zero-field screen lets through less than a
rigid barrier does, by an amount that hangs on the geometry, which the ratio cancels;
these checks measure how far it does.

- ``full_wave``: the axisymmetric Helmholtz equation in air whose effective speed of
  sound changes with height, with thin rigid ring screens and rigid flat-topped ring
  blocks, solved by stepwise coupled modes in range (Evans 1983, J. Acoust. Soc. Am.
  74, 188): exact in range (Hankel functions), finite volumes in height (cell centres)
  with a perfectly matched layer on top. No paraxial approximation, rigid screens, the
  wind everywhere, rigid ground and grass (Miki, 200 kPa s/m2), 63 to 500 Hz. Where a
  case says, also along a plane (``plane``: a line source, straight screens and blocks,
  exponentials in range), where nothing focuses what goes back between two screens.
  The engine's effect against its effect (each the level in the weather less that in
  still air), per band at each listener.
- ``started``: a split-step parabolic equation of its own (the exact square-root
  propagator, rigid ground as an even field), started 1 m behind a rigid straight
  screen from the exact field there in still air (the uniform theory of diffraction
  for a half-plane, four paths: source and image to listener and image; Kouyoumjian
  and Pathak 1974), then marched in a wind that blows behind the screen only, so that
  the starting field holds; 125 Hz to 8 kHz. The engine's march takes the same wind,
  by range.
- ``engine``: :func:`acoustic_spaces.simulate`'s early paths end to end (a straight
  barrier or building 1 km long, rigid ground, reflections off both sides of an edge)
  in the wind and in still air, the difference against ``full_wave``'s effect.
- ``convergence``: the engine's parabolic equation alone, its grid and its frequencies
  against finer ones.

Each check also tests itself: the full-wave solution against the exact two-path field
over open rigid ground, against the engine's parabolic equation over open ground in
wind, with a screen in still air against a straight screen's four paths, and on its
domain against a taller one deep in an upwind shadow; the started equation's still air
against the four paths it starts from.
Acceptance, per band, at the listeners where the reference's effect is 25 dB or less
(deeper shadows are listed, not judged): :data:`LIMITS`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings

import numpy as np
import scipy.linalg as la
from scipy import special
from scipy.special import fresnel

from acoustic_spaces.engine import parabolic
from acoustic_spaces.engine.diffraction import band_frequencies
from acoustic_spaces.engine.refracted import (
    BARRIER_COARSE,
    BARRIER_COARSE_FROM_HZ,
    BARRIER_HEADROOM,
    BARRIER_SAMPLES,
    WAVE_MIN_HEADROOM_M,
    WAVE_TOP_SLOPE,
)
from acoustic_spaces.materials.ground import surface_impedance
from acoustic_spaces.scene import Atmosphere

CENTRES = (63, 125, 250, 500, 1000, 2000, 4000, 8000)
GROUND_C = 20.0
#: The grass of the full-wave cases: Miki's model, this flow resistivity (kPa s/m2).
GRASS = 200.0
#: Listeners' effects deeper than this (dB below still air) are listed, not judged.
DEEPEST_DB = 25.0
#: Acceptance (dB): the largest error of the weather's effect measured, rounded up, so
#: that a change that makes any of them worse fails, by (case, reference, wind) for the
#: full-wave cases (``one``: near, far, rail and roof), ``started`` and, end to end,
#: ``engine`` beyond its parabolic equation's own error. The design aimed for 0.6 dB
#: in 5 m/s and 0.8 dB in 10 m/s behind one barrier or building.
LIMITS = {
    ("one", "ring", 5.0): 0.65,
    ("one", "ring", 10.0): 0.75,
    ("one", "plane", 5.0): 0.65,
    ("one", "plane", 10.0): 0.55,
    ("near_up", "ring", 5.0): 0.75,
    ("near_up", "ring", 10.0): 1.65,
    ("steep45", "ring", 5.0): 0.5,
    ("steep45", "ring", 10.0): 1.0,
    ("steep60", "ring", 5.0): 0.55,
    ("steep60", "ring", 10.0): 0.5,
    ("two", "ring", 5.0): 1.7,
    ("two", "ring", 10.0): 1.65,
    ("two", "plane", 5.0): 1.65,
    ("two", "plane", 10.0): 1.75,
    ("two40", "ring", 5.0): 2.25,
    ("two40", "ring", 8.0): 4.95,
    ("two40", "ring", 10.0): 1.15,
    ("two40", "plane", 5.0): 2.2,
    ("two40", "plane", 8.0): 4.95,
    ("two40", "plane", 10.0): 1.1,
    ("started",): 0.5,
    ("engine",): 0.2,
}

#: (a) The cases: the source's height (m), what stands on the way (near, far, top: a
#: screen where near is far), the listeners (range, height); where given, the
#: full-wave domain's height under its absorbing layer (``top``, default
#: :data:`FULL_WAVE_TOP_M`; ``top_upwind`` upwind, where the sound comes down from high up
#: in the shadow), the winds (default :data:`WINDS`), the bands (indices of
#: :data:`CENTRES`, default 63 to 500 Hz) and the references: ``ring`` round an axis
#: (the default), ``plane`` along a plane (:func:`full_wave`).
FULL_WAVE = {
    "near": {
        "zs": 1.0,
        "cut": [(15.0, 15.0, 4.0)],
        "at": [(30, 1.5), (60, 1.5), (120, 1.5), (60, 4)],
        "references": ("ring", "plane"),
    },
    "far": {
        "zs": 1.0,
        "cut": [(100.0, 100.0, 4.0)],
        "at": [(110, 1.5), (130, 1.5), (130, 4)],
        "top_upwind": 60.0,
    },
    "rail": {"zs": 0.5, "cut": [(4.0, 4.0, 3.0)], "at": [(25, 1.5), (50, 1.5), (100, 1.5)]},
    "steep45": {"zs": 0.5, "cut": [(4.0, 4.0, 4.5)], "at": [(15, 1.5), (30, 1.5), (60, 1.5)]},
    "steep60": {"zs": 0.5, "cut": [(4.0, 4.0, 7.43)], "at": [(15, 1.5), (30, 1.5), (60, 1.5)]},
    "roof": {
        "zs": 1.0,
        "cut": [(40.0, 52.0, 6.0)],
        "at": [(80, 1.5), (150, 1.5)],
        "top_upwind": 60.0,
    },
    "two": {
        "zs": 1.0,
        "cut": [(45.0, 45.0, 8.0), (55.0, 55.0, 8.0)],
        "at": [(100, 1.5), (150, 1.5)],
        "references": ("ring", "plane"),
        "top_upwind": 60.0,
    },
    "two40": {
        "zs": 1.0,
        "cut": [(20.0, 20.0, 3.0), (60.0, 60.0, 5.0)],
        "at": [(100, 1.5), (160, 1.5)],
        "bands": (1, 2),
        "winds": (5.0, -5.0, 8.0, 10.0),
        "references": ("ring", "plane"),
        "top_upwind": 60.0,
    },
    "near_up": {
        "zs": 1.0,
        "cut": [(15.0, 15.0, 4.0)],
        "at": [(60, 1.5), (120, 1.5), (200, 1.5), (120, 4)],
        "top": 60.0,
        "winds": (-5.0, -10.0),
    },
}
FULL_WAVE_TOP_M = 30.0
#: The domain's check: a source 1 m up, a screen 4 m high 15 m away, rigid ground, one
#: frequency of the 500 Hz band, listeners deep in the upwind shadow; the level on
#: :data:`FULL_WAVE_TOP_M` and the first of ``tops`` against that on the last.
DOMAIN_CHECK = {
    "zs": 1.0,
    "cut": [(15.0, 15.0, 4.0)],
    "band": 3,
    "sample": 4,
    "winds": (-5.0, -10.0),
    "at": [(120.0, 1.5), (150.0, 1.5), (200.0, 1.5)],
    "tops": (60.0, 90.0),
}
#: Winds of the full-wave cases (m/s at 10 m, a log profile over 5 cm; negative:
#: upwind).
WINDS = (5.0, -5.0, 10.0)

#: (b) The cases: the weather and the way's cosine to the wind, the screen's range and
#: height (m); the source 1 m up.
STARTED = {
    "down_3m_20": ("windy", 1.0, 20.0, 3.0),
    "down_6m_20": ("windy", 1.0, 20.0, 6.0),
    "down_3m_50": ("windy", 1.0, 50.0, 3.0),
    "up_3m_20": ("windy", -1.0, 20.0, 3.0),
    "up_3m_50": ("windy", -1.0, 50.0, 3.0),
    "night_3m_20": ("clear_night", 1.0, 20.0, 3.0),
    "day_3m_20": ("sunny_day", 1.0, 20.0, 3.0),
}
STARTED_AT = [(50.0, 1.5), (50.0, 4.0), (100.0, 1.5), (100.0, 4.0), (200.0, 1.5), (200.0, 4.0)]
#: From 4 kHz up the started cases stop at 100 m (the reference's grid grows as f^2).
STARTED_HIGH_REACH_M = 100.0


# -- the engine's runs -------------------------------------------------------------------


def samples(band: int) -> np.ndarray:
    """The frequencies the engine's runs behind barriers take in ``band``."""
    dense = [n for start, n in BARRIER_SAMPLES if CENTRES[band] >= start]
    return band_frequencies(dense[-1])[0][band] if dense else band_frequencies()[0][band]


def engine_levels(band, speed, impedance, zs, at, cut, speed_at=None) -> np.ndarray:
    """(n,) the band level (dB over free field) of the engine's parabolic equation with
    ``cut`` inside, at the points ``at`` (range, height), as its runs behind barriers
    take it: its grid, samples, absorbing layer, every point on its own last step."""
    f = samples(band)
    ranges = np.array([r for r, _ in at], dtype=float)
    heights = np.array([z for _, z in at], dtype=float)
    top = max(
        1.5 * max(zs, heights.max()),
        max(zs, heights.max()) + WAVE_MIN_HEADROOM_M,
        WAVE_TOP_SLOPE * ranges.max(),
        BARRIER_HEADROOM * max((t for _, _, t in cut), default=0.0),
    )
    coarse = CENTRES[band] >= BARRIER_COARSE_FROM_HZ
    steps = parabolic.STEPS_PER_WAVELENGTH * (BARRIER_COARSE if coarse else 1.0)
    p = parabolic.march(
        f,
        speed,
        impedance,
        zs,
        ranges,
        heights,
        top,
        speed_at=speed_at,
        independent=True,
        obstacles=cut,
        refracting=True,
        steps_per_wavelength=steps,
    )
    distance = np.hypot(ranges, heights - zs)
    return 10 * np.log10(np.mean(np.abs(p) ** 2, axis=0) * distance**2)


def wind(u10: float, towards: float = 1.0):
    """The effective speed of sound (m/s, a function of height) along a way at cosine
    ``towards`` to a wind of ``u10`` m/s at 10 m, logarithmic over a roughness of 5 cm,
    the temperature the same at every height."""
    atmosphere = Atmosphere.model_validate(
        {
            "wind": {"speed_m_s": abs(u10), "direction_deg": 0.0 if u10 >= 0 else 180.0},
            "roughness_m": 0.05,
        }
    )
    direction = np.array([towards, math.sqrt(max(1.0 - towards * towards, 0.0))])
    return lambda z: np.asarray(
        atmosphere.sound_speed(np.atleast_1d(np.asarray(z, dtype=float)), GROUND_C, direction),
        dtype=float,
    )


def preset(name: str, towards: float):
    """As :func:`wind`, for one of the atmosphere's presets blowing along +x."""
    atmosphere = Atmosphere.model_validate({"preset": name, "wind": {"direction_deg": 0.0}})
    direction = np.array([towards, math.sqrt(max(1.0 - towards * towards, 0.0))])
    return lambda z: np.asarray(
        atmosphere.sound_speed(np.atleast_1d(np.asarray(z, dtype=float)), GROUND_C, direction),
        dtype=float,
    )


def uniform(c0: float):
    return lambda z: np.full(np.shape(np.atleast_1d(z)), float(c0))


def grass_impedance(f: np.ndarray) -> np.ndarray:
    """The grass's normalised impedance, time as exp(-i omega t)."""
    return np.conj(surface_impedance(f, GRASS))


# -- (a) the full-wave solution --------------------------------------------------------


class _Modes:
    """The modes of one stretch of range: on the cells from ``floor`` up (the ground's
    impedance ``ground`` at the bottom when ``floor`` is 0, else a rigid roof), with a
    perfectly matched layer of thickness ``layer`` above ``zmax``; waves in range round
    an axis (Hankel functions), or along a plane (``plane``: exponentials)."""

    def __init__(self, f, speed, h, cells, floor, ground, zmax, layer, c_ref, plane=False):
        self.plane = plane
        omega = 2 * math.pi * f
        z = (np.arange(cells) + 0.5) * h
        faces = np.arange(cells + 1) * h
        strongest = 20.7 * c_ref / layer

        def stretch(x):
            return 1 + 1j * strongest * np.clip((x - zmax) / layer, 0.0, None) ** 2 / omega

        k2 = (omega / speed(z)) ** 2
        conductance = 1.0 / (h * stretch(faces))
        n = cells - floor
        index = np.arange(n)
        cell = index + floor
        up = conductance[cell + 1] / h
        down = conductance[cell] / h
        down[0] = 0.0  # the floor
        operator = np.zeros((n, n), dtype=complex)
        operator[index, index] = -(up + down)
        operator[index[:-1], index[:-1] + 1] = up[:-1]
        operator[index[:-1] + 1, index[:-1]] = up[:-1]
        if floor == 0 and ground is not None:
            k0 = omega / float(speed(np.array([0.0]))[0])
            operator[0, 0] += 1j * k0 / (ground * h)
        weight = stretch(z)[cell]
        scale = 1.0 / np.sqrt(weight)
        values, vectors = la.eig(
            scale[:, None] * (operator + np.diag(weight * k2[cell])) * scale[None, :]
        )
        self.shapes = scale[:, None] * vectors
        self.norms = np.sum(vectors * vectors, axis=0)
        kappa = np.sqrt(values.astype(complex))
        # Outgoing (a trapped mode's imaginary part is round-off of either sign).
        self.kappa = np.where(kappa.imag < -1e-9 * np.abs(kappa), -kappa, kappa)
        self.weight = weight
        self.floor = floor
        self.z = z[cell]

    def out_ratio(self, b):
        if self.plane:
            return 1j * self.kappa
        x = self.kappa * b
        return -self.kappa * special.hankel1e(1, x) / special.hankel1e(0, x)

    def in_ratio(self, b):
        if self.plane:
            return -1j * self.kappa
        x = self.kappa * b
        return -self.kappa * special.hankel2e(1, x) / special.hankel2e(0, x)

    def outgoing(self, r, b):  # H0(kappa r) / H0(kappa b)
        if self.plane:
            return np.exp(1j * self.kappa * (r - b))
        return (
            special.hankel1e(0, self.kappa * r)
            / special.hankel1e(0, self.kappa * b)
            * np.exp(1j * self.kappa * (r - b))
        )

    def incoming(self, r, b):  # H0^(2)(kappa r) / H0^(2)(kappa b)
        if self.plane:
            return np.exp(-1j * self.kappa * (r - b))
        return (
            special.hankel2e(0, self.kappa * r)
            / special.hankel2e(0, self.kappa * b)
            * np.exp(-1j * self.kappa * (r - b))
        )


def full_wave(f, speed, zs, at, zmax, cut=(), ground=None, h=None, plane=False) -> np.ndarray:
    """(n,) the pressure (time as exp(-i omega t), free field exp(ikR) / (4 pi R)) at
    the points ``at`` (range, height) from a source ``zs`` m up on the axis, at ``f`` Hz
    in air of effective speed ``speed(z)``, over a ground of normalised impedance
    ``ground`` (None: rigid), with ``cut`` (near, far, top) round the axis: thin rigid
    ring screens (near is far) and rigid ring blocks. Tops lie on the cells' faces (a
    multiple of ``h``, by default a twentieth of the wavelength). With ``plane``, a line
    source across a plane instead (free field i H0(kR) / 4) with straight screens and
    blocks along it: the two dimensions of the engine's parabolic equation.

    A screen blocks the radial flux of the cells below its top (dp/dr = 0 on both its
    faces); above it p and dp/dr are continuous at each cell's centre. Over a block the
    field lives on the cells above its roof, over a rigid floor. Sound a screen sends
    back towards the source goes in as H0^(2) and is not sent out again from the axis
    (a ring would focus it there; a straight barrier sends it away); along a plane it
    goes on past the source."""
    c_ref = float(np.min(speed(np.linspace(0.0, zmax, 400))))
    wavelength = c_ref / f
    h = wavelength / 20.0 if h is None else h
    layer = 3.0 * wavelength
    cells = int(math.ceil((zmax + layer) / h))
    common = {"ground": ground, "zmax": zmax, "layer": layer, "c_ref": c_ref, "plane": plane}
    base = _Modes(f, speed, h, cells, 0, **common)
    walls = []
    for near, far, top in cut:
        if far == near:
            walls.append((near, "screen", int(round(top / h))))
        else:
            walls.append((near, "front", int(round(top / h))))
            walls.append((far, "back", int(round(top / h))))
    walls.sort()
    regions = [base]
    for _, kind, roof in walls:
        regions.append(_Modes(f, speed, h, cells, roof, **common) if kind == "front" else base)
    last = len(regions) - 1
    kinds = ["in" if i == 0 else "out" if i == last else "both" for i in range(len(regions))]
    offsets, total = [], 0
    for i, region in enumerate(regions):
        offsets.append(total)
        total += len(region.kappa) * (2 if kinds[i] == "both" else 1)
    # The source in region 0, as a sum of modes.
    t = zs / h - 0.5
    below = int(math.floor(t))
    point = np.zeros(cells, dtype=complex)
    point[below] = (1 - (t - below)) / h
    point[below + 1] = (t - below) / h
    alpha = (base.shapes.T @ (base.weight * point)) / base.norms

    def source(region, r, derivative):
        """Each mode's part of the source's own field at range ``r`` (or its range
        derivative), per unit of its amplitude ``alpha``."""
        kappa = region.kappa
        if plane:
            wave = np.exp(1j * kappa * r)
            return -0.5 * wave if derivative else 0.5j / kappa * wave
        x = kappa * r
        return 0.25j * (-kappa * special.hankel1(1, x) if derivative else special.hankel1(0, x))

    def terms(i, b, derivative):
        region, n = regions[i], len(regions[i].kappa)
        out = []
        if kinds[i] == "in":
            out.append((offsets[i], region.in_ratio(b) if derivative else np.ones(n)))
        elif kinds[i] == "out":
            out.append((offsets[i], region.out_ratio(b) if derivative else np.ones(n)))
        else:
            left, right = walls[i - 1][0], walls[i][0]
            outgoing, incoming = region.outgoing(b, left), region.incoming(b, right)
            if derivative:
                outgoing = outgoing * region.out_ratio(b)
                incoming = incoming * region.in_ratio(b)
            out += [(offsets[i], outgoing), (offsets[i] + n, incoming)]
        known = None
        if i == 0:
            known = alpha * source(region, b, derivative)
        return out, known

    rows, right_side = [], []
    for w, (b, _, roof) in enumerate(walls):
        left_region, right_region = regions[w], regions[w + 1]
        for derivative in (False, True):
            left, known = terms(w, b, derivative)
            right, _ = terms(w + 1, b, derivative)
            for j in range(cells):
                in_left, in_right = j >= left_region.floor, j >= right_region.floor
                if j >= roof and in_left and in_right:
                    row = np.zeros(total, dtype=complex)
                    for offset, coefficient in left:
                        row[offset : offset + len(coefficient)] += (
                            left_region.shapes[j - left_region.floor] * coefficient
                        )
                    for offset, coefficient in right:
                        row[offset : offset + len(coefficient)] -= (
                            right_region.shapes[j - right_region.floor] * coefficient
                        )
                    value = 0.0
                    if known is not None:
                        value = -(left_region.shapes[j - left_region.floor] @ known)
                    rows.append(row)
                    right_side.append(value)
                elif derivative:
                    # A rigid face on each side that has the cell.
                    for region, side, given in (
                        (left_region, left, known),
                        (right_region, right, None),
                    ):
                        if j < region.floor:
                            continue
                        row = np.zeros(total, dtype=complex)
                        for offset, coefficient in side:
                            row[offset : offset + len(coefficient)] += (
                                region.shapes[j - region.floor] * coefficient
                            )
                        value = 0.0 if given is None else -(region.shapes[j - region.floor] @ given)
                        rows.append(row)
                        right_side.append(value)
    unknowns = None
    if walls:
        # The modes deep in the absorbing layer make the system ill-conditioned by its
        # scale alone; the self-checks measure the solution.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", la.LinAlgWarning)
            unknowns = la.solve(np.array(rows), np.array(right_side, dtype=complex))
    bounds = [0.0] + [w[0] for w in walls] + [math.inf]

    def pressure(r, z):
        i = int(np.searchsorted(bounds, r, side="right") - 1)
        region, n = regions[i], len(regions[i].kappa)
        if kinds[i] == "in":
            coefficient = alpha * source(region, r, False)
            if walls:
                coefficient = coefficient + unknowns[:n] * region.incoming(r, bounds[1])
        elif kinds[i] == "out":
            coefficient = unknowns[offsets[i] : offsets[i] + n] * region.outgoing(r, bounds[i])
        else:
            coefficient = unknowns[offsets[i] : offsets[i] + n] * region.outgoing(
                r, bounds[i]
            ) + unknowns[offsets[i] + n : offsets[i] + 2 * n] * region.incoming(r, bounds[i + 1])
        shapes = np.array(
            [
                np.interp(z, region.z, s.real) + 1j * np.interp(z, region.z, s.imag)
                for s in region.shapes.T
            ]
        )
        return shapes @ coefficient

    return np.array([pressure(float(r), float(z)) for r, z in at])


def cell(band: int) -> float:
    """The full-wave solution's cell height (m) in ``band``: a twentieth of the shortest
    wavelength at the ground, the same in the wind and in still air."""
    c0 = float(wind(0.0)(np.array([0.0]))[0])
    return c0 / band_frequencies()[0][band].max() / 20.0


def free(k: float, distance, plane: bool):
    """The free field of a point source (exp(ikR) / (4 pi R)) or a line source (i
    H0(kR) / 4) ``distance`` away, at wavenumber ``k``."""
    distance = np.asarray(distance, dtype=float)
    if plane:
        return 0.25j * special.hankel1(0, k * distance)
    return np.exp(1j * k * distance) / (4 * math.pi * distance)


def full_wave_levels(band, speed, zs, at, cut, ground, top=None, plane=False) -> np.ndarray:
    """(n,) the full-wave band level (dB over free field) at the engine's samples, the
    tops on its cells (as the engine's are given them here), on a domain ``top`` m high
    under its absorbing layer (default :data:`FULL_WAVE_TOP_M`)."""
    f = band_frequencies()[0][band]
    h = cell(band)
    top = FULL_WAVE_TOP_M if top is None else top
    ranges = np.array([r for r, _ in at], dtype=float)
    heights = np.array([z for _, z in at], dtype=float)
    distance = np.hypot(ranges, heights - zs)
    c0 = float(speed(np.array([0.0]))[0])
    energy = []
    for k, fk in enumerate(f):
        z = None if ground is None else ground[k]
        p = full_wave(fk, speed, zs, at, top, cut, z, h, plane)
        energy.append(np.abs(p / free(2 * math.pi * fk / c0, distance, plane)) ** 2)
    return 10 * np.log10(np.mean(energy, axis=0))


def snapped(cut, band: int):
    """``cut`` with its tops on the full-wave solution's cells in ``band``."""
    h = cell(band)
    return [(a, b, round(t / h) * h) for a, b, t in cut]


def case_options(case: dict, quick: bool, filters: dict):
    """A ``full_wave`` case's bands (indices), winds and references, narrowed by
    ``filters`` (``bands``: centres in Hz, ``winds``, ``references``; None: all)."""
    bands = case.get("bands", (1, 3) if quick else (0, 1, 2, 3))
    winds = case.get("winds", WINDS)
    references = case.get("references", ("ring",))
    if filters.get("bands"):
        bands = [b for b in bands if CENTRES[b] in filters["bands"]]
    if filters.get("winds"):
        winds = [u for u in winds if u in filters["winds"]]
    if filters.get("references"):
        references = [r for r in references if r in filters["references"]]
    return bands, winds, references


def domain_of(case: dict, u: float) -> float:
    """The full-wave domain's height (m) under its absorbing layer for ``case`` in a
    wind of ``u`` (negative: upwind)."""
    top = case.get("top", FULL_WAVE_TOP_M)
    return case.get("top_upwind", top) if u < 0 else top


def check_full_wave(names, quick: bool, filters: dict | None = None, own: bool = True) -> dict:
    """The engine's weather effect against the full-wave solution's (rigid ground and
    grass, 63 to 500 Hz unless the case says, the winds of :data:`WINDS` unless it
    says), round an axis and, where the case says, along a plane; and (``own``) the
    solution's own checks."""
    print("(a) Full-wave mode matching: the engine's effect less the solution's (dB)")
    filters = filters or {}
    out = {"self": full_wave_self_checks()} if own else {}
    for name in names:
        case = FULL_WAVE[name]
        zs, at = case["zs"], case["at"]
        bands, winds, references = case_options(case, quick, filters)
        tops = {u: domain_of(case, u) for u in winds}
        for reference_name in references:
            plane = reference_name == "plane"
            for ground_name in ("rigid", "grass"):
                for band in bands:
                    started = time.time()
                    f = band_frequencies()[0][band]
                    ground = None if ground_name == "rigid" else grass_impedance(f)
                    impedance = None if ground is None else (lambda r, z=ground: z)
                    cut = snapped(case["cut"], band)
                    still = uniform(float(wind(0.0)(np.array([0.0]))[0]))
                    reference_still = {
                        top: full_wave_levels(band, still, zs, at, cut, ground, top, plane)
                        for top in sorted(set(tops.values()))
                    }
                    engine_still = engine_levels(band, still, impedance, zs, at, cut)
                    for u in winds:
                        speed = wind(u)
                        top = tops[u]
                        reference = (
                            full_wave_levels(band, speed, zs, at, cut, ground, top, plane)
                            - reference_still[top]
                        )
                        engine = engine_levels(band, speed, impedance, zs, at, cut) - engine_still
                        key = f"{name} {ground_name} {CENTRES[band]} Hz {u:+g} m/s"
                        if plane:
                            key += " plane"
                        out[key] = {
                            "case": name,
                            "reference": reference_name,
                            "ground": ground_name,
                            "band": CENTRES[band],
                            "wind": u,
                            "at": at,
                            "top_m": top,
                            "reference_effect_db": reference.tolist(),
                            "engine_effect_db": engine.tolist(),
                            "error_db": (engine - reference).tolist(),
                            "reference_still_db": reference_still[top].tolist(),
                            "engine_still_db": engine_still.tolist(),
                        }
                        print(
                            f"  {key:38s} effect"
                            + "".join(f"{v:7.2f}" for v in reference)
                            + "   error"
                            + "".join(f"{v:+7.2f}" for v in engine - reference),
                            flush=True,
                        )
                    print(f"    ({time.time() - started:.0f} s)", flush=True)
    return out


def full_wave_self_checks() -> dict:
    """The full-wave solution's own checks: with nothing on the way, over rigid ground
    against the exact field of the source and its image (round an axis and along a
    plane), and in wind against the engine's parabolic equation (each sample); a screen
    in still air against a straight one's four paths (the uniform theory of
    diffraction; a ring round an axis, a straight screen along a plane); and in the
    upwind shadow behind a screen, the level on :data:`FULL_WAVE_TOP_M` and on
    :data:`DOMAIN_CHECK`'s domains against that on the tallest."""
    out = {}
    at = [(30.0, 1.5), (100.0, 1.5), (100.0, 4.0)]
    still = uniform(343.0)
    for plane in (False, True):
        errors = []
        for f in (63.0, 250.0):
            p = full_wave(f, still, 1.0, at, FULL_WAVE_TOP_M, plane=plane)
            k = 2 * math.pi * f / 343.0
            exact = [sum(free(k, math.hypot(r, z - s), plane) for s in (1.0, -1.0)) for r, z in at]
            errors += list(20 * np.log10(np.abs(p) / np.abs(exact)))
        out["plane_two_paths_db" if plane else "two_paths_db"] = float(np.max(np.abs(errors)))
    speed = wind(5.0)
    f = band_frequencies()[0][1]
    p = 4 * math.pi * np.array([full_wave(fk, speed, 1.0, at, FULL_WAVE_TOP_M) for fk in f])
    q = parabolic.march(
        f, speed, None, 1.0, np.array([r for r, _ in at]), np.array([z for _, z in at]), 30.0
    )
    out["open_wind_pe_db"] = float(np.max(np.abs(20 * np.log10(np.abs(q) / np.abs(p)))))
    # A screen in still air against a straight one's four paths: round the axis the two
    # differ in the edge's curve only.
    for plane in (False, True):
        errors = []
        for band in (1, 2):
            f, h = band_frequencies()[0][band], cell(band)
            top, at = round(4.0 / h) * h, [(30.0, 1.5), (60.0, 1.5)]
            screen = [(15.0, 15.0, top)]
            solved = [
                np.abs(full_wave(fk, still, 1.0, at, FULL_WAVE_TOP_M, screen, None, h, plane)) ** 2
                for fk in f
            ]
            scale = 1.0 if plane else 1.0 / (4 * math.pi)
            straight = [
                [
                    abs(scale * four_paths(2 * math.pi * fk / 343.0, 1.0, 15.0, top, x, plane)) ** 2
                    for x in at
                ]
                for fk in f
            ]
            errors += list(10 * np.log10(np.mean(solved, axis=0) / np.mean(straight, axis=0)))
        name = "plane_screen_four_paths_db" if plane else "ring_screen_four_paths_db"
        out[name] = [float(min(errors)), float(max(errors))]
    # The domain: deep in the upwind shadow the sound comes down from high up.
    f = band_frequencies()[0][DOMAIN_CHECK["band"]][DOMAIN_CHECK["sample"]]
    h = cell(DOMAIN_CHECK["band"])
    cut = snapped(DOMAIN_CHECK["cut"], DOMAIN_CHECK["band"])
    domain = {}
    for u in DOMAIN_CHECK["winds"]:
        level = {}
        for top in (FULL_WAVE_TOP_M, *DOMAIN_CHECK["tops"]):
            p = full_wave(f, wind(u), DOMAIN_CHECK["zs"], DOMAIN_CHECK["at"], top, cut, None, h)
            level[top] = 20 * np.log10(np.abs(p))
        tallest = max(level)
        domain[f"{u:+g}"] = {
            f"{top:g}": (level[top] - level[tallest]).tolist() for top in level if top != tallest
        }
    out["domain_db"] = domain
    print(
        f"  self: two paths {out['two_paths_db']:.3f} dB round an axis, "
        f"{out['plane_two_paths_db']:.3f} dB along a plane; open ground in wind against the "
        f"engine's equation, each sample of 125 Hz, {out['open_wind_pe_db']:.2f} dB; a screen "
        f"in still air against a straight one's four paths, 125 and 250 Hz, "
        + "{:+.2f} to {:+.2f} dB round an axis, ".format(*out["ring_screen_four_paths_db"])
        + "{:+.2f} to {:+.2f} dB along a plane".format(*out["plane_screen_four_paths_db"]),
        flush=True,
    )
    for u, rows in domain.items():
        print(
            f"  self: {f:.0f} Hz, {u} m/s, at {DOMAIN_CHECK['at']}, against a domain "
            f"{max(DOMAIN_CHECK['tops']):g} m high: "
            + "; ".join(
                f"{top} m " + " ".join(f"{v:+.2f}" for v in values) for top, values in rows.items()
            ),
            flush=True,
        )
    return out


# -- (b) the started parabolic equation --------------------------------------------------


def _tail(a: float) -> complex:
    """The integral of exp(i t^2) from a to infinity."""
    u = a * math.sqrt(2 / math.pi)
    s, c = fresnel(u)
    return math.sqrt(math.pi / 2) * ((0.5 - c) + 1j * (0.5 - s))


def _transition(x: float) -> complex:
    """Kouyoumjian and Pathak's transition function, time as exp(-i omega t)."""
    if x < 1e-12:
        return 0.0
    a = math.sqrt(x)
    return -2j * a * np.exp(-1j * x) * _tail(a)


def half_plane(k: float, source, edge, listener, line: bool = False) -> complex:
    """The field (time as exp(-i omega t), source exp(ikR) / R, or with ``line`` a line
    source along the edge, i H0(kR) / 4) at ``listener`` (x, z) from ``source`` behind a
    rigid half-plane hanging down from ``edge`` (x, z): the incident wave where it is
    lit, and the edge's (the uniform theory of diffraction)."""

    def angle(p):
        return math.atan2(-(p[0] - edge[0]), -(p[1] - edge[1])) % (2 * math.pi)

    phi_s, phi = angle(source), angle(listener)
    s1 = math.hypot(source[0] - edge[0], source[1] - edge[1])
    s = math.hypot(listener[0] - edge[0], listener[1] - edge[1])
    length = s * s1 / (s + s1)
    d = 0.0
    for beta in (phi - phi_s, phi + phi_s):
        d += _transition(k * length * 2 * math.cos(beta / 2) ** 2) / math.cos(beta / 2)
    d *= -np.exp(1j * math.pi / 4) / (2 * math.sqrt(2 * math.pi * k))
    direct = math.hypot(listener[0] - source[0], listener[1] - source[1])
    lit = phi < phi_s + math.pi
    if line:
        edge_wave = 0.25j * special.hankel1(0, k * s1) * d * np.exp(1j * k * s) / math.sqrt(s)
        return (0.25j * special.hankel1(0, k * direct) if lit else 0.0) + edge_wave
    edge_wave = np.exp(1j * k * s1) / s1 * d * math.sqrt(s1 / (s * (s + s1))) * np.exp(1j * k * s)
    return (np.exp(1j * k * direct) / direct if lit else 0.0) + edge_wave


def four_paths(k, zs, screen, height, listener, line: bool = False) -> complex:
    """The rigid screen over rigid ground: the source and its image to the listener and
    its image, each over the half-plane (``line``: a line source, see
    :func:`half_plane`)."""
    x, z = listener
    return sum(
        half_plane(k, (0.0, a), (screen, height), (x, b), line) for a in (zs, -zs) for b in (z, -z)
    )


def started_reference(f, c0, speed, zs, screen, height, at) -> dict:
    """The field at ``at`` from the split-step equation started 1 m behind the screen
    from :func:`four_paths`, marched in ``speed(z)`` (rigid ground as an even field,
    an absorbing sponge above the domain)."""
    k = 2 * math.pi * f / c0
    wavelength = c0 / f
    zmax = max(height, 4.0) + 25.0
    layer = max(30.0, 20 * wavelength)
    extent = zmax + layer
    count = 1 << int(math.ceil(math.log2(2 * extent / (wavelength / 16))))
    dz = 2 * extent / count
    z = -extent + dz * (np.arange(count) + 0.5)
    up = z > 0
    start = screen + 1.0
    field = np.zeros(count, dtype=complex)
    field[up] = [four_paths(k, zs, screen, height, (start, zz)) for zz in z[up]]
    field[~up] = field[up][::-1]
    psi = field * math.sqrt(start) * np.exp(-1j * k * start)
    kz = 2 * math.pi * np.fft.fftfreq(count, d=dz)
    kx = np.sqrt((k * k - kz * kz).astype(complex))
    kx = np.where(kx.imag < 0, np.conj(kx), kx)
    dk = 2 * math.pi * f / speed(np.abs(z)) - k
    sponge = np.ones(count)
    beyond = np.abs(z) > zmax
    sponge[beyond] = np.exp(-(((np.abs(z[beyond]) - zmax) / layer) ** 2) * 3.0)
    longest = min(0.5, wavelength)
    r = start
    out = {}
    for x, zr in sorted(at):
        while r < x - 1e-9:
            dr = min(longest, x - r)
            half = np.exp(0.5j * dr * dk)
            psi = sponge * half * np.fft.ifft(np.fft.fft(half * psi) * np.exp(1j * dr * (kx - k)))
            r += dr
        value = np.interp(zr, z, psi.real) + 1j * np.interp(zr, z, psi.imag)
        out[(x, zr)] = value * np.exp(1j * k * x) / math.sqrt(x)
    return out


def check_started(names, quick: bool, only_bands=None) -> dict:
    """The engine's weather effect behind a screen with the wind behind it only,
    against the equation started from the exact field (in the bands of centres
    ``only_bands``, Hz, when given)."""
    print("(b) Started parabolic equation: the engine's effect less the reference's (dB)")
    out = {}
    bands = (1, 3, 5) if quick else range(1, 8)
    if only_bands:
        bands = [b for b in bands if CENTRES[b] in only_bands]
    for name in names:
        weather, towards, screen, height = STARTED[name]
        speed = preset(weather, towards)
        c0 = float(speed(np.array([0.0]))[0])
        calm = uniform(c0)
        zs = 1.0
        for band in bands:
            started = time.time()
            reach = STARTED_HIGH_REACH_M if CENTRES[band] >= 4000 else math.inf
            at = [p for p in STARTED_AT if screen + 2.0 < p[0] <= reach]
            f = samples(band)
            energy = {"wind": [], "still": [], "paths": []}
            for fk in f:
                windy = started_reference(fk, c0, speed, zs, screen, height, at)
                still = started_reference(fk, c0, calm, zs, screen, height, at)
                k = 2 * math.pi * fk / c0
                energy["wind"].append([abs(windy[p]) ** 2 for p in at])
                energy["still"].append([abs(still[p]) ** 2 for p in at])
                energy["paths"].append([abs(four_paths(k, zs, screen, height, p)) ** 2 for p in at])
            level = {key: 10 * np.log10(np.mean(v, axis=0)) for key, v in energy.items()}
            reference = level["wind"] - level["still"]

            def behind(r, speed=speed, calm=calm, screen=screen):
                return speed if r >= screen else calm

            cut = [(screen, screen, height)]
            engine = engine_levels(band, calm, None, zs, at, cut, speed_at=behind) - engine_levels(
                band, calm, None, zs, at, cut
            )
            key = f"{name} {CENTRES[band]} Hz"
            out[key] = {
                "case": name,
                "band": CENTRES[band],
                "at": at,
                "reference_effect_db": reference.tolist(),
                "engine_effect_db": engine.tolist(),
                "error_db": (engine - reference).tolist(),
                "starter_db": (level["still"] - level["paths"]).tolist(),
            }
            print(
                f"  {key:22s} effect"
                + "".join(f"{v:7.2f}" for v in reference)
                + "   error"
                + "".join(f"{v:+7.2f}" for v in engine - reference)
                + f"   starter {np.abs(level['still'] - level['paths']).max():.2f}"
                + f"  ({time.time() - started:.0f} s)",
                flush=True,
            )
    return out


# -- (c) the engine end to end ---------------------------------------------------------


def engine_scene(case: dict, u10: float | None):
    """A scene of a ``full_wave`` case: its obstacles 1 km long across the x axis, rigid
    ground, the listeners along the axis, in a wind of ``u10`` (None: still air)."""
    from acoustic_spaces import Scene

    barriers, buildings = [], []
    for k, (near, far, top) in enumerate(case["cut"]):
        if far == near:
            barriers.append(
                {"name": f"b{k}", "path": [[near, -500.0], [near, 500.0]], "height": top}
            )
        else:
            footprint = [[near, -500.0], [far, -500.0], [far, 500.0], [near, 500.0]]
            buildings.append({"name": f"h{k}", "footprint": footprint, "height": top})
    reach = max(r for r, _ in case["at"])
    data = {
        "geometry": {
            "ground": {"extent": [[-20.0, -600.0], [reach + 30.0, 600.0]], "sky": 40.0},
            "barriers": barriers,
            "buildings": buildings,
        },
        "materials": {"rigid": {"absorption": [0.0] * 8, "scattering": 0.0}},
        "default_material": "rigid",
        "air": {"temperature_c": GROUND_C},
        "sources": [{"name": "s", "position": [0.0, 0.0, case["zs"]]}],
        "listeners": [
            {"name": f"l{k}", "position": [r, 0.0, z]} for k, (r, z) in enumerate(case["at"])
        ],
        "simulation": {
            "seed": 1,
            "early": {
                "method": "image_sources",
                "max_order": 1,
                "diffraction": "second_order"
                if len(case["cut"]) > 1 or buildings
                else "first_order",
                "diffraction_reflections": "both_sides",
            },
            "late": {"rays": 100},
            "ir_length_s": 0.2 + reach / 343.0,
        },
    }
    if u10 is not None:
        data["atmosphere"] = {
            "wind": {"speed_m_s": abs(u10), "direction_deg": 0.0 if u10 >= 0 else 180.0},
            "roughness_m": 0.05,
        }
    return Scene.model_validate(data)


def check_engine(names, full: dict, quick: bool = False, filters: dict | None = None) -> dict:
    """:func:`acoustic_spaces.simulate`'s early paths in wind and in still air against
    the full-wave effect over rigid ground (from ``full``, the ``full_wave`` results),
    for each reference there."""
    from acoustic_spaces import Room
    from acoustic_spaces.engine.coherent import total_energy
    from acoustic_spaces.engine.simulate import early_paths

    print("(c) The engine end to end: its effect less the full-wave solution's (dB)")
    out = {}
    for name in names:
        case = FULL_WAVE[name]
        bands, winds, references = case_options(case, quick, filters or {})
        started = time.time()
        levels = {}
        for u in (None, *winds):
            room = Room.from_scene(engine_scene(case, u))
            levels[u] = np.array(
                [
                    10 * np.log10(total_energy(early_paths(room, listener=f"l{k}")))
                    for k in range(len(case["at"]))
                ]
            )  # (listeners, bands)
        for reference_name in references:
            suffix = " plane" if reference_name == "plane" else ""
            for u in winds:
                for band in bands:
                    reference = full.get(f"{name} rigid {CENTRES[band]} Hz {u:+g} m/s{suffix}")
                    if reference is None:
                        continue
                    effect = levels[u][:, band] - levels[None][:, band]
                    error = effect - np.array(reference["reference_effect_db"])
                    key = f"{name} {CENTRES[band]} Hz {u:+g} m/s{suffix}"
                    out[key] = {
                        "case": name,
                        "reference": reference_name,
                        "band": CENTRES[band],
                        "wind": u,
                        "engine_effect_db": effect.tolist(),
                        "error_db": error.tolist(),
                        "reference_effect_db": reference["reference_effect_db"],
                        "own_error_db": reference["error_db"],
                    }
                    print(
                        f"  {key:34s} effect"
                        + "".join(f"{v:7.2f}" for v in effect)
                        + "   error"
                        + "".join(f"{v:+7.2f}" for v in error),
                        flush=True,
                    )
        print(f"    ({time.time() - started:.0f} s)", flush=True)
    return out


# -- (d) convergence of the engine's runs --------------------------------------------------


#: (d) The cases (as :data:`FULL_WAVE`'s), and ``high``: listeners high and close
#: behind a barrier, the sound reaching them steeply.
CONVERGENCE = {
    **{name: FULL_WAVE[name] for name in ("near", "rail", "roof", "steep60")},
    "high": {"zs": 1.0, "cut": [(10.0, 10.0, 3.0)], "at": [(25, 10.0), (25, 14.0), (40, 1.5)]},
}
#: The winds of the convergence check (m/s at 10 m).
CONVERGENCE_WINDS = (5.0, -5.0)


def weather_effect(f, u, ground_name, case, steps) -> np.ndarray:
    """(n,) the engine's parabolic equation's weather effect (dB) at ``case``'s
    listeners at the frequencies ``f``, with a grid of ``steps`` a wavelength."""
    impedance = None
    if ground_name == "grass":
        z = grass_impedance(f)
        impedance = lambda r, z=z: z  # noqa: E731
    c0 = float(wind(0.0)(np.array([0.0]))[0])
    levels = []
    for speed in (wind(u), uniform(c0)):
        ranges = np.array([r for r, _ in case["at"]], dtype=float)
        heights = np.array([z for _, z in case["at"]], dtype=float)
        zs = case["zs"]
        top = max(
            1.5 * max(zs, heights.max()),
            max(zs, heights.max()) + WAVE_MIN_HEADROOM_M,
            WAVE_TOP_SLOPE * ranges.max(),
            BARRIER_HEADROOM * max(t for _, _, t in case["cut"]),
        )
        p = parabolic.march(
            f,
            speed,
            impedance,
            zs,
            ranges,
            heights,
            top,
            independent=True,
            obstacles=case["cut"],
            refracting=True,
            steps_per_wavelength=steps,
        )
        levels.append(10 * np.log10(np.mean(np.abs(p) ** 2, axis=0)))
    return levels[0] - levels[1]


def check_convergence(names, quick: bool, only_bands=None) -> dict:
    """The engine's weather effect from its parabolic equation alone, against finer
    runs: from 1 kHz up a tenth of a wavelength's grid against its quarter, and in each
    band its frequencies against 32 an octave (64 where it takes 32)."""
    print("(d) Convergence: the engine's effect less a finer run's (dB)")
    out: dict = {"grid": {}, "samples": {}}
    bands = (3, 5, 7) if quick else range(8)
    if only_bands:
        bands = [b for b in bands if CENTRES[b] in only_bands]
    fine = parabolic.STEPS_PER_WAVELENGTH
    for name in names:
        case = CONVERGENCE[name]
        for ground_name in ("rigid", "grass"):
            for u in CONVERGENCE_WINDS:
                started = time.time()
                for band in bands:
                    f = samples(band)
                    coarse = CENTRES[band] >= BARRIER_COARSE_FROM_HZ
                    steps = fine * (BARRIER_COARSE if coarse else 1.0)
                    own = weather_effect(f, u, ground_name, case, steps)
                    key = f"{name} {ground_name} {CENTRES[band]} Hz {u:+g} m/s"
                    if coarse:
                        finer = weather_effect(f, u, ground_name, case, fine)
                        out["grid"][key] = (own - finer).tolist()
                    count = max(32, 2 * len(f))
                    denser = weather_effect(
                        band_frequencies(count)[0][band], u, ground_name, case, steps
                    )
                    out["samples"][key] = {"count": len(f), "against": count}
                    out["samples"][key]["error_db"] = (own - denser).tolist()
                    grid = out["grid"].get(key)
                    print(
                        f"  {key:34s} grid"
                        + ("".join(f"{v:+7.2f}" for v in grid) if grid else "      -")
                        + f"   {len(f)} against {count}"
                        + "".join(f"{v:+7.2f}" for v in own - denser),
                        flush=True,
                    )
                print(f"    ({time.time() - started:.0f} s)", flush=True)
    return out


# -- acceptance ------------------------------------------------------------------------


def group_of(check: str, row: dict) -> tuple:
    """The key in :data:`LIMITS` of a row of ``check``."""
    if check in ("started", "engine"):
        return (check,)
    case = row["case"]
    family = case if case in ("steep45", "steep60", "two", "two40", "near_up") else "one"
    return (family, row.get("reference", "ring"), abs(row["wind"]))


def acceptance(results: dict) -> bool:
    """Prints the largest error of each check and case where the reference's effect is
    within :data:`DEEPEST_DB` of still air, against its limit in :data:`LIMITS`; end to
    end, the error beyond its parabolic equation's own; True when all hold."""
    print("\nAcceptance (largest |error|, dB, where the effect is within 25 dB; limit):")
    ok = True
    for check in ("full_wave", "started", "engine"):
        worst: dict[str, list] = {}
        for key, row in results.get(check, {}).items():
            if key == "self":
                continue
            error = np.array(row["error_db"])
            if check == "engine":
                error = error - np.array(row["own_error_db"])
            judged = np.abs(np.array(row["reference_effect_db"])) <= DEEPEST_DB
            if not judged.any():
                continue
            name = row["case"]
            if check != "started":
                name += f" {row.get('reference', 'ring')} {abs(row['wind']):g} m/s"
            value = float(np.abs(error[judged]).max())
            limit = LIMITS.get(group_of(check, row), -1.0)
            if name not in worst or value > worst[name][0]:
                worst[name] = [value, limit, key]
        for name, (value, limit, key) in sorted(worst.items()):
            mark = "ok" if value <= limit else "OVER"
            ok &= value <= limit
            print(f"  {check:9s} {name:28s} {value:5.2f}  (limit {limit:.2f}) {mark}  [{key}]")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", choices=["full_wave", "started", "engine", "convergence"], help="one check"
    )
    parser.add_argument("--case", action="append", help="these cases only (repeatable)")
    parser.add_argument("--quick", action="store_true", help="fewer bands")
    parser.add_argument(
        "--band", type=int, action="append", help="these bands only (centre, Hz; repeatable)"
    )
    parser.add_argument(
        "--wind", type=float, action="append", help="(a), (c): these winds only (repeatable)"
    )
    parser.add_argument(
        "--reference",
        choices=["ring", "plane"],
        action="append",
        help="(a), (c): this full-wave reference only",
    )
    parser.add_argument("--full-wave-json", help="(c): the full-wave results to compare with")
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    results: dict = {}
    filters = {"bands": args.band, "winds": args.wind, "references": args.reference}
    if args.only in (None, "full_wave"):
        names = [n for n in FULL_WAVE if not args.case or n in args.case]
        own = not args.case or "self" in args.case
        results["full_wave"] = check_full_wave(names, args.quick, filters, own)
    if args.only in (None, "started"):
        names = [n for n in STARTED if not args.case or n in args.case]
        results["started"] = check_started(names, args.quick, args.band)
    if args.only in (None, "engine"):
        full = results.get("full_wave")
        if full is None and args.full_wave_json:
            with open(args.full_wave_json, encoding="utf-8") as given:
                full = json.load(given)["full_wave"]
        names = [n for n in FULL_WAVE if not args.case or n in args.case]
        results["engine"] = check_engine(names, full or {}, args.quick, filters)
    if args.only in (None, "convergence"):
        names = [n for n in CONVERGENCE if not args.case or n in args.case]
        results["convergence"] = check_convergence(names, args.quick, args.band)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=1)
    return 0 if acceptance(results) else 1


if __name__ == "__main__":
    sys.exit(main())
