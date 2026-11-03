"""Prototype (probe only): the engine's narrow-angle GFPE (parabolic._march, main 3fc4e7d)
with obstacles in the vertical plane.

- thin screen  ("screen", x, top): at range x the field below ``top`` is set to zero
  (Kirchhoff / Salomons 1994).  mask="smooth" ramps the one grid node that straddles the
  top linearly (m = clip((z - top)/dz + 0.5, 0, 1)) so the result is continuous in ``top``;
  mask="hard" zeroes the nodes with z < top.
- rigid flat roof ("roof", x0, x1, top): the front face zeroes the field below the top at
  x0; between x0 and x1 the GFPE step runs on the sub-grid z >= top with a rigid floor
  there (the same mirror the engine uses for a rigid ground, parabolic.py:338-343); at x1
  the field below the top starts from zero (rear face).  The roof top is snapped to the
  nearest node (sub-grid floor on a node).
- start=(x0, psi0): start from a given field psi(z) (F, n) at range x0 instead of Greene's.

Everything else as parabolic._march: grid, steps, Greene starter, impedance ground,
absorbing layer, speed(z).  Not the steep-point hand-off (receivers here are flat).
Returns (F, n_receivers) complex pressure, free field exp(ikR)/R, time exp(-i w t).
Optionally returns the field at given ranges (snapshots) for starters.
"""
from __future__ import annotations

import math

import numpy as np

from acoustic_spaces.engine import parabolic as P


def _ground_parts(zg, ka, nn, dz):
    kz = 2.0 * math.pi * np.fft.fftfreq(2 * nn, d=dz)
    z = np.arange(nn) * dz
    width = 2.0 * math.pi / (2 * nn * dz)
    g = P._Ground(zg, ka, kz, z, width)
    vertical = np.sqrt((ka[:, None] ** 2 - kz[None, :] ** 2).astype(complex))
    vertical = np.where(vertical.imag < 0.0, -vertical, vertical)
    mirror = (-np.arange(2 * nn)) % (2 * nn)
    return g, kz, vertical, mirror


def march(frequencies, speed, impedance, zs, ranges, heights, top, obstacles=(),
          mask="smooth", start=None, snapshots=(), dz_scale=1.0, step_scale=1.0):
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    c0 = float(np.asarray(speed(np.array([0.0])))[0])
    ka = 2.0 * math.pi * f / c0
    shortest = c0 / f.max()
    n, dz, thickness = P._grid(f, c0, top)
    if dz_scale != 1.0:
        dz *= dz_scale
        n = int(2 ** math.ceil(math.log2((top + thickness) / dz)))
    z = np.arange(n) * dz
    kfull = 2.0 * math.pi * f[:, None] / np.asarray(speed(z), dtype=float)[None, :]
    depth = np.clip((z - top) / thickness, 0.0, 1.0)
    delta = (kfull - ka[:, None]) + 1j * P.LAYER_STRENGTH * depth**2

    def impedances(at):
        value = impedance(at) if impedance is not None else None
        if value is None:
            return np.full(len(f), np.inf, dtype=complex)
        return np.broadcast_to(np.asarray(value, dtype=complex), (len(f),))

    grids = {}

    def parts(j0, zg):
        key = (j0, zg.tobytes())
        if key not in grids:
            if len(grids) > 6:
                grids.clear()
            grids[key] = _ground_parts(zg, ka, n - j0, dz)
        return grids[key]

    def advance(psi, r, dr, j0, zg):
        # One GFPE step on the (sub-)grid from node j0 up, floor impedance zg.
        g, kz, vertical, mirror = parts(j0, zg)
        nn = n - j0
        half = np.exp(0.5j * dr * delta[:, j0:])
        propagate = np.exp(1j * dr * (vertical - ka[:, None]))
        screened = half * psi[:, j0:]
        tilted = np.zeros((len(f), 2 * nn), dtype=complex)
        tilted[:, :nn] = screened
        tilted[g.plain, 0] *= 0.5
        if g.image.any():
            image = g.image
            inner = g.growth[image] * screened[image]
            integral = (np.cumsum(inner, axis=1) - 0.5 * (inner[:, :1] + inner)) * dz
            below = screened[image] + g.scale[image, None] * g.shrink[image] * integral
            below *= g.fade[image]
            tilted[image, nn + 1:] = below[:, 1:][:, ::-1]
        spectrum = np.fft.fft(tilted, axis=1) * dz
        field = np.fft.ifft((spectrum + g.reflection * spectrum[:, mirror]) * propagate, axis=1) / dz
        nxt = field[:, :nn]
        if g.surface.any():
            s = g.surface
            amplitude = np.sum(tilted[s, :nn] * g.decay[s], axis=1) * dz
            lift = 2j * g.beta[s] * amplitude * np.exp(1j * dr * (g.wave[s] - ka[s]))
            nxt[s] += lift[:, None] * g.decay[s]
        out = np.zeros_like(psi)
        out[:, j0:] = half * nxt
        return out

    def cut(psi, h):
        if mask == "smooth":
            m = np.clip((z - h) / dz + 0.5, 0.0, 1.0)
        else:
            m = (z >= h).astype(float)
        return psi * m[None, :]

    events = []
    for ob in obstacles:
        if ob[0] == "screen":
            events.append((ob[1], 0, ("screen", ob[2])))
        elif ob[0] == "roof":
            events.append((ob[1], 0, ("roof_on", ob[3], ob[4] if len(ob) > 4 else None)))
            events.append((ob[2], 1, ("roof_off", ob[3])))
    for i, x in enumerate(ranges):
        events.append((x, 2, ("rec", i)))
    for i, x in enumerate(snapshots):
        events.append((x, 3, ("snap", i)))
    events.sort(key=lambda e: (e[0], e[1]))

    if start is None:
        zg0 = impedances(0.0)
        g0 = P._Ground(zg0, ka, 2.0 * math.pi * np.fft.fftfreq(2 * n, d=dz), z,
                       2.0 * math.pi / (2 * n * dz))
        psi = np.sqrt(1j * ka)[:, None] * (
            P._starter(ka[:, None] * (z - zs)) + g0.normal[:, None] * P._starter(ka[:, None] * (z + zs)))
        r = 0.0
    else:
        r, psi = start
        psi = psi.copy()
    step = max(P.RANGE_STEP_WAVELENGTHS * shortest, P.RANGE_STEP_MIN_M) * step_scale
    out = np.zeros((len(f), len(ranges)), dtype=complex)
    snaps = [None] * len(snapshots)
    j0, roof_zg = 0, None
    for x, _, ev in events:
        while x - r > 1e-9:
            dr = min(step, x - r)
            if j0 == 0:
                zg = impedances(r + 0.5 * dr)
            else:
                zg = roof_zg
            psi = advance(psi, r, dr, j0, zg)
            r += dr
        kind = ev[0]
        if kind == "screen":
            psi = cut(psi, ev[1])
        elif kind == "roof_on":
            j0 = int(round(ev[1] / dz))
            psi[:, :j0] = 0.0
            roof_zg = np.full(len(f), np.inf, dtype=complex) if ev[2] is None else impedances(0.0) * 0 + ev[2]
        elif kind == "roof_off":
            psi[:, :j0] = 0.0
            j0 = 0
        elif kind == "rec":
            i = ev[1]
            p = psi * (np.exp(1j * ka * x) / math.sqrt(max(x, 1e-9)))[:, None]
            h = heights[i]
            out[:, i] = [np.interp(h, z, row.real) + 1j * np.interp(h, z, row.imag) for row in p]
        else:
            snaps[ev[1]] = (x, psi.copy())
    if snapshots:
        return out, snaps, z
    return out


def octave_samples(centre, per=8):
    """The engine's linear samples of an octave band (diffraction.band_frequencies)."""
    from acoustic_spaces.engine.diffraction import band_frequencies
    freqs, _ = band_frequencies(per)
    from acoustic_spaces.materials import OCTAVE_BANDS_HZ
    return freqs[list(OCTAVE_BANDS_HZ).index(centre)]
