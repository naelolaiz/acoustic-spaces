"""A Green's-function parabolic equation (GFPE) solver: the reference for refraction.

Outdoor sound over a flat ground of normalised impedance Z under a stratified
atmosphere, in the vertical plane through the source and the listener (axisymmetric
round the source), after Gilbert and Di (1993) and Salomons (2001, ch. 5 and app. H).
Time goes as e^(-iwt); the effective speed of sound c(z) takes the wind along the plane.
Not shipped with the package: validation only.

The field p = psi(r, z) exp(i ka r) / sqrt(r) is marched in range by

    psi(r + dr, z) = exp(i dr dk(z) / 2) { F^-1[(Psi(kz) + R(kz) Psi(-kz))
                     exp(i dr (sqrt(ka^2 - kz^2) - ka))]
                     + 2 i beta Psi(beta) exp(i dr (sqrt(ka^2 - beta^2) - ka)) exp(-i beta z) }

with Psi the Fourier transform of exp(i dr dk(z) / 2) psi(r, z) over the height,
dk = k(z) - ka, R(kz) = (kz Z - ka) / (kz Z + ka) and beta = ka / Z (the surface wave,
kept where it decays with height). An absorbing layer at the top takes what goes up.
The starting field is Gaussian (Salomons' eq. G.67), whose image in the ground has the
plane-wave reflection factor at normal incidence; its scale makes p the field of a
source with exp(ikR)/R in free field, which :func:`level` checks against.
"""

from __future__ import annotations

import math

import numpy as np

#: Grid steps per wavelength in height, and range steps in wavelengths.
STEPS_PER_WAVELENGTH = 10
RANGE_STEP_WAVELENGTHS = 5.0
#: The absorbing layer at the top: its thickness in wavelengths, and its strength (1/m
#: at the top, growing as the square of the depth into it).
LAYER_WAVELENGTHS = 50.0
LAYER_STRENGTH = 1.0


def gfpe(
    frequency: float,
    speed,
    impedance: complex | None,
    zs: float,
    ranges: np.ndarray,
    heights: np.ndarray,
    top: float = 100.0,
) -> np.ndarray:
    """(len(ranges), len(heights)) the complex pressure (e^(-iwt), a source of exp(ikR)/R
    in free field) at ``frequency`` Hz, for a profile ``speed(z)`` (m/s, z in m above
    the ground), a ground of normalised impedance ``impedance`` (e^(-iwt); None: rigid)
    and a source at height ``zs``, at the horizontal ``ranges`` (m, increasing) and the
    ``heights``; ``top`` is the height (m) where the absorbing layer starts."""
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    c0 = float(speed(np.array([0.0]))[0])
    ka = 2.0 * math.pi * frequency / c0
    wavelength = c0 / frequency
    dz = wavelength / STEPS_PER_WAVELENGTH
    z_max = top + LAYER_WAVELENGTHS * wavelength
    n = int(2 ** math.ceil(math.log2(z_max / dz)))
    z = np.arange(n) * dz
    k = 2.0 * math.pi * frequency / speed(z)
    delta = k - ka
    layer = z > top
    depth = (z - top) / (z_max - top)
    delta = delta + 1j * np.where(layer, LAYER_STRENGTH * np.minimum(depth, 1.0) ** 2, 0.0)

    # Wave numbers of a 2n-point transform (the field padded with zeros above z_max).
    kz = 2.0 * math.pi * np.fft.fftfreq(2 * n, d=dz)
    vertical = np.sqrt((ka**2 - kz**2).astype(complex))
    vertical = np.where(vertical.imag < 0.0, -vertical, vertical)
    if impedance is None:
        reflection = np.ones_like(kz, dtype=complex)
        pole = None
    else:
        zg = complex(impedance)
        reflection = (kz * zg - ka) / (kz * zg + ka)
        pole = ka / zg
        if pole.imag > 0.0:
            pole = None  # it would grow with height: no surface wave
    mirror = (-np.arange(2 * n)) % (2 * n)  # index of -kz

    r_start = 0.0
    if impedance is None:
        r0 = 1.0
    else:
        r0 = (complex(impedance) - 1.0) / (complex(impedance) + 1.0)
    psi = np.sqrt(1j * ka) * (
        np.exp(-(ka**2) * (z - zs) ** 2 / 2.0) + r0 * np.exp(-(ka**2) * (z + zs) ** 2 / 2.0)
    )
    out = np.zeros((len(ranges), len(heights)), dtype=complex)
    step = RANGE_STEP_WAVELENGTHS * wavelength
    r = r_start
    targets = list(enumerate(ranges))
    half = None
    while targets:
        index, goal = targets[0]
        dr = min(step, goal - r)
        if dr <= 1e-9:
            # At a requested range: interpolate the field in height.
            p = psi * np.exp(1j * ka * r) / math.sqrt(max(r, 1e-9))
            out[index] = np.interp(heights, z, p.real) + 1j * np.interp(heights, z, p.imag)
            targets.pop(0)
            continue
        half = np.exp(0.5j * dr * delta)
        tilted = np.zeros(2 * n, dtype=complex)
        tilted[:n] = half * psi
        tilted[0] *= 0.5  # the trapezoidal rule: the ground's sample weighs half
        spectrum = np.fft.fft(tilted) * dz
        propagate = np.exp(1j * dr * (vertical - ka))
        field = np.fft.ifft((spectrum + reflection * spectrum[mirror]) * propagate) / dz
        psi_next = field[:n]
        if pole is not None:
            amplitude = np.sum(tilted[:n] * np.exp(-1j * pole * z)) * dz
            wave = np.sqrt(complex(ka**2 - pole**2))
            if wave.imag < 0.0:
                wave = -wave
            psi_next = psi_next + 2j * pole * amplitude * np.exp(1j * dr * (wave - ka)) * np.exp(
                -1j * pole * z
            )
        psi = half * psi_next
        r += dr
    return out


def level(pressure: np.ndarray, r: np.ndarray, zs: float, zr) -> np.ndarray:
    """The level (dB) of ``pressure`` over the free field's at the same distance."""
    distance = np.hypot(np.asarray(r, dtype=float)[:, None], np.asarray(zr, dtype=float)[None] - zs)
    return 20.0 * np.log10(np.abs(pressure) * distance)
