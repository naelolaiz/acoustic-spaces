"""The ground wave at low frequencies in a refracting atmosphere: a parabolic equation.

Below a few hundred hertz rays fail outdoors in wind and temperature gradients: the
profiles change fastest within a wavelength of the ground, where the direct sound and its
reflection meet, so the ground dip and the shadow zone come out wrong (by 10 dB and more
against a full-wave solution). There the sound over open ground is marched out from the
source with the Green's-function parabolic equation (GFPE; Gilbert and Di 1993;
Salomons 2001, ch. 5 and app. H), in the vertical plane through the source and the
listener, with the effective speed of sound along it.

The field p = psi(r, z) exp(i ka r) / sqrt(r) (time as exp(-i omega t)) goes from range r
to r + dr by

    psi(r + dr, z) = exp(i dr dk(z) / 2) { F^-1[(Psi(kz) + R(kz) Psi(-kz))
                     exp(i dr (sqrt(ka^2 - kz^2) - ka))]
                     + 2 i beta Psi(beta) exp(i dr (sqrt(ka^2 - beta^2) - ka)) exp(-i beta z) }

with Psi the Fourier transform over the height of exp(i dr dk(z) / 2) psi(r, z) (the
ground's sample weighing half), dk = k(z) - ka, R(kz) = (kz Z - ka) / (kz Z + ka) for the
ground's normalised impedance Z, and beta = ka / Z, the surface wave, kept where it
decays with height. Over a hard ground the pole of R, at kz = -beta, comes close to the
real axis, where the samples miss it: there the field is continued below the ground in
height instead (see the code), and marched in free field. An absorbing layer at the top
takes the sound that goes up; the ground's impedance may change along the way. The
Gaussian starting field (Salomons' eq. G.67) makes the free field exp(ikR) / R. Over
grass, and over hard grounds, in still air it is within 0.2 dB of the exact solution
from 20 m.

The frequencies of one band are marched together, on the grid and in the steps of the
highest, under the absorbing layer of the lowest.
"""

from __future__ import annotations

import math

import numpy as np

#: Height steps per wavelength.
STEPS_PER_WAVELENGTH = 10
#: Range step in wavelengths.
RANGE_STEP_WAVELENGTHS = 5.0
#: Within this many sample spacings of the real axis, the pole of the ground's
#: reflection factor is left to the field continued below the ground.
NEAR_POLE = 3.0
#: The absorbing layer: its thickness in wavelengths (at least LAYER_MIN_M) and its
#: strength (1/m at its top; growing as the square of the depth into it).
LAYER_WAVELENGTHS = 50.0
LAYER_MIN_M = 20.0
LAYER_STRENGTH = 1.0


def march(
    frequencies,
    speed,
    impedance,
    zs: float,
    ranges: np.ndarray,
    heights: np.ndarray,
    top: float,
) -> np.ndarray:
    """(F, n) the complex pressure at ``frequencies`` (F,) Hz (time as exp(-i omega t),
    free field exp(ikR) / R) at the points ``ranges`` (n,, m along the ground from the
    source) and ``heights`` (n,, m above it), from a source at height ``zs``, under the
    effective speed of sound ``speed(z)`` (m/s, an array function of the height).
    ``impedance(r)`` gives the ground's normalised impedance at the frequencies (F,,
    time as exp(-i omega t); infinite: rigid) at range r, or None where it is rigid.
    The absorbing layer starts at height ``top``."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    c0 = float(np.asarray(speed(np.array([0.0])))[0])
    ka = 2.0 * math.pi * f / c0  # (F,)
    shortest, longest = c0 / f.max(), c0 / f.min()
    dz = shortest / STEPS_PER_WAVELENGTH
    thickness = max(LAYER_WAVELENGTHS * longest, LAYER_MIN_M)
    n = int(2 ** math.ceil(math.log2((top + thickness) / dz)))
    z = np.arange(n) * dz
    k = 2.0 * math.pi * f[:, None] / np.asarray(speed(z), dtype=float)[None, :]
    depth = np.clip((z - top) / thickness, 0.0, 1.0)
    delta = (k - ka[:, None]) + 1j * LAYER_STRENGTH * depth**2  # (F, n)

    kz = 2.0 * math.pi * np.fft.fftfreq(2 * n, d=dz)
    vertical = np.sqrt((ka[:, None] ** 2 - kz[None, :] ** 2).astype(complex))
    vertical = np.where(vertical.imag < 0.0, -vertical, vertical)  # (F, 2n)
    mirror = (-np.arange(2 * n)) % (2 * n)
    width = 2.0 * math.pi / (2 * n * dz)

    order = np.argsort(ranges, kind="stable")
    out = np.zeros((len(f), len(ranges)), dtype=complex)
    step = RANGE_STEP_WAVELENGTHS * shortest
    r = 0.0
    tilted = np.zeros((len(f), 2 * n), dtype=complex)
    steps: dict[float, tuple] = {}

    def stepping(dr: float):
        # The phase screen's half step and the propagator over dr (most steps are full).
        if dr not in steps:
            if len(steps) > 4:
                steps.clear()
            steps[dr] = (np.exp(0.5j * dr * delta), np.exp(1j * dr * (vertical - ka[:, None])))
        return steps[dr]

    def impedances(at: float) -> np.ndarray:
        value = impedance(at) if impedance is not None else None
        if value is None:
            return np.full(len(f), np.inf, dtype=complex)
        return np.broadcast_to(np.asarray(value, dtype=complex), (len(f),))

    here = _Ground(impedances(0.0), ka, kz, z, width)
    psi = np.sqrt(1j * ka)[:, None] * (
        np.exp(-(ka[:, None] ** 2) * (z - zs) ** 2 / 2.0)
        + here.normal[:, None] * np.exp(-(ka[:, None] ** 2) * (z + zs) ** 2 / 2.0)
    )
    key = here.key
    for index in order:
        goal = ranges[index]
        while goal - r > 1e-9:
            dr = min(step, goal - r)
            value = impedances(r + 0.5 * dr)
            if not np.array_equal(value, key):
                here = _Ground(value, ka, kz, z, width)
                key = here.key
            half, propagate = stepping(dr)
            screened = half * psi
            tilted[:, :n] = screened
            tilted[:, n:] = 0.0
            # The ground's sample weighs half (the trapezoidal rule), as the reflection
            # adds its mirror image.
            tilted[here.plain, 0] *= 0.5
            if here.image.any():
                image = here.image
                inner = here.growth[image] * screened[image]
                integral = (np.cumsum(inner, axis=1) - 0.5 * (inner[:, :1] + inner)) * dz
                below = screened[image] + here.scale[image, None] * here.shrink[image] * integral
                tilted[image, n + 1 :] = below[:, 1:][:, ::-1]
            spectrum = np.fft.fft(tilted, axis=1) * dz
            field = np.fft.ifft((spectrum + here.reflection * spectrum[:, mirror]) * propagate, axis=1) / dz
            psi_next = field[:, :n]
            surface = here.surface
            if surface.any():
                amplitude = np.sum(tilted[surface, :n] * here.decay[surface], axis=1) * dz
                lift = 2j * here.beta[surface] * amplitude * np.exp(1j * dr * (here.wave[surface] - ka[surface]))
                psi_next[surface] += lift[:, None] * here.decay[surface]
            psi = half * psi_next
            r += dr
        p = psi * (np.exp(1j * ka * r) / math.sqrt(max(r, 1e-9)))[:, None]
        h = heights[index]
        out[:, index] = [np.interp(h, z, row.real) + 1j * np.interp(h, z, row.imag) for row in p]
    return out


class _Ground:
    """How a ground of normalised impedances ``zg`` (F,; infinite: rigid) reflects at
    the march's frequencies: R (F, 2n) and the surface wave (rows ``surface``), or for
    the rows ``image`` the field continued below the ground.

    R = 1 - (2 ka / Z) / (kz - kp) has a pole at kz = kp = -ka / Z. Over a hard ground it
    comes within a few samples' spacing of the real axis, where the samples miss it and
    the march blows up. There the field is continued below the ground instead, as the
    reflection factor has it (the image of R),

        psi(-y) = psi(y) + 2 i (ka / Z) int_0^y exp(-i kp (y - t)) psi(t) dt,

    and marched in free field: what the march does in kz, done in z."""

    def __init__(self, zg, ka, kz, z, width) -> None:
        zg = np.asarray(zg, dtype=complex)
        self.key = zg.copy()
        rigid = ~np.isfinite(zg)
        safe = np.where(rigid, 1.0, zg)
        kp = np.where(rigid, 0.0, -ka / safe)
        self.image = ~rigid & (np.abs(kp.imag) < NEAR_POLE * width)
        self.plain = ~self.image
        self.normal = np.where(rigid, 1.0, (safe - 1.0) / (safe + 1.0))
        reflection = (kz[None, :] * safe[:, None] - ka[:, None]) / (kz[None, :] * safe[:, None] + ka[:, None])
        self.reflection = np.where(rigid[:, None], 1.0, np.where(self.image[:, None], 0.0, reflection))
        self.beta = -kp
        # The surface wave goes as exp(-i beta z): kept where that decays.
        self.surface = ~rigid & self.plain & (self.beta.imag < 0.0)
        wave = np.sqrt((ka**2 - self.beta**2).astype(complex))
        self.wave = np.where(wave.imag < 0.0, -wave, wave)
        self.decay = np.ones((len(zg), len(z)), dtype=complex)
        self.decay[self.surface] = np.exp(-1j * self.beta[self.surface, None] * z[None, :])
        self.growth = np.ones((len(zg), len(z)), dtype=complex)
        self.shrink = np.ones((len(zg), len(z)), dtype=complex)
        self.growth[self.image] = np.exp(1j * kp[self.image, None] * z[None, :])
        self.shrink[self.image] = np.exp(-1j * kp[self.image, None] * z[None, :])
        self.scale = 2j * ka / safe


__all__ = ["march"]
