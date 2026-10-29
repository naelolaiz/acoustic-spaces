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
takes the sound that goes up; the ground's impedance may change along the way. Greene's
starting field (Greene 1984; Jensen et al., Computational Ocean Acoustics, ch. 6), the
Gaussian's of Salomons' eq. G.67 widened, makes the free field exp(ikR) / R up to some
35 degrees from the horizontal (0.2 dB; the Gaussian's falls 2 dB short there). In
still air over grass and over hard grounds it is within 0.2 dB of the exact solution
from a source 1 m up, 25 to 400 m away, and within 0.45 dB from a wind turbine's hub
178 m away, where the sound comes in at 34 degrees (``validation/ground_reflection.py
--parabolic``).

The frequencies of one band are marched together, on the grid and in the steps of the
highest, under the absorbing layer of the lowest.
"""

from __future__ import annotations

import math

import numpy as np

#: Height steps per wavelength.
STEPS_PER_WAVELENGTH = 10
#: Range step in wavelengths, at least RANGE_STEP_MIN_M (finer changes nothing: within
#: 0.25 dB of steps of a wavelength at 2 kHz, 0.4 dB at 63 Hz, in 5 m/s of wind).
RANGE_STEP_WAVELENGTHS = 5.0
RANGE_STEP_MIN_M = 1.2
#: The pole of the ground's reflection factor is left to the field continued below the
#: ground within this many sample spacings below the real axis (where that field stays
#: bounded) and above it (where it grows with depth). The sampled reflection factor
#: holds from 4 spacings below the axis and 2 above (to 0.1 dB at 400 m).
BELOW_AXIS_POLE = 6.0
ABOVE_AXIS_POLE = 3.0
#: The continued field is faded out over the lower half of its depth, or, where it grows
#: with depth, of the depth at which it has grown by this factor: what lies deeper does
#: not reach the ground in a step, and its edge would ring through the transform.
FADE_GROWTH = 10.0
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
    screen=None,
    speed_at=None,
    independent: bool = False,
) -> np.ndarray:
    """(F, n) the complex pressure at ``frequencies`` (F,) Hz (time as exp(-i omega t),
    free field exp(ikR) / R) at the points ``ranges`` (n,, m along the ground from the
    source) and ``heights`` (n,, m above it), from a source at height ``zs``, under the
    effective speed of sound ``speed(z)`` (m/s, an array function of the height).
    ``impedance(r)`` gives the ground's normalised impedance at the frequencies (F,,
    time as exp(-i omega t); infinite: rigid) at range r, or None where it is rigid.
    The absorbing layer starts at height ``top``. ``screen(r, dr, z)``, when given,
    returns the fluctuation of the index of refraction integrated over the step from r to
    r + dr at heights z (m): a turbulent phase screen, half of whose phase each half step
    takes (for validation, :mod:`validation.turbulence`). ``speed_at(r)``, when given,
    returns the speed function that holds at range r (the same object wherever the
    weather is the same), ``speed`` being the one at the source: each step takes the
    profile of its own middle, the reference wavenumber staying the source's.

    The march goes out in full steps, and a range between two of them is reached by a
    shorter last step. Unless ``independent``, the march goes on from there, so that the
    ranges asked for break up each other's steps; ``independent`` takes each one's last
    step off to the side, so that each range gets what a march to it alone gives, whatever
    else is asked for (the places of a walk sharing a run, say)."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    c0 = float(np.asarray(speed(np.array([0.0])))[0])
    ka = 2.0 * math.pi * f / c0  # (F,)
    shortest = c0 / f.max()
    n, dz, thickness = _grid(f, c0, top)
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
    step = max(RANGE_STEP_WAVELENGTHS * shortest, RANGE_STEP_MIN_M)
    r = 0.0
    tilted = np.zeros((len(f), 2 * n), dtype=complex)
    steps: dict[float, tuple] = {}
    current = speed

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
        _starter(ka[:, None] * (z - zs)) + here.normal[:, None] * _starter(ka[:, None] * (z + zs))
    )
    key = here.key

    def advance(psi: np.ndarray, r: float, dr: float) -> np.ndarray:
        # The field a step dr further on from r, as a new array.
        nonlocal here, key, current, k, delta
        value = impedances(r + 0.5 * dr)
        if not np.array_equal(value, key):
            here = _Ground(value, ka, kz, z, width)
            key = here.key
        if speed_at is not None:
            there = speed_at(r + 0.5 * dr)
            if there is not current:
                current = there
                k = 2.0 * math.pi * f[:, None] / np.asarray(current(z), dtype=float)[None, :]
                delta = (k - ka[:, None]) + 1j * LAYER_STRENGTH * depth**2
                steps.clear()
        half, propagate = stepping(dr)
        if screen is not None:
            half = half * np.exp(0.5j * ka[:, None] * screen(r, dr, z)[None, :])
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
            below *= here.fade[image]
            tilted[image, n + 1 :] = below[:, 1:][:, ::-1]
        spectrum = np.fft.fft(tilted, axis=1) * dz
        field = (
            np.fft.ifft((spectrum + here.reflection * spectrum[:, mirror]) * propagate, axis=1) / dz
        )
        psi_next = field[:, :n]
        surface = here.surface
        if surface.any():
            amplitude = np.sum(tilted[surface, :n] * here.decay[surface], axis=1) * dz
            lift = (
                2j
                * here.beta[surface]
                * amplitude
                * np.exp(1j * dr * (here.wave[surface] - ka[surface]))
            )
            psi_next[surface] += lift[:, None] * here.decay[surface]
        return half * psi_next

    for index in order:
        goal = ranges[index]
        while goal - r > 1e-9:
            dr = min(step, goal - r)
            if independent and dr < step:
                break
            psi = advance(psi, r, dr)
            r += dr
        reached, at = psi, r
        if goal - r > 1e-9:
            # Its own last step, off to the side of the march.
            reached, at = advance(psi, r, goal - r), goal
        p = reached * (np.exp(1j * ka * at) / math.sqrt(max(at, 1e-9)))[:, None]
        h = heights[index]
        out[:, index] = [np.interp(h, z, row.real) + 1j * np.interp(h, z, row.imag) for row in p]
    return out


def _starter(x: np.ndarray) -> np.ndarray:
    """Greene's starting field at ``x`` = ka times the height from the source."""
    return (1.4467 - 0.4201 * x**2) * np.exp(-(x**2) / 3.0512)


def _grid(f: np.ndarray, c0: float, top: float) -> tuple[int, float, float]:
    """The heights' count and spacing, and the absorbing layer's thickness, for the
    frequencies ``f`` under an absorbing layer from ``top``."""
    dz = c0 / f.max() / STEPS_PER_WAVELENGTH
    thickness = max(LAYER_WAVELENGTHS * c0 / f.min(), LAYER_MIN_M)
    return int(2 ** math.ceil(math.log2((top + thickness) / dz))), dz, thickness


def footprint(frequencies, c0: float, top: float) -> int:
    """About the memory (bytes) :func:`march` takes for ``frequencies`` under an
    absorbing layer from ``top`` in air of speed ``c0`` on the ground."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    n = _grid(f, c0, top)[0]
    return 24 * len(f) * 2 * n * 16


class _Ground:
    """How a ground of normalised impedances ``zg`` (F,; infinite: rigid) reflects at
    the march's frequencies: R (F, 2n) and the surface wave (rows ``surface``), or for
    the rows ``image`` the field continued below the ground.

    R = 1 - (2 ka / Z) / (kz - kp) has a pole at kz = kp = -ka / Z. Over a hard ground it
    comes within a few samples' spacing of the real axis, where the samples miss it and
    the march blows up. There the field is continued below the ground instead, as the
    reflection factor has it (the image of R),

        psi(-y) = psi(y) + 2 i (ka / Z) int_0^y exp(-i kp (y - t)) psi(t) dt,

    and marched in free field: what the march does in kz, done in z. With the pole above
    the axis that holds the surface wave too, continued into the ground, where it grows
    as exp(Im kp y). Deep down it is faded out (see FADE_GROWTH)."""

    def __init__(self, zg, ka, kz, z, width) -> None:
        zg = np.asarray(zg, dtype=complex)
        self.key = zg.copy()
        rigid = ~np.isfinite(zg)
        safe = np.where(rigid, 1.0, zg)
        kp = np.where(rigid, 0.0, -ka / safe)
        self.image = (
            ~rigid & (kp.imag > -BELOW_AXIS_POLE * width) & (kp.imag < ABOVE_AXIS_POLE * width)
        )
        self.plain = ~self.image
        self.normal = np.where(rigid, 1.0, (safe - 1.0) / (safe + 1.0))
        reflection = (kz[None, :] * safe[:, None] - ka[:, None]) / (
            kz[None, :] * safe[:, None] + ka[:, None]
        )
        self.reflection = np.where(
            rigid[:, None], 1.0, np.where(self.image[:, None], 0.0, reflection)
        )
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
        self.fade = np.ones((len(zg), len(z)))
        for row in np.flatnonzero(self.image):
            depth = (
                min(math.log(FADE_GROWTH) / kp[row].imag, z[-1]) if kp[row].imag > 0.0 else z[-1]
            )
            t = np.clip((z - 0.5 * depth) / (0.5 * depth), 0.0, 1.0)
            self.fade[row] = 0.5 * (1.0 + np.cos(math.pi * t))


__all__ = ["footprint", "march"]
