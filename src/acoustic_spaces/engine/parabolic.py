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

Barriers and buildings stand in the way as obstacles across the vertical plane. A barrier
is a thin screen: the march stops at it and keeps of the field only what passes above its
top, each height step the share of its cell above it, so that the field changes smoothly
with the top's height (a zero-field, or Kirchhoff, screen: Salomons 1994, J. Acoust. Soc.
Am. 95, 3109). A building's front face does the same at its roof's height; over the roof
the field marches on the heights from the roof up, over a rigid floor there, and behind
its back face the field below the roof starts from nothing. Such a screen lets through
less than a rigid one does (1 to 7 dB less 2 to 120 m behind it), by an amount that
hangs on the geometry and hardly on the weather.

The frequencies of one band are marched together, on the grid and in the steps of the
highest, under the absorbing layer of the lowest.
"""

from __future__ import annotations

import math

import numpy as np

#: Height steps per wavelength.
STEPS_PER_WAVELENGTH = 10
#: Range step in wavelengths, at least RANGE_STEP_MIN_M. The march does not converge as
#: it shrinks: each step also adds an error at the ground that shrinks with the height
#: step, not the range step, so on this grid shorter steps add up more of it. In the
#: ``windy`` preset downwind over grass, from 1 m up to 1.5 m up 200 m away, the 63 Hz
#: band is 0.21 dB under the fast field program (``validation/ffp.py``) at this step,
#: 0.21 dB over it at a wavelength and 0.93 dB over it at a 32nd of this step.
RANGE_STEP_WAVELENGTHS = 5.0
RANGE_STEP_MIN_M = 1.2
#: In wind or temperature gradients the phase shift taken at each step strays for sound
#: that comes in steeply, at any step. The points that the reflection off the ground
#: reaches more steeply than STEEP_DEG (degrees from the horizontal) get the wide-angle
#: equation's field (:mod:`~acoustic_spaces.engine.wide_angle`): in the ``windy`` preset
#: their octave bands are within 0.05 dB of the fast field program, where this equation
#: is up to 0.46 dB off (0.56 dB at the flatter points; docs/validation.md, "Steep
#: sound").
STEEP_DEG = 15.0
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
    threads: int = 1,
    obstacles=(),
    refracting: bool | None = None,
    steps_per_wavelength: float = STEPS_PER_WAVELENGTH,
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
    else is asked for (the places of a walk sharing a run, say).

    In wind or temperature gradients the points the reflection off the ground reaches
    more steeply than STEEP_DEG are given by the wide-angle equation
    (:func:`acoustic_spaces.engine.wide_angle.march`, on ``threads`` threads); the others
    get what the march to all the points gives. ``refracting`` True or False says
    whether the air bends sound, in place of finding it from ``speed`` and ``speed_at``:
    two runs that differ in their weather alone then take the same equations.

    ``obstacles`` (near, far, top) (m) stand across the way, ``top`` m above the ground:
    a thin screen at ``near`` where ``far`` is ``near``, otherwise a building from
    ``near`` to ``far`` along the way with its rigid roof at ``top`` (rounded to a height
    step). ``steps_per_wavelength`` sets the height step at the highest frequency (the
    wide-angle equation's in proportion)."""
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    common = (frequencies, speed, impedance, zs)
    obstacles = _obstacles(obstacles)
    steep = zs + heights > math.tan(math.radians(STEEP_DEG)) * ranges
    if refracting is None and steep.any():
        refracting = _refracting(
            frequencies, speed, speed_at, top, ranges.max(), steps_per_wavelength
        )
    if not steep.any() or not refracting:
        return _march(
            *common,
            ranges,
            heights,
            top,
            screen,
            speed_at,
            independent,
            obstacles,
            steps_per_wavelength,
        )
    from acoustic_spaces.engine import wide_angle  # which imports this module

    out = np.zeros((len(np.atleast_1d(frequencies)), len(ranges)), dtype=complex)
    if not steep.all():
        # The flat points' march, with the steep points among them breaking its steps as
        # they would without the wide-angle equation; those beyond the last flat one
        # change nothing.
        before = ranges <= ranges[~steep].max()
        out[:, before] = _march(
            *common,
            ranges[before],
            heights[before],
            top,
            screen,
            speed_at,
            independent,
            obstacles,
            steps_per_wavelength,
        )
    out[:, steep] = wide_angle.march(
        *common,
        ranges[steep],
        heights[steep],
        top,
        screen=screen,
        speed_at=speed_at,
        threads=threads,
        independent=independent,
        obstacles=obstacles,
        steps_per_wavelength=wide_angle.STEPS_PER_WAVELENGTH
        * steps_per_wavelength
        / STEPS_PER_WAVELENGTH,
    )
    return out


def open_share(z: np.ndarray, top: float) -> np.ndarray:
    """(n,) the share of each height step's cell (the ground's sample: half a cell) on
    the grid of heights ``z`` (m) that lies above a thin screen's ``top`` (m)."""
    dz = float(z[1] - z[0])
    low = np.maximum(z - 0.5 * dz, 0.0)
    high = z + 0.5 * dz
    return np.clip((high - top) / (high - low), 0.0, 1.0)


def _obstacles(obstacles) -> tuple[tuple[float, float, float], ...]:
    """``obstacles`` as (near, far, top) floats in order along the way, those that stand
    beyond the source and above the ground."""
    kept = [(float(a), float(b), float(h)) for a, b, h in obstacles]
    return tuple(sorted(o for o in kept if o[1] > 0.0 and o[2] > 0.0 and o[1] >= o[0]))


def _refracting(
    frequencies,
    speed,
    speed_at,
    top: float,
    reach: float,
    steps_per_wavelength: float = STEPS_PER_WAVELENGTH,
) -> bool:
    """Whether the march goes through air that bends sound: a speed of sound that changes
    with height on its grid, or from the source's along the way out to ``reach`` (m),
    ``speed_at`` read every half metre."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    c0 = float(np.asarray(speed(np.array([0.0])))[0])
    n, dz, _ = _grid(f, c0, top, steps_per_wavelength)
    found = [speed]
    if speed_at is not None:
        for r in np.arange(0.0, float(reach) + 1.0, 0.5):
            there = speed_at(float(r))
            if all(there is not seen for seen in found):
                found.append(there)
    z = np.arange(n) * dz
    return any(bool(np.any(np.asarray(each(z)) != c0)) for each in found)


def _march(
    frequencies,
    speed,
    impedance,
    zs: float,
    ranges: np.ndarray,
    heights: np.ndarray,
    top: float,
    screen,
    speed_at,
    independent: bool,
    obstacles: tuple = (),
    steps_per_wavelength: float = STEPS_PER_WAVELENGTH,
) -> np.ndarray:
    """:func:`march` by the narrow-angle equation alone, past ``obstacles`` (as
    :func:`_obstacles` gives them)."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    c0 = float(np.asarray(speed(np.array([0.0])))[0])
    ka = 2.0 * math.pi * f / c0  # (F,)
    shortest = c0 / f.max()
    n, dz, thickness = _grid(f, c0, top, steps_per_wavelength)
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
    floor = 0  # the height step of the roof the field marches over, or 0 over the ground
    roofs: dict[tuple[int, float], tuple] = {}

    def over_roof(psi: np.ndarray, r: float, dr: float) -> np.ndarray:
        # A step over a rigid roof at height step ``floor``: the march on the heights
        # from there up, reflecting off it as off a rigid ground.
        part = n - floor
        if (floor, dr) not in roofs:
            if len(roofs) > 4:
                roofs.clear()
            heights_kz = 2.0 * math.pi * np.fft.fftfreq(2 * part, d=dz)
            up = np.sqrt((ka[:, None] ** 2 - heights_kz[None, :] ** 2).astype(complex))
            up = np.where(up.imag < 0.0, -up, up)
            roofs[(floor, dr)] = (
                np.exp(0.5j * dr * delta[:, floor:]),
                np.exp(1j * dr * (up - ka[:, None])),
                (-np.arange(2 * part)) % (2 * part),
            )
        half, propagate, flip = roofs[(floor, dr)]
        if screen is not None:
            half = half * np.exp(0.5j * ka[:, None] * screen(r, dr, z[floor:])[None, :])
        screened = half * psi[:, floor:]
        tilted_roof = np.zeros((len(f), 2 * part), dtype=complex)
        tilted_roof[:, :part] = screened
        tilted_roof[:, 0] *= 0.5
        spectrum = np.fft.fft(tilted_roof, axis=1) * dz
        field = np.fft.ifft((spectrum + spectrum[:, flip]) * propagate, axis=1) / dz
        out = np.zeros_like(psi)
        out[:, floor:] = half * field[:, :part]
        return out

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
                roofs.clear()
        if floor:
            return over_roof(psi, r, dr)
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

    # Where the march meets the obstacles: (range, what: "screen" its top (m), "roof" or
    # "behind" a roof's height step).
    events = []
    for near, far, top_at in obstacles:
        if far == near:
            events.append((near, "screen", top_at))
        elif int(round(top_at / dz)) > 0:
            events.append((near, "roof", min(int(round(top_at / dz)), n - 2)))
            events.append((far, "behind", min(int(round(top_at / dz)), n - 2)))
    events.sort(key=lambda e: e[0])
    roofs_over: list[int] = []  # the roofs the march is over (height steps)

    def meet(psi: np.ndarray, event) -> np.ndarray:
        # The field just past an obstacle the march has reached: above a screen's top,
        # or above the highest roof it is over.
        nonlocal floor
        _, what, value = event
        if what == "screen":
            return psi * open_share(z, value)[None, :]
        if what == "roof":
            roofs_over.append(value)
        else:
            roofs_over.remove(value)
        floor = max(roofs_over, default=0)
        psi = psi.copy()
        psi[:, :floor] = 0.0
        return psi

    for index in order:
        goal = ranges[index]
        while True:
            if events and events[0][0] <= r + 1e-9:
                psi = meet(psi, events.pop(0))
                continue
            if goal - r <= 1e-9:
                break
            # The march stops at each obstacle on the way, one at the goal included, and
            # at the goal.
            obstacle = bool(events) and events[0][0] <= goal + 1e-9
            dr = min(step, (events[0][0] if obstacle else goal) - r)
            if independent and dr < step and not obstacle:
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


def _grid(
    f: np.ndarray, c0: float, top: float, steps_per_wavelength: float = STEPS_PER_WAVELENGTH
) -> tuple[int, float, float]:
    """The heights' count and spacing, and the absorbing layer's thickness, for the
    frequencies ``f`` under an absorbing layer from ``top``."""
    dz = c0 / f.max() / steps_per_wavelength
    thickness = max(LAYER_WAVELENGTHS * c0 / f.min(), LAYER_MIN_M)
    return int(2 ** math.ceil(math.log2((top + thickness) / dz))), dz, thickness


def footprint(
    frequencies, c0: float, top: float, steps_per_wavelength: float = STEPS_PER_WAVELENGTH
) -> int:
    """About the memory (bytes) :func:`march` takes for ``frequencies`` under an
    absorbing layer from ``top`` in air of speed ``c0`` on the ground: the narrow-angle
    march's, or the wide-angle one's for steep points after it, the larger, whether or
    not a run has steep points."""
    from acoustic_spaces.engine import wide_angle

    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    n = _grid(f, c0, top, steps_per_wavelength)[0]
    wide = wide_angle.STEPS_PER_WAVELENGTH * steps_per_wavelength / STEPS_PER_WAVELENGTH
    return max(24 * len(f) * 2 * n * 16, wide_angle.footprint(f, c0, top, wide))


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


__all__ = ["footprint", "march", "open_share"]
