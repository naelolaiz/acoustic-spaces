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
less than a rigid one does (0.7 to 7.4 dB less 2 to 90 m behind it, up to 11 dB 3 to
10 m behind it steeply below its top), by an amount that hangs on the geometry and hardly
on the weather.

The frequencies of one band are marched together, on the grid and in the steps of the
highest, under the absorbing layer of the lowest.
"""

from __future__ import annotations

import inspect
import math
import threading

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
#: A step takes the frequencies (the rows of the field) this many at a time, so that
#: what it works on stays in the processor's cache from one pass to the next. Each row
#: comes out with the bytes it has when all the rows are taken at once (numpy 1.24, 1.26,
#: 2.0 and 2.5 on x86-64: its transforms and its element-wise loops give an element the
#: same whatever the rows around it). A multiple of 8, the most elements of a vector
#: there, so that a block starts where a vector does in the whole array.
BLOCK_ROWS = 8
#: Whether numpy's transforms write into an array given to them (``out``: numpy 2.0 on).
_TRANSFORMS_INTO = "out" in inspect.signature(np.fft.fft).parameters


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
    trunk: Trunk | None = None,
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
    ``near`` to ``far`` along the way with its rigid roof at ``top``, between the height
    steps where it falls. ``steps_per_wavelength`` sets the height step at the highest
    frequency (the wide-angle equation's in proportion).

    ``trunk``, a :class:`Trunk` given to the runs that differ in their obstacles and
    points alone, keeps the narrow-angle march over the open ground from one of them to
    the next; each gets what it gets without it. A run with a ``screen`` or with
    ``speed_at`` marches without it, and one with steep points clears it before the
    wide-angle equation runs."""
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
            trunk,
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
            trunk,
        )
    if trunk is not None:
        # What the trunk keeps would stay beside the wide-angle march's memory, which
        # footprint does not count with it.
        trunk.clear()
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


def roof_share(z: np.ndarray, roof: float) -> np.ndarray:
    """(n,) the share of each height step's cell on the grid of heights ``z`` (m) that
    lies above a rigid roof ``roof`` m up: the sample under the roof's mirror plane
    counts for the share of its cell above it (half a cell with the roof on it), the
    way the ground's sample does."""
    dz = float(z[1] - z[0])
    return np.clip((z + 0.5 * dz - roof) / dz, 0.0, 1.0)


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


class Trunk:
    """The march from the source over the open ground, kept between runs of
    :func:`march` that differ in their obstacles and points alone. A run begins with
    full steps, up to its first shorter one (to its nearest obstacle or point): where
    the runs before it took the same steps (the same frequencies, speed of sound, source
    height, absorbing layer and height step, over the same grounds), it takes the field
    where they left it instead of marching there again. The runs take their turns, one
    at a time. One whose full steps end nearer the source than those already taken
    marches from the source by itself, and with one that differs in anything else the
    trunk starts anew."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: _Steps | None = None
        self._restart()

    def clear(self) -> None:
        """Drops the march kept, its grid and air included: the next run starts anew."""
        with self._lock:
            self._steps = None
            self._restart()

    def _restart(self) -> None:
        #: The field after ``_count`` full steps from the source, at range ``_range``.
        self._psi: np.ndarray | None = None
        self._count = 0
        self._range = 0.0
        #: The ground's impedances under the source and at the middle of each of those
        #: steps, (range, their bytes), and the ground the last of them was taken over.
        self._grounds: list[tuple[float, bytes]] = []
        self._here: _Ground | None = None


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
    trunk: Trunk | None = None,
) -> np.ndarray:
    """:func:`march` by the narrow-angle equation alone, past ``obstacles`` (as
    :func:`_obstacles` gives them)."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if trunk is None or screen is not None or speed_at is not None:
        alone = _Steps(f, speed, impedance, zs, top, screen, speed_at, steps_per_wavelength)
        return alone.run(ranges, heights, independent, obstacles, None)
    with trunk._lock:
        steps = trunk._steps
        if steps is None or not steps.fits(f, speed, zs, top, steps_per_wavelength):
            steps = _Steps(f, speed, impedance, zs, top, None, None, steps_per_wavelength)
            trunk._steps = steps
            trunk._restart()
        steps.impedance = impedance
        return steps.run(ranges, heights, independent, obstacles, trunk)


class _Steps:
    """The narrow-angle march's grid, air and ground for the frequencies ``f`` (F,) of
    a run (the other arguments as :func:`march`'s), and its steps."""

    def __init__(
        self, f, speed, impedance, zs, top, screen, speed_at, steps_per_wavelength
    ) -> None:
        self.f = f
        self.zs = zs
        self.top = top
        self.steps_per_wavelength = steps_per_wavelength
        self.impedance = impedance
        self.screen = screen
        self.speed_at = speed_at
        self.c0 = c0 = float(np.asarray(speed(np.array([0.0])))[0])
        self.ka = ka = 2.0 * math.pi * f / c0  # (F,)
        shortest = c0 / f.max()
        n, dz, thickness = _grid(f, c0, top, steps_per_wavelength)
        self.n, self.dz = n, dz
        self.z = z = np.arange(n) * dz
        #: (n,) the speed of sound at the source, on the heights.
        self.c = np.asarray(speed(z), dtype=float)
        k = 2.0 * math.pi * f[:, None] / self.c[None, :]
        self.depth = depth = np.clip((z - top) / thickness, 0.0, 1.0)
        self.delta = (k - ka[:, None]) + 1j * LAYER_STRENGTH * depth**2  # (F, n)

        self.kz = kz = 2.0 * math.pi * np.fft.fftfreq(2 * n, d=dz)
        vertical = np.sqrt((ka[:, None] ** 2 - kz[None, :] ** 2).astype(complex))
        self.vertical = np.where(vertical.imag < 0.0, -vertical, vertical)  # (F, 2n)
        self.width = 2.0 * math.pi / (2 * n * dz)
        self.step = max(RANGE_STEP_WAVELENGTHS * shortest, RANGE_STEP_MIN_M)
        #: The speed of sound the steps are taken in (``speed_at``'s, where it changes).
        self.current = speed
        #: By step length, the phase screen's half step and the propagator.
        self.steps: dict[float, tuple] = {}
        #: By roof and step length, the same over a roof (:meth:`over_roof`).
        self.roofs: dict[tuple[float, float], tuple] = {}
        #: The ground the last step was taken over.
        self.here: _Ground | None = None
        #: The height (m) of the roof the field marches over, or 0 over the ground.
        self.roof = 0.0
        self.room(min(BLOCK_ROWS, len(f)))
        #: The same over a roof, for the heights from the sample under it up: how many
        #: they are, and the arrays (the field's upper half stays zero).
        self.over: tuple | None = None

    def room(self, rows: int) -> None:
        """What a step works on, for ``rows`` rows at a time (the first rows of each):
        the field on the heights and, in the upper half, zero or the field continued
        below the ground (its first sample there is never written and stays zero); its
        spectrum; the spectrum a step on; and two arrays of the heights for the rest."""
        n = self.n
        self.tilted = np.zeros((rows, 2 * n), dtype=complex)
        self.spectrum = np.empty((rows, 2 * n), dtype=complex)
        self.mirrored = np.empty((rows, 2 * n), dtype=complex)
        self.work = np.empty((2, rows, n), dtype=complex)
        #: How many of the first rows of ``tilted`` may hold a field in their upper half.
        self.below = 0

    def fits(self, f, speed, zs, top, steps_per_wavelength) -> bool:
        """Whether a run at the frequencies ``f`` in air of speed ``speed`` from a source
        ``zs`` up, under an absorbing layer from ``top``, with that height step, is on
        this grid and in this air (where the speed changes with height alone)."""
        return (
            np.array_equal(f, self.f)
            and zs == self.zs
            and top == self.top
            and steps_per_wavelength == self.steps_per_wavelength
            and float(np.asarray(speed(np.array([0.0])))[0]) == self.c0
            and np.array_equal(np.asarray(speed(self.z), dtype=float), self.c)
        )

    def stepping(self, dr: float) -> tuple:
        """The phase screen's half step and the propagator over ``dr`` (most steps are
        full: theirs stay while the shorter ones' come and go)."""
        steps = self.steps
        if dr not in steps:
            if len(steps) > 4:
                for other in [d for d in steps if d != self.step]:
                    del steps[other]
            steps[dr] = (
                _exp_rows(0.5j * dr * self.delta),
                _exp_mirrored(1j * dr * (self.vertical - self.ka[:, None])),
            )
        return steps[dr]

    def impedances(self, at: float) -> np.ndarray:
        value = self.impedance(at) if self.impedance is not None else None
        if value is None:
            return np.full(len(self.f), np.inf, dtype=complex)
        return np.broadcast_to(np.asarray(value, dtype=complex), (len(self.f),))

    def ground(self, at: float) -> _Ground:
        """The ground at range ``at``: the last one while its impedances hold."""
        value = self.impedances(at)
        if self.here is None or not np.array_equal(value, self.here.key):
            self.here = _Ground(value, self.ka, self.kz, self.z, self.width)
        return self.here

    def start(self) -> np.ndarray:
        """(F, n) the starting field, at the source."""
        ka, z, zs = self.ka, self.z, self.zs
        here = self.ground(0.0)
        return np.sqrt(1j * ka)[:, None] * (
            _starter(ka[:, None] * (z - zs))
            + here.normal[:, None] * _starter(ka[:, None] * (z + zs))
        )

    def over_roof(self, psi: np.ndarray, r: float, dr: float, out: np.ndarray) -> np.ndarray:
        """A step over a rigid roof ``roof`` m up, between the grid's samples: the march
        on the heights from the sample under it up, reflecting off it as off a rigid
        ground (the field mirrored about it, each sample weighed by the share of its
        cell above it, as a screen's are)."""
        n, dz, z, ka, roof, roofs = self.n, self.dz, self.z, self.ka, self.roof, self.roofs
        floor = int(roof / dz)
        part = n - floor
        if (roof, dr) not in roofs:
            if len(roofs) > 4:
                for other in [key for key in roofs if key[1] != self.step]:
                    del roofs[other]
            heights_kz = 2.0 * math.pi * np.fft.fftfreq(2 * part, d=dz)
            up = np.sqrt((ka[:, None] ** 2 - heights_kz[None, :] ** 2).astype(complex))
            up = np.where(up.imag < 0.0, -up, up)
            roofs[(roof, dr)] = (
                _exp_rows(0.5j * dr * self.delta[:, floor:]),
                _exp_mirrored(1j * dr * (up - ka[:, None])),
                np.exp(-2j * heights_kz * (roof - z[floor])),
                roof_share(z[floor:], roof),
            )
        half, propagate, mirror, share = roofs[(roof, dr)]
        if self.over is None or self.over[0] != part:
            # A block of rows of the field on the heights, its upper half staying zero,
            # of its spectrum and of the mirror, and of the field itself.
            rows = min(BLOCK_ROWS, len(psi))
            self.over = (
                part,
                np.zeros((rows, 2 * part), dtype=complex),
                np.empty((rows, 2 * part), dtype=complex),
                np.empty((rows, 2 * part), dtype=complex),
                np.empty((rows, part), dtype=complex),
            )
        _, tilted, spectrum, mirrored, screened = self.over
        if self.screen is not None:
            half = half * np.exp(0.5j * ka[:, None] * self.screen(r, dr, z[floor:])[None, :])
        for lo in range(0, len(psi), BLOCK_ROWS):
            hi = min(lo + BLOCK_ROWS, len(psi))
            rows = hi - lo
            np.multiply(half[lo:hi], psi[lo:hi, floor:], out=screened[:rows])
            np.multiply(share[None, :], screened[:rows], out=tilted[:rows, :part])
            found = _transform(tilted[:rows], spectrum[:rows])
            found *= dz
            field = _inverse(_reflected(found, mirror, propagate[lo:hi], mirrored[:rows]))
            above = field[:, :part]
            above /= dz
            out[lo:hi, :floor] = 0.0
            np.multiply(half[lo:hi], above, out=out[lo:hi, floor:])
        return out

    def advance(self, psi: np.ndarray, r: float, dr: float, out: np.ndarray) -> np.ndarray:
        """The field a step ``dr`` further on from ``r``, written into ``out`` (which
        may be ``psi`` itself)."""
        ka, z = self.ka, self.z
        here = self.ground(r + 0.5 * dr)
        if self.speed_at is not None:
            there = self.speed_at(r + 0.5 * dr)
            if there is not self.current:
                self.current = there
                k = 2.0 * math.pi * self.f[:, None] / np.asarray(there(z), dtype=float)[None, :]
                self.delta = (k - ka[:, None]) + 1j * LAYER_STRENGTH * self.depth**2
                self.steps.clear()
                self.roofs.clear()
        if self.roof > 0.0:
            return self.over_roof(psi, r, dr, out)
        half, propagate = self.stepping(dr)
        if self.screen is not None:
            half = half * np.exp(0.5j * ka[:, None] * self.screen(r, dr, z)[None, :])
        some_image = bool(here.image.any())
        some_surface = bool(here.surface.any())
        surface = None
        if some_surface:
            # The surface wave's factors: 2 i beta, and its phase over the step.
            surface = (
                2j * here.beta[here.surface],
                np.exp(1j * dr * (here.wave[here.surface] - ka[here.surface])),
            )
        # In blocks of rows where the ground is of one kind at every frequency; else all
        # the rows at once, those of each kind picked out of them.
        rows = len(psi)
        if (here.every_image or not some_image) and (here.every_surface or not some_surface):
            rows = min(BLOCK_ROWS, rows)
        if rows > len(self.tilted):
            self.room(rows)
        for lo in range(0, len(psi), rows):
            hi = min(lo + rows, len(psi))
            self._advance(psi, here, half, propagate, surface, some_image, lo, hi, out)
        return out

    def _advance(
        self, psi, here: _Ground, half, propagate, surface, some_image, lo: int, hi: int, out
    ) -> None:
        """The rows ``lo`` to ``hi`` of :meth:`advance` over the ground ``here``; all the
        rows, where the field is continued below the ground, or the surface wave kept, at
        some of the frequencies and not at others."""
        n, dz = self.n, self.dz
        tilted = self.tilted[: hi - lo]
        screened = np.multiply(half[lo:hi], psi[lo:hi], out=tilted[:, :n])
        # The ground's sample weighs half (the trapezoidal rule), as the reflection
        # adds its mirror image.
        if not here.every_image:
            if self.below:
                self.tilted[: self.below, n:] = 0.0
                self.below = 0
            if some_image:
                tilted[here.plain, 0] *= 0.5
            else:
                tilted[:, 0] *= 0.5
        if some_image:
            self.below = max(self.below, hi - lo)
            image = here.image
            on = screened if here.every_image else screened[image]
            inner = np.multiply(here.growth_below[lo:hi], on, out=self.work[0, : len(on)])
            integral = np.cumsum(inner, axis=1, out=self.work[1, : len(on)])
            first = inner[:, :1].copy()
            np.add(first, inner, out=inner)
            np.multiply(0.5, inner, out=inner)
            np.subtract(integral, inner, out=integral)
            np.multiply(integral, dz, out=integral)
            np.multiply(here.lift_below[lo:hi], integral, out=integral)
            below = np.add(on, integral, out=integral)
            below *= here.fade_below[lo:hi]
            if here.every_image:
                tilted[:, n + 1 :] = below[:, :0:-1]
            else:
                tilted[image, n + 1 :] = below[:, :0:-1]
        spectrum = _transform(tilted, self.spectrum[: hi - lo])
        spectrum *= dz
        field = _inverse(
            _reflected(spectrum, here.reflection[lo:hi], propagate[lo:hi], self.mirrored[: hi - lo])
        )
        psi_next = field[:, :n]
        psi_next /= dz
        if surface is not None:
            decay = here.decay_surface[lo:hi]
            on = tilted[:, :n] if here.every_surface else tilted[here.surface, :n]
            product = np.multiply(on, decay, out=self.work[0, : len(decay)])
            amplitude = np.sum(product, axis=1) * dz
            lift = surface[0][lo:hi] * amplitude * surface[1][lo:hi]
            np.multiply(lift[:, None], decay, out=product)
            if here.every_surface:
                psi_next += product
            else:
                psi_next[here.surface] += product
        np.multiply(half[lo:hi], psi_next, out=out[lo:hi])

    def meet(self, psi: np.ndarray, event, roofs_over: list[float]) -> np.ndarray:
        """The field just past an obstacle the march has reached (``event``: its range,
        what, and a height), as a new array: above a screen's top (each sample weighed
        by the share of its cell above it), above the highest roof it is over
        (``roofs_over``, m), or above the roof it has just left."""
        z, dz = self.z, self.dz
        _, what, value = event
        if what == "screen":
            return psi * open_share(z, value)[None, :]
        if what == "roof":
            # The march over the roof weighs the sample under it itself.
            roofs_over.append(value)
            self.roof = max(roofs_over)
            psi = psi.copy()
            psi[:, : int(self.roof / dz)] = 0.0
            return psi
        roofs_over.remove(value)
        self.roof = max(roofs_over, default=0.0)
        return psi * roof_share(z, value)[None, :]

    def run(
        self,
        ranges: np.ndarray,
        heights: np.ndarray,
        independent: bool,
        obstacles: tuple,
        trunk: Trunk | None,
    ) -> np.ndarray:
        """(F, n) the pressure at the points ``ranges`` and ``heights`` past
        ``obstacles``, as :func:`_march` gives it; with ``trunk``, from where it has the
        field over the open ground."""
        f, n, dz, z, ka, step = self.f, self.n, self.dz, self.z, self.ka, self.step
        order = np.argsort(ranges, kind="stable")
        out = np.zeros((len(f), len(ranges)), dtype=complex)

        # Where the march meets the obstacles: (range, what: "screen" its top (m),
        # "roof" or "behind" a roof's height (m)).
        events = []
        highest = (n - 2) * dz
        for near, far, top_at in obstacles:
            if far == near:
                events.append((near, "screen", top_at))
            else:
                events.append((near, "roof", min(top_at, highest)))
                events.append((far, "behind", min(top_at, highest)))
        events.sort(key=lambda e: e[0])
        roofs_over: list[float] = []  # the roofs the march is over (m)
        self.roof = 0.0
        # Of the runs before it, a run keeps the full steps' tables alone.
        for other in [dr for dr in self.steps if dr != step]:
            del self.steps[other]
        for key in [key for key in self.roofs if key[1] != step]:
            del self.roofs[key]

        psi = None
        r = 0.0
        if trunk is not None and len(order):
            # The full steps the march begins with, before a shorter one to its nearest
            # obstacle or point: the trunk's, where it has not gone further, over the
            # grounds this run has there.
            goal = ranges[order[0]]
            nearest = events[0][0] if events and events[0][0] <= goal + 1e-9 else goal
            count, at = 0, 0.0
            while nearest - at >= step:
                at += step
                count += 1
            if any(self.impedances(a).tobytes() != value for a, value in trunk._grounds):
                trunk._restart()
            if trunk._count <= count:
                self.here = trunk._here
                if trunk._psi is None:
                    trunk._psi = self.start()
                    trunk._grounds.append((0.0, self.impedances(0.0).tobytes()))
                while trunk._count < count:
                    middle = trunk._range + 0.5 * step
                    self.advance(trunk._psi, trunk._range, step, trunk._psi)
                    trunk._grounds.append((middle, self.impedances(middle).tobytes()))
                    trunk._range += step
                    trunk._count += 1
                trunk._here = self.here
                psi, r = trunk._psi.copy(), trunk._range
        if psi is None:
            self.here = None
            psi = self.start()
        side = None  # a point's own last step
        last = None  # the range of the last point, while the march stays where it was
        for index in order:
            goal = ranges[index]
            while True:
                if events and events[0][0] <= r + 1e-9:
                    psi = self.meet(psi, events.pop(0), roofs_over)
                    last = None
                    continue
                if goal - r <= 1e-9:
                    break
                # The march stops at each obstacle on the way, one at the goal included,
                # and at the goal.
                obstacle = bool(events) and events[0][0] <= goal + 1e-9
                dr = min(step, (events[0][0] if obstacle else goal) - r)
                if independent and dr < step and not obstacle:
                    break
                self.advance(psi, r, dr, psi)
                r += dr
                last = None
            # A point at the range of the one before it, the march where it was, has the
            # same pressure on the heights.
            if last is None or goal != last:
                reached, at = psi, r
                if goal - r > 1e-9:
                    # Its own last step, off to the side of the march.
                    if side is None:
                        side = np.empty_like(psi)
                    reached, at = self.advance(psi, r, goal - r, side), goal
                p = reached * (np.exp(1j * ka * at) / math.sqrt(max(at, 1e-9)))[:, None]
                last = goal
            h = heights[index]
            out[:, index] = [
                np.interp(h, z, row.real) + 1j * np.interp(h, z, row.imag) for row in p
            ]
        return out


def _exp_rows(a: np.ndarray) -> np.ndarray:
    """exp of the complex ``a`` (F, n), a new array: where every row has the first one's
    bytes (the half step in air of one speed of sound), the first row's, taken once."""
    out = np.empty_like(a)
    first = a[0].tobytes()
    if len(a) > 1 and all(row.tobytes() == first for row in a[1:]):
        np.exp(a[:1], out=out[:1])
        out[1:] = out[:1]
    else:
        np.exp(a, out=out)
    return out


def _exp_mirrored(a: np.ndarray) -> np.ndarray:
    """exp of the complex ``a`` (F, 2n), a new array: where the columns at -kz have the
    bytes of those at kz (index -i mod 2n: the propagator), theirs, taken once."""
    out = np.empty_like(a)
    n = a.shape[1] // 2
    if a.shape[1] == 2 * n and a[:, n + 1 :].tobytes() == a[:, n - 1 : 0 : -1].tobytes():
        np.exp(a[:, : n + 1], out=out[:, : n + 1])
        out[:, n + 1 :] = out[:, n - 1 : 0 : -1]
    else:
        np.exp(a, out=out)
    return out


def _transform(a: np.ndarray, out: np.ndarray) -> np.ndarray:
    """The Fourier transform of each row of ``a``: in ``out`` where numpy writes into
    an array given to it, else in a new array."""
    if _TRANSFORMS_INTO:
        return np.fft.fft(a, axis=1, out=out)
    return np.fft.fft(a, axis=1)


def _inverse(a: np.ndarray) -> np.ndarray:
    """The inverse Fourier transform of each row of ``a``: in ``a`` itself where numpy
    writes into an array given to it, else in a new array."""
    if _TRANSFORMS_INTO:
        return np.fft.ifft(a, axis=1, out=a)
    return np.fft.ifft(a, axis=1)


def _reflected(spectrum, reflection, propagate, out):
    """(spectrum + reflection * mirrored spectrum) * propagate, into ``out`` (the same
    shape as ``spectrum``): the field's spectrum and its reflection off the ground at
    kz = 0, a step on. The mirror takes the spectrum at -kz (index -i mod 2n)."""
    out[:, 0] = spectrum[:, 0]
    out[:, 1:] = spectrum[:, :0:-1]
    np.multiply(reflection, out, out=out)
    np.add(spectrum, out, out=out)
    np.multiply(out, propagate, out=out)
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
        # (Rigid rows take 1, where on a coarse grid kz may meet -ka.)
        with np.errstate(divide="ignore", invalid="ignore"):
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
        # What the march takes of the rows ``image`` at every step.
        self.every_image = bool(self.image.all())
        self.every_surface = bool(self.surface.all())
        self.decay_surface = self.decay[self.surface]
        self.growth_below = self.growth[self.image]
        self.lift_below = self.scale[self.image, None] * self.shrink[self.image]
        self.fade_below = self.fade[self.image]


__all__ = ["Trunk", "footprint", "march", "open_share", "roof_share"]
