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
its back face the field below the roof starts from nothing; a source over a roof starts
with its image in it. Such a screen lets through less than a rigid one does, by an
amount that hangs on the geometry and hardly on the weather: in still air over rigid
ground, 1.5 m up behind a barrier 4 m high 15 m from a source 1 m up, 1.2 to 3.2 dB less
10 to 90 m behind it and 4.1 to 9.4 dB less 2 to 5 m behind it, and 2 m behind one 6 m
high 20 m away 10.9 to 14.6 dB less (the octave bands from 63 Hz to 8 kHz against the
four paths of the uniform theory of diffraction; a one-off check). The engine's runs
behind barriers in the weather (:func:`acoustic_spaces.engine.refracted.solve`) march
past their obstacles in bounded steps (:func:`march`, ``bounded``), so that the sound an
obstacle's top bends steeply up does not come back down from the top of the grid; the
marches over a row of edges (:mod:`acoustic_spaces.engine.rows`) are plain ones under a
layer made thicker for that sound (``layer``).

The frequencies of one band are marched together, on the grid and in the steps of the
highest, under the absorbing layer of the lowest.
"""

from __future__ import annotations

import inspect
import math
import threading

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.special import hankel1e

#: Height steps per wavelength.
STEPS_PER_WAVELENGTH = 10
#: Range step in wavelengths, at least RANGE_STEP_MIN_M. The march does not converge as
#: it shrinks: each step also adds an error at the ground that shrinks with the height
#: step, not the range step, so on this grid shorter steps add up more of it. In the
#: ``windy`` preset downwind over grass, from 1 m up to 1.5 m up 200 m away, the 63 Hz
#: band is 0.26 dB under the fast field program at this step, 0.14 dB over it at a
#: wavelength and 0.59 dB over it at a 32nd of this step
#: (``validation/steep_sound.py --only steps``).
RANGE_STEP_WAVELENGTHS = 5.0
RANGE_STEP_MIN_M = 1.2
#: In wind or temperature gradients the phase shift taken at each step strays for sound
#: that comes in steeply, at any step. The points that the reflection off the ground
#: reaches more steeply than STEEP_DEG (degrees from the horizontal) get the wide-angle
#: equation's field (:mod:`~acoustic_spaces.engine.wide_angle`): in 28 cases of wind and
#: temperature gradients their octave bands (63 to 500 Hz in ten of the cases, 63 and
#: 125 Hz in the others) are within 0.09 dB of the fast field program, where this
#: equation is up to 0.96 dB off (1.10 dB at the flatter points;
#: ``validation/steep_sound.py``, docs/validation.md, "Steep sound").
STEEP_DEG = 15.0
#: The pole of the ground's reflection factor is left to the field continued below the
#: ground within this many sample spacings below the real axis (where that field stays
#: bounded) and above it (where it grows with depth). The sampled reflection factor
#: holds from 4 spacings below the axis and 2 above (to 0.1 dB at 400 m).
BELOW_AXIS_POLE = 6.0
ABOVE_AXIS_POLE = 3.0
#: The continued field is faded out over the lower half of its depth, or, where it grows
#: with depth, of the depth at which it has grown by this factor: what lies deeper does
#: not reach the ground in a step, and its edge would ring through the transform. (A
#: point's own step, as long as many, keeps it whole as deep as the layer starts.)
FADE_GROWTH = 10.0
#: The absorbing layer: its thickness in wavelengths (at least LAYER_MIN_M) and its
#: strength (1/m at its top; growing as the square of the depth into it).
LAYER_WAVELENGTHS = 50.0
LAYER_MIN_M = 20.0
LAYER_STRENGTH = 1.0
#: A march that begins from a given field (``start``) takes it up to the absorbing layer;
#: it fades out over this share of the layer's thickness.
START_FADE = 0.1
#: A march in bounded steps (``bounded``: the narrow-angle runs past obstacles of
#: :func:`acoustic_spaces.engine.refracted.solve`). A bounded step
#: takes all of the sound that goes up or down less than BOUNDED_WHOLE of the absorbing
#: layer's thickness and a share that falls smoothly to none at BOUNDED_WINDOW of it.
BOUNDED_WHOLE = 0.25
BOUNDED_WINDOW = 0.7
#: Such a step's table is larger than 1 for waves some 65 to 80 degrees from the
#: horizontal (by up to 5 per cent): each grows from step to step until it has gone up
#: into the layer, where it ends. Where a wave would grow by more than BOUNDED_GROWTH on
#: its way from the ground to the layer's top, the table is brought down to what makes
#: it grow by that, by a factor that spreads the sound by no more than BOUNDED_SMOOTH of
#: the layer's thickness (:func:`_bounded`), so that no sound goes further up or down in
#: a step than the layer is thick. Under a layer whose top is a kilometre up or less
#: the factor changes a full step's table by 1.3e-7 at most (the waves along the
#: ground, which the window makes larger than 1 by that); it takes hold of the steep
#: waves where the top is some 2 km up at 1 kHz, 4 km at 2 kHz and 8 km at 250 Hz.
#: Where the factor made that way would pass below 0 (a top several km up), it is made
#: from the largest excess among each wavenumber's neighbours, those that hold
#: BOUNDED_COVER of the smoothing (:func:`_covered`).
BOUNDED_GROWTH = 1e10
BOUNDED_SMOOTH = 0.3
BOUNDED_COVER = 0.9
#: A bounded step takes a wave whole where what it goes up or down in the step, and
#: BOUNDED_FRESNEL times the width of its Fresnel zone over the step (at the lowest
#: frequency) beside that, lie within BOUNDED_WHOLE of the layer's thickness
#: (:meth:`_Steps.takes`): in full steps the waves up to 51 degrees from the horizontal
#: in the bands up to 1 kHz and 55, 60 and 64 degrees in those of 2, 4 and 8 kHz, for
#: which the table is within 1e-4 of the plain step's.
BOUNDED_FRESNEL = 2.7
#: Between two obstacles the march takes steps no shorter than this many of the shortest
#: wavelengths for the sound's sake (:meth:`_Steps.run`).
BOUNDED_SHORTEST = 0.1
#: A point's own step looks at the air along its way every so many metres, and in at
#: least OWN_LOOKS places (:meth:`_Steps.stretches`).
OWN_LOOK_M = 0.1
OWN_LOOKS = 8
#: In bounded steps the absorbing layer also damps what stands in it at a step's end, by
#: exp(-LAYER_DEEP (2 depth)^LAYER_DEEP_POWER) a step (the depth 0 at the layer's start and
#: 1 at its top): next to nothing over the layer's start, which stays as it is, and from
#: its middle on 30 dB and more. Sound that passes the top of the grid in a step comes
#: back down from there, as the transform repeats the grid in height: a bounded step
#: takes it no further than into that upper half.
LAYER_DEEP = 3.5
LAYER_DEEP_POWER = 6
#: A bounded step's table is the transform of its kernel, sampled BOUNDED_FINE of the
#: shortest wavelength over the step's length times as finely as the heights (the kernel
#: is as narrow as the step is short), at most BOUNDED_FINEST times; a shorter step's
#: is made from the plain step's table instead (:func:`_bounded`).
BOUNDED_FINE = 0.366
BOUNDED_FINEST = 8
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
    edges: list | None = None,
    start: tuple | None = None,
    layer: float | None = None,
    bounded: bool = False,
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
    profile of its own middle (a point's own step, :meth:`_Steps.own`, each height's
    sound through the airs along its way), the reference wavenumber staying the
    source's.

    The march goes out in full steps, and a range between two of them is reached by a
    shorter last step. Unless ``independent``, the march goes on from there, so that the
    ranges asked for break up each other's steps; ``independent`` takes each one's last
    step off to the side, so that each range gets what a march to it alone gives, whatever
    else is asked for (the engine's runs: :func:`acoustic_spaces.engine.refracted.solve`),
    but, with ``refracting`` None, for which equation the steep points take (below).

    In wind or temperature gradients the points the reflection off the ground reaches
    more steeply than STEEP_DEG are given by the wide-angle equation
    (:func:`acoustic_spaces.engine.wide_angle.march`, on ``threads`` threads); the others
    get what the march to all the points gives. Whether the air bends sound is found
    from ``speed`` and ``speed_at`` at the grid's heights out to the farthest point, so
    it depends on ``top`` and on the points asked for together; ``refracting`` True or
    False says it instead. The runs of :func:`acoustic_spaces.engine.refracted.solve` say
    True: a steep point then takes the wide-angle equation whatever else its run holds, in
    still air too, and two runs that differ in their weather alone take the same
    equations. The solver for rows (:mod:`acoustic_spaces.engine.rows`) says False, which
    ``edges``, ``start`` and ``layer`` need (below).

    ``obstacles`` (near, far, top) (m) stand across the way, ``top`` m above the ground:
    a thin screen at ``near`` where ``far`` is ``near``, otherwise a building from
    ``near`` to ``far`` along the way with its rigid roof at ``top``, between the height
    steps where it falls. ``steps_per_wavelength`` sets the height step at the highest
    frequency (the wide-angle equation's in proportion).

    ``trunk``, a :class:`Trunk` given to the runs that differ in their obstacles and
    points alone, keeps the narrow-angle march over the open ground from one of them to
    the next; each gets what it gets without it. A run with a ``screen``, with
    ``speed_at``, or with a ``start`` or a ``layer`` of its own marches without it, and
    one with steep points clears it before the wide-angle equation runs.

    ``edges``, a list, takes what the march finds at each obstacle it meets beyond the
    source on its way to the farthest point, in order: (range, what, top, pressure), the
    pressure (F,) at the obstacle's top just before the obstacle changes the field,
    ``what`` being "screen" (the field that reaches a thin screen's top), "roof" (at a
    building's near corner: the field without the building) or "behind" (at its far
    corner: the field on its roof; of a roof the source stands over, the far corner
    alone). It needs ``refracting`` False, with which the narrow-angle march gives
    every point.

    ``start``, (range, field), begins the march ``range`` m from the source with the
    pressure ``field(z)`` gives on the heights z (m) there, (F, len(z)), in place of the
    starting field at the source, which holds for sound within some 35 degrees of the
    horizontal: the field is taken up to the absorbing layer and fades out over the
    layer's first ``START_FADE``. What stands before ``range`` does nothing; an obstacle
    at it is met at once, and a building over it keeps its roof under the field. The
    start lies beyond the source and no point before it (a ValueError otherwise). It
    needs ``refracting`` False too.

    ``layer`` (m) makes the absorbing layer thicker than its ``LAYER_WAVELENGTHS``
    wavelengths. Of a wave that goes up through the layer at a slope s and comes back
    down from the top of the grid, the layer leaves exp(-2 ``LAYER_STRENGTH`` thickness
    / (3 s)) at most (less where the grid, a power of two of height steps, goes on
    above the layer, at the layer's full strength): a start with waves that rise
    steeply needs it thicker. With ``refracting`` False alone.

    ``bounded`` (with ``independent``) marches a run past obstacles in bounded steps; a
    run over open ground is the plain one. An obstacle's top bends some of the sound
    almost straight up and down, far more of it than it bends down to a point in its
    shadow, and the source sends some up as steeply. A plain step takes that sound any
    way up in one step: past the absorbing layer, which takes sound by the metre along
    the range, and past the top of the grid, from where it comes back down, as the
    transform repeats the grid in height. It ends in the shadow, by an amount that goes
    with where the layer starts. A bounded step takes no sound further up or down than
    the layer is thick, and the layer damps what stands in its upper half at a step's
    end (LAYER_DEEP), so that none comes back. The bounded step takes whole the sound
    that goes up or down gently enough (:meth:`_Steps.takes`: in full steps to 51
    degrees from the horizontal in the bands up to 1 kHz and to 55, 60 and 64 degrees in
    those of 2, 4 and 8 kHz) and less of the steeper. A point that the source or the top
    of an obstacle met reaches more steeply, straight or by way of the ground, gets
    one step of its own from the source or the last obstacle, which takes the sound of
    every height (:meth:`_Steps.own`); and where the next obstacle's top lies as
    steeply, the march goes there in shorter steps (:meth:`_Steps.run`).
    Without ``start``, ``layer`` and ``edges``."""
    if edges is not None and refracting is not False:
        raise ValueError("the field at the obstacles' edges needs refracting=False")
    if start is not None and refracting is not False:
        raise ValueError("a march from a given field needs refracting=False")
    if layer is not None and refracting is not False:
        raise ValueError("a march with a thicker absorbing layer needs refracting=False")
    if bounded and not independent:
        raise ValueError("a march in bounded steps takes each point by its own last step")
    if bounded and (start is not None or layer is not None or edges is not None):
        raise ValueError("a march in bounded steps begins at the source, under its own layer")
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if start is not None and not start[0] > 0.0:
        raise ValueError("a march from a given field begins beyond the source")
    if start is not None and np.any(ranges < start[0] - 1e-9):
        raise ValueError("a march from a given field gives no point before its start")
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
            edges,
            start,
            layer,
            bounded,
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
            bounded=bounded,
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
        #: Whether the march kept was made in bounded steps.
        self._bound = False
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
    edges: list | None = None,
    start: tuple | None = None,
    layer: float | None = None,
    bounded: bool = False,
) -> np.ndarray:
    """:func:`march` by the narrow-angle equation alone, past ``obstacles`` (as
    :func:`_obstacles` gives them)."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if bounded:
        # The steps are bounded where the march meets an obstacle: a point short of the
        # first one gets the plain march, as it does by itself.
        nearest = min((near for near, _, _ in obstacles), default=math.inf)
        short = ranges < nearest - 1e-9
        if short.any() and not short.all():
            both = np.empty((len(f), len(ranges)), dtype=complex)
            for part, which, kept in ((short, False, None), (~short, True, trunk)):
                both[:, part] = _march(
                    f,
                    speed,
                    impedance,
                    zs,
                    ranges[part],
                    heights[part],
                    top,
                    screen,
                    speed_at,
                    independent,
                    obstacles,
                    steps_per_wavelength,
                    kept,
                    bounded=which,
                )
            return both
        bounded = not short.any()
    apart = start is not None or layer is not None
    if trunk is None or screen is not None or speed_at is not None or apart:
        alone = _Steps(f, speed, impedance, zs, top, screen, speed_at, steps_per_wavelength, layer)
        return alone.run(ranges, heights, independent, obstacles, None, edges, start, bounded)
    with trunk._lock:
        steps = trunk._steps
        if steps is None or not steps.fits(f, speed, zs, top, steps_per_wavelength):
            steps = _Steps(f, speed, impedance, zs, top, None, None, steps_per_wavelength)
            trunk._steps = steps
            trunk._restart()
        steps.impedance = impedance
        return steps.run(ranges, heights, independent, obstacles, trunk, edges, None, bounded)


class _Steps:
    """The narrow-angle march's grid, air and ground for the frequencies ``f`` (F,) of
    a run (the other arguments as :func:`march`'s), and its steps."""

    def __init__(
        self, f, speed, impedance, zs, top, screen, speed_at, steps_per_wavelength, layer=None
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
        n, dz, thickness = _grid(f, c0, top, steps_per_wavelength, layer)
        self.n, self.dz = n, dz
        self.z = z = np.arange(n) * dz
        #: (n,) the speed of sound at the source, on the heights.
        self.c = np.asarray(speed(z), dtype=float)
        k = 2.0 * math.pi * f[:, None] / self.c[None, :]
        self.depth = depth = np.clip((z - top) / thickness, 0.0, 1.0)
        self.delta = (k - ka[:, None]) + 1j * LAYER_STRENGTH * depth**2  # (F, n)
        #: The absorbing layer's thickness (m), and what it also takes at each half of a
        #: bounded step (n,).
        self.thickness = thickness
        self.deeper = np.exp(-0.5 * LAYER_DEEP * (2.0 * depth) ** LAYER_DEEP_POWER)
        #: Whether the run in hand takes bounded steps (a bounded run past obstacles).
        self.bound = False
        #: The longest and the shortest wavelength (m).
        self.longest = c0 / f.min()
        self.shortest = shortest

        self.kz = kz = 2.0 * math.pi * np.fft.fftfreq(2 * n, d=dz)
        vertical = np.sqrt((ka[:, None] ** 2 - kz[None, :] ** 2).astype(complex))
        self.vertical = np.where(vertical.imag < 0.0, -vertical, vertical)  # (F, 2n)
        self.width = 2.0 * math.pi / (2 * n * dz)
        self.step = max(RANGE_STEP_WAVELENGTHS * shortest, RANGE_STEP_MIN_M)
        #: The speed of sound the steps are taken in (``speed_at``'s, where it changes).
        self.current = speed
        #: By step length, whether the step is bounded and the height it is read at
        #: alone (or None), the phase screen's half step and the propagator.
        self.steps: dict[tuple, tuple] = {}
        #: By roof and the same, the same over a roof (:meth:`over_roof`).
        self.roofs: dict[tuple, tuple] = {}
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

    def stepping(self, dr: float, reads: float | None = None) -> tuple:
        """The phase screen's half step and the propagator over ``dr`` (most steps are
        full: theirs stay while the shorter ones' and the points' own come and go);
        ``reads``: of a bounded step read at that height (m) alone
        (:meth:`propagator`)."""
        steps = self.steps
        key = (dr, self.bound, reads)
        if key not in steps:
            if len(steps) > 4:
                for other in [each for each in steps if not self.full(each[0], each[2])]:
                    del steps[other]
            if self.bound:
                steps[key] = (
                    _exp_rows(0.5j * dr * self.delta) * self.deeper[None, :],
                    self.propagator(dr, self.n, 0.0, reads),
                )
            else:
                steps[key] = (
                    _exp_rows(0.5j * dr * self.delta),
                    _exp_mirrored(1j * dr * (self.vertical - self.ka[:, None])),
                )
        return steps[key]

    def full(self, dr: float, reads: float | None) -> bool:
        """Whether a step over ``dr`` is one of the march's full steps, whose tables
        stay: not a shorter one, and not a point's own (``reads``), one as long as a
        full step included."""
        return dr == self.step and reads is None

    def propagator(self, dr: float, n: int, base: float, reads: float | None) -> np.ndarray:
        """(F, 2n) a bounded step's table over ``dr`` on the ``n`` heights from ``base``
        (m) up. With ``reads``, of a step whose field is read at that height (m) alone, a
        point's own from the last obstacle: it takes the sound of every height of the
        grid, straight and by way of the ground, whole from those under the absorbing
        layer and through the lower part of it, and less up to the top of the grid. The
        sound comes straight from as far down as the point is above ``base`` and from as
        far up as the grid goes on above it, and by the ground from as far below
        ``base`` as its height is above: together the transform's length, so that its
        window lies about ``base`` as seen from the point and no repeat of the grid
        falls within it. Its table is left as the window makes it, as no step follows
        it."""
        thickness = self.thickness
        if reads is None:
            return _bounded(
                self.ka,
                n,
                self.dz,
                dr,
                BOUNDED_WHOLE * thickness,
                BOUNDED_WINDOW * thickness,
                BOUNDED_SMOOTH * thickness,
                math.log(BOUNDED_GROWTH) / (self.top + thickness),
            )
        end = (n - 1) * self.dz
        whole = max(end - (BOUNDED_WINDOW - BOUNDED_WHOLE) * thickness, 0.5 * end)
        return _bounded(self.ka, n, self.dz, dr, whole, end, centre=max(reads - base, 0.0))

    def takes(self, slope: float, dr: float) -> bool:
        """Whether a bounded step over ``dr`` (m) takes whole the sound that goes up or
        down by ``slope`` (m a metre): what it goes up or down in the step, and
        BOUNDED_FRESNEL times its Fresnel zone's width beside that, the square root of
        the wavelength times the step's length along the sound's way, seen along the
        height, lie within what the step's window lets through whole."""
        width = math.sqrt(self.longest * dr) * (1.0 + slope * slope) ** 0.75
        return slope * dr + BOUNDED_FRESNEL * width <= BOUNDED_WHOLE * self.thickness

    def shorter(self, slope: float, span: float) -> int:
        """In how many equal steps the march goes ``span`` (m) on, for each to take
        whole (:meth:`takes`) the sound that goes up or down by ``slope`` there: as few
        as do, none shorter than BOUNDED_SHORTEST of the shortest wavelength."""
        spread = BOUNDED_FRESNEL * math.sqrt(self.longest) * (1.0 + slope * slope) ** 0.75
        whole = BOUNDED_WHOLE * self.thickness
        # The step's length dr: slope dr + spread sqrt(dr) = whole.
        root = 2.0 * whole / (spread + math.sqrt(spread * spread + 4.0 * slope * whole))
        dr = min(max(root * root, BOUNDED_SHORTEST * self.shortest), self.step)
        return math.ceil(span / dr - 1e-9)

    def impedances(self, at: float) -> np.ndarray:
        value = self.impedance(at) if self.impedance is not None else None
        if value is None:
            return np.full(len(self.f), np.inf, dtype=complex)
        return np.broadcast_to(np.asarray(value, dtype=complex), (len(self.f),))

    def ground(self, at: float) -> _Ground:
        """The ground at range ``at``: the last one while its impedances hold."""
        value = self.impedances(at)
        if self.here is None or not np.array_equal(value, self.here.key):
            self.here = _Ground(value, self.ka, self.kz, self.z, self.width, self.top)
        return self.here

    def start(self, roof: float = 0.0) -> np.ndarray:
        """(F, n) the starting field, at the source: with its image in the ground, or
        over a rigid roof ``roof`` m up (above 0) in the roof."""
        ka, z, zs = self.ka, self.z, self.zs
        here = self.ground(0.0)
        if roof > 0.0:
            return np.sqrt(1j * ka)[:, None] * (
                _starter(ka[:, None] * (z - zs)) + _starter(ka[:, None] * (z - (2.0 * roof - zs)))
            )
        return np.sqrt(1j * ka)[:, None] * (
            _starter(ka[:, None] * (z - zs))
            + here.normal[:, None] * _starter(ka[:, None] * (z + zs))
        )

    def over_roof(
        self,
        psi: np.ndarray,
        r: float,
        dr: float,
        out: np.ndarray,
        reads: float | None = None,
        taken: bool = False,
    ) -> np.ndarray:
        """A step over a rigid roof ``roof`` m up, between the grid's samples: the march
        on the heights from the sample under it up, reflecting off it as off a rigid
        ground (the field mirrored about it, each sample weighed by the share of its
        cell above it, as a screen's are)."""
        n, dz, z, ka, roof, roofs = self.n, self.dz, self.z, self.ka, self.roof, self.roofs
        floor = int(roof / dz)
        part = n - floor
        key = (roof, dr, self.bound, reads)
        if key not in roofs:
            if len(roofs) > 4:
                for other in [each for each in roofs if not self.full(each[1], each[3])]:
                    del roofs[other]
            heights_kz = 2.0 * math.pi * np.fft.fftfreq(2 * part, d=dz)
            if self.bound:
                roofs[key] = (
                    _exp_rows(0.5j * dr * self.delta[:, floor:]) * self.deeper[None, floor:],
                    self.propagator(dr, part, float(z[floor]), reads),
                    np.exp(-2j * heights_kz * (roof - z[floor])),
                    roof_share(z[floor:], roof),
                )
            else:
                up = np.sqrt((ka[:, None] ** 2 - heights_kz[None, :] ** 2).astype(complex))
                up = np.where(up.imag < 0.0, -up, up)
                roofs[key] = (
                    _exp_rows(0.5j * dr * self.delta[:, floor:]),
                    _exp_mirrored(1j * dr * (up - ka[:, None])),
                    np.exp(-2j * heights_kz * (roof - z[floor])),
                    roof_share(z[floor:], roof),
                )
        half, propagate, mirror, share = roofs[key]
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
        if taken:
            half = np.ones_like(half)
        elif self.screen is not None:
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

    def air(self, at: float) -> None:
        """Takes the speed of sound that holds at range ``at``, where it changes along
        the way (``speed_at``)."""
        if self.speed_at is not None:
            there = self.speed_at(at)
            if there is not self.current:
                self.current = there
                speed = np.asarray(there(self.z), dtype=float)
                k = 2.0 * math.pi * self.f[:, None] / speed[None, :]
                self.delta = (k - self.ka[:, None]) + 1j * LAYER_STRENGTH * self.depth**2
                self.steps.clear()
                self.roofs.clear()

    def advance(
        self,
        psi: np.ndarray,
        r: float,
        dr: float,
        out: np.ndarray,
        reads: float | None = None,
        taken: bool = False,
        lands: float | None = None,
    ) -> np.ndarray:
        """The field a step ``dr`` further on from ``r``, written into ``out`` (which
        may be ``psi`` itself), over the ground at the step's middle; ``reads``: by a
        bounded step read at that height alone (:meth:`propagator`); ``taken``: of a
        field that has the phase screen's half steps in it already, over the ground at
        the range ``lands`` (:meth:`own`)."""
        ka, z = self.ka, self.z
        here = self.ground(r + 0.5 * dr if lands is None else lands)
        self.air(r + 0.5 * dr)
        if self.roof > 0.0:
            return self.over_roof(psi, r, dr, out, reads, taken)
        half, propagate = self.stepping(dr, reads)
        if taken:
            half = np.ones_like(half)
        elif self.screen is not None:
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
            self._advance(psi, here, half, propagate, surface, some_image, lo, hi, out, taken)
        return out

    def _advance(
        self,
        psi,
        here: _Ground,
        half,
        propagate,
        surface,
        some_image,
        lo: int,
        hi: int,
        out,
        own: bool = False,
    ) -> None:
        """The rows ``lo`` to ``hi`` of :meth:`advance` over the ground ``here``; all the
        rows, where the field is continued below the ground, or the surface wave kept, at
        some of the frequencies and not at others. ``own``: in a point's own step, for
        which the field continued below the ground stays whole deeper down
        (:class:`_Ground`)."""
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
            below *= here.fade_own if own else here.fade_below[lo:hi]
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

    def stretches(self, r: float, dr: float, what) -> list[tuple[float, float]]:
        """The stretches of a step over ``dr`` from ``r`` (m) along which ``what(at)``
        (bytes, of the range ``at``) stays the same: for each, the share of the step at
        which it ends and a range (m) in it. It is looked at in the middle of each of
        the equal parts the step is cut in, no longer than OWN_LOOK_M and at least
        OWN_LOOKS of them (a stretch shorter than a part may be passed over), and
        where two neighbours differ, the place between them where it changes is found
        to a thousandth of a part."""
        count = max(int(math.ceil(dr / OWN_LOOK_M)), OWN_LOOKS)
        part = dr / count
        found: list[tuple[float, float]] = []
        first = before = r + 0.5 * part
        held = what(first)
        for index in range(1, count):
            middle = r + (index + 0.5) * part
            here = what(middle)
            if here != held:
                lo, hi = before, middle
                for _ in range(10):
                    between = 0.5 * (lo + hi)
                    if what(between) == held:
                        lo = between
                    else:
                        hi = between
                found.append(((0.5 * (lo + hi) - r) / dr, first))
                first, held = middle, here
            before = middle
        found.append((1.0, first))
        return found

    def airs(self, r: float, dr: float) -> list[tuple[float, np.ndarray]]:
        """The airs along a step over ``dr`` from ``r`` (m), where the speed of sound
        changes along the way (``speed_at``): for each stretch of it in one air
        (:meth:`stretches`), the share of the step at which it ends and (F, n) that
        air's wavenumber less the reference one. Empty where the air is the same all
        along (the same speeds at the grid's heights, from one function or from two)."""
        speed_at = self.speed_at
        if speed_at is None:
            return []
        # By the speed function's identity (each kept, so that none's is given again):
        # its speeds at the heights, and their bytes.
        seen: dict[int, tuple] = {}

        def air(at: float) -> tuple:
            there = speed_at(at)
            kept = seen.get(id(there))
            if kept is None:
                speed = np.asarray(there(self.z), dtype=float)
                kept = seen[id(there)] = (there, speed, speed.tobytes())
            return kept

        found = self.stretches(r, dr, lambda at: air(at)[2])
        if len(found) < 2:
            return []
        return [
            (end, 2.0 * math.pi * self.f[:, None] / air(at)[1][None, :] - self.ka[:, None])
            for end, at in found
        ]

    def own(
        self,
        psi: np.ndarray,
        r: float,
        dr: float,
        out: np.ndarray,
        reads: float,
        lands: float | None = None,
    ) -> np.ndarray:
        """The field one step ``dr`` on from ``r``, into ``out``, as it is at the height
        ``reads`` (m) alone: a point's own step from the source or the last obstacle,
        which may be many full steps long (:meth:`run`). Its table takes the sound of
        every height (:meth:`propagator`), and in place of the phase screen's two half
        steps, which stand for the air at the step's two ends, each height's sound takes
        the air's phase along its own straight way to the point: the direct sound along
        the line from its height to the point's, and the reflected sound along the way
        down to the ground (or the roof the march is over) and up to the point, each
        through the airs it meets where the air changes along the step (:meth:`airs`).
        The reflected sound of every height takes one ground, the one at the range
        ``lands`` (m; None: the step's middle), where :meth:`run` has the reflection of
        the sound's way to the point land. It takes that ground whole, as a reflection
        off a ground without end: close to a change of ground, where the two grounds
        share the reflection, the step is off."""
        ka, z, dz, n = self.ka, self.z, self.dz, self.n
        self.air(r + 0.5 * dr)
        floor = int(self.roof / dz) if self.roof > 0.0 else 0
        heights = z[floor:]
        base = self.roof
        across = reads - heights
        near = np.abs(across) < 0.5 * dz
        both = (heights - base) + (reads - base)
        low = both < 0.5 * dz
        airs = self.airs(r, dr)
        if not airs:
            # The air's wavenumber less the reference one, and its integral over the
            # height.
            air = self.delta.real[:, floor:]
            summed = np.zeros_like(air)
            np.cumsum(0.5 * (air[:, 1:] + air[:, :-1]) * dz, axis=1, out=summed[:, 1:])
            at_point = np.array([np.interp(reads, heights, row) for row in summed])[:, None]
            at_base = np.array([np.interp(base, heights, row) for row in summed])[:, None]
            air_point = np.array([np.interp(reads, heights, row) for row in air])[:, None]
            air_base = np.array([np.interp(base, heights, row) for row in air])[:, None]
            # Its mean along the straight way from each height to the point, and along
            # the way by the ground: (two heights less than half a step apart take the
            # air there).
            direct = np.where(
                near[None, :], air_point, (at_point - summed) / np.where(near, 1.0, across)[None, :]
            )
            bounced = np.where(
                low[None, :],
                air_base,
                (summed + at_point - 2.0 * at_base) / np.where(low, 1.0, both)[None, :],
            )
        else:
            # The same through the airs along the step: each takes the stretch of the
            # way that lies in it, the heights between those the way has at its two
            # ends (down to the ground and up again, by the ground).
            direct = np.zeros((len(ka), len(heights)))
            bounced = np.zeros((len(ka), len(heights)))
            down = heights - base
            begin = 0.0
            for end, whole in airs:
                air = whole[:, floor:]
                summed = np.zeros_like(air)
                np.cumsum(0.5 * (air[:, 1:] + air[:, :-1]) * dz, axis=1, out=summed[:, 1:])

                def integral(y, summed=summed):
                    return np.array([np.interp(y, heights, row) for row in summed])

                air_point = np.array([np.interp(reads, heights, row) for row in air])[:, None]
                air_base = np.array([np.interp(base, heights, row) for row in air])[:, None]
                at_base = integral(np.array([base]))
                stretch = integral(heights + end * across) - integral(heights + begin * across)
                direct += np.where(
                    near[None, :],
                    (end - begin) * air_point,
                    stretch / np.where(near, 1.0, across)[None, :],
                )

                def along(share, summed=summed, integral=integral, at_base=at_base):
                    gone = share * both
                    return np.where(
                        (gone <= down)[None, :],
                        summed - integral(np.maximum(heights - gone, base)),
                        summed - 2.0 * at_base + integral(np.maximum(base + gone - down, base)),
                    )

                bounced += np.where(
                    low[None, :],
                    (end - begin) * air_base,
                    (along(end) - along(begin)) / np.where(low, 1.0, both)[None, :],
                )
                begin = end
        turbulent = None
        if self.screen is not None:
            turbulent = np.exp(0.5j * ka[:, None] * self.screen(r, dr, heights)[None, :])
        # (What the absorbing layer takes at the step's start, as in a half step.)
        start = psi[:, floor:] * (
            np.exp(-0.5 * dr * self.delta.imag[:, floor:]) * self.deeper[None, floor:]
        )
        if turbulent is not None:
            start = start * turbulent
        # The whole step of the field with the reflected sound's phase, and the direct
        # sound of the difference: what the step does to a field less what the ground
        # (the roof) adds to it.
        reflected = np.zeros_like(psi)
        reflected[:, floor:] = start * np.exp(1j * dr * bounced)
        self.advance(reflected, r, dr, out, reads, taken=True, lands=lands)
        # (Over a roof lower than the first height above the ground too.)
        if self.roof > 0.0:
            propagate = self.roofs[(self.roof, dr, self.bound, reads)][1]
        else:
            propagate = self.steps[(dr, self.bound, reads)][1]
        rest = start * (np.exp(1j * dr * direct) - np.exp(1j * dr * bounced))
        if self.roof > 0.0:
            rest = rest * roof_share(heights, base)[None, :]
        part = n - floor
        padded = np.zeros((len(psi), 2 * part), dtype=complex)
        padded[:, :part] = rest
        out[:, floor:] += np.fft.ifft(np.fft.fft(padded, axis=1) * propagate, axis=1)[:, :part]
        if turbulent is not None:
            out[:, floor:] *= turbulent
        return out

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
        edges: list | None = None,
        start: tuple | None = None,
        bounded: bool = False,
    ) -> np.ndarray:
        """(F, n) the pressure at the points ``ranges`` and ``heights`` past
        ``obstacles``, as :func:`_march` gives it; with ``trunk``, from where it has the
        field over the open ground. ``edges`` takes the field at the obstacles' tops,
        and ``start`` is where the march begins and its field there (see
        :func:`march`).

        ``bounded``: every step is a bounded one (:func:`march`; the points lie at or
        past the first obstacle). The sound's ways start at the source and at the top
        of each obstacle met, and go on to the next obstacle's top and to the points,
        straight and by way of the ground (or of the roof the march is over). The march
        goes on in full steps, with a shorter one to the next obstacle and each point's
        own last one, as a plain march does, where those take the steepest such way
        whole (:meth:`takes`). To the next obstacle, where they do not, the march goes
        in equal shorter steps that do (:meth:`shorter`), from the field as it was at
        the source or just past the last obstacle. A point that the source or a top
        met reaches too steeply gets one step of its own from that field
        (:meth:`own`), whatever the other points are, one at its own range included."""
        f, n, dz, z, step = self.f, self.n, self.dz, self.z, self.step
        #: Where the march begins (m from the source), and where its obstacles do.
        begin, first = 0.0, 1e-9
        if start is not None:
            begin = float(start[0])
            first = begin - 1e-9
            obstacles = tuple(o for o in obstacles if o[1] >= first)
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
        # A source over a roof starts with its image in it, which the trunk (started over
        # the ground) does not hold.
        under = max((e[2] for e in events if e[1] == "roof" and e[0] <= 1e-9), default=0.0)
        if under > 0.0:
            trunk = None
        # Of the runs before it, a run keeps the full steps' tables alone, and over
        # the roofs it meets itself.
        for other in [key for key in self.steps if not self.full(key[0], key[2])]:
            del self.steps[other]
        over = {e[2] for e in events if e[1] == "roof"}
        for key in [
            key for key in self.roofs if not (self.full(key[1], key[3]) and key[0] in over)
        ]:
            del self.roofs[key]
        self.bound = bounded
        if trunk is not None and trunk._bound != bounded:
            trunk._restart()
            trunk._bound = bounded

        psi = None
        r = 0.0
        if start is not None:
            self.here = None
            psi, r = self.given(start[1], begin), begin
        elif trunk is not None and len(order):
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
            psi = self.start(under)
        side = None  # a point's own last step
        last = None  # the last point's range and own step, while the march stays where it was
        # In bounded steps, where the sound's ways start from: the source and the tops
        # of the obstacles met, the range and the height (m) of each; and the range of
        # the last of them and the field there as it was (the starting field is made
        # again where the march has left it).
        met = [(0.0, self.zs)]
        origin = 0.0
        edge = psi if bounded and r == 0.0 else None
        for index in order:
            goal = ranges[index]
            # A point that takes a step of its own from the last obstacle: its height (m).
            own = None
            while True:
                if events and events[0][0] <= r + 1e-9:
                    event = events.pop(0)
                    if edges is not None and event[0] >= first:
                        at_top = _at_height(self.pressure(psi, r), z, event[2])
                        edges.append((*event, np.array(at_top)))
                    psi = self.meet(psi, event, roofs_over)
                    last = None
                    if bounded:
                        if r > 1e-9:
                            met.append((r, event[2]))
                            origin = r
                        edge = psi
                    continue
                # The march stops at each obstacle on the way, one at the goal included,
                # and at the goal.
                obstacle = bool(events) and events[0][0] <= goal + 1e-9
                steep, slope, span, above = False, 0.0, 0.0, 0.0
                if bounded and (obstacle or goal - origin > 1e-9):
                    # The steepest of the ways from the source and from the tops met
                    # to the next obstacle's top, or to the point, by way of the ground
                    # (the roof): a top higher than those after it sends its sound
                    # down over them.
                    there = events[0][2] if obstacle else heights[index]
                    target = events[0][0] if obstacle else goal
                    span = target - origin
                    above = max(there - self.roof, 0.0)
                    slope = max(
                        (max(top - self.roof, 0.0) + above) / (target - at)
                        for at, top in met
                        if target - at > 1e-9
                    )
                    steep = not self.takes(slope, min(step, span))
                if steep and not obstacle:
                    # Whatever the march has done for the points before it, one at
                    # this range included.
                    own = float(heights[index])
                    break
                if goal - r <= 1e-9:
                    break
                if steep:
                    if edge is None:
                        edge = self.start(under)
                    count = self.shorter(slope, span)
                    if side is None:
                        side = np.empty_like(edge)
                    for part in range(count):
                        at = origin + part * span / count
                        self.advance(edge if part == 0 else side, at, span / count, side)
                    psi, side, r = side, None, events[0][0]
                    last = None
                    continue
                dr = min(step, (events[0][0] if obstacle else goal) - r)
                if independent and dr < step and not obstacle:
                    break
                if psi is edge:
                    # The field there stays as it is for the steps that start from it.
                    psi = self.advance(psi, r, dr, np.empty_like(psi))
                else:
                    self.advance(psi, r, dr, psi)
                r += dr
                last = None
            # A point at the range of the one before it, the march where it was and its
            # own step the same, has the same pressure on the heights.
            if last is None or (goal, own) != last:
                reached, at = psi, r
                if own is not None:
                    # One step of its own from the source or the last obstacle.
                    if edge is None:
                        edge = self.start(under)
                    if side is None:
                        side = np.empty_like(psi)
                    # The ground its reflection lands on: where the steepest of the
                    # ways, the last stretch of the sound's shortest way over the tops
                    # to the point by way of the ground, comes down (over a roof, the
                    # roof).
                    lands = None if self.roof > 0.0 else goal - above / slope
                    self.own(edge, origin, goal - origin, side, own, lands)
                    reached, at = side, goal
                elif goal - r > 1e-9:
                    # Its own last step, off to the side of the march.
                    if side is None:
                        side = np.empty_like(psi)
                    reached, at = self.advance(psi, r, goal - r, side), goal
                p = self.pressure(reached, at)
                last = (goal, own)
            out[:, index] = _at_height(p, z, heights[index])
        return out

    def given(self, field, at: float) -> np.ndarray:
        """(F, n) the field of a march that begins at range ``at`` with the pressure
        ``field(z)`` gives on the heights there: up to the absorbing layer, and fading
        out over its first ``START_FADE``."""
        p = np.asarray(field(self.z), dtype=complex)
        fade = np.sin(0.5 * np.pi * np.clip(1.0 - self.depth / START_FADE, 0.0, 1.0)) ** 2
        return p * fade[None, :] * (math.sqrt(at) * np.exp(-1j * self.ka * at))[:, None]

    def pressure(self, psi: np.ndarray, at: float) -> np.ndarray:
        """(F, n) the pressure on the heights at range ``at``, of the field ``psi``
        there."""
        return psi * (np.exp(1j * self.ka * at) / math.sqrt(max(at, 1e-9)))[:, None]


def _at_height(p: np.ndarray, z: np.ndarray, h: float) -> list:
    """The pressure ``p`` (F, n) on the heights ``z``, at the height ``h``: a value for
    each frequency."""
    return [np.interp(h, z, row.real) + 1j * np.interp(h, z, row.imag) for row in p]


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


def _bounded(
    ka: np.ndarray,
    n: int,
    dz: float,
    dr: float,
    whole: float,
    end: float,
    smooth: float = 0.0,
    rate: float = 0.0,
    centre: float = 0.0,
) -> np.ndarray:
    """(F, 2n) a bounded step's propagator over ``dr`` (m) at the wavenumbers ``ka``
    (F,), for the field on ``n`` heights ``dz`` apart and as many below them (the
    transform's wavenumbers). A step on, the field is the field's convolution over the
    height with the step's kernel (Rayleigh's integral, with the march's phase taken off),

        K(d) = (i ka dr / (2 R)) H1(ka R) exp(-i ka dr),    R = sqrt(dr^2 + d^2),

    whose transform is the plain step's exp(i dr (sqrt(ka^2 - kz^2) - ka)). The bounded
    step's is the transform of K times a window: 1 where the sound goes up or down by
    less than ``whole`` (m), 0 beyond ``end`` (m), and falling smoothly between them.
    With ``centre`` (m) above 0 the window lies about the sound that comes up by that
    much instead: about the ground under a point that high, for a step read at the point
    alone, so that ``whole`` and ``end`` are heights the sound comes from, straight or
    by its image below the ground (``end`` no further than the heights reach).

    A window with a flat top makes the table larger than 1 for some steep waves, which
    grow from step to step as they go up. With ``smooth`` (m) above 0 a wave may grow by
    exp(``rate``) for each metre it goes up, so its table be as large as exp(``rate``
    times what it goes up in the step). Where the table is larger, it is multiplied by
    1 - B, B being its excess 1 - allowed / |table|, smoothed over the wavenumbers by a
    bump that is nowhere negative and scaled up to cover the excess (where that would
    make B larger than 1, B is made as :func:`_covered` says): the table is then
    nowhere larger than allowed, and as the bump is the transform of a function that
    ends at ``smooth``, the step takes no sound further than ``end`` + ``smooth``.

    The kernel is as narrow as the step is short: it is sampled finer than the heights
    by what BOUNDED_FINE says, and the transform kept at the heights' wavenumbers. For a
    step so short that BOUNDED_FINEST times as fine is too coarse, the kernel is the
    plain table's inverse transform instead, which repeats over the transform's length:
    of the sound that goes that far up in a step, which such a step has next to none of,
    it keeps what lands within the window."""
    rows = len(ka)
    index = np.arange(2 * n)
    apart = np.minimum(index, 2 * n - index) * dz
    kz = 2.0 * math.pi * np.fft.fftfreq(2 * n, d=dz)
    fine = math.ceil(0.5 + BOUNDED_FINE * (2.0 * math.pi / float(ka.max())) / dr)
    if fine > BOUNDED_FINEST:
        vertical = np.sqrt((ka[:, None] ** 2 - kz[None, :] ** 2).astype(complex))
        vertical = np.where(vertical.imag < 0.0, -vertical, vertical)
        kernel = np.fft.ifft(np.exp(1j * dr * (vertical - ka[:, None])), axis=1)
        if centre > 0.0:
            # (The transform's repeat of each sample that lies nearest ``centre``.)
            apart = np.abs((index * dz - centre + n * dz) % (2 * n * dz) - n * dz)
        kernel *= (1.0 - _smooth((apart - whole) / (end - whole)))[None, :]
        out = np.fft.fft(kernel, axis=1)
    else:
        h = dz / fine
        # The kernel's samples up (``m`` of them beside the one at 0) and down, as far
        # as the window reaches: the same with ``centre`` 0.
        m = min(int((centre + end) / h), (n if centre == 0.0 else 2 * n) * fine - 1)
        down = m if centre == 0.0 else min(int(max(end - centre, 0.0) / h), 2 * n * fine - 1 - m)
        d = np.arange(m + 1) * h
        slant = np.hypot(dr, d)
        # R - dr, without the loss of digits where the step is long.
        beyond = d * d / (slant + dr)
        window = 1.0 - _smooth((np.abs(d - centre) - whole) / (end - whole))
        above = window
        if centre > 0.0:
            # (The sound from above the point: the samples down from 0.)
            above = 1.0 - _smooth((d[: down + 1] + centre - whole) / (end - whole))
        out = np.empty((rows, 2 * n), dtype=complex)
        block = min(BLOCK_ROWS, rows)
        samples = np.zeros((block, 2 * n * fine), dtype=complex)
        for lo in range(0, rows, block):
            hi = min(lo + block, rows)
            k = ka[lo:hi, None]
            kernel = (0.5j * dr * h) * k / slant[None, :] * hankel1e(1, k * slant[None, :])
            if centre > 0.0:
                kernel *= np.exp(1j * k * beyond[None, :])
                if down:
                    samples[: hi - lo, -down:] = (kernel[:, 1 : down + 1] * above[None, 1:])[
                        :, ::-1
                    ]
                kernel *= window[None, :]
                samples[: hi - lo, : m + 1] = kernel
            else:
                kernel *= np.exp(1j * k * beyond[None, :]) * window[None, :]
                samples[: hi - lo, : m + 1] = kernel
                samples[: hi - lo, -m:] = kernel[:, m:0:-1]
            spectrum = np.fft.fft(samples[: hi - lo], axis=1)
            out[lo:hi, : n + 1] = spectrum[:, : n + 1]
            out[lo:hi, n + 1 :] = spectrum[:, -(n - 1) :]
    if smooth > 0.0:
        # What each wave goes up or down in the step (m): dr times its slope; the waves
        # that fade along the range grow by nothing.
        along = np.sqrt(np.maximum(ka[:, None] ** 2 - kz[None, :] ** 2, 0.0))
        going = along > 0.0
        rise = np.full(along.shape, np.inf)
        grown = np.full(along.shape, 700.0)
        with np.errstate(over="ignore"):
            np.divide(dr * np.abs(kz)[None, :], along, out=rise, where=going)
            np.multiply(rate, rise, out=grown, where=going)
        size = np.abs(out)
        allowed = np.exp(np.minimum(grown, 700.0))
        excess = 1.0 - np.divide(allowed, size, out=np.ones_like(size), where=size > allowed)
        if excess.any():
            bump = _bump(apart / smooth)
            cover = np.fft.fft(np.fft.ifft(excess, axis=1) * bump[None, :], axis=1)
            cover = np.maximum(cover.real, 0.0)
            for row in np.flatnonzero(excess.any(axis=1)):
                above = excess[row] > 0.0
                scale = float(np.max(excess[row, above] / cover[row, above]))
                factor = 1.0 - scale * cover[row]
                if factor.min() < 0.0:
                    factor = _covered(excess[row], bump)
                out[row] *= factor
    return out


def _covered(excess: np.ndarray, bump: np.ndarray) -> np.ndarray:
    """(2n,) 1 - B for a table's ``excess`` (2n,) where the smoothed excess, scaled up
    to cover it, would pass 1 (:func:`_bounded`: an excess at a few wavenumbers beside
    a wide one, under an absorbing layer that starts kilometres up). B is the bump's
    smoothing of the largest excess among each wavenumber's neighbours, those that hold
    BOUNDED_COVER of the bump's transform, divided by that share: nowhere less than the
    excess and nowhere more than 1 while the excess is less than the share, and the
    transform of a function that ends where ``bump`` does, as the scaled one is. An
    excess larger than the share is taken off as it is, which no function that ends
    there does."""
    weights = np.fft.fft(bump).real / len(bump)
    half = len(bump) // 2
    held = weights[0] + 2.0 * np.cumsum(weights[1:half])
    reach = min(int(np.searchsorted(held, BOUNDED_COVER * held[-1])) + 1, half - 1)
    share = float(weights[0] + 2.0 * weights[1 : reach + 1].sum())
    largest = maximum_filter1d(excess, size=2 * reach + 1, mode="wrap")
    spread = np.fft.fft(np.fft.ifft(np.minimum(largest / share, 1.0)) * bump).real
    return np.minimum(1.0 - np.clip(spread, 0.0, 1.0), 1.0 - excess)


def _smooth(x: np.ndarray) -> np.ndarray:
    """A smooth step from 0 at ``x`` = 0 and below to 1 at 1 and above, flat at both
    ends up to its second derivative."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _bump(x: np.ndarray) -> np.ndarray:
    """A bump from 1 at ``x`` = 0 to 0 at 1 and beyond whose transform is nowhere
    negative (Wendland 1995, Adv. Comput. Math. 4, 389: the function of smoothness 6 that
    is positive definite in up to three dimensions)."""
    x = np.clip(np.abs(x), 0.0, 1.0)
    return (1.0 - x) ** 8 * (x * (x * (32.0 * x + 25.0) + 8.0) + 1.0)


def _starter(x: np.ndarray) -> np.ndarray:
    """Greene's starting field at ``x`` = ka times the height from the source."""
    return (1.4467 - 0.4201 * x**2) * np.exp(-(x**2) / 3.0512)


def _grid(
    f: np.ndarray,
    c0: float,
    top: float,
    steps_per_wavelength: float = STEPS_PER_WAVELENGTH,
    layer: float | None = None,
) -> tuple[int, float, float]:
    """The heights' count and spacing, and the absorbing layer's thickness (``layer``,
    where that is more than the usual), for the frequencies ``f`` under an absorbing
    layer from ``top``."""
    dz = c0 / f.max() / steps_per_wavelength
    thickness = layer_thickness(f, c0)
    if layer is not None:
        thickness = max(thickness, float(layer))
    return int(2 ** math.ceil(math.log2((top + thickness) / dz))), dz, thickness


def layer_thickness(frequencies, c0: float) -> float:
    """The absorbing layer's usual thickness (m) for ``frequencies`` in air of speed
    ``c0``."""
    return max(LAYER_WAVELENGTHS * c0 / np.min(frequencies), LAYER_MIN_M)


def footprint(
    frequencies,
    c0: float,
    top: float,
    steps_per_wavelength: float = STEPS_PER_WAVELENGTH,
    bounded: bool = False,
) -> int:
    """About the memory (bytes) :func:`march` takes for ``frequencies`` under an
    absorbing layer from ``top`` in air of speed ``c0`` on the ground: the narrow-angle
    march's, or the wide-angle one's for steep points after it, the larger, whether or
    not a run has steep points. ``bounded``: of a run in bounded steps, which also
    keeps the field at the last obstacle, makes its tables from the steps' kernels and
    takes a point's own step from that field."""
    from acoustic_spaces.engine import wide_angle

    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    n = _grid(f, c0, top, steps_per_wavelength)[0]
    wide = wide_angle.STEPS_PER_WAVELENGTH * steps_per_wavelength / STEPS_PER_WAVELENGTH
    arrays = 34 if bounded else 24
    return max(arrays * len(f) * 2 * n * 16, wide_angle.footprint(f, c0, top, wide))


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
    as exp(Im kp y). Deep down it is faded out (see FADE_GROWTH).

    A point's own step (:meth:`_Steps.own`) takes the sound of every height under the
    absorbing layer by its image too, so it needs the continued field whole as deep as
    the layer starts above the ground (``top``, m): for that step the field fades from
    there to the deepest height (from the middle of the heights, where the layer starts
    below it), however much it has grown by then (``fade_own``, (n,)): nothing of it is
    carried on to a next step, so its growth does not build up as it would along a
    march."""

    def __init__(self, zg, ka, kz, z, width, top: float = 0.0) -> None:
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
        start = max(0.5 * z[-1], top)
        t = np.clip((z - start) / (z[-1] - start), 0.0, 1.0)
        self.fade_own = 0.5 * (1.0 + np.cos(math.pi * t))
        # What the march takes of the rows ``image`` at every step.
        self.every_image = bool(self.image.all())
        self.every_surface = bool(self.surface.all())
        self.decay_surface = self.decay[self.surface]
        self.growth_below = self.growth[self.image]
        self.lift_below = self.scale[self.image, None] * self.shrink[self.image]
        self.fade_below = self.fade[self.image]


__all__ = [
    "Trunk",
    "footprint",
    "layer_thickness",
    "march",
    "open_share",
    "roof_share",
]
