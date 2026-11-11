"""Wave-based simulation of the lowest octave bands (or their third-octave bands).

Rays and image sources treat sound as rays, which holds while the room is large next
to the wavelength. At low frequencies it is not: small rooms ring at their modes, and
sound bends round everything. The wave solver computes the sound field itself, with
the finite-difference time-domain method (FDTD) on a cubic grid over the room, and its
octave bands replace the ray-traced ones.

- One grid covers every octave band up to ``max_band_hz``, with
  ``points_per_wavelength`` cells per wavelength at the top of that band's crossover.
  In a simulation in third-octave bands, those are the third octaves of these octave
  bands, and the grid is set by the top of the highest of them.
  Each octave higher costs about 16 times as much (cells 8 times smaller, steps twice
  as many). All bands share the grid, so a mode near a crossover is the same mode on
  both sides of it.
- Waves on the grid travel slower the shorter they are. ``scheme`` picks the stencil:
  the interpolated isotropic one (19 points, :data:`ISOTROPIC_COURANT`) is 0.67 % slow
  at 8 cells per wavelength, within 0.03 % the same in every direction; the standard
  one (7 points, :data:`COURANT`) up to 1.7 %, along the grid's axes. ``auto`` takes the
  isotropic one but round an open model, where only the standard one has the matched
  layer. Along the axes both are slowest (:func:`slow_by`); the run's details give how
  slow the top of the highest band is there (``wave_slow_percent``).
- The surfaces absorb as their materials do in each band: each material becomes a
  locally reacting surface whose admittance changes with frequency, fitted so that its
  random-incidence absorption matches the material's in every wave band
  (:mod:`acoustic_spaces.engine.impedance`). One run covers all the wave bands.
- Cells are air when the source reaches them without crossing a surface, so closed
  rooms, open models and thin screens all work. Round an open model, whose air reaches
  the edge of the grid, the grid grows by a perfectly matched layer of
  :data:`LAYER_CELLS` cells that takes in the sound leaving it, and leaves room
  (:data:`LAYER_ROOM`) for sound that runs along it, on the sides where no surface of
  the model lies between that sound and the layer (:func:`_open_sides`). Openings are not
  surfaces here: the air flows through them. Surfaces that let only part of the sound
  through are not modelled, so a model with any stops the wave solver with an error.
- At these frequencies the source radiates the same way in all directions. The
  listener records the pressure and, from its gradient, the direction the sound comes
  from (first-order Ambisonics), so spatial outputs keep the low end's direction.
- The bands join the ray-traced ones through the zero-phase crossovers the impulse
  response synthesis uses, which sum to one; the lowest starts at :data:`LOW_EDGE_HZ`
  (22 Hz) instead of 0 Hz. Their energies, for the room parameters, are counted
  through the octave filters of the analysis of measured responses.
- Outdoors, in wind and temperature that change with height
  (:class:`~acoustic_spaces.scene.Atmosphere`), the speed of sound changes from cell to
  cell: the still air's at the cell's height above the ground, plus the wind's
  component along the horizontal from the source, the effective speed of sound the rays
  and the parabolic equation take (:mod:`acoustic_spaces.engine.refraction`). The grid
  is then fine enough for the slowest sound and steps in time for the fastest, so it
  costs a little more than in still air. Sound that turns back towards the source, off
  a building, keeps the speed of sound of sound going away from it.
- With ``wind="moving"`` the air moves instead: each cell keeps the still air's speed
  of sound, and before each step both of the leapfrog's pressure levels are carried
  along with the wind, dq/dt = -v . grad q (fourth-order central differences, second
  order or upwind next to a face, the strong-stability-preserving Runge-Kutta method
  of third order); the still air's step follows, its boundaries, the matched layer and
  the source fixed to the ground. In the air's own frame that is the still air's step
  exactly, so a uniform wind gives the convected wave equation: downwind sound at
  c + v, upwind at c - v whichever way it turned. Where the wind changes, the term
  of the linearised Euler equations that the wind's shear puts on the air's velocity is
  left out, about ``|dv/dz| / omega`` of the rest. The air moves no more than half a cell a
  step, and the matched layer's air is still. It costs three to four times the
  effective speed of sound's run, and runs on the CPU (asked for the GPU, it says so and stays
  there).
- The grid is laid out on the CPU and stepped there, or with ``device`` on a GPU,
  through WebGPU (:mod:`acoustic_spaces.engine.gpu`); the two agree to rounding, in
  wind and temperature too (with the effective speed of sound).

The levels follow the ray tracer's convention: in free field, the direct sound at r
metres is an impulse of height 1 / r.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from acoustic_spaces import synthesis
from acoustic_spaces.engine.gpu import Device, GpuWaveSolver, gpu_name, use_gpu
from acoustic_spaces.engine.impedance import (
    MAX_ABSORPTION,
    Admittance,
    absorption_of_impedance,
    admittance,
    fit_admittance,
    fit_impedance,
)
from acoustic_spaces.materials.bands import BANDS_PER_OCTAVE, OCTAVE_BANDS_HZ, band_centres

#: Courant number of the 7-point (standard rectilinear) stencil: its stable limit.
COURANT = 1.0 / math.sqrt(3.0)
#: Courant number of the 19-point (interpolated isotropic) stencil: its stable limit.
ISOTROPIC_COURANT = math.sqrt(0.75)
#: Which stencil steps the grid (see :func:`wave_responses`).
Scheme = Literal["auto", "standard", "isotropic"]
#: How the wave solver takes the wind (see :func:`wave_responses`).
WindModel = Literal["effective", "moving"]
#: The stencils' points, and their Courant numbers, by scheme.
STENCILS = {"standard": (7, COURANT), "isotropic": (19, ISOTROPIC_COURANT)}
#: Each stencil's link weights, by its points: of an axial link and of an edge link (the
#: Courant number squared shared out over the links, as the update takes them).
LINK_WEIGHTS = {7: (1.0 / 3.0, 0.0), 19: (0.25, 0.125)}
#: Grids with more cells than this are refused (about 20 bytes each).
MAX_CELLS = 40_000_000
#: The highest bands the solver may compute (``simulation.wave.max_band_hz``).
WAVE_BANDS_HZ = OCTAVE_BANDS_HZ[:4]
#: The source's high-pass, applied twice: once so that the air the source pushes into
#: the room stops flowing, twice so that as much flows out again and a closed room's
#: pressure comes back to rest. Its length in seconds; twice, it is flat within 0.2 dB
#: above 20 Hz.
HIGH_PASS_S = 0.12
#: Cells of the matched layer round an open model, and its damping at the grid's edge
#: (the damping rate times half a time step). Sound leaving the model comes back from
#: it 72 to 74 dB down; from the grid's edge alone, 7 to 14 dB down (docs/validation.md,
#: Open models).
LAYER_CELLS = 8
LAYER_DAMPING = 0.5
#: Round an open model, the grid reaches past the source and the listeners, on each
#: side, by at least this share of the farthest a listener lies from the source along
#: that side (across its axis), before its margin and its matched layer; under an
#: outdoor ground, not at all. Sound that runs along the layer meets it near grazing,
#: where the layer sends much of it back: in one-off checks over a rigid plate, the
#: source and the listener 1.5 m above it and the layer a few cells above them (20 cm
#: cells), the level read 0.26 dB below the exact one at 125 Hz 20 m away, and 0.50 and
#: 1.71 dB below it at 63 and 125 Hz 40 m away. With this room it is within 0.03 dB of
#: the exact one, 20 to 160 m away (docs/validation.md, Open models). A side that the
#: model's own surfaces close off from the source and the listeners gets none
#: (:func:`_open_sides`).
LAYER_ROOM = 0.15
#: The wave solver's part of a response starts here (Hz), with the synthesis's
#: crossover shape round it: an octave below the 63 Hz band's lower edge, so that band,
#: skirts and all, comes from the wave solver. Further down a closed room's air is
#: squeezed as a whole (the wave solver's source leaves no air in it, so this falls
#: away), and the rays' response stays.
LOW_EDGE_HZ = OCTAVE_BANDS_HZ[0] / (2.0 * math.sqrt(2.0))
#: Length of the source's low-pass (taps; odd), which keeps the grid free of the
#: frequencies it cannot carry, and its cutoff (of the grid's sample rate) with the
#: 7-point stencil; it scales with the stencil's Courant number (0.3 with the 19-point
#: one), so that the top of the highest band lies as far below it with either.
_LOW_PASS_TAPS = 33
_LOW_PASS_CUTOFF = 0.2
#: Runs of more cell updates than this (a few seconds' work) report their progress.
_LONG_RUN = 2e9


def band_top(band: int, total: int = len(OCTAVE_BANDS_HZ)) -> float:
    """Highest frequency band ``band`` of the ``total`` (8 octave or 24 third-octave)
    bands has (Hz): the top of its upper crossover."""
    edge = synthesis.crossover_frequencies(total)[band]
    return edge * 2.0 ** (synthesis.crossover_width(total) / 2)


def wave_gains(
    n_fft: int, sample_rate: float, bands: int, total: int = len(OCTAVE_BANDS_HZ)
) -> np.ndarray:
    """Zero-phase gains (bands, n_fft // 2 + 1) of the lowest ``bands`` of the ``total``
    (8 octave or 24 third-octave) bands as the wave solver covers them: the synthesis's
    band gains, the lowest one starting at :data:`LOW_EDGE_HZ`."""
    gains = synthesis.band_gains(n_fft, sample_rate, total)[:bands]
    f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    width = synthesis.CROSSOVER_WIDTH
    x = (np.log2(np.maximum(f, 1e-6)) - math.log2(LOW_EDGE_HZ) + width / 2) / width
    gains[0] *= np.sin(0.5 * np.pi * np.clip(x, 0.0, 1.0)) ** 2
    return gains


def _filter(x: np.ndarray, gain: np.ndarray, n_fft: int) -> np.ndarray:
    """x (n, ...) through a zero-phase gain on the rFFT grid of n_fft."""
    shape = (-1,) + (1,) * (x.ndim - 1)
    spectrum = np.fft.rfft(x, n_fft, axis=0) * gain.reshape(shape)
    return np.fft.irfft(spectrum, n_fft, axis=0)[: len(x)]


def grid_phase(theta, stencil: int, ratio: float = 1.0) -> np.ndarray:
    """The phase w T a wave turns through in a time step on the ``stencil``-point grid,
    from the phase it turns through across a cell along each axis, ``theta`` (..., 3),
    k h: the stencil's dispersion relation, 2 - 2 cos(w T) = ratio^2 sum over the links
    of g_k (1 - cos(k . d_k)). ``ratio`` is the speed of sound over the one the grid's
    time step is set by (in wind and temperature, below 1 in all but the fastest air)."""
    axial, edge = LINK_WEIGHTS[stencil]
    cos = np.cos(np.asarray(theta, dtype=float))
    total = 2 * axial * (1 - cos).sum(axis=-1)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        total += 4 * edge * (1 - cos[..., a] * cos[..., b])
    return np.arccos(1 - ratio**2 * total / 2)


def slow_by(stencil: int, per_wavelength: float, ratio: float = 1.0) -> float:
    """How much slower than it should (a fraction) a wave of ``per_wavelength`` cells per
    wavelength travels on the ``stencil``-point grid along its axes, the way it is
    slowest with either stencil; ``ratio`` as in :func:`grid_phase` (below 1, slower
    still). The 19-point stencil at 8 cells per wavelength: 0.67 %, at 5: 1.85 %; the
    7-point one 1.74 % and 4.56 %."""
    courant = dict(STENCILS.values())[stencil]
    theta = 2.0 * math.pi / per_wavelength
    turned = float(grid_phase([theta, 0.0, 0.0], stencil, ratio))
    return 1.0 - turned / (ratio * courant * theta)


@dataclass(frozen=True)
class BandGrid:
    """The grid of one band's run."""

    band: int
    #: Cell size (m).
    cell: float
    #: Sample rate of the run (Hz): the output rate over or times a whole number.
    rate: float


def band_grid(
    band: int,
    sample_rate: float,
    speed_of_sound: float,
    points_per_wavelength: float = 8.0,
    total: int = len(OCTAVE_BANDS_HZ),
    courant: float = COURANT,
) -> BandGrid:
    """The coarsest grid with ``points_per_wavelength`` cells per wavelength at the top of
    band ``band`` (of ``total``, 8 octave or 24 third-octave bands) whose time step is a
    whole number of the output's sample periods, or one of them over a whole number
    where the grid needs more steps per second than the output has samples. ``courant``
    is the stencil's Courant number."""
    cell = speed_of_sound / (points_per_wavelength * band_top(band, total))
    rate = speed_of_sound / (courant * cell)  # one step crosses `courant` cells
    if rate <= sample_rate:
        rate = sample_rate / math.floor(sample_rate / rate)
    else:
        rate = sample_rate * math.ceil(rate / sample_rate)
    return BandGrid(band, speed_of_sound / (courant * rate), rate)


def _cutoff(courant: float) -> float:
    """The source's low-pass cutoff (of the grid's sample rate) on the grid of the
    stencil whose Courant number is ``courant``."""
    return _LOW_PASS_CUTOFF * courant / COURANT


def source_signal(
    rate: float, cell: float, courant: float = COURANT, speed: float = 1.0, *, level: bool = True
) -> tuple[np.ndarray, int]:
    """What the source adds to the grid on each step, and its delay in steps.

    An impulse, low-passed (so the grid carries no frequencies it cannot) and
    high-passed twice (so it leaves no air in the room), scaled so that in free field the
    pressure at r metres is that impulse over r. ``courant`` is the stencil's Courant
    number, which says which stencil it is for and sets the low-pass's cutoff
    (:func:`_cutoff`); ``speed`` the speed of sound at the source over the grid's (1
    where it is the same everywhere).

    On the 19-point grid a point source's sound comes out louder than what the source
    adds, the more the higher the frequency: by 9 / (5 + 4 cos wT) in pressure, w T the
    frequency's turn over a time step (the far field of the scheme's own response to one
    cell, by stationary phase: 3 / (2 + cos kh) along an axis, kh the turn over a cell
    there, which the scheme ties to the frequency by cos kh = 1 - 4 (1 - cos wT) / 3).
    That is the gain along the axes and the cube's diagonals; along a face's diagonals
    it is up to 0.03 dB more at eight cells per wavelength, and the same within 0.01 dB
    over a band: 0.06 and 0.35 dB over the 63 and 125 Hz bands of a grid up to the
    125 Hz band. Three taps, 2/9, 5/9 and 2/9, have the inverse gain exactly and are
    part of the signal, with a step of delay (where the source's speed of sound is
    ``speed`` times the grid's, the side taps are over the square of ``speed``, the
    middle one what is left of one); ``level=False`` gives the signal without them, the
    impulse that the signal with them stands for. The 7-point grid's gain changes with
    the direction (twelve cells away, over that 125 Hz band: none along an axis, 0.25 dB
    along a face's diagonals and 0.32 to 0.34 dB along the cube's, by the places of the
    source and the listener within their cells; validation/wave_solver.py), so no signal
    can undo it.
    """
    half = _LOW_PASS_TAPS // 2
    k = np.arange(-half, half + 1)
    cutoff = _cutoff(courant)
    low = 2 * cutoff * np.sinc(2 * cutoff * k) * np.blackman(_LOW_PASS_TAPS)
    low /= low.sum()
    length = max(3, int(round(HIGH_PASS_S * rate)))
    window = np.hanning(length + 2)[1:-1]
    high = -window / window.sum()
    high[0] += 1.0
    signal = np.convolve(np.convolve(low, high), high)
    delay = half
    # (The 19-point stencil, known by its Courant number.)
    if level and math.isclose(courant, ISOTROPIC_COURANT, rel_tol=1e-9):
        side = 2.0 / (9.0 * speed**2)
        signal = np.convolve(signal, [side, 1.0 - 2.0 * side, side])
        delay += 1
    return 4.0 * math.pi * courant**2 / cell * signal, delay


@dataclass
class WaveResponse:
    """The lowest octave bands (or third-octave bands) of one response, from the wave
    solver.

    ``signals`` holds the run at ``rate``: the pressure and the first-order directional
    parts (samples, 4), AmbiX W, X, Y and Z in room coordinates, from ``lead_s`` before
    emission on. Only the frequencies of the wave bands count.
    """

    rate: float
    signals: np.ndarray
    #: The bands 0 to bands - 1 come from the wave solver.
    bands: int
    #: Cell size of the grid (m).
    cell: float
    #: How many bands the whole response has: 8 octave or 24 third-octave bands.
    total_bands: int = len(OCTAVE_BANDS_HZ)
    #: How long before emission ``signals`` start (s): the source's pulse is centred on
    #: emission, so a listener near the source hears it start before then.
    lead_s: float = 0.0
    #: The energy gain of each wave band from the source's feed, or None for none.
    band_gain: np.ndarray | None = None
    #: How long after emission the source starts playing (s).
    delay_s: float = 0.0
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def fed(self, gain: np.ndarray, delay_s: float) -> WaveResponse:
        """These bands with the energy ``gain`` of each and ``delay_s`` later."""
        gain = np.asarray(gain, dtype=float)
        total = gain if self.band_gain is None else self.band_gain * gain
        return dataclasses.replace(self, band_gain=total, delay_s=self.delay_s + delay_s)

    def _amplitudes(self) -> np.ndarray:
        return np.ones(self.bands) if self.band_gain is None else np.sqrt(self.band_gain)

    def _resampled(self, sample_rate: float, n: int) -> tuple[np.ndarray, int]:
        """The run at ``sample_rate`` (before + n, 4), still broadband, and ``before``:
        how many of its samples come before emission (those of the run that do)."""
        from acoustic_spaces.render import resample

        key = (float(sample_rate), int(n))
        if key not in self._cache:
            # Where the run starts, in samples from emission, rounded once, so within half
            # a sample of its time at any sample_rate (lead_s need not be a whole number
            # of samples at a rate other than the run's: 16 steps at 2000 Hz are 352.8
            # samples at 44.1 kHz).
            start = int(round((self.delay_s - self.lead_s) * sample_rate))
            before = max(0, -start)
            out = np.zeros((before + n, 4))
            # Impulse responses scale with the sample period: an impulse of height a at
            # one rate is a * rate / sample_rate at another.
            x = resample(self.signals, self.rate, sample_rate)[: max(0, n - start)]
            out[before + start : before + start + len(x)] = x * (self.rate / sample_rate)
            self._cache = {key: (out, before)}
        return self._cache[key]

    def field(self, sample_rate: float, n: int) -> np.ndarray:
        """The first-order sound field (n, 4) of the wave bands together, from emission:
        W, X, Y, Z. The bands' filters take the run before emission too, and what they
        spread to before emission is left out, as it is of the rays' sound."""
        run, before = self._resampled(sample_rate, n)
        n_fft = synthesis.fft_size(len(run), sample_rate)
        gains = wave_gains(n_fft, sample_rate, self.bands, self.total_bands)
        gain = (gains * self._amplitudes()[:, None]).sum(axis=0)
        spectrum = np.fft.rfft(run, n_fft, axis=0) * gain[:, None]
        return np.fft.irfft(spectrum, n_fft, axis=0)[before : before + n]

    def high_pass(self, signals: np.ndarray, sample_rate: float, late: bool = False) -> np.ndarray:
        """``signals`` (n, channels) without the frequencies the wave bands cover, which
        take them at a gain g: the rays' early paths at 1 - g, as they add to the wave
        bands' sound in pressure, or with ``late`` their noise at sqrt(1 - g^2), as it
        adds to the wave bands' tail in energy."""
        n_fft = synthesis.fft_size(len(signals), sample_rate)
        g = wave_gains(n_fft, sample_rate, self.bands, self.total_bands).sum(axis=0)
        return _filter(signals, np.sqrt(1.0 - g**2) if late else 1.0 - g, n_fft)

    def energy(
        self,
        time_step: float,
        slots: int,
        axis: np.ndarray | None = None,
        exponent: int = 2,
    ) -> np.ndarray:
        """Energy per band and time slot (bands, slots), as the ray tracer counts it: the
        energy of the pressure through each band's filter (those of the analysis of
        measured responses, :func:`acoustic_spaces.parameters.band_filter_gains`) over
        that filter's share of the spectrum. The filters spread each arrival a
        little both ways in time; what they spread to before emission counts in the
        first slot.

        With ``axis`` (a unit vector in room coordinates), that of a figure of eight
        facing it instead, from X, Y and Z: with ``exponent`` 1, the magnitude of its
        product with the pressure."""
        from acoustic_spaces.parameters import band_filter_gains

        out = np.zeros((self.bands, slots))
        pad = int(round(synthesis._PAD_S * self.rate))
        n_fft = synthesis.fft_size(len(self.signals) + pad, self.rate)
        gains = band_filter_gains(n_fft, self.rate, self.total_bands)[: self.bands]
        weights = np.full(gains.shape[1], 2.0)
        weights[0] = weights[-1] = 1.0
        # Each sample's time from emission (whole steps counted first, so that a sample
        # on a slot's edge falls in the same slot whatever the lead).
        lead = int(round(self.lead_s * self.rate))
        time = (np.arange(len(self.signals) + pad) - pad - lead) / self.rate + self.delay_s
        slot = np.maximum(time / time_step, 0).astype(np.int64)
        amplitude = self._amplitudes()
        keep = slot < slots
        padded = np.concatenate([np.zeros(pad), self.signals[:, 0]])
        if axis is not None:
            figure8 = np.concatenate([np.zeros(pad), self.signals[:, 1:] @ axis])
        for b, gain in enumerate(gains):
            pressure = _filter(padded, gain, n_fft)
            share = float((gain**2 * weights).sum() / n_fft)
            power = pressure[keep] ** 2
            if axis is not None:
                side = _filter(figure8, gain, n_fft)[keep]
                power = side**2 if exponent == 2 else np.abs(side * pressure[keep])
            energy = power * amplitude[b] ** 2 / (synthesis.ENERGY_TO_IR * share)
            out[b] = np.bincount(slot[keep], energy, minlength=slots)[:slots]
        return out


class _Weather:
    """The effective speed of sound over an outdoor scene's grid, from the source at
    ``source``: the still air's at each height above the ground (``ground``, its
    temperature ``ground_c`` °C), plus the wind's component along the horizontal from
    the source (none straight above it). Where the weather changes from place to place
    (``zones``, a :class:`~acoustic_spaces.engine.weather.Weather`), each cell takes
    the weather it stands over."""

    def __init__(self, atmosphere, ground_c: float, ground: float, source: np.ndarray, zones=None):
        self.atmosphere = atmosphere
        self.ground_c = ground_c
        self.ground = ground
        self.source = np.asarray(source, dtype=float)
        self.wind = atmosphere.to_wind()
        self.zones = zones
        if zones is not None:
            self.weathers = zones.atmosphere.weathers()
            self.grounds = zones.atmosphere.ground_temperatures(ground_c)
            self.winds = zones.towards  # (K, 2)

    @classmethod
    def of(cls, model, source: np.ndarray) -> _Weather | None:
        """The weather of ``model``'s scene, or None where the speed of sound is the
        same everywhere: indoors, or outdoors in still air."""
        from acoustic_spaces.engine.weather import Weather

        atmosphere = getattr(model.room.scene, "atmosphere", None)
        if model.room.outdoor is None or atmosphere is None or atmosphere.still:
            return None
        zones = Weather.of(model) if atmosphere.zones else None
        return cls(
            atmosphere, float(model.room.scene.air.temperature_c), model.ground, source, zones
        )

    def _profiles(self, z: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The still air's speed of sound and the wind's speed at heights z (m, in the
        room's coordinates), of the atmosphere's own weather or of weather ``k``."""
        height = np.asarray(z, dtype=float) - self.ground
        if k is None:
            still = np.asarray(self.atmosphere.sound_speed(height, self.ground_c), dtype=float)
            return still, np.asarray(self.atmosphere.wind_speed(height), dtype=float)
        weather = self.weathers[k]
        still = np.asarray(weather.sound_speed(height, self.grounds[k]), dtype=float)
        return still, np.asarray(weather.wind_speed(height), dtype=float)

    def _along(self, x: np.ndarray, y: np.ndarray, wind=None) -> np.ndarray:
        """The cosine between the wind (``wind`` (2,), or the atmosphere's own) and the
        horizontal from the source to (x, y)."""
        wind = self.wind if wind is None else wind
        dx, dy = x - self.source[0], y - self.source[1]
        distance = np.hypot(dx, dy)
        along = wind[0] * dx + wind[1] * dy
        return np.divide(along, distance, out=np.zeros_like(along), where=distance > 0)

    def at(self, points: np.ndarray) -> np.ndarray:
        """The speed of sound (m/s) at ``points`` (n, 3)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.zones is None:
            still, wind = self._profiles(points[:, 2])
            return still + wind * self._along(points[:, 0], points[:, 1])
        out = np.zeros(len(points))
        zone = self.zones.zone_at(points[:, :2])
        for k in np.unique(zone).tolist():
            rows = zone == k
            still, wind = self._profiles(points[rows, 2], k)
            along = self._along(points[rows, 0], points[rows, 1], self.winds[k])
            out[rows] = still + wind * along
        return out

    def _grid(self, origin: np.ndarray, h: float, shape) -> tuple:
        centres = [origin[a] + (np.arange(shape[a]) + 0.5) * h for a in range(3)]
        if self.zones is None:
            still, wind = self._profiles(centres[2])
            along = self._along(centres[0][None, :], centres[1][:, None])  # (ny, nx)
            return still, wind, along, None
        # Each weather's profiles and cosines, and which holds over each column.
        profiles = [self._profiles(centres[2], k) for k in range(len(self.weathers))]
        still = np.array([p[0] for p in profiles])  # (K, nz)
        wind = np.array([p[1] for p in profiles])  # (K, nz)
        along = np.array(
            [
                self._along(centres[0][None, :], centres[1][:, None], self.winds[k])
                for k in range(len(self.weathers))
            ]
        )  # (K, ny, nx)
        x, y = np.meshgrid(centres[0], centres[1])  # (ny, nx)
        zone = self.zones.zone_at(np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)
        return still, wind, along, zone

    def extremes(self, origin: np.ndarray, h: float, shape) -> tuple[float, float]:
        """The slowest and the fastest speed of sound over the grid's cells."""
        still, wind, along, zone = self._grid(origin, h, shape)
        if zone is None:
            ends = wind[:, None] * np.array([along.min(), along.max()])[None, :]
            return float((still + ends.min(axis=1)).min()), float((still + ends.max(axis=1)).max())
        low, high = np.inf, -np.inf
        for k in np.unique(zone).tolist():
            cosines = along[k][zone == k]
            ends = wind[k][:, None] * np.array([cosines.min(), cosines.max()])[None, :]
            low = min(low, float((still[k] + ends.min(axis=1)).min()))
            high = max(high, float((still[k] + ends.max(axis=1)).max()))
        return low, high

    def ratios(self, origin: np.ndarray, h: float, shape, speed: float) -> np.ndarray:
        """(nx ny nz,) float32, x fastest: each cell's speed of sound over ``speed``,
        the grid's (at most 1)."""
        still, wind, along, zone = self._grid(origin, h, shape)
        out = np.empty((int(shape[2]), int(shape[1]), int(shape[0])), np.float32)
        if zone is None:
            for z in range(len(still)):  # a layer at a time, to keep the memory down
                out[z] = np.minimum((still[z] + wind[z] * along) / speed, 1.0)
            return out.reshape(-1)
        iy, ix = np.indices(zone.shape)
        cosines = along[zone, iy, ix]  # (ny, nx) each column's cosine to its own wind
        for z in range(still.shape[1]):
            out[z] = np.minimum((still[zone, z] + wind[zone, z] * cosines) / speed, 1.0)
        return out.reshape(-1)

    # Moving air (``simulation.wave.wind: moving``): the still air's speed of sound in
    # each cell, and the wind as the air's own motion.

    def _still_grid(self, origin: np.ndarray, h: float, shape) -> tuple:
        """The still air's speed of sound (nz,) or (K, nz), the wind's speed the same
        way, each weather's wind direction (K, 2) and the weather over each column
        (ny, nx), or None and the atmosphere's own direction."""
        z = origin[2] + (np.arange(shape[2]) + 0.5) * h
        if self.zones is None:
            still, wind = self._profiles(z)
            return still, wind, self.wind[None, :], None
        profiles = [self._profiles(z, k) for k in range(len(self.weathers))]
        centres = [origin[a] + (np.arange(shape[a]) + 0.5) * h for a in range(2)]
        x, y = np.meshgrid(centres[0], centres[1])
        zone = self.zones.zone_at(np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)
        still = np.array([q[0] for q in profiles])
        wind = np.array([q[1] for q in profiles])
        return still, wind, np.asarray(self.winds, dtype=float), zone

    def still_at(self, point: np.ndarray) -> float:
        """The still air's speed of sound (m/s) at ``point`` (3,)."""
        point = np.asarray(point, dtype=float).reshape(3)
        k = None if self.zones is None else int(self.zones.zone_at(point[None, :2])[0])
        return float(self._profiles(point[2:3], k)[0][0])

    def moving_extremes(self, origin: np.ndarray, h: float, shape) -> tuple[float, float, float]:
        """The slowest and the fastest still air's speed of sound over the grid's
        cells, and the wind's fastest speed (m/s)."""
        still, wind, _, zone = self._still_grid(origin, h, shape)
        if zone is not None:
            used = np.unique(zone)
            still, wind = still[used], wind[used]
        return float(still.min()), float(still.max()), float(np.abs(wind).max())

    def moving_fields(self, origin: np.ndarray, h: float, shape, speed: float, shift: float):
        """(nx ny nz,) float32 each, x fastest: each cell's still air's speed of sound
        over ``speed`` (the grid's, at most 1), and the air's velocity along x and y in
        cells per step (the wind's speed times ``shift``, a step over a cell)."""
        still, wind, towards, zone = self._still_grid(origin, h, shape)
        nz, ny, nx = int(shape[2]), int(shape[1]), int(shape[0])
        speeds = np.empty((nz, ny, nx), np.float32)
        fx = np.empty((nz, ny, nx), np.float32)
        fy = np.empty((nz, ny, nx), np.float32)
        for z in range(nz):  # a layer at a time, to keep the memory down
            if zone is None:
                speeds[z] = min(still[z] / speed, 1.0)
                fx[z] = wind[z] * towards[0, 0] * shift
                fy[z] = wind[z] * towards[0, 1] * shift
            else:
                speeds[z] = np.minimum(still[zone, z] / speed, 1.0)
                fx[z] = wind[zone, z] * towards[zone, 0] * shift
                fy[z] = wind[zone, z] * towards[zone, 1] * shift
        return speeds.reshape(-1), fx.reshape(-1), fy.reshape(-1)


#: Points each listener takes the pressure at (see :func:`_probe_points`).
_PROBES = 13
#: A point round a listener that lies nearer the source than this many cells, and nearer
#: it than the listener itself, is left out of the listener's gradient (see
#: :func:`_off_the_source`), by the stencil's points: it reads the cells the source
#: itself feeds, whose sound is not a point's. Measured against a point source's exact
#: field over 18 places of the source within a cell and 50 directions, the grid's 26 and
#: others (``places`` of validation/wave_solver.py, with this at 0 and at 1.5; the worse
#: of the 63 and 125 Hz bands). On the 19-point grid, 1.5, 2, 2.5 and 3 cells from the
#: source, the gradient's main axis is up to 26, 21, 10 and 7 degrees off the line to
#: the source with those points and 16, 13, 10 and 7 without them, and its energy over
#: the pressure's up to 9.2, 7.6, 13.2 and 6.6 dB off with them and 6.3, 4.5, 3.7 and
#: 1.2 dB without. A listener nearer the source than 1.5 cells keeps the points that lie
#: no nearer it than itself: 1 and 1.4 cells away its axis is then up to 11.4 and 16.0
#: degrees off and its energy 7.30 and 6.54 dB (with every point 17.6 and 26.0 degrees,
#: 7.56 and 9.25 dB; without any point nearer than 1.5 cells 27.7 and 20.4 degrees), and
#: from one listener to the next 0.1 cells further out the energy changes by 2.4 dB at
#: most from 0.9 cells on (by 15.2 dB at 1.5 cells if a listener nearer than that kept
#: every point; one-off scans every 0.1 cells from 0.5 to 2, as ``places`` measures).
#: On the 7-point grid 1.5, 2, 2.5 and 3 cells give 21, 15, 5 and 4 degrees and 6.2,
#: 5.4, 3.4 and 1.7 dB with the points, 17, 17, 5 and 4 degrees and 5.4, 4.3, 3.5 and
#: 1.9 dB without: no clear gain, so it keeps them.
_SOURCE_CELLS = {7: 0.0, 19: 1.5}


def _off_the_source(
    found: np.ndarray, probes: np.ndarray, source: np.ndarray, cell: float, stencil: int
) -> np.ndarray:
    """``found`` (whether each of ``probes``, the :func:`_probe_points` of each listener
    in turn, counts) without the points among the source's own cells: those nearer the
    source than :data:`_SOURCE_CELLS` cells and nearer it than their listener (that many
    cells, or as far as the listener, to within rounding counts as not nearer). Of the
    two points a cell from a listener along an axis at most one lies nearer the source
    than the listener, so the listener loses at most that one. Where the other one is
    not found either (a wall behind the listener), the axis keeps its points."""
    limit = (_SOURCE_CELLS[stencil] - 1e-9) * cell
    away = np.linalg.norm(probes - source, axis=1).reshape(-1, _PROBES)
    near = (away < limit) & (away < away[:, :1] - 1e-9 * cell)
    before = np.asarray(found, dtype=bool).reshape(-1, _PROBES)
    after = before & ~near
    for axis in range(3):
        points = [1 + 2 * axis, 2 + 2 * axis, 7 + 2 * axis, 8 + 2 * axis]
        empty = ~(after[:, points[0]] | after[:, points[1]])
        after[np.ix_(empty, points)] = before[np.ix_(empty, points)]
    return after.reshape(-1)


def _probe_points(listener: np.ndarray, cell: float) -> np.ndarray:
    """The listener, the points one cell to either side of it along each axis (+x, -x,
    +y, -y, +z, -z), then those two cells to either side, in the same order:
    :data:`_PROBES` points."""
    steps = np.repeat(np.eye(3), 2, axis=0) * np.tile([1, -1], 3)[:, None]
    return listener[None, :] + cell * np.vstack([np.zeros(3), steps, 2 * steps])


def _spans_across(source: np.ndarray, listeners: np.ndarray) -> np.ndarray:
    """(3,) the farthest any of ``listeners`` lies from ``source`` across each axis: in
    the plane of the other two."""
    offsets = listeners - source
    across = np.sum(offsets**2, axis=1)[:, None] - offsets**2
    return np.sqrt(np.max(across, axis=0, initial=0.0))


def _open_sides(
    bvh,
    points: np.ndarray,
    room: np.ndarray,
    origin: np.ndarray,
    cell: float,
    shape,
    layer: int,
    threads: int = 0,
) -> np.ndarray:
    """(3, 2) for each axis's low and high side, whether sound that runs between
    ``points`` (the source, then the listeners) can meet the matched layer on that side
    of the grid (of ``shape`` cells of ``cell`` metres from ``origin``, ``layer`` of them
    matched layer inside each edge) near grazing: whether the layer there needs its
    ``room`` (3,) (see :data:`LAYER_ROOM`).

    It can where the air round the points reaches the layer inside the box that lies the
    room past them along every axis (and five cells, for the cells the points take: a
    listener's gradient points lie two cells from it, and each reads up to two further). The
    air is flooded from the source over the grid's cells in that box, as the solver
    floods its own, and from every listener it does not reach. Where it reaches no
    layer, a surface of the model lies between: sound from a point to that side's layer
    and back to a point leaves the box across one of its faces, so it runs at least as
    far as a reflection off a layer that lies that face's axis's room past the points,
    which a grid with room on every side has too. A side of the box that ends short of
    the layer counts as open (its room lies inside the grid).
    """
    from acoustic_spaces import _native

    shape = np.asarray(shape, dtype=np.int64)
    first, last = np.full(3, layer + 1), shape - 2 - layer  # the cells inside the layer
    low = np.floor((points.min(axis=0) - room - origin) / cell).astype(np.int64) - 5
    high = np.floor((points.max(axis=0) + room - origin) / cell).astype(np.int64) + 5
    start, stop = np.maximum(first, low), np.minimum(last, high)
    reached = np.zeros((3, 2), dtype=bool)
    left = list(range(len(points)))  # the points whose air is not flooded yet
    while left:
        seed, left = left[0], left[1:]
        try:
            # A grid of the box's cells and one more round them, where its air ends.
            part = _native.WaveSolver(
                bvh,
                np.zeros(bvh.face_count, np.int64),
                np.zeros(1),
                np.zeros((0, 3)),
                np.zeros(0, np.int64),
                origin + (start - 1) * cell,
                cell,
                *(int(n) for n in stop - start + 3),
                np.ascontiguousarray(points[seed]),
                np.ascontiguousarray(points[left].reshape(-1, 3)),
                1.0,
                0,
                0.0,
                threads,
            )
        except ValueError:
            continue  # a point in no air: the run's own grid says which, and stops
        reached |= np.array(part.open_sides, dtype=bool).reshape(3, 2)
        left = [p for k, p in enumerate(left) if not part.probe_found(k)]
    at_layer = np.column_stack([start == first, stop == last])
    return reached | ~at_layer


def _first_order(
    probes: np.ndarray,
    found: Sequence[bool],
    name: str,
    courant: float = COURANT,
    cutoff: float = _LOW_PASS_CUTOFF,
) -> np.ndarray:
    """W, X, Y, Z (samples, 4) from the pressure at the listener and round it, at the
    :func:`_probe_points` (``probes``, one row each).

    X, Y and Z are c times the time integral of the pressure gradient: X = -rho c u_x,
    against the particle velocity, so that it points towards the sound (as in AmbiX).
    The gradient is a central difference of fourth order over two cells either side,
    (8 (p(h) - p(-h)) - (p(2h) - p(-2h))) / 12h, of second order where a point two cells
    away is not ``found`` (out of the air, or across a surface from the listener), and
    one-sided where a point one cell away is not. The integral is the trapezoidal rule,
    so that it falls on the pressure's samples, with the rule's gain, (wT/2) / tan(wT/2),
    divided out up to ``cutoff`` (of the sample rate), the source's low-pass's cutoff:
    above it the source gives next to nothing. ``courant`` is the Courant number of the
    speed of sound at the listener.
    """
    if not found[0]:
        raise ValueError(f"listener {name!r} is not in the air of the wave grid")
    out = np.zeros((probes.shape[1], 4))
    out[:, 0] = probes[0]
    for axis in range(3):
        plus, minus = 1 + 2 * axis, 2 + 2 * axis
        if found[plus] and found[minus]:
            difference = 0.5 * (probes[plus] - probes[minus])
            if found[plus + 6] and found[minus + 6]:
                difference = difference * (4 / 3) - (probes[plus + 6] - probes[minus + 6]) / 12
        elif found[plus]:
            difference = probes[plus] - probes[0]
        elif found[minus]:
            difference = probes[0] - probes[minus]
        else:
            continue
        out[:, 1 + axis] = courant * (np.cumsum(difference) - 0.5 * difference)
    n_fft = 1 << int(math.ceil(math.log2(2 * len(out) + 1)))
    turn = np.pi * np.fft.rfftfreq(n_fft)  # wT / 2
    gain = np.ones(len(turn))
    below = (turn > 0) & (turn <= np.pi * cutoff)
    gain[below] = np.tan(turn[below]) / turn[below]
    spectrum = np.fft.rfft(out[:, 1:], n_fft, axis=0) * gain[:, None]
    out[:, 1:] = np.fft.irfft(spectrum, n_fft, axis=0)[: len(out)]
    return out


def wave_responses(
    model,
    source: np.ndarray,
    listeners: np.ndarray,
    names: Sequence[str],
    *,
    max_band_hz: int,
    length_s: float,
    sample_rate: float,
    speed_of_sound: float,
    points_per_wavelength: float = 8.0,
    threads: int = 0,
    device: Device = "cpu",
    scheme: Scheme = "auto",
    say: Callable[[str], None] | None = None,
    label: str = "",
    wind: WindModel = "effective",
) -> tuple[list[WaveResponse], dict[str, float]]:
    """The wave bands of the responses from ``source`` to each of ``listeners`` in the
    room of ``model`` (a :class:`~acoustic_spaces.engine.model.TraceModel`), and run
    details. ``device`` says where the grid is stepped (see :mod:`.gpu`); ``scheme``
    which stencil steps it: ``"isotropic"`` (19 points), ``"standard"`` (7 points) or
    ``"auto"``, the isotropic one but round an open model, where only the standard one
    has the matched layer. ``wind`` says how it takes the wind outdoors: ``"effective"``,
    as a speed of sound changed by the wind's component along the horizontal from the
    source, or ``"moving"``, as air that moves (see the module's notes)."""
    from acoustic_spaces import _native

    say = say or (lambda _: None)
    use_gpu(device)  # an unknown device, or the GPU missing, before any work
    if max_band_hz not in WAVE_BANDS_HZ:
        raise ValueError(f"max_band_hz must be one of {WAVE_BANDS_HZ}, got {max_band_hz}")
    # The scene's schema checks these; a caller of this function may pass anything.
    if scheme not in ("auto", "standard", "isotropic"):
        raise ValueError(f"unknown scheme {scheme!r}; use auto, standard or isotropic")
    if wind not in ("effective", "moving"):
        raise ValueError(f"unknown wind {wind!r}; use effective or moving")
    for what, value in (
        ("points_per_wavelength", points_per_wavelength),
        ("length_s", length_s),
        ("sample_rate", sample_rate),
        ("speed_of_sound", speed_of_sound),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{what} must be a positive number, not {value!r}")
    listeners = np.asarray(listeners, dtype=float).reshape(-1, 3)
    if len(names) != len(listeners):
        raise ValueError(f"{len(names)} names for {len(listeners)} listeners")
    mesh = model.room.mesh
    solid = ~model.transparent
    if not solid.any():
        raise ValueError(
            "the wave solver needs a surface that reflects: every face of the model lets "
            "all sound through (an opening)"
        )
    if model.transmission is not None:
        partial = np.unique(model.room.face_materials[solid & np.any(model.transmission > 0, 1)])
        if len(partial):
            materials = ", ".join(repr(model.room.material_names[m]) for m in partial)
            raise ValueError(
                f"the wave solver cannot pass sound through a surface; {materials} let "
                "some through (openings, with transmission 1, are fine): set "
                "simulation.wave.enabled to false or give them no transmission"
            )
    bvh = model.bvh
    if not solid.all():  # openings: holes in the grid's walls
        bvh = _native.Bvh(
            mesh.vertices, np.ascontiguousarray(mesh.faces[solid]), split=model.ray_tree
        )
    points = np.vstack([np.asarray(source, dtype=float).reshape(1, 3), listeners])
    centres = band_centres(model.bands)
    bands = (OCTAVE_BANDS_HZ.index(max_band_hz) + 1) * BANDS_PER_OCTAVE[model.bands]
    total = len(centres)
    # Each material's admittance, fitted to its absorption in the wave bands, or for a
    # material with an impedance (a ground's) to that impedance's admittance.
    coefficients = np.round(model.absorption[solid, :bands], 6)
    impedance = None
    if model.face_impedance is not None:
        keys = np.column_stack([coefficients, model.face_impedance[solid]])
        keys, face_material = np.unique(keys, axis=0, return_inverse=True)
        rows, impedance = keys[:, :bands], keys[:, bands].astype(np.int64)
    else:
        rows, face_material = np.unique(coefficients, axis=0, return_inverse=True)

    outdoor = model.room.outdoor is not None
    weather = _Weather.of(model, source)
    moving = False
    if weather is not None and wind == "moving":
        # The wind blows somewhere over the scene (temperature alone needs no moving air).
        heights = model.room.mesh.vertices[:, 2]
        top = max(float(heights.max()), float(points[:, 2].max())) + 100.0
        z = np.linspace(model.ground, top, 512)
        weathers = weather.weathers if weather.zones is not None else [weather.atmosphere]
        moving = any(np.any(np.asarray(w.wind_speed(z - model.ground)) != 0.0) for w in weathers)

    open_sides: list[np.ndarray] = []  # the sides whose layer needs room, once found

    def layout(courant: float, layer: int, speed: float, per_wavelength: float):
        """The grid for the stencil's ``courant``, with ``layer`` cells of matched
        layer, its time step set by ``speed``, ``per_wavelength`` cells per wavelength
        at that speed; where it starts, and its shape."""
        grid = band_grid(bands - 1, sample_rate, speed, per_wavelength, total, courant)
        h = grid.cell
        # The model, and two cells round the source and listeners (in an open model they
        # may lie outside it); then two cells of margin (one for an open model's air, one
        # that is never air) and in an open model the matched layer, shifted off the
        # model's planes.
        low = np.minimum(mesh.vertices.min(axis=0), points.min(axis=0) - 2.0 * h)
        high = np.maximum(mesh.vertices.max(axis=0), points.max(axis=0) + 2.0 * h)
        offset = h * (2.0 + layer + np.array([0.0137, 0.0219, 0.0291]))
        if layer:
            # Room for sound that runs along the layer (see LAYER_ROOM), below in whole
            # cells so that the grid keeps its place on the model; none under an outdoor
            # scene, whose ground closes the grid there, nor on a side the model's own
            # surfaces close (found once, on the first grid that would grow).
            room = LAYER_ROOM * _spans_across(points[0], points[1:])
            below = np.ceil(np.maximum(low - (points.min(axis=0) - room), 0.0) / h)
            if outdoor:
                below[2] = 0.0
            above = np.maximum(high, points.max(axis=0) + room)
            if not open_sides and (below.any() or np.any(above > high)):
                bare = np.ceil((high - low + offset) / h).astype(np.int64) + 2 + layer
                if int(np.prod(bare)) > MAX_CELLS:  # refused below, with or without room
                    open_sides.append(np.ones((3, 2), dtype=bool))
                else:
                    open_sides.append(
                        _open_sides(bvh, points, room, low - offset, h, bare, layer, threads)
                    )
            if open_sides:
                low = low - h * np.where(open_sides[0][:, 0], below, 0.0)
                high = np.where(open_sides[0][:, 1], above, high)
        origin = low - offset
        shape = np.ceil((high - origin) / h).astype(np.int64) + 2 + layer
        return grid, origin, shape

    def check_size(shape, layer: int) -> None:
        """Refuses a grid of more than MAX_CELLS cells, before the weather's fields or
        the solver, each the size of the grid, are made for it."""
        if int(np.prod(shape)) > MAX_CELLS:
            what = " with the matched layer round this open model" if layer else ""
            raise ValueError(
                f"the wave grid up to the {max_band_hz} Hz band would have "
                f"{int(np.prod(shape)):,} cells{what} (at most {MAX_CELLS:,}); lower "
                "max_band_hz or points_per_wavelength"
            )

    def prepare(stencil: str, layer: int):
        """The solver, grid, fits and probes for the stencil, with ``layer`` cells of
        matched layer."""
        size, courant = STENCILS[stencil]
        speeds, extremes, flow = None, None, (None, None)
        still = speed_of_sound  # the slowest still air's speed of sound on the grid
        if weather is None:
            grid, origin, shape = layout(courant, layer, speed_of_sound, points_per_wavelength)
            check_size(shape, layer)
        elif moving:
            # Fine enough for sound going against the fastest wind in the slowest air, in
            # time steps for the fastest still air (the air's motion is carried apart).
            slowest = fastest = float(weather.at(source)[0])
            for _ in range(8):
                grid, origin, shape = layout(
                    courant, layer, fastest, points_per_wavelength * fastest / slowest
                )
                low, high, top = weather.moving_extremes(origin, grid.cell, shape)
                if low - top >= slowest and high <= fastest:
                    break
                slowest, fastest = min(slowest, low - top), max(fastest, high)
            check_size(shape, layer)
            step = courant * grid.cell * grid.rate
            speeds, *flow = weather.moving_fields(
                origin, grid.cell, shape, step, 1.0 / (grid.cell * grid.rate)
            )
            extremes, still = (low - top, high + top), low
        else:
            # Fine enough for the slowest sound and in time steps for the fastest, over
            # the grid those give (which grows a little as its cells shrink).
            slowest = fastest = float(weather.at(source)[0])
            for _ in range(8):
                grid, origin, shape = layout(
                    courant, layer, fastest, points_per_wavelength * fastest / slowest
                )
                low_speed, high_speed = weather.extremes(origin, grid.cell, shape)
                if low_speed >= slowest and high_speed <= fastest:
                    break
                slowest, fastest = min(slowest, low_speed), max(fastest, high_speed)
            check_size(shape, layer)
            speeds = weather.ratios(origin, grid.cell, shape, courant * grid.cell * grid.rate)
            extremes, still = (low_speed, high_speed), low_speed
        h = grid.cell
        fits = [fit_admittance(row, total, grid.rate) for row in rows]
        if impedance is not None:
            for m in np.flatnonzero(impedance >= 0).tolist():
                fits[m] = fit_impedance(
                    _impedance_of(model, int(impedance[m])), bands, total, grid.rate
                )
        branches = [fit.branches(grid.rate) for fit in fits]
        branch_material = np.concatenate(
            [np.full(len(b), m, dtype=np.int64) for m, b in enumerate(branches)]
        )
        probes = np.vstack([_probe_points(p, h) for p in listeners])
        solver = _native.WaveSolver(
            bvh,
            np.ascontiguousarray(face_material.reshape(-1), dtype=np.int64),
            np.array([fit.beta for fit in fits]),
            np.ascontiguousarray(np.vstack(branches)),
            branch_material,
            origin,
            h,
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
            np.asarray(source, dtype=float),
            np.ascontiguousarray(probes),
            1.0,
            layer,
            LAYER_DAMPING,
            threads,
            size,
            speeds,
            *flow,
            _PROBES,
        )
        return solver, grid, fits, probes, courant, extremes, still

    def kept(stencil: str, layer: int):
        """What ``prepare`` gives, where its grid is closed without a matched layer or
        open with one; otherwise None, and its grid goes before the next is made."""
        built = prepare(stencil, layer)
        return built if built[0].open == bool(layer) else None

    # The isotropic stencil, but round an open model the standard one with its matched
    # layer, which the isotropic one has none of. An outdoor model's sky is an opening,
    # so its grid is open unless the source is shut in: the one with the layer first,
    # and the isotropic stencil refused before any grid is made.
    if scheme == "isotropic" and outdoor:
        raise ValueError(
            "outdoors the isotropic stencil is not available (it has no matched layer for "
            "the sky): set simulation.wave.scheme to auto or standard"
        )
    built = kept("standard", LAYER_CELLS) if outdoor else None
    if built is None:
        built = kept("standard" if scheme == "standard" else "isotropic", 0)
    if built is None:
        if scheme == "isotropic":
            raise ValueError(
                "the isotropic stencil has no matched layer for an open model: set "
                "simulation.wave.scheme to auto or standard"
            )
        built = prepare("standard", LAYER_CELLS)
    solver, grid, fits, probes, courant, extremes, still = built
    h = grid.cell
    at_source = 1.0  # the speed of sound at the source over the grid's
    if weather is not None:
        here = weather.still_at(source) if moving else weather.at(source)[0]
        at_source = min(float(here) / (courant * h * grid.rate), 1.0)
    signal, delay = source_signal(grid.rate, h, courant, at_source)
    samples = int(math.ceil(length_s * grid.rate))
    # The run starts `lead` steps before emission, where the source's pulse starts (or a
    # little before, so that the lead is a whole number of the output's samples), and
    # keeps all of it: a listener within `delay` steps' travel of the source hears it
    # start before emission.
    whole = max(1, round(grid.rate / sample_rate))  # steps per output sample, at least 1
    lead = -(-delay // whole) * whole
    steps = lead + samples  # the source's term on step n stands for its value at step n
    drive = np.zeros(steps, np.float32)
    drive[lead - delay : lead - delay + len(signal)] = signal[: steps - lead + delay]
    # Before any step: a listener outside the source's air, or a source closed in. A
    # point round a listener counts only where the listener sees it, from the side of a
    # surface each is read on: a point across a thin surface (a screen, a balcony's front)
    # can have air of its own, and a difference across the surface is no gradient of the
    # listener's air.
    found = np.array(
        [solver.probe_found(p) and solver.probe_in_sight(p) for p in range(len(probes))]
    )
    found = _off_the_source(found, probes, np.asarray(source, dtype=float), h, solver.stencil)
    found = found.tolist()
    missing = [listener for i, listener in enumerate(names) if not found[_PROBES * i]]
    if missing:
        # A source on a surface sends sound into the space on one side of it, which the
        # listeners it reaches choose where they can (see the solver's place_on_face).
        note = (
            " (the source lies on a surface of the model, and sends sound into the space on"
            " one side of it only: move it off the surface into the air)"
            if solver.source_on_face
            else ""
        )
        air, volume = solver.air_cells * h**3, model.room.volume_m3
        if volume and air < 0.1 * volume:
            who = f"source {label!r}" if label else "the source"
            raise ValueError(
                f"{who} is closed in: the wave grid has {air:.2g} m³ of air round it, "
                f"in a room of {volume:.3g} m³{note}"
            )
        raise ValueError(f"listener {missing[0]!r} is not in the air of the wave grid{note}")
    name = f"{centres[0]} Hz" if bands == 1 else f"{centres[0]}-{centres[bands - 1]} Hz"
    # The GPU steps still air only: moving air stays on the CPU.
    gpu_wanted = use_gpu(device, steps)
    on_gpu = gpu_wanted and not solver.flowing
    where = f" on {gpu_name()}" if on_gpu else ""
    if extremes is not None:
        where += f", sound at {extremes[0]:.0f} to {extremes[1]:.0f} m/s"
    if solver.flowing:
        where += ", the air moving" + (
            " (on the CPU: the GPU steps still air)" if gpu_wanted else ""
        )
    say(
        f"{label}: wave solver, {name} ({solver.air_cells:,} cells of {100 * h:.0f} cm, "
        f"{steps:,} steps{where})"
    )
    stepper = GpuWaveSolver(solver) if on_gpu else solver
    # Long runs report every tenth (which is also where a cancelled run stops).
    chunks = 10 if solver.air_cells * steps > _LONG_RUN else 1
    if chunks == 1:
        recorded = stepper.run(drive)
    else:
        # Into one array as each tenth comes, not all ten joined at the end (twice the
        # memory).
        recorded = np.empty((len(probes), steps))
        bounds = np.linspace(0, steps, chunks + 1).astype(np.int64)
        for c in range(chunks):
            if c:
                say(f"{label}: wave solver, {name}, {100 * c // chunks} %")
            recorded[:, bounds[c] : bounds[c + 1]] = stepper.run(drive[bounds[c] : bounds[c + 1]])
    info: dict[str, float] = {
        "wave_cells": float(solver.air_cells),
        "wave_cell_m": h,
        "wave_steps": float(steps),
        "wave_stencil": float(solver.stencil),
    }
    if solver.open:
        info["wave_open"] = 1.0
    if on_gpu:
        info["wave_gpu"] = 1.0
    if extremes is not None:
        info["wave_speed_min_m_s"], info["wave_speed_max_m_s"] = extremes
    if solver.flowing:
        info["wave_moving_air"] = 1.0
    # How much slower than it should sound travels at the top of the highest band along
    # the grid's axes: the slowest sound's cells per wavelength there, stepped as the
    # slowest air is (in moving air, the still air's step in the air's own frame;
    # carrying it with the wind adds an error of its own).
    slowest = speed_of_sound if extremes is None else extremes[0]
    per_wavelength = slowest / (band_top(bands - 1, total) * h)
    ratio = still / (courant * h * grid.rate)
    info["wave_slow_percent"] = 100.0 * slow_by(solver.stencil, per_wavelength, ratio)
    # How far the fitted surfaces' absorption is from the materials', at worst.
    used = np.unique(face_material)
    wanted = np.minimum(rows[used], MAX_ABSORPTION)
    info["wave_absorption_miss"] = float(
        np.max(np.abs(np.array([fits[m].achieved for m in used]) - wanted))
    )
    responses = []
    for i, listener in enumerate(names):
        part = slice(_PROBES * i, _PROBES * (i + 1))
        local = courant  # the Courant number at the listener
        if weather is not None:
            # In moving air the still air's (the wind's share of the gradient is left out).
            here = weather.still_at(listeners[i]) if moving else weather.at(listeners[i])[0]
            local *= float(here) / (courant * h * grid.rate)
        # X, Y and Z from the first step on, where the grid is still at rest.
        signals = _first_order(recorded[part], found[part], listener, local, _cutoff(courant))
        responses.append(WaveResponse(grid.rate, signals, bands, h, total, lead / grid.rate))
    return responses, info


def _impedance_of(model, row: int):
    """The admittance at frequencies f (Hz) of the material with the impedance in row
    ``row`` of the model's (see
    :attr:`~acoustic_spaces.engine.model.TraceModel.face_impedance`)."""
    room = model.room
    face = int(np.flatnonzero(model.face_impedance == row)[0])
    material = room.scene.materials[room.material_names[room.face_materials[face]]]
    return lambda f: material.impedance.admittance(f, room.speed_of_sound())


__all__ = [
    "COURANT",
    "ISOTROPIC_COURANT",
    "HIGH_PASS_S",
    "LAYER_CELLS",
    "LAYER_DAMPING",
    "LOW_EDGE_HZ",
    "MAX_ABSORPTION",
    "MAX_CELLS",
    "LINK_WEIGHTS",
    "STENCILS",
    "WAVE_BANDS_HZ",
    "Scheme",
    "WindModel",
    "Admittance",
    "BandGrid",
    "WaveResponse",
    "absorption_of_impedance",
    "admittance",
    "fit_admittance",
    "band_grid",
    "band_top",
    "grid_phase",
    "slow_by",
    "source_signal",
    "wave_gains",
    "wave_responses",
]
