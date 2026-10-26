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
  layer.
- The surfaces absorb as their materials do in each band: each material becomes a
  locally reacting surface whose admittance changes with frequency, fitted so that its
  random-incidence absorption matches the material's in every wave band
  (:mod:`acoustic_spaces.engine.impedance`). One run covers all the wave bands.
- Cells are air when the source reaches them without crossing a surface, so closed
  rooms, open models and thin screens all work. Round an open model, whose air reaches
  the edge of the grid, the grid grows by a perfectly matched layer of
  :data:`LAYER_CELLS` cells that takes in the sound leaving it. Openings are not
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
#: Grids with more cells than this are refused (about 20 bytes each).
MAX_CELLS = 40_000_000
#: The highest bands the solver may compute (``simulation.wave.max_band_hz``).
WAVE_BANDS_HZ = OCTAVE_BANDS_HZ[:4]
#: The source's high-pass, applied twice: once so that the air the source pushes into
#: the room stops flowing, twice so that as much flows out again and a closed room's
#: pressure comes back to rest (but for a slow swing from rounding L^2 down, far below
#: the wave bands). Its length in seconds; twice, it is flat within 0.2 dB above 20 Hz.
HIGH_PASS_S = 0.12
#: Cells of the matched layer round an open model, and its damping at the grid's edge
#: (the damping rate times half a time step). Sound leaving the model comes back from
#: it some 60 dB down; from the grid's edge alone, 7 to 14 dB down.
LAYER_CELLS = 8
LAYER_DAMPING = 0.5
#: The wave solver's part of a response starts here (Hz), with the synthesis's
#: crossover shape round it: an octave below the 63 Hz band's lower edge, so that band,
#: skirts and all, comes from the wave solver. Further down a closed room's air is
#: squeezed as a whole (the wave solver's source leaves no air in it, so this falls
#: away), and the rays' response stays.
LOW_EDGE_HZ = OCTAVE_BANDS_HZ[0] / (2.0 * math.sqrt(2.0))
#: Length of the source's low-pass (taps; odd), which keeps the grid free of the
#: frequencies it cannot carry.
_LOW_PASS_TAPS = 33
_LOW_PASS_CUTOFF = 0.2  # of the grid's sample rate
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


def source_signal(rate: float, cell: float, courant: float = COURANT) -> tuple[np.ndarray, int]:
    """What the source adds to the grid on each step, and its delay in steps.

    An impulse, low-passed (so the grid carries no frequencies it cannot) and
    high-passed twice (so it leaves no air in the room), scaled so that in free field the
    pressure at r metres is that impulse over r.
    """
    half = _LOW_PASS_TAPS // 2
    k = np.arange(-half, half + 1)
    low = 2 * _LOW_PASS_CUTOFF * np.sinc(2 * _LOW_PASS_CUTOFF * k) * np.blackman(_LOW_PASS_TAPS)
    low /= low.sum()
    length = max(3, int(round(HIGH_PASS_S * rate)))
    window = np.hanning(length + 2)[1:-1]
    high = -window / window.sum()
    high[0] += 1.0
    signal = np.convolve(np.convolve(low, high), high)
    return 4.0 * math.pi * courant**2 / cell * signal, half


@dataclass
class WaveResponse:
    """The lowest octave bands (or third-octave bands) of one response, from the wave
    solver.

    ``signals`` holds the run at ``rate``: the pressure and the first-order directional
    parts (samples, 4), AmbiX W, X, Y and Z in room coordinates, from emission on. Only
    the frequencies of the wave bands count.
    """

    rate: float
    signals: np.ndarray
    #: The bands 0 to bands - 1 come from the wave solver.
    bands: int
    #: Cell size of the grid (m).
    cell: float
    #: How many bands the whole response has: 8 octave or 24 third-octave bands.
    total_bands: int = len(OCTAVE_BANDS_HZ)
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

    def _resampled(self, sample_rate: float, n: int) -> np.ndarray:
        """The run at ``sample_rate``, n samples long (n, 4), from emission; still
        broadband."""
        from acoustic_spaces.render import resample

        key = (float(sample_rate), int(n))
        if key not in self._cache:
            out = np.zeros((n, 4))
            start = min(int(round(self.delay_s * sample_rate)), n)
            # Impulse responses scale with the sample period: an impulse of height a at
            # one rate is a * rate / sample_rate at another.
            x = resample(self.signals, self.rate, sample_rate)[: n - start]
            out[start : start + len(x)] = x * (self.rate / sample_rate)
            self._cache = {key: out}
        return self._cache[key]

    def field(self, sample_rate: float, n: int) -> np.ndarray:
        """The first-order sound field (n, 4) of the wave bands together: W, X, Y, Z."""
        run = self._resampled(sample_rate, n)
        n_fft = synthesis.fft_size(n, sample_rate)
        gains = wave_gains(n_fft, sample_rate, self.bands, self.total_bands)
        gain = (gains * self._amplitudes()[:, None]).sum(axis=0)
        spectrum = np.fft.rfft(run, n_fft, axis=0) * gain[:, None]
        return np.fft.irfft(spectrum, n_fft, axis=0)[:n]

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
        lead = int(round(synthesis._PAD_S * self.rate))
        n_fft = synthesis.fft_size(len(self.signals) + lead, self.rate)
        gains = band_filter_gains(n_fft, self.rate, self.total_bands)[: self.bands]
        weights = np.full(gains.shape[1], 2.0)
        weights[0] = weights[-1] = 1.0
        time = (np.arange(len(self.signals) + lead) - lead) / self.rate + self.delay_s
        slot = np.maximum(time / time_step, 0).astype(np.int64)
        amplitude = self._amplitudes()
        keep = slot < slots
        padded = np.concatenate([np.zeros(lead), self.signals[:, 0]])
        if axis is not None:
            figure8 = np.concatenate([np.zeros(lead), self.signals[:, 1:] @ axis])
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


def _probe_points(listener: np.ndarray, cell: float) -> np.ndarray:
    """The listener and a point one cell to either side along each axis."""
    offsets = np.vstack(
        [np.zeros(3), np.repeat(np.eye(3), 2, axis=0) * np.tile([1, -1], 3)[:, None]]
    )
    return listener[None, :] + cell * offsets


def _first_order(
    probes: np.ndarray, found: Sequence[bool], name: str, courant: float = COURANT
) -> np.ndarray:
    """W, X, Y, Z (samples, 4) from the pressure at the listener and round it.

    X, Y and Z are c times the time integral of the pressure gradient (central
    differences, or one-sided ones next to a surface), by the trapezoidal rule so that
    they fall on the pressure's samples: X = -rho c u_x, against the particle velocity,
    so that it points towards the sound (as in AmbiX). ``courant`` is the Courant number
    of the speed of sound at the listener.
    """
    if not found[0]:
        raise ValueError(f"listener {name!r} is not in the air of the wave grid")
    out = np.zeros((probes.shape[1], 4))
    out[:, 0] = probes[0]
    for axis in range(3):
        plus, minus = 1 + 2 * axis, 2 + 2 * axis
        if found[plus] and found[minus]:
            difference = 0.5 * (probes[plus] - probes[minus])
        elif found[plus]:
            difference = probes[plus] - probes[0]
        elif found[minus]:
            difference = probes[0] - probes[minus]
        else:
            continue
        out[:, 1 + axis] = courant * (np.cumsum(difference) - 0.5 * difference)
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
    mesh = model.room.mesh
    solid = ~model.transparent
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
    listeners = np.asarray(listeners, dtype=float).reshape(-1, 3)
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

    weather = _Weather.of(model, source)
    moving = False
    if weather is not None and wind == "moving":
        # The wind blows somewhere over the scene (temperature alone needs no moving air).
        heights = model.room.mesh.vertices[:, 2]
        top = max(float(heights.max()), float(points[:, 2].max())) + 100.0
        z = np.linspace(model.ground, top, 512)
        weathers = weather.weathers if weather.zones is not None else [weather.atmosphere]
        moving = any(np.any(np.asarray(w.wind_speed(z - model.ground)) != 0.0) for w in weathers)

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
        origin = low - h * (2.0 + layer + np.array([0.0137, 0.0219, 0.0291]))
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
            extremes = (low - top, high + top)
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
            extremes = (low_speed, high_speed)
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
        )
        return solver, grid, fits, probes, courant, extremes

    # The isotropic stencil, but round an open model the standard one with its matched
    # layer, which the isotropic one has none of.
    stencil = "standard" if scheme == "standard" else "isotropic"
    solver, grid, fits, probes, courant, extremes = prepare(stencil, 0)
    if solver.open:
        if scheme == "isotropic":
            raise ValueError(
                "the isotropic stencil has no matched layer for an open model: set "
                "simulation.wave.scheme to auto or standard"
            )
        stencil = "standard"
        solver, grid, fits, probes, courant, extremes = prepare(stencil, LAYER_CELLS)
    h = grid.cell
    signal, delay = source_signal(grid.rate, h, courant)
    samples = int(math.ceil(length_s * grid.rate))
    steps = samples + delay  # the source's term on step n stands for its value at time n
    drive = np.zeros(steps, np.float32)
    drive[: min(len(signal), steps)] = signal[:steps]
    # Before any step: a listener outside the source's air, or a source closed in.
    found = [solver.probe_found(p) for p in range(len(probes))]
    missing = [listener for i, listener in enumerate(names) if not found[7 * i]]
    if missing:
        air, volume = solver.air_cells * h**3, model.room.volume_m3
        if volume and air < 0.1 * volume:
            who = f"source {label!r}" if label else "the source"
            raise ValueError(
                f"{who} is closed in: the wave grid has {air:.2g} m³ of air round it, "
                f"in a room of {volume:.3g} m³"
            )
        raise ValueError(f"listener {missing[0]!r} is not in the air of the wave grid")
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
    recorded = []
    bounds = np.linspace(0, steps, chunks + 1).astype(np.int64)
    for c in range(chunks):
        if c:
            say(f"{label}: wave solver, {name}, {100 * c // chunks} %")
        recorded.append(stepper.run(drive[bounds[c] : bounds[c + 1]]))
    run = np.concatenate(recorded, axis=1)[:, delay:][:, :samples]
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
    # How far the fitted surfaces' absorption is from the materials', at worst.
    used = np.unique(face_material)
    wanted = np.minimum(rows[used], MAX_ABSORPTION)
    info["wave_absorption_miss"] = float(
        np.max(np.abs(np.array([fits[m].achieved for m in used]) - wanted))
    )
    responses = []
    for i, listener in enumerate(names):
        part = slice(7 * i, 7 * i + 7)
        local = courant  # the Courant number at the listener
        if weather is not None:
            # In moving air the still air's (the wind's share of the gradient is left out).
            here = weather.still_at(listeners[i]) if moving else weather.at(listeners[i])[0]
            local *= float(here) / (courant * h * grid.rate)
        signals = _first_order(run[part], found[part], listener, local)
        responses.append(WaveResponse(grid.rate, signals, bands, h, total))
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
    "source_signal",
    "wave_gains",
    "wave_responses",
]
