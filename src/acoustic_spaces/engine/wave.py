"""Wave-based simulation of the lowest octave bands.

Rays and image sources treat sound as rays, which holds while the room is large next
to the wavelength. At low frequencies it is not: small rooms ring at their modes, and
sound bends round everything. The wave solver computes the sound field itself, with
the finite-difference time-domain method (FDTD) on a cubic grid over the room, and its
octave bands replace the ray-traced ones.

- One grid covers every octave band up to ``max_band_hz``, with
  ``points_per_wavelength`` cells per wavelength at the top of that band's crossover.
  Each octave higher costs about 16 times as much (cells 8 times smaller, steps twice
  as many). All bands share the grid, so a mode near a crossover is the same mode on
  both sides of it.
- The surfaces absorb as their materials do in each band: each absorption coefficient
  becomes the locally reacting admittance with that random-incidence absorption
  (Paris's formula). Bands whose coefficients differ get runs of their own on the same
  grid; bands with the same coefficients (63 and 125 Hz in most material tables) share
  one.
- Cells are air when the source reaches them without crossing a surface, so closed
  rooms, open models and thin screens all work; where an open model's air reaches the
  edge of the grid, the edge absorbs.
- At these frequencies the source radiates the same way in all directions. The
  listener records the pressure and, from its gradient, the direction the sound comes
  from (first-order Ambisonics), so spatial outputs keep the low end's direction.
- The bands join the ray-traced ones through the zero-phase crossovers the impulse
  response synthesis uses, which sum to one; the lowest starts at :data:`LOW_EDGE_HZ`
  (22 Hz) instead of 0 Hz. Their energies, for the room parameters, are counted
  through the octave filters of the analysis of measured responses.

The levels follow the ray tracer's convention: in free field, the direct sound at r
metres is an impulse of height 1 / r.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces import synthesis
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ

#: Courant number of the scheme: the stable limit of the cubic grid.
COURANT = 1.0 / math.sqrt(3.0)
#: Grids with more cells than this are refused (about 20 bytes each).
MAX_CELLS = 40_000_000
#: The source's high-pass: it adds no air to the room, so nothing builds up at 0 Hz.
#: Its length in seconds; it is flat within 0.3 dB above 20 Hz.
HIGH_PASS_S = 0.1
#: Highest random-incidence absorption a locally reacting surface with a real
#: admittance reaches (Paris's formula), and its normalised impedance.
MAX_ABSORPTION = 0.951
_BEST_IMPEDANCE = 1.5672
#: The wave solver's part of a response starts here (Hz), with the synthesis's
#: crossover shape round it: an octave below the 63 Hz band's lower edge, so that band,
#: skirts and all, comes from the wave solver. Further down a closed room's air is
#: squeezed as a whole (the wave solver's source adds no air, so this falls away), and
#: the rays' response stays.
LOW_EDGE_HZ = OCTAVE_BANDS_HZ[0] / (2.0 * math.sqrt(2.0))
#: Length of the source's low-pass (taps; odd), which keeps the grid free of the
#: frequencies it cannot carry.
_LOW_PASS_TAPS = 33
_LOW_PASS_CUTOFF = 0.2  # of the grid's sample rate
#: Runs of more cell updates than this (a few seconds' work) report their progress.
_LONG_RUN = 2e9


def absorption_of_impedance(impedance: np.ndarray) -> np.ndarray:
    """Random-incidence absorption of a locally reacting surface with a real normalised
    impedance (Paris's formula, for a diffuse field)."""
    z = np.asarray(impedance, dtype=float)
    return 8.0 / z * (1.0 + 1.0 / (1.0 + z) - 2.0 / z * np.log1p(z))


def admittance(absorption: np.ndarray) -> np.ndarray:
    """Specific (normalised) admittance of a locally reacting surface whose
    random-incidence absorption is ``absorption``: the inverse of Paris's formula, on
    the side of hard surfaces. Coefficients from :data:`MAX_ABSORPTION` up get the most
    absorbing admittance."""
    alpha = np.clip(np.asarray(absorption, dtype=float), 0.0, 1.0)
    low = np.full(alpha.shape, math.log(_BEST_IMPEDANCE))
    high = np.full(alpha.shape, math.log(1e9))
    for _ in range(60):  # bisection in log(impedance); absorption falls as it grows
        mid = 0.5 * (low + high)
        above = absorption_of_impedance(np.exp(mid)) > alpha
        low = np.where(above, mid, low)
        high = np.where(above, high, mid)
    beta = np.exp(-0.5 * (low + high))
    return np.where(alpha <= 0.0, 0.0, beta)


def band_top(band: int) -> float:
    """Highest frequency band ``band`` has (Hz): the top of its upper crossover."""
    edge = OCTAVE_BANDS_HZ[band] * math.sqrt(2.0)
    return edge * 2.0 ** (synthesis.CROSSOVER_WIDTH / 2)


def wave_gains(n_fft: int, sample_rate: float, bands: int) -> np.ndarray:
    """Zero-phase gains (bands, n_fft // 2 + 1) of the lowest ``bands`` octave bands as
    the wave solver covers them: the synthesis's band gains, the lowest one starting at
    :data:`LOW_EDGE_HZ`."""
    gains = synthesis.band_gains(n_fft, sample_rate)[:bands]
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
    #: Sample rate of the run (Hz): the output rate over a whole number.
    rate: float


def band_grid(
    band: int, sample_rate: float, speed_of_sound: float, points_per_wavelength: float = 8.0
) -> BandGrid:
    """The coarsest grid with ``points_per_wavelength`` cells per wavelength at the top of
    band ``band``, whose time step divides the output's sample period."""
    cell = speed_of_sound / (points_per_wavelength * band_top(band))
    rate = speed_of_sound / (COURANT * cell)  # one step crosses COURANT cells
    ratio = max(1, math.floor(sample_rate / rate))
    rate = sample_rate / ratio
    return BandGrid(band, speed_of_sound / (COURANT * rate), rate)


def source_signal(rate: float, cell: float) -> tuple[np.ndarray, int]:
    """What the source adds to the grid on each step, and its delay in steps.

    An impulse, low-passed (so the grid carries no frequencies it cannot) and
    high-passed (so it puts no air in the room), scaled so that in free field the
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
    signal = np.convolve(low, high)
    return 4.0 * math.pi * COURANT**2 / cell * signal, half


@dataclass
class WaveResponse:
    """The lowest octave bands of one response, from the wave solver.

    ``signals[k]`` holds run k at ``rate``: the pressure and the first-order directional
    parts (samples, 4), AmbiX W, X, Y and Z in room coordinates, from emission on.
    Octave band b comes from run ``band_runs[b]``; only the frequencies of its bands
    count.
    """

    rate: float
    signals: np.ndarray
    band_runs: tuple[int, ...]
    #: Cell size of the grid (m).
    cell: float
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def bands(self) -> int:
        """The octave bands 0 to bands - 1 come from the wave solver."""
        return len(self.band_runs)

    def _resampled(self, sample_rate: float, n: int) -> np.ndarray:
        """The runs at ``sample_rate``, n samples long (runs, n, 4); still broadband."""
        from acoustic_spaces.render import resample

        key = (float(sample_rate), int(n))
        if key not in self._cache:
            out = np.zeros((len(self.signals), n, 4))
            for k, run in enumerate(self.signals):
                # Impulse responses scale with the sample period: an impulse of height a
                # at one rate is a * rate / sample_rate at another.
                x = resample(run, self.rate, sample_rate)[:n] * (self.rate / sample_rate)
                out[k, : len(x)] = x
            self._cache = {key: out}
        return self._cache[key]

    def field(self, sample_rate: float, n: int) -> np.ndarray:
        """The first-order sound field (n, 4) of the wave bands together: W, X, Y, Z."""
        runs = self._resampled(sample_rate, n)
        n_fft = synthesis.fft_size(n, sample_rate)
        gains = wave_gains(n_fft, sample_rate, self.bands)
        spectrum = 0.0
        for k, run in enumerate(runs):
            gain = sum(gains[b] for b, r in enumerate(self.band_runs) if r == k)
            spectrum = spectrum + np.fft.rfft(run, n_fft, axis=0) * gain[:, None]
        return np.fft.irfft(spectrum, n_fft, axis=0)[:n]

    def high_pass(self, signals: np.ndarray, sample_rate: float) -> np.ndarray:
        """``signals`` (n, channels) without the frequencies the wave bands cover."""
        n_fft = synthesis.fft_size(len(signals), sample_rate)
        return _filter(signals, 1.0 - wave_gains(n_fft, sample_rate, self.bands).sum(axis=0), n_fft)

    def energy(self, time_step: float, slots: int) -> np.ndarray:
        """Energy per band and time slot (bands, slots), as the ray tracer counts it: the
        energy of the pressure through each octave band's filter (those of the analysis
        of measured responses, :func:`acoustic_spaces.parameters.octave_filter_gains`)
        over that filter's share of the spectrum. The filters spread each arrival a
        little both ways in time; what they spread to before emission counts in the
        first slot."""
        from acoustic_spaces.parameters import octave_filter_gains

        out = np.zeros((self.bands, slots))
        lead = int(round(synthesis._PAD_S * self.rate))
        n_fft = synthesis.fft_size(self.signals.shape[1] + lead, self.rate)
        gains = octave_filter_gains(n_fft, self.rate)[: self.bands]
        weights = np.full(gains.shape[1], 2.0)
        weights[0] = weights[-1] = 1.0
        time = (np.arange(self.signals.shape[1] + lead) - lead) / self.rate
        slot = np.maximum(time / time_step, 0).astype(np.int64)
        keep = slot < slots
        for b, (gain, k) in enumerate(zip(gains, self.band_runs, strict=True)):
            padded = np.concatenate([np.zeros(lead), self.signals[k, :, 0]])
            pressure = _filter(padded, gain, n_fft)
            share = float((gain**2 * weights).sum() / n_fft)
            energy = pressure[keep] ** 2 / (synthesis.ENERGY_TO_IR * share)
            out[b] = np.bincount(slot[keep], energy, minlength=slots)[:slots]
        return out


def _probe_points(listener: np.ndarray, cell: float) -> np.ndarray:
    """The listener and a point one cell to either side along each axis."""
    offsets = np.vstack(
        [np.zeros(3), np.repeat(np.eye(3), 2, axis=0) * np.tile([1, -1], 3)[:, None]]
    )
    return listener[None, :] + cell * offsets


def _first_order(probes: np.ndarray, found: Sequence[bool], name: str) -> np.ndarray:
    """W, X, Y, Z (samples, 4) from the pressure at the listener and round it.

    The velocity is the time integral of the pressure gradient (central differences,
    or one-sided ones next to a surface), by the trapezoidal rule so that it falls on
    the pressure's samples; X = rho c u_x points towards the sound.
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
        out[:, 1 + axis] = COURANT * (np.cumsum(difference) - 0.5 * difference)
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
    say: Callable[[str], None] | None = None,
    label: str = "",
) -> tuple[list[WaveResponse], dict[str, float]]:
    """The wave bands of the responses from ``source`` to each of ``listeners`` in the
    room of ``model`` (a :class:`~acoustic_spaces.engine.model.TraceModel`), and run
    details."""
    from acoustic_spaces import _native

    say = say or (lambda _: None)
    if max_band_hz not in OCTAVE_BANDS_HZ:
        raise ValueError(f"max_band_hz must be one of {OCTAVE_BANDS_HZ}, got {max_band_hz}")
    mesh = model.room.mesh
    listeners = np.asarray(listeners, dtype=float).reshape(-1, 3)
    points = np.vstack([np.asarray(source, dtype=float).reshape(1, 3), listeners])
    bands = OCTAVE_BANDS_HZ.index(max_band_hz) + 1
    grid = band_grid(bands - 1, sample_rate, speed_of_sound, points_per_wavelength)
    h = grid.cell
    # The model, and two cells round the source and listeners (in an open model they may
    # lie outside it); then two cells of margin (one for an open model's air, one that is
    # never air), shifted off the model's planes.
    low = np.minimum(mesh.vertices.min(axis=0), points.min(axis=0) - 2.0 * h)
    high = np.maximum(mesh.vertices.max(axis=0), points.max(axis=0) + 2.0 * h)
    origin = low - h * (2.0 + np.array([0.0137, 0.0219, 0.0291]))
    shape = np.ceil((high - origin) / h).astype(np.int64) + 2
    if int(np.prod(shape)) > MAX_CELLS:
        raise ValueError(
            f"the wave grid up to the {max_band_hz} Hz band would have "
            f"{int(np.prod(shape)):,} cells (at most {MAX_CELLS:,}); lower max_band_hz or "
            "points_per_wavelength"
        )
    signal, delay = source_signal(grid.rate, h)
    samples = int(math.ceil(length_s * grid.rate))
    steps = samples + delay  # the source's term on step n stands for its value at time n
    drive = np.zeros(steps, np.float32)
    drive[: min(len(signal), steps)] = signal[:steps]
    probes = np.vstack([_probe_points(p, h) for p in listeners])
    # One run per set of absorption coefficients: bands with the same ones share it.
    coefficients = np.round(model.absorption[:, :bands], 6)
    sets: list[np.ndarray] = []
    band_runs = []
    for b in range(bands):
        same = [k for k, c in enumerate(sets) if np.array_equal(c, coefficients[:, b])]
        if not same:
            sets.append(coefficients[:, b])
        band_runs.append(same[0] if same else len(sets) - 1)
    recorded_runs = []
    info: dict[str, float] = {}
    for k, absorption in enumerate(sets):
        solver = _native.WaveSolver(
            model.bvh,
            np.ascontiguousarray(admittance(absorption)),
            origin,
            h,
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
            np.asarray(source, dtype=float),
            np.ascontiguousarray(probes),
            1.0,
            threads,
        )
        covered = [OCTAVE_BANDS_HZ[b] for b, r in enumerate(band_runs) if r == k]
        name = " and ".join(f"{f}" for f in covered) + " Hz"
        say(
            f"{label}: wave solver, {name} ({solver.air_cells:,} cells of {100 * h:.0f} cm, "
            f"{steps:,} steps)"
        )
        # Long runs report every tenth (which is also where a cancelled run stops).
        chunks = 10 if solver.air_cells * steps > _LONG_RUN else 1
        recorded = []
        bounds = np.linspace(0, steps, chunks + 1).astype(np.int64)
        for c in range(chunks):
            if c:
                say(f"{label}: wave solver, {name}, {100 * c // chunks} %")
            recorded.append(solver.run(drive[bounds[c] : bounds[c + 1]]))
        recorded_runs.append(np.concatenate(recorded, axis=1)[:, delay:][:, :samples])
        found = [solver.probe_found(p) for p in range(len(probes))]
        info["wave_cells"] = float(solver.air_cells)
        if solver.open:
            info["wave_open"] = 1.0
    info["wave_cell_m"] = h
    info["wave_runs"] = float(len(sets))
    responses = []
    for i, listener in enumerate(names):
        part = slice(7 * i, 7 * i + 7)
        signals = np.stack(
            [_first_order(run[part], found[part], listener) for run in recorded_runs]
        )
        responses.append(WaveResponse(grid.rate, signals, tuple(band_runs), h))
    return responses, info


__all__ = [
    "COURANT",
    "HIGH_PASS_S",
    "LOW_EDGE_HZ",
    "MAX_ABSORPTION",
    "MAX_CELLS",
    "BandGrid",
    "WaveResponse",
    "absorption_of_impedance",
    "admittance",
    "band_grid",
    "band_top",
    "source_signal",
    "wave_gains",
    "wave_responses",
]
