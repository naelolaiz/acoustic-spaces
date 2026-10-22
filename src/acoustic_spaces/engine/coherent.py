"""Early paths that add in pressure at low frequencies.

The echogram adds the early paths in energy, band by band, as if their phases were
random. At low frequencies they are not: a band's filter rings longer than the early
paths lie apart, and their pressures add. The impulse response has always added them
that way (its paths are impulses that add sample by sample), so the parameters of the
63 and 125 Hz bands read from it differed from the echogram's by up to 2 dB.

In the lowest bands, up to ``simulation.early.coherent_max_band_hz``, every early path
therefore carries its pressure (``EarlyReflections.spectrum``) at the frequencies the
edge diffraction is evaluated at: the phase of each reflection factor on the way, and
for paths over an edge the whole complex UTD coefficient, whose terms then add to the
specular paths in pressure, as the theory has them, rather than through the cross
terms of the higher bands. Its level in each band stays the energy that band gives it.
A surface reflects like a locally reacting one whose admittance has the material's
absorption, fitted as for the wave solver
(:func:`~acoustic_spaces.engine.impedance.fit_admittance`): R = (cos theta - Y) /
(cos theta + Y) at the angle of incidence theta. A hard surface reflects in phase at
every angle; an absorbing one turns the phase of sound that grazes it, by half a turn at
grazing incidence. Above those bands each path keeps the phase it has at their top
(:func:`held_phase`).

The impulse response builds those bands' early part from the pressures, and the
echogram reads each band's early energy from the same signal through the band filters
of the analysis of measured responses
(:func:`~acoustic_spaces.parameters.band_filter_gains`), as for the wave solver's bands:
an impulse response and the echogram it came from now give the same parameters in those
bands. The late part still adds in energy, and so do the uncorrelated points of a line
or surface source.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import numpy as np

from acoustic_spaces import synthesis
from acoustic_spaces.materials.bands import (
    BANDS_PER_OCTAVE,
    OCTAVE_BANDS_HZ,
    band_centres,
    resolution_of,
)

if TYPE_CHECKING:
    from acoustic_spaces.engine.early import EarlyReflections
    from acoustic_spaces.engine.model import TraceModel

#: Paths whose spectra are built at once.
_CHUNK = 256
#: Outdoors, where every band adds in pressure, at most this many values (paths times
#: frequencies) of the impulse response's spectra are built at once.
_MAX_VALUES = 1 << 22
#: The coherent bands up to this octave band are read through the analysis's band
#: filters (:class:`EarlyPressure`); outdoors, the bands above it from the interference
#: of each pair of paths averaged over the band (:func:`pair_energy`), which costs the
#: same however far the sound travels.
FILTERED_MAX_BAND_HZ = 500
#: How much longer than the last early path the echogram's early part is taken (s),
#: for the band filters' ringing.
_TAIL_S = 0.3


def band_count(max_band_hz: int | None, bands: int, wave_max_band_hz: int | None = None) -> int:
    """How many of the lowest of ``bands`` (8 octave or 24 third-octave bands) add in
    pressure: those of the octave bands up to ``max_band_hz`` (None: none), and none
    when the wave solver runs up to that band or higher (``wave_max_band_hz``), as its
    bands take their place."""
    if max_band_hz is None:
        return 0
    if wave_max_band_hz is not None and wave_max_band_hz >= max_band_hz:
        return 0
    octaves = OCTAVE_BANDS_HZ.index(int(max_band_hz)) + 1
    return octaves * BANDS_PER_OCTAVE[resolution_of(bands)]


def filtered_bands(paths: EarlyReflections) -> int:
    """How many of the paths' coherent bands are read through the band filters: those
    up to :data:`FILTERED_MAX_BAND_HZ`."""
    bands = paths.energy.shape[1]
    return min(paths.coherent_bands, band_count(FILTERED_MAX_BAND_HZ, bands))


def sample_frequencies(bands: int, count: int) -> np.ndarray:
    """(count, M) the frequencies (Hz) a spectrum holds in each of the lowest ``count``
    of ``bands`` bands: those the edge diffraction is evaluated at."""
    from acoustic_spaces.engine.diffraction import band_frequencies

    return band_frequencies(bands=resolution_of(bands))[0][:count]


def reflection_phases(
    model: TraceModel, faces: np.ndarray, cosines: np.ndarray, count: int
) -> np.ndarray:
    """(n, count, M) e^(j arg R) of the reflections off ``faces`` (n,) of sound that
    meets them at incidence cosines ``cosines`` (n,), at the frequencies of
    :func:`sample_frequencies`: R is that of a locally reacting surface whose
    admittance gives the face's absorption in the lowest ``count`` bands; 1 for the
    faces whose reflection :func:`ground_factors` gives (see
    :meth:`~acoustic_spaces.engine.model.TraceModel.factor_faces`)."""
    faces = np.asarray(faces, dtype=np.int64)
    total = model.absorption.shape[1]
    f = sample_frequencies(total, count)
    out = np.ones((len(faces), *f.shape), dtype=complex)
    if len(faces) == 0 or count == 0:
        return out
    if model.face_impedance is not None or model.ground_part is not None:
        fitted = ~model.factor_faces(faces)
        if fitted.any():
            out[fitted] = _fitted_phases(
                model, faces[fitted], np.asarray(cosines)[fitted], count, f, out[fitted]
            )
        return out
    return _fitted_phases(model, faces, cosines, count, f, out)


def _fitted_phases(model, faces, cosines, count, f, out):
    """:func:`reflection_phases` into ``out`` for faces without an impedance."""
    from acoustic_spaces.engine.impedance import fit_admittance

    total = model.absorption.shape[1]
    rows, inverse = np.unique(model.absorption[faces, :count], axis=0, return_inverse=True)
    admittance = np.array([fit_admittance(row, total, None)(f) for row in rows])
    y = admittance[inverse.reshape(-1)]
    cos = np.clip(np.asarray(cosines, dtype=float), 0.0, 1.0)[:, None, None]
    total_ = cos + y
    hard = np.abs(total_) < 1e-12  # a hard surface met at grazing incidence
    r = (cos - y) / np.where(hard, 1.0, total_)
    magnitude = np.abs(r)
    out[...] = np.where(hard | (magnitude < 1e-12), 1.0, r / np.maximum(magnitude, 1e-300))
    return out


def ground_factors(
    model: TraceModel,
    faces: np.ndarray,
    cosines: np.ndarray,
    distance: np.ndarray | None,
    spherical: bool = True,
    ends: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """(n, bands, M) the reflection factors off ``faces`` (n,) of sound that meets them
    at incidence cosines ``cosines`` (n,), at the frequencies of every band's samples
    (:func:`~acoustic_spaces.engine.diffraction.band_frequencies`): for surfaces with an
    impedance, that of the spherical wave (Weyl-van der Pol, see
    :func:`~acoustic_spaces.materials.ground.spherical_reflection`) along paths
    ``distance`` (n,) m long from the image source, or with ``spherical`` False the
    plane wave's; 1 for the others.

    On the ground of a scene with ground zones, a reflection is the sum of the factors
    of the grounds its Fresnel zone covers, each weighted by the share it covers (see
    :meth:`~acoustic_spaces.geometry.outdoor.Layout.fresnel_shares`), from the points
    ``ends`` (n, 3) before the reflection, of the reflection and after it; without
    ``ends``, the ground's own. There, a ground without an impedance reflects with the
    square root of (1 - its absorption) and the phase of :func:`reflection_phases`."""
    from acoustic_spaces.engine.diffraction import band_frequencies

    faces = np.asarray(faces, dtype=np.int64)
    f = band_frequencies(bands=model.bands)[0]
    out = np.ones((len(faces), *f.shape), dtype=complex)
    mixed = np.zeros(len(faces), dtype=bool)
    if model.ground_part is not None:
        mixed = model.ground_part[faces] >= 0
    rows = np.flatnonzero(model.impedance_faces(faces) & ~mixed)
    if len(rows):
        length = None if distance is None else np.asarray(distance, dtype=float)[rows]
        out[rows] = _surface_factors(model, faces[rows], cosines, rows, length, spherical, f)
    rows = np.flatnonzero(mixed)
    if len(rows) == 0:
        return out
    part = model.ground_part[faces[rows]]
    if ends is None:
        shares = np.zeros((len(rows), f.shape[0], len(model.part_face)))
        shares[np.arange(len(rows)), :, part] = 1.0
    else:
        before, at, after = (np.asarray(p, dtype=float)[rows] for p in ends)
        wavelength = model.room.speed_of_sound() / np.asarray(model.band_hz, dtype=float)
        shares = model.room.outdoor.fresnel_shares(before, at, after, wavelength)
        # A part with no faces (the ground outside zones that cover it all) is the one
        # the reflection lies in.
        empty = model.part_face < 0
        if empty.any():
            shares[np.arange(len(rows)), :, part] += shares[..., empty].sum(axis=-1)
            shares[..., empty] = 0.0
    mix = np.zeros((len(rows), *f.shape), dtype=complex)
    length = None if distance is None else np.asarray(distance, dtype=float)[rows]
    for p, face in enumerate(model.part_face.tolist()):
        some = np.flatnonzero(np.any(shares[:, :, p] > 0.0, axis=1))
        if face < 0 or len(some) == 0:
            continue
        factor = _surface_factors(
            model,
            np.full(len(some), face),
            cosines,
            rows[some],
            None if length is None else length[some],
            spherical,
            f,
        )
        mix[some] += shares[some, :, p, None] * factor
    out[rows] = mix
    return out


def _surface_factors(model, faces, cosines, rows, distance, spherical, f) -> np.ndarray:
    """(n, bands, M) :func:`ground_factors` off ``faces`` (n,) on their own, at the
    incidence cosines ``cosines[rows]``: an impedance's spherical (or plane) wave, or for
    a surface without one the square root of (1 - its absorption) with the phase of its
    fitted admittance."""
    from acoustic_spaces.materials.ground import plane_reflection, spherical_reflection

    cos = np.asarray(cosines, dtype=float)[rows]
    out = np.empty((len(faces), *f.shape), dtype=complex)
    own = model.impedance_faces(faces)
    if own.any():
        which = model.face_impedance[faces[own]]
        admittance = model.admittance[which]  # (r, bands, M)
        extended = model.impedance_extended[which]
        if extended.any():
            # The extended reaction: the admittance each path meets at its own angle.
            admittance = admittance.copy()
            c = model.room.speed_of_sound()
            angles = cos[own]
            for k in np.unique(which[extended]):
                at = np.flatnonzero(extended & (which == k))
                admittance[at] = model.impedances[k].admittance(
                    f[None], c, cosine=angles[at, None, None]
                )
        if spherical:
            k = 2.0 * np.pi * f / model.room.speed_of_sound()
            length = np.asarray(distance, dtype=float)[own, None, None]
            out[own] = spherical_reflection(admittance, cos[own, None, None], k, length)
        else:
            out[own] = plane_reflection(admittance, cos[own, None, None])
    if not own.all():
        other = ~own
        bands = f.shape[0]
        phase = _fitted_phases(
            model, faces[other], cos[other], bands, f, np.ones((other.sum(), *f.shape), complex)
        )
        level = np.sqrt(np.clip(1.0 - model.absorption[faces[other]], 0.0, None))
        out[other] = level[:, :, None] * phase
    return out


def shaped(
    spectrum: np.ndarray, bands: int, f: np.ndarray, amplitude: np.ndarray, gains: np.ndarray
) -> np.ndarray:
    """(n, len(f)) the coherent bands' amplitudes (n, count) through the synthesis's
    band ``gains`` (count, len(f)) at frequencies ``f`` (Hz), each band's times the
    shape its spectra (n, count, M, :attr:`EarlyReflections.spectrum`) give it: the
    magnitude and the unwrapped phase interpolated linearly in log frequency between
    the band's own samples and held beyond them. A band so carries its own sound into
    the crossovers with its neighbours, not theirs: outdoors a path's bands below the
    parabolic equation's top band hold the whole sound over the ground and those above
    it the path's own, and a shape drawn across that edge would notch the sum."""
    n, count, per_band = spectrum.shape
    samples = np.log2(sample_frequencies(bands, count))  # (count, M)
    x = np.log2(np.maximum(np.asarray(f, dtype=float), 1e-6))
    out = np.zeros((n, len(f)), dtype=complex)
    for b in range(count):
        support = np.flatnonzero(gains[b] > 0.0)
        if len(support) == 0:
            continue
        lo, hi = int(support[0]), int(support[-1]) + 1
        own = spectrum[:, b]
        if per_band == 1:
            shape = np.repeat(own, hi - lo, axis=1)
        else:
            xs = np.clip(x[lo:hi], samples[b, 0], samples[b, -1])
            j = np.clip(np.searchsorted(samples[b], xs, side="right") - 1, 0, per_band - 2)
            w = (xs - samples[b, j]) / (samples[b, j + 1] - samples[b, j])
            magnitude = np.abs(own)
            phase = np.unwrap(np.angle(own), axis=1)
            level = magnitude[:, j] * (1.0 - w) + magnitude[:, j + 1] * w
            angle = phase[:, j] * (1.0 - w) + phase[:, j + 1] * w
            shape = level * np.exp(1j * angle)
        out[:, lo:hi] += (amplitude[:, b, None] * gains[b, lo:hi]) * shape
    return out


def held_phase(paths: EarlyReflections) -> np.ndarray:
    """(n,) the phase factor of each path at the top of the coherent bands, which it
    keeps in the bands above them: there the paths add in energy, and a path whose
    phase stepped back to zero at the crossover would cancel itself in it, as two bands
    half a turn apart do (ones without coherent bands)."""
    if paths.coherent_bands == 0 or len(paths) == 0:
        return np.ones(len(paths), dtype=complex)
    top = paths.spectrum[:, -1, -1]
    magnitude = np.abs(top)
    return np.where(magnitude > 1e-12, top / np.maximum(magnitude, 1e-300), 1.0)


def amplitude_spectra(
    paths: EarlyReflections, f: np.ndarray, gains: np.ndarray, only_coherent: bool = False
) -> np.ndarray:
    """(n, len(f)) each path's pressure at frequencies ``f`` relative to its delay, as
    the impulse response has it: its amplitude in each band through the synthesis's
    band ``gains`` (bands, len(f)) at those frequencies, in the coherent bands times its
    spectrum's shape in that band (:func:`shaped`) and above them times its
    :func:`held_phase`; with ``only_coherent``, the coherent bands alone."""
    bands = paths.energy.shape[1]
    count = paths.coherent_bands
    amplitude = np.sqrt(synthesis.ENERGY_TO_IR * paths.energy)
    if count:
        low = shaped(paths.spectrum, bands, f, amplitude[:, :count], gains[:count])
    else:
        low = np.zeros((len(paths), len(f)), dtype=complex)
    if not only_coherent:
        low = low + (amplitude[:, count:] @ gains[count:]) * held_phase(paths)[:, None]
    lagged = lag_phase(paths, f)
    return low if lagged is None else low * lagged


def lag_phase(paths: EarlyReflections, f: np.ndarray) -> np.ndarray | None:
    """(n, len(f)) at a moment of turbulence, the phase factor of each path's
    :attr:`~acoustic_spaces.engine.early.EarlyReflections.lag` at frequencies ``f``
    (Hz), the lag interpolated linearly in log frequency between the bands' centres and
    held beyond the first and the last; None without one."""
    if paths.lag is None:
        return None
    f = np.asarray(f, dtype=float)
    centres = np.log2(np.asarray(band_centres(resolution_of(paths.energy.shape[1])), float))
    x = np.log2(np.clip(f, 2.0 ** centres[0], 2.0 ** centres[-1]))
    j = np.clip(np.searchsorted(centres, x, side="right") - 1, 0, len(centres) - 2)
    w = (x - centres[j]) / (centres[j + 1] - centres[j])
    lag = paths.lag[:, j] * (1.0 - w) + paths.lag[:, j + 1] * w
    return np.exp(-2j * np.pi * lag * f)


def band_impulses(
    paths: EarlyReflections, n: int, sample_rate: float, gains: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """The band signals (bands, n) of the early paths above the coherent bands, each
    path times ``gains`` (paths,) (default ones), as :func:`synthesis.place_impulses`
    has them, and those of the quarter turn of their :func:`held_phase` for
    :func:`synthesis.combine_bands` (None when no path has one)."""
    amplitudes = np.sqrt(synthesis.ENERGY_TO_IR * paths.energy)
    amplitudes[:, : paths.coherent_bands] = 0.0  # those come from their pressure
    if gains is not None:
        amplitudes = amplitudes * np.asarray(gains, dtype=float)[:, None]
    held = held_phase(paths)

    def place(values: np.ndarray) -> np.ndarray:
        if paths.lag is None:
            return synthesis.place_impulses(paths.delay_s, values, n, sample_rate)
        # At a moment of turbulence each band's paths come at their own delays.
        out = np.zeros((values.shape[1], n))
        for b in range(paths.coherent_bands, values.shape[1]):
            delay = paths.delay_s + paths.lag[:, b]
            out[b] = synthesis.place_impulses(delay, values[:, b : b + 1], n, sample_rate)[0]
        return out

    if np.all(np.abs(held.imag) < 1e-12):
        return place(amplitudes * held.real[:, None]), None
    return place(amplitudes * held.real[:, None]), place(amplitudes * held.imag[:, None])


def signal(
    paths: EarlyReflections, n: int, sample_rate: float, weights: np.ndarray | None = None
) -> np.ndarray:
    """(n, K) the coherent bands' part of the early paths' impulse response, at
    ``sample_rate``, each path weighted by ``weights`` (paths, K) (default: one column
    of ones). Zero without coherent bands."""
    k = 1 if weights is None else np.asarray(weights).shape[1]
    count = paths.coherent_bands
    if count == 0 or len(paths) == 0:
        return np.zeros((n, k))
    bands = paths.energy.shape[1]
    n_fft = synthesis.fft_size(n, sample_rate)
    gains = synthesis.band_gains(n_fft, sample_rate, bands)
    support = np.flatnonzero(gains[:count].sum(axis=0) > 0.0)
    top = int(support.max()) + 1
    f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)[:top]
    weights = np.ones((len(paths), 1)) if weights is None else np.asarray(weights, dtype=float)
    out = np.zeros((n_fft // 2 + 1, k), dtype=complex)
    chunk = _CHUNK
    if count > filtered_bands(paths):  # every band: long spectra
        chunk = max(1, min(_CHUNK, _MAX_VALUES // top))
    for start in range(0, len(paths), chunk):
        part = slice(start, start + chunk)
        spectra = amplitude_spectra(paths[part], f, gains[:, :top], only_coherent=True)
        spectra *= np.exp(-2j * np.pi * np.outer(paths.delay_s[part], f))
        for column in range(k):  # one at a time: a mono response and W add alike
            out[:top, column] += spectra.T @ weights[part, column]
    return np.fft.irfft(out, n_fft, axis=0)[:n]


def total_energy(paths: EarlyReflections) -> np.ndarray:
    """(bands,) the paths' energy together in each band: their sum, and in the coherent
    bands that of their pressures added at each of the band's frequencies, averaged over
    them; each frequency stands for its slice of the band, over which the interference
    of two paths that arrive far apart averages out; in turbulence it takes their
    coherence factor (:meth:`~acoustic_spaces.engine.turbulence.Wander.coherence`).
    Paths of different points of a line or a surface source add in energy."""
    from acoustic_spaces.engine.diffraction import band_frequencies

    out = paths.energy.sum(axis=0)
    count = paths.coherent_bands
    if count == 0 or len(paths) == 0:
        return out
    bands = paths.energy.shape[1]
    f, width = band_frequencies(bands=resolution_of(bands))
    f, width = f[:count], width[:count]
    pressure = np.sqrt(paths.energy[:, :count, None]) * paths.spectrum
    pressure = pressure * np.exp(-2j * np.pi * paths.delay_s[:, None, None] * f[None])
    apart = paths.delay_s[:, None] - paths.delay_s[None, :]
    same = paths.point[:, None] == paths.point[None, :]
    for b in range(count):
        here = pressure[:, b]
        lagged = lag_phase(paths, f[b])
        if lagged is not None:
            here = here * lagged
        products = np.real(here[:, None] * np.conj(here[None, :]))
        if paths.wander is not None:
            products = products * paths.wander.coherence(f[b])
        out[b] = float(np.sum(products.mean(axis=2) * np.sinc(apart * width[b]) * same))
    return out


def pair_energy(
    paths: EarlyReflections,
    bands: range,
    time_step: float,
    slots: int,
    axis: np.ndarray | None = None,
    exponent: int = 2,
    origin: float = 0.0,
) -> np.ndarray:
    """(len(bands), slots) energy per time slot of ``time_step`` (slot 0 at ``origin``,
    s from emission) in ``bands`` of the paths' coherent ones, adding them in pressure:
    each path's own energy where it arrives, and the cross term of each pair of paths,
    averaged over the band's frequencies as in :func:`total_energy` (with their
    coherence factor in turbulence), shared between the two paths' slots as their
    energies are (a slot that would go below zero is left at zero). With ``axis``, that
    of a figure of eight facing it (``exponent`` as in :meth:`EarlyPressure.energy`; with
    1 the cross terms take the geometric mean of the two paths' weights)."""
    from acoustic_spaces.engine.diffraction import band_frequencies

    out = np.zeros((len(bands), slots))
    if len(paths) == 0:
        return out
    f, width = band_frequencies(bands=resolution_of(paths.energy.shape[1]))
    slot = np.floor((paths.delay_s - origin) / time_step + 1e-9).astype(np.int64)
    keep = (slot >= 0) & (slot < slots)
    apart = paths.delay_s[:, None] - paths.delay_s[None, :]
    weight = np.ones(len(paths))
    if axis is not None:
        cosine = paths.direction @ np.asarray(axis, dtype=float)
        weight = cosine if exponent == 2 else np.sqrt(np.abs(cosine))
    for row, b in enumerate(bands):
        energy = paths.energy[:, b]
        pressure = np.sqrt(energy)[:, None] * paths.spectrum[:, b]
        pressure = pressure * np.exp(-2j * np.pi * np.outer(paths.delay_s, f[b]))
        lagged = lag_phase(paths, f[b])
        if lagged is not None:
            pressure = pressure * lagged
        if paths.wander is None:
            cross = np.real(pressure @ pressure.conj().T) / f.shape[1]
        else:
            together = paths.wander.coherence(f[b])
            cross = np.einsum("im,jm,ijm->ij", pressure, pressure.conj(), together).real
            cross /= f.shape[1]
        cross *= np.sinc(apart * width[b]) * np.outer(weight, weight)
        # A pair's cross terms, 2 Re(p_i p_j*), shared between its two paths.
        total = energy[:, None] + energy[None, :]
        share = np.divide(2.0 * energy[:, None], total, out=np.zeros_like(cross), where=total > 0)
        np.fill_diagonal(share, 1.0)
        mine = np.sum(cross * share, axis=1)
        np.add.at(out[row], slot[keep], mine[keep])
    return np.maximum(out, 0.0)


def coherent_energy(
    paths: EarlyReflections,
    time_step: float,
    slots: int,
    axis: np.ndarray | None = None,
    exponent: int = 2,
    origin: float = 0.0,
    start: float | None = None,
    cache: dict | None = None,
) -> np.ndarray:
    """(coherent bands, slots) the coherent bands' energy per time slot (arguments as
    for :meth:`EarlyPressure.energy`): through the band filters up to
    :data:`FILTERED_MAX_BAND_HZ` and by :func:`pair_energy` above. The paths of each of
    a line's or a surface's points add in pressure, those of different points in
    energy. ``cache`` (a dict) keeps the :class:`EarlyPressure` of each point between
    calls on the same paths."""
    count = paths.coherent_bands
    filtered = filtered_bands(paths)
    points = np.unique(paths.point)
    if len(points) <= 1 and count == filtered:
        pressure = _pressure(paths, 0, cache)
        return pressure.energy(time_step, slots, axis, exponent, origin, start)
    out = np.zeros((count, slots))
    for point in points:
        part = paths[paths.point == point] if len(points) > 1 else paths
        if filtered:
            pressure = _pressure(part, int(point), cache)
            out[:filtered] += pressure.energy(time_step, slots, axis, exponent, origin, start)
        if count > filtered:
            out[filtered:] += pair_energy(
                part, range(filtered, count), time_step, slots, axis, exponent, origin
            )
    return out


def _pressure(paths: EarlyReflections, point: int, cache: dict | None) -> EarlyPressure:
    if cache is None:
        return EarlyPressure(paths)
    if point not in cache:
        cache[point] = EarlyPressure(paths)
    return cache[point]


@functools.lru_cache(maxsize=8)
def _band_filters(n_fft: int, rate: float, bands: int) -> np.ndarray:
    """The analysis's band filters (:func:`~acoustic_spaces.parameters.band_filter_gains`),
    kept for the next point: read-only."""
    from acoustic_spaces.parameters import band_filter_gains

    gains = band_filter_gains(n_fft, rate, bands)
    gains.flags.writeable = False
    return gains


class EarlyPressure:
    """The early part of an impulse response at a low sample rate, as the echogram
    reads the coherent bands from it: as spectra from 0.2 s before emission, the
    pressure and the three components of the paths' arrival directions times it, or
    with ``weights`` (paths, K) the sums of the paths weighted by each column.

    Each band is scaled so that every path alone carries its energy in it: the analysis
    filters also let in a little of the neighbouring bands, whose levels differ, and the
    filters of the impulse response's bands overlap with them. What is left is the
    paths' interference, which in turbulence takes each two paths' coherence factor:
    there each path's spectrum is kept, and :meth:`energy` adds the filtered paths'
    products."""

    def __init__(self, paths: EarlyReflections, weights: np.ndarray | None = None) -> None:
        bands = paths.energy.shape[1]
        self.bands = bands
        self.count = filtered_bands(paths)
        top = band_centres(resolution_of(bands))[max(self.count, 1) - 1]
        # Rate high enough for the band filters of the coherent bands to have died out,
        # and a whole number of samples to a millisecond; each sample stands for the
        # interval it is the middle of.
        self.rate = 1000.0 * 2.0 ** math.ceil(math.log2(max(32.0 * top / 1000.0, 1.0)))
        self.lead = synthesis._PAD_S
        last = float(paths.delay_s.max()) if len(paths) else 0.0
        length = int(math.ceil((self.lead + last + _TAIL_S) * self.rate))
        self.n_fft = 1 << max(int(math.ceil(math.log2(length))), 4)
        f = np.fft.rfftfreq(self.n_fft, 1.0 / self.rate)
        gains = synthesis.band_gains(self.n_fft, self.rate, bands)
        if weights is None:
            weights = np.column_stack([np.ones(len(paths)), paths.direction])
        self.spectra = np.zeros((np.shape(weights)[1], len(f)), dtype=complex)
        self._alone = np.zeros(len(f))  # the paths' squared spectra, summed
        self._energy = paths.energy[:, : self.count].sum(axis=0)
        self._wander = paths.wander if len(paths) > 1 else None
        if self._wander is not None:
            self._each = np.zeros((len(paths), len(f)), dtype=complex)
            self._weights = np.asarray(weights, dtype=float)
        # A path that carries nothing in a band is left out of that band's reading:
        # what its neighbouring bands' filters let into this one would otherwise count
        # against the paths that do carry it (outdoors, where the parabolic equation's
        # bands sit on one path of a family and the others are silent in them).
        self._silent = paths.energy[:, : self.count] <= 0.0  # (n, count)
        partly = [
            b
            for b in range(self.count)
            if self._silent[:, b].any() and not self._silent[:, b].all()
        ]
        self._spectra_of = {b: np.zeros_like(self.spectra) for b in partly}
        self._alone_of = {b: np.zeros(len(f)) for b in partly}
        for start in range(0, len(paths), _CHUNK):
            part = slice(start, start + _CHUNK)
            spectra = amplitude_spectra(paths[part], f, gains)
            squared = np.abs(spectra) ** 2
            self._alone += np.sum(squared, axis=0)
            shift = paths.delay_s[part] + self.lead - 0.5 / self.rate
            spectra *= np.exp(-2j * np.pi * np.outer(shift, f))
            self.spectra += weights[part].T @ spectra
            for b in partly:
                heard = ~self._silent[part, b]
                self._alone_of[b] += np.sum(squared[heard], axis=0)
                self._spectra_of[b] += np.asarray(weights[part])[heard].T @ spectra[heard]
            if self._wander is not None:
                self._each[part] = spectra

    def band(
        self,
        b: int,
        time_step: float,
        slots: int,
        origin: float = 0.0,
        start: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """The spectra through band b's analysis filter (K, T) at the samples that fall
        into the ``slots`` of ``time_step``, slot 0 at ``origin`` (s from emission); the
        slot of each sample (T,); and the factor that makes a squared sample energy.

        The filters ring before an arrival as much as after it: what they spread to
        before ``start`` (s from emission; default ``origin``), the arrival of the
        direct sound, counts as arriving with it, as in the analysis of a measured
        response."""
        gain, keep, slot, scale = self._filter(b, time_step, slots, origin, start)
        spectra = self._spectra_of.get(b, self.spectra)
        pressure = np.fft.irfft(gain * spectra, self.n_fft, axis=1)[:, keep]
        return pressure, slot, scale

    def _filter(self, b, time_step, slots, origin, start):
        """Band b's analysis filter, which samples fall into the slots, their slots,
        and the factor that makes a squared sample energy (see :meth:`band`)."""
        gain = _band_filters(self.n_fft, self.rate, self.bands)[b]
        time = np.arange(self.n_fft) / self.rate - self.lead  # where each interval begins
        time = np.maximum(time, origin if start is None else max(start, origin)) - origin
        slot = (time / time_step + 1e-9).astype(np.int64)
        keep = slot < slots
        weights = np.full(len(gain), 2.0)
        weights[0] = weights[-1] = 1.0
        alone = float(np.sum(weights * gain**2 * self._alone_of.get(b, self._alone))) / self.n_fft
        scale = float(self._energy[b]) / alone if alone > 0.0 else 0.0
        return gain, keep, slot[keep], scale

    def _turbulent(self, b, time_step, slots, axis, exponent, origin, start) -> np.ndarray:
        """(slots,) band b's energy with each two paths' products taken by their
        coherence factor, averaged over the band's frequencies."""
        from acoustic_spaces.engine.diffraction import band_frequencies

        gain, keep, slot, scale = self._filter(b, time_step, slots, origin, start)
        signals = np.fft.irfft(gain * self._each, self.n_fft, axis=1)[:, keep]
        signals[self._silent[:, b]] = 0.0
        f = band_frequencies(bands=resolution_of(self.bands))[0][b]
        together = self._wander.coherence(f).mean(axis=2)
        plain = self._weights[:, 0]
        if axis is None:
            first = second = plain
        else:
            side = self._weights[:, 1:] @ np.asarray(axis, dtype=float)
            first, second = side, (side if exponent == 2 else plain)
        power = np.sum((first[:, None] * signals) * (together @ (second[:, None] * signals)), 0)
        if axis is not None and exponent != 2:
            power = np.abs(power)
        return np.bincount(slot, power * scale, minlength=slots)[:slots]

    def energy(
        self,
        time_step: float,
        slots: int,
        axis: np.ndarray | None = None,
        exponent: int = 2,
        origin: float = 0.0,
        start: float | None = None,
    ) -> np.ndarray:
        """(count, slots) energy per coherent band and time slot, as
        :meth:`~acoustic_spaces.engine.wave.WaveResponse.energy` reads the wave
        solver's bands, through each band's analysis filter (``origin`` and ``start``
        as for :meth:`band`). With ``axis``, that of a figure of eight facing it
        (``exponent`` as there)."""
        out = np.zeros((self.count, slots))
        for b in range(self.count):
            if self._wander is not None:
                out[b] = self._turbulent(b, time_step, slots, axis, exponent, origin, start)
                continue
            pressure, slot, scale = self.band(b, time_step, slots, origin, start)
            if axis is None:
                power = pressure[0] ** 2
            else:
                side = np.asarray(axis) @ pressure[1:]
                power = side**2 if exponent == 2 else np.abs(side * pressure[0])
            out[b] = np.bincount(slot, power * scale, minlength=slots)[:slots]
        return out


__all__ = [
    "FILTERED_MAX_BAND_HZ",
    "EarlyPressure",
    "amplitude_spectra",
    "band_count",
    "band_impulses",
    "coherent_energy",
    "filtered_bands",
    "ground_factors",
    "held_phase",
    "pair_energy",
    "reflection_phases",
    "sample_frequencies",
    "shaped",
    "signal",
    "total_energy",
]
