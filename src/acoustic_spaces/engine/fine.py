"""The parabolic equation's field over the open ground at every frequency it ran at.

Over the open ground in wind and temperature gradients, the parabolic equation's field
(:mod:`~acoustic_spaces.engine.refracted`) has the ground's interference fringes in it:
dips a few hundred hertz wide far downwind. A band's eight samples
(:attr:`~acoustic_spaces.engine.early.EarlyReflections.spectrum`), held beyond the
band's first and last, miss them in the half-octave crossovers between the bands
(:func:`~acoustic_spaces.synthesis.band_gains`): above the equation's top band, where
the rays take over, and at every join between its own bands. Out of the shadow the
path that carries the field therefore keeps the equation's pressure at every frequency
its runs took (:attr:`~acoustic_spaces.engine.early.EarlyReflections.fine`), on a
layout (:func:`fine_frequencies`): in each band the frequencies of the runs at
``OPEN_DENSE`` times the early paths' from ``OPEN_DENSE_FROM_HZ`` up (theirs below),
then the same spacing on through the band's upper crossover, its skirt, which one more
run takes above each listener's top band of the equation where a response is rendered
or the band filters read it (:func:`skirt_frequencies`). Each band's shape is drawn
through its own and its neighbours' samples (:func:`fine_shape`), so the two bands of a
crossover give the same field, each at its own level.

The lowest band's gain goes on down to 0 Hz, and its band filter, through which the
echogram reads it, down to an octave below it, 33 dB down there in octave bands: its
run goes on down that far at its spacing, held below. 100 m downwind in the benchmark's
case 2 the 63 Hz band's C50 and C80 so come within 0.004 and 0.07 dB of a run of the
equation 27 times as dense going on down to 11 Hz, read the same way, where its own
samples held below it were 0.30 and 0.28 dB off; stopping an octave lower moves them by
at most 0.03 dB, and EDT by 0.5 ms.

Between two samples h apart the shape is the complex pressure taken linearly, relative
to the path's delay plus its :func:`reference_delay` (the delay its phase follows on
average). For a field of arrivals a_i at delays dtau_i past that, the error is at most
h^2 / 8 times the sum of |a_i| (2 pi dtau_i)^2: on uniform samples the second
difference of the samples over 8, which :func:`interpolation_bound` reads from the
samples themselves. The samples resolve the field only where that is small beside it:
a field that turns by half a period or more from one sample to the next (two arrivals
more than 1 / (2 h) apart, 8.5 ms with samples 59 Hz apart) is not resolved, and a dip
narrower than the spacing that falls between two samples is not seen. Far downwind in
the windy preset, 300 m away, the equation's dip to -10.1 dB re free field at 2225 Hz
reads -9.9 dB through samples 59 Hz apart.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from acoustic_spaces import synthesis
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, octave_of, resolution_of

#: Over the open ground (and along the ways round ends) the runs in the octave bands
#: from this one (centre, Hz) up take ``OPEN_DENSE`` times the early paths' frequencies
#: (an odd number, so that theirs are among them), with
#: ``simulation.early.open_ground_frequencies: 24``, the default (8 and ``rays`` take
#: theirs): far downwind the ground's interference fringes are dense in frequency, and
#: in the benchmark's case 2 1 km away the 1 kHz band of 8 an octave was 2.2 dB below
#: the fast field program's (``validation/benchmark_1995.py``).
OPEN_DENSE_FROM_HZ = 1000
OPEN_DENSE = 3


def _dense(bands: int, band: int) -> int:
    """How many times the early paths' frequencies the layout has in ``band``."""
    octave = octave_of(resolution_of(bands))[band]
    return OPEN_DENSE if octave >= OCTAVE_BANDS_HZ.index(OPEN_DENSE_FROM_HZ) else 1


@functools.lru_cache(maxsize=64)
def _grid(bands: int, band: int) -> tuple[float, float, int, int]:
    """The low edge (Hz) of ``band``, the layout's spacing in it (Hz), its count in the
    band and how many times the early paths' that is."""
    from acoustic_spaces.engine.diffraction import band_frequencies

    resolution = resolution_of(bands)
    dense = _dense(bands, band)
    early = band_frequencies(bands=resolution)[0].shape[1]
    frequencies, width = band_frequencies(dense * early, resolution)
    step = float(width[band])
    return float(frequencies[band, 0]) - step / 2, step, dense * early, dense


def _past(top: float, low: float, step: float, every: int, first: int) -> int:
    """The first slot ``first + every j`` (j >= 0) of the grid ``low + step (i + 1/2)``
    at or past ``top``."""
    j = math.ceil(((top - low) / step - 0.5 - first) / every - 1e-9)
    return first + every * max(j, 0)


def band_fine_frequencies(bands: int, band: int, dense: bool, skirt: bool = True) -> np.ndarray:
    """The frequencies (Hz) the open ground's runs of the parabolic equation take in
    ``band`` of ``bands``: ``OPEN_DENSE`` times the early paths' (``dense``, from
    ``OPEN_DENSE_FROM_HZ`` up) or theirs; with ``skirt`` the same spacing on to the
    first at or past the top of the band's upper crossover (none for the top band) and,
    in the lowest band, whose gain goes on down to 0 Hz, down to the last at or above
    half its lower edge (:func:`skirt_frequencies`). All lie on the layout
    (:func:`fine_frequencies`)."""
    low, step, count, times = _grid(bands, band)
    every, first = (1, 0) if dense or times == 1 else (times, (times - 1) // 2)
    start, stop = first, count
    if skirt and band + 1 < bands:
        top = synthesis.crossover_frequencies(bands)[band] * 2.0 ** (
            synthesis.crossover_width(bands) / 2
        )
        stop = max(stop, _past(top, low, step, every, first) + 1)
    if skirt and band == 0:
        # The slots low + step (i + 1/2) >= low / 2, on the run's own (first + every j).
        lowest = math.ceil(-low / (2.0 * step) - 0.5 - 1e-9)
        start = first + every * math.ceil((lowest - first) / every - 1e-9)
    return low + step * (np.arange(start, stop, every) + 0.5)


def skirt_frequencies(bands: int, band: int, dense: bool, below: bool) -> np.ndarray:
    """The frequencies (Hz) of ``band``'s skirt over the open ground, at the spacing of
    its runs (``dense`` as :func:`band_fine_frequencies`): ``below`` its own in the
    lowest band, through the octave under its lower edge, which the analysis's band
    filter of that band reads (an octave band's 33 dB down at its foot); else above
    them, through its upper crossover."""
    every = band_fine_frequencies(bands, band, dense)
    own = band_fine_frequencies(bands, band, dense, skirt=False)
    return every[every < own[0]] if below else every[every > own[-1]]


def fine_frequencies(bands: int, count: int) -> np.ndarray:
    """(count, K) the layout of :attr:`~acoustic_spaces.engine.early.EarlyReflections.
    fine` in the lowest ``count`` of ``bands`` bands, NaN past a band's last: in each
    band every frequency a run takes there at either spacing
    (:func:`band_fine_frequencies`), the lowest band's skirt below it, and below the top
    one its skirt above too."""
    return _fine_frequencies(int(bands), int(count)).copy()


@functools.lru_cache(maxsize=16)
def _fine_frequencies(bands: int, count: int) -> np.ndarray:
    rows = []
    for band in range(count):
        row = np.union1d(
            band_fine_frequencies(bands, band, True),
            band_fine_frequencies(bands, band, False),
        )
        if band + 1 == count:
            # The top band takes no skirt above (the lowest, alone, keeps its own below).
            row = row[row <= band_fine_frequencies(bands, band, True, skirt=False)[-1]]
        rows.append(row)
    out = np.full((count, max(len(row) for row in rows)), np.nan)
    for band, row in enumerate(rows):
        out[band, : len(row)] = row
    out.flags.writeable = False
    return out


def placed(bands: int, count: int, runs: dict, delay: float) -> np.ndarray:
    """(count, K) a path's :attr:`~acoustic_spaces.engine.early.EarlyReflections.fine`:
    ``runs`` by band, each a list of (frequencies (Hz), complex pressures, time as
    exp(j omega t)), taken relative to the path's ``delay`` (s) and put in their slots of
    the layout (:func:`fine_frequencies`); NaN elsewhere. A frequency off the layout is
    an error."""
    layout = _fine_frequencies(int(bands), int(count))
    out = np.full(layout.shape, np.nan, dtype=complex)
    for band, pieces in runs.items():
        row = layout[band][np.isfinite(layout[band])]
        for at, values in pieces:
            at = np.asarray(at, dtype=float)
            slot = np.clip(np.searchsorted(row, at), 1, len(row) - 1)
            slot = np.where(np.abs(row[slot - 1] - at) < np.abs(row[slot] - at), slot - 1, slot)
            off = ~np.isclose(row[slot], at, rtol=1e-9, atol=0.0)
            if off.any():
                raise ValueError(f"frequencies {at[off]} Hz of band {band} are not on its layout")
            out[band, slot] = np.asarray(values) * np.exp(2j * np.pi * at * delay)
    return out


def reference_delay(fine: np.ndarray, bands: int) -> float:
    """The delay (s) past the path's own that the phase of its samples ``fine`` (C, K)
    follows on average, which :func:`fine_shape` draws them relative to: over each two
    neighbouring samples of a band, h apart, the delay their phase changes by, weighted
    by h^2 and their magnitudes over the band's mean square, which makes the sum of their
    interpolation bounds least; 0 without two."""
    layout = _fine_frequencies(int(bands), fine.shape[0])
    total = weight = 0.0
    for band in range(fine.shape[0]):
        has = np.isfinite(fine[band].real)
        if has.sum() < 2:
            continue
        f, v = layout[band][has], fine[band][has]
        level = float(np.mean(np.abs(v) ** 2))
        if level <= 0.0:
            continue
        h = np.diff(f)
        w = h**2 * np.abs(v[1:] * v[:-1]) / level
        total += float(np.sum(w * np.angle(v[1:] * np.conj(v[:-1])) / (2.0 * np.pi * h)))
        weight += float(np.sum(w))
    return total / weight if weight > 0.0 else 0.0


def fine_rows(fine: np.ndarray, band: int, bands: int) -> np.ndarray:
    """The rows of ``fine`` (n, C, K) with samples of their own in ``band``, short of
    its upper crossover's middle, that are not all zero: those whose shape there
    :func:`fine_shape` draws."""
    own = _own(fine.shape[1], band, bands)
    values = fine[:, band][:, own]
    has = np.isfinite(values.real)
    return np.flatnonzero((has & (np.abs(np.where(has, values, 0.0)) > 0.0)).any(axis=1))


def _own(count: int, band: int, bands: int) -> np.ndarray:
    """(K,) the layout's slots of ``band`` from its lower edge (below which the lowest
    band's skirt lies) short of its upper crossover's middle."""
    layout = _fine_frequencies(int(bands), count)[band]
    edges = synthesis.crossover_frequencies(bands)
    upper = edges[band] if band < len(edges) else np.inf
    return np.isfinite(layout) & (layout >= _grid(int(bands), band)[0]) & (layout <= upper)


def _drawn(row: np.ndarray, band: int, bands: int) -> tuple[np.ndarray, np.ndarray, float]:
    """A row's samples (C, K) that ``band``'s shape is drawn through, its own and its
    neighbours' in order of frequency, and its own root mean square."""
    count = row.shape[0]
    layout = _fine_frequencies(int(bands), count)
    at, values = [], []
    for c in (band - 1, band, band + 1):
        if 0 <= c < count:
            has = np.isfinite(row[c].real)
            at.append(layout[c][has])
            values.append(row[c][has])
    f, v = np.concatenate(at), np.concatenate(values)
    order = np.argsort(f, kind="stable")
    own = _own(count, band, bands) & np.isfinite(row[band].real)
    level = float(np.sqrt(np.mean(np.abs(row[band][own]) ** 2))) if own.any() else 0.0
    return f[order], v[order], level


def fine_shape(
    fine: np.ndarray, delay: np.ndarray, bands: int, band: int, f: np.ndarray
) -> np.ndarray:
    """(n, len(f)) ``band``'s shape at frequencies ``f`` (Hz) for paths with samples
    ``fine`` (n, C, K) there (:func:`fine_rows`): the field through the band's own
    samples and its neighbours' (one field across the crossovers between them), the
    complex pressure relative to the path's delay plus its own ``delay`` (n,) s
    (:func:`reference_delay`) taken linearly in frequency and held beyond the outermost,
    over its root mean square at the band's own frequencies short of its upper
    crossover's middle. Each band so keeps its own level (the air's, the source's
    directivity), and in a crossover the two bands give the same field."""
    f = np.asarray(f, dtype=float)
    out = np.zeros((len(fine), len(f)), dtype=complex)
    for i in range(len(fine)):
        at, values, level = _drawn(fine[i], band, bands)
        if level <= 0.0:
            continue
        u = values * np.exp(-2j * np.pi * at * delay[i])
        held = np.clip(f, at[0], at[-1])
        field = np.interp(held, at, u.real) + 1j * np.interp(held, at, u.imag)
        out[i] = field * np.exp(2j * np.pi * f * delay[i]) / level
    return out


def interpolation_bound(fine: np.ndarray, delay: np.ndarray, bands: int) -> np.ndarray:
    """(n, C) for each path and band with samples (:func:`fine_rows`), how far
    :func:`fine_shape` can be off between them inside the band's gain, over the band's
    level, as the samples give it: h^2 / 8 times |P''|, the larger second divided
    difference at the two ends of each gap h (on uniform samples their second difference
    over 8), exact where |P''| is the same across the gap; NaN without samples."""
    count = fine.shape[1]
    out = np.full((len(fine), count), np.nan)
    edges = synthesis.crossover_frequencies(bands)
    half = synthesis.crossover_width(bands) / 2
    for band in range(count):
        lo = edges[band - 1] * 2.0**-half if band > 0 else 0.0
        hi = edges[band] * 2.0**half if band < len(edges) else np.inf
        for i in fine_rows(fine, band, bands):
            at, values, level = _drawn(fine[i], band, bands)
            if len(at) < 3:
                out[i, band] = 0.0
                continue
            u = values * np.exp(-2j * np.pi * at * delay[i])
            first = (u[1:] - u[:-1]) / np.diff(at)
            second = np.abs(2.0 * (first[1:] - first[:-1]) / (at[2:] - at[:-2]))  # ~ |u''|
            # Each gap's |u''|: the larger of its two ends' (the outermost gaps their one).
            curve = np.maximum(np.append(second[:1], second), np.append(second, second[-1:]))
            gaps = np.diff(at)
            inside = (at[1:] > lo) & (at[:-1] < hi)
            bound = gaps**2 / 8.0 * curve
            out[i, band] = float(bound[inside].max()) / level if inside.any() else 0.0
    return out


__all__ = [
    "OPEN_DENSE",
    "OPEN_DENSE_FROM_HZ",
    "band_fine_frequencies",
    "fine_frequencies",
    "fine_rows",
    "fine_shape",
    "interpolation_bound",
    "placed",
    "reference_delay",
    "skirt_frequencies",
]
