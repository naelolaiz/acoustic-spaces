"""The Doppler shift of a walk: each early path through a delay that changes as the
walker goes.

A source moving towards a listener shortens the way its sound has to go while it is
on it, so the listener hears its pitch raised by 1 / (1 - M cos theta), M its speed
over the speed of sound and theta the angle between the way it goes and the way to the
listener; going away, lowered. A listener moving past a source hears the same. Each
early path has its own shift (the sound reflected off a wall behind a car falls in
pitch while the direct sound still rises), so :func:`render` follows every path from
place to place along the walk:

- **The paths.** A path at one place is the same path at the next when it meets the
  same planes (the ground, a facade) in the same order over the same edge; failing
  that, the path of the same kind whose delay carries on from it. Where a path appears
  or goes, it fades in or out between two places.
- **The delay.** Each path's delay is known at every place: for a walking source, the
  time its sound takes from there, heard that much later (the retarded time); for a
  walking listener, at the moment it is there. Between the places a cubic through
  them gives the delay at every sample, and the signal is read through it with a
  windowed sinc (Kaiser, 32 zero crossings each side) whose cutoff falls with the
  shift, so a raised pitch does not alias.
- **The filter.** What else the path does to the sound (the air, the ground's
  reflection, a diffracting edge, the source's directivity, the direction it arrives
  from in the output format, and in turbulence how the eddies delay its other bands
  from its top one at that moment) goes into a filter, the path's spectrum without its
  delay as the impulse response has it. The filter changes every :data:`HOP_S`, taken
  between the places on either side in proportion to how near they are; the delayed
  signal goes through it in windows that half overlap (squared sines, which add up to
  one), so a path heard from one place alone gives exactly the convolution with its
  response.
- **The ears.** Where each ear's response starts (its onset, as the ears are mixed
  between measured directions: :meth:`~acoustic_spaces.spatial.hrtf.HRTF.interpolate`)
  is a delay too: it goes into that ear's delay, read through the cubic and the sinc
  as the path's is, and out of its filter. Measured ears can start a few samples apart
  from one direction to the next; filters that start apart, mixed between two places
  or in the overlapping windows, would cancel the treble there.
- **The eddies.** In turbulence the eddies move each path's delay from one moment to
  the next (by microseconds, as :class:`~acoustic_spaces.engine.turbulence.Gusts`
  draws them): the top band's goes into the path's delay, taken between the places and
  the moments on either side as the filter is, so that two moments' filters mixed do
  not cancel the treble either.
- **The sound sent.** Every path carries the same signal, unless it is an
  :class:`~acoustic_spaces.emission.Emission`, the sound a source sends each way: then
  each path carries what is sent the way it leaves the source (its first leg, in the
  source's frame at each place, in between turning from one place's to the next's) at
  the time it leaves, and the source's directivity is taken out of its filter.

The late part and the wave solver's bands have no Doppler shift here: the late part
of each place is crossfaded as in a walkthrough without it, and the wave solver's
bands are not taken (a walk with Doppler and the wave solver stops with an error).
The source radiates as it would standing at each place, as road and rail emission data
have it: the convective amplification of a moving source (for a monopole up to 0.8 dB
ahead at 30 m/s, 2 dB at 70 m/s, less behind) is left out, and emission measured in a
pass-by already carries it, as an emission may.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from acoustic_spaces import synthesis
from acoustic_spaces.emission import Emission
from acoustic_spaces.engine import coherent
from acoustic_spaces.engine.model import source_weights

#: How often each path's filter changes (s): the hop of the half-overlapping windows.
HOP_S = 0.02
#: Length of a path's filter (s), as the binaural early paths have it.
KERNEL_S = 0.3
#: Zero crossings of the reading sinc on each side, and its Kaiser window's beta.
_ZEROS = 32
_BETA = 9.0
#: Phases of the reading sinc's table; a delay rounds to the nearest.
_PHASES = 4096
#: Samples read through a delay at once.
_CHUNK = 1 << 15


@dataclass
class Track:
    """One path followed along a walk."""

    #: (N,) whether the path is heard from each place.
    present: np.ndarray
    #: (N,) its row among the paths of each place (-1 where it is not heard).
    rows: np.ndarray
    #: (N,) when the sound of each place along it is heard (s from the start).
    received: np.ndarray
    #: (N,) its delay at each place (s); where it is not heard, carried on from where
    #: it is.
    delays: np.ndarray

    @property
    def span(self) -> tuple[int, int]:
        """The first and last place it is heard from."""
        heard = np.flatnonzero(self.present)
        return int(heard[0]), int(heard[-1])


def plane_ids(mesh) -> np.ndarray:
    """(F,) the plane each face of ``mesh`` lies in, numbered: faces in one plane (the
    triangles of a ground or a wall) share it."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return np.zeros(0, dtype=np.int64)
    normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(length > 0.0, length, 1.0)
    offsets = np.einsum("ij,ij->i", normals, vertices[faces[:, 0]])
    keys = np.column_stack([np.round(normals * 1e6), np.round(offsets * 1e3)]).astype(np.int64)
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    return ids.reshape(-1).astype(np.int64)


def _key(paths, row: int, planes: np.ndarray) -> tuple:
    faces = paths.faces[row] if paths.faces else np.zeros(0, dtype=np.int64)
    met = tuple(int(planes[f]) if 0 <= f < len(planes) else -1 for f in np.asarray(faces))
    return int(paths.point[row]), int(paths.edge[row]), int(paths.order[row]), met


def tracks(
    places: list, times: np.ndarray, *, source_walks: bool, planes: np.ndarray, step_s: float
) -> list[Track]:
    """The paths of ``places`` (the early paths at each place of a walk, in order),
    followed from place to place. ``times`` (N,) are when the walker is at each place;
    with ``source_walks`` its sound is heard a path's delay later, otherwise then.
    ``step_s`` is the most a delay may jump between places and stay the same path's."""
    count = len(places)
    found: list[dict] = []  # each track: {place: row}
    open_tracks: dict[tuple, list[int]] = {}  # key -> tracks heard at the previous place
    previous: dict[int, int] = {}  # track -> its row at the previous place
    for k, paths in enumerate(places):
        keys = [_key(paths, i, planes) for i in range(len(paths))]
        order = np.argsort(paths.delay_s, kind="stable")
        taken: set[int] = set()
        now: dict[int, int] = {}
        reopened: dict[tuple, list[int]] = {}
        loose = []
        for i in order:
            candidates = [t for t in open_tracks.get(keys[i], []) if t not in taken]
            if candidates:
                t = candidates[0]
            else:
                loose.append(int(i))
                continue
            taken.add(t)
            found[t][k] = int(i)
            now[t] = int(i)
            reopened.setdefault(keys[i], []).append(t)
        # Failing the same planes, a path of the same kind whose delay carries on.
        for i in loose:
            kind = keys[i][:3]
            best, gap = None, step_s
            for t, row in previous.items():
                if t in taken or _kind(found[t], places, k) != kind:
                    continue
                jump = abs(float(paths.delay_s[i]) - float(places[k - 1].delay_s[row]))
                if jump <= gap:
                    best, gap = t, jump
            if best is None:
                best = len(found)
                found.append({})
            taken.add(best)
            found[best][k] = i
            now[best] = i
            reopened.setdefault(keys[i], []).append(best)
        open_tracks, previous = reopened, now
    out = []
    for rows_of in found:
        rows = np.full(count, -1, dtype=np.int64)
        for k, i in rows_of.items():
            rows[k] = i
        present = rows >= 0
        heard = np.flatnonzero(present)
        delays = np.array([places[k].delay_s[rows[k]] for k in heard], dtype=float)
        filled = _carried(times[heard], delays, times)
        received = times + filled if source_walks else times.astype(float)
        out.append(Track(present, rows, received, filled))
    return out


def _kind(rows_of: dict, places: list, k: int) -> tuple:
    last = max(rows_of)
    paths, row = places[last], rows_of[last]
    return int(paths.point[row]), int(paths.edge[row]), int(paths.order[row])


def _carried(at: np.ndarray, values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """``values`` given at times ``at`` (rising), at ``times``: as given, linearly in
    between, and carried on along the slope at either end (held with one value)."""
    out = np.interp(times, at, values)
    if len(at) >= 2:
        before, after = times < at[0], times > at[-1]
        first = (values[1] - values[0]) / (at[1] - at[0])
        last = (values[-1] - values[-2]) / (at[-1] - at[-2])
        out[before] = values[0] + first * (times[before] - at[0])
        out[after] = values[-1] + last * (times[after] - at[-1])
    return out


def hermite(x: np.ndarray, y: np.ndarray, at: np.ndarray, *, hold: bool = True) -> np.ndarray:
    """The cubic Hermite curve through (``x``, ``y``) (``x`` rising) at ``at``, its slope
    at each point the weighted mean of the chords on either side; held at the ends'
    values beyond them (or carried on along the end slopes when ``hold`` is false)."""
    x, y, at = (np.asarray(v, dtype=float) for v in (x, y, at))
    if len(x) == 1:
        return np.full(at.shape, y[0])
    h = np.diff(x)
    chords = np.diff(y) / h
    slopes = np.empty(len(x))
    slopes[0], slopes[-1] = chords[0], chords[-1]
    slopes[1:-1] = (h[1:] * chords[:-1] + h[:-1] * chords[1:]) / (h[:-1] + h[1:])
    j = np.clip(np.searchsorted(x, at, side="right") - 1, 0, len(x) - 2)
    u = (at - x[j]) / h[j]
    u2, u3 = u * u, u * u * u
    out = (
        (2 * u3 - 3 * u2 + 1) * y[j]
        + (u3 - 2 * u2 + u) * h[j] * slopes[j]
        + (-2 * u3 + 3 * u2) * y[j + 1]
        + (u3 - u2) * h[j] * slopes[j + 1]
    )
    before, after = at < x[0], at > x[-1]
    if hold:
        out[before], out[after] = y[0], y[-1]
    else:
        out[before] = y[0] + slopes[0] * (at[before] - x[0])
        out[after] = y[-1] + slopes[-1] * (at[after] - x[-1])
    return out


def _reach(cutoff: float) -> int:
    """How far the reading sinc for ``cutoff`` (share of Nyquist) reaches back from the
    sample before the position, and on from it."""
    return int(math.ceil(_ZEROS / cutoff))


@functools.lru_cache(maxsize=8)
def _table(cutoff: float) -> tuple[np.ndarray, int]:
    """The reading sinc's taps (phases, taps) for ``cutoff`` (share of Nyquist), and how
    far it reaches back from the sample before the position: kept for the last few
    cutoffs, as a path's signals are read one after another with its own."""
    reach = _reach(cutoff)
    offsets = np.arange(-reach + 1, reach + 1)
    frac = np.arange(_PHASES) / _PHASES
    t = offsets[None, :] - frac[:, None]
    half = _ZEROS / cutoff
    window = np.i0(_BETA * np.sqrt(np.clip(1.0 - (t / half) ** 2, 0.0, 1.0))) / np.i0(_BETA)
    taps = cutoff * np.sinc(cutoff * t) * window
    taps.flags.writeable = False
    return taps, reach


def read(dry: np.ndarray, positions: np.ndarray, cutoff: float) -> np.ndarray:
    """``dry`` (n,) read at the fractional sample ``positions`` through a windowed sinc
    cut off at ``cutoff`` (a share of Nyquist, below one where the reading runs faster
    than the samples, so nothing aliases); zero before and after it."""
    table, reach = _table(min(float(cutoff), 1.0))
    taps = table.shape[1]
    padded = np.concatenate(
        [np.zeros(reach + 1), np.asarray(dry, dtype=float), np.zeros(reach + 1)]
    )
    rows = sliding_window_view(padded, taps)
    out = np.zeros(len(positions))
    for start in range(0, len(positions), _CHUNK):
        x = positions[start : start + _CHUNK]
        whole = np.floor(x)
        phase = np.round((x - whole) * _PHASES).astype(np.int64)
        whole = whole.astype(np.int64) + phase // _PHASES
        phase %= _PHASES
        # Row r of ``rows`` starts at dry sample r - reach - 1, so sample `whole` - reach + 1
        # is row `whole` + 2.
        index = whole + 2
        inside = (index >= 0) & (index < len(rows))
        part = np.zeros(len(x))
        if np.any(inside):
            part[inside] = np.einsum("kt,kt->k", table[phase[inside]], rows[index[inside]])
        out[start : start + _CHUNK] = part
    return out


class Filters:
    """The filters of the paths at each place (and, in turbulence, each moment of the
    eddies): (paths, channels, bins) spectra on one FFT grid, kept for the places in
    use. With the source's ``directivity`` and ``axes`` (its frame at each place), the
    directivity is taken out of them."""

    def __init__(
        self,
        responses,
        fmt,
        sample_rate: int,
        n_fft: int,
        delay: int,
        gusts,
        seed,
        directivity=None,
        axes=None,
    ):
        self.responses = responses
        self.directivity = directivity
        self.axes = axes
        self.fmt = fmt
        self.rate = int(sample_rate)
        self.n_fft = n_fft
        self.f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self.cycles = np.fft.rfftfreq(n_fft)  # per sample
        # The filters are zero-phase about the path's delay: ``delay`` samples later.
        self.shift = np.exp(-2j * np.pi * self.f * delay / sample_rate)
        self.gusts = gusts
        self.seed = seed
        self.cache: OrderedDict = OrderedDict()
        self.size = 8
        self.roots: OrderedDict = OrderedDict()
        self.moves: OrderedDict = OrderedDict()

    def at(self, k: int, moment: int = 0) -> np.ndarray:
        key = (k, moment)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        spectra = self._spectra(k, moment)
        self.cache[key] = spectra
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return spectra

    def _place_roots(self, k: int) -> list:
        """The eddies' :meth:`~acoustic_spaces.engine.early.EarlyReflections.wander_roots`
        at place ``k``, kept for the places in use."""
        if k in self.roots:
            self.roots.move_to_end(k)
            return self.roots[k]
        roots = self.responses[k].early.wander_roots()
        self.roots[k] = roots
        while len(self.roots) > self.size:
            self.roots.popitem(last=False)
        return roots

    def _paths(self, k: int, moment: int):
        """The early paths at place ``k`` and, in turbulence, moment ``moment``."""
        response = self.responses[k]
        paths = response.early
        if paths.wander is None:
            return paths
        if self.gusts is None:
            return response.moment(self.seed)
        roots = self._place_roots(k)
        return paths.at_moment(roots, self.gusts.draws(roots, paths.point, moment))

    def moved(self, k: int, moment: int = 0) -> np.ndarray:
        """(paths,) how far the eddies move each path's delay (its top band's, s) at
        place ``k`` and moment ``moment``: zero without turbulence."""
        paths = self.responses[k].early
        if paths.wander is None:
            return np.zeros(len(paths))
        key = (k, moment if self.gusts is not None else 0)
        if key in self.moves:
            self.moves.move_to_end(key)
            return self.moves[key]
        out = self._paths(k, moment).delay_s - paths.delay_s
        self.moves[key] = out
        while len(self.moves) > 64 * self.size:
            self.moves.popitem(last=False)
        return out

    def _spectra(self, k: int, moment: int) -> np.ndarray:
        from acoustic_spaces.spatial.directions import to_head
        from acoustic_spaces.spatial.microphone import Microphone
        from acoustic_spaces.spatial.render import Binaural

        response = self.responses[k]
        # In turbulence the top band's delay at the moment is in the path's delay
        # (:meth:`moved`); its other bands' lags behind it stay here.
        paths = self._paths(k, moment)
        paths = coherent.with_own_phases(paths, response.seed if self.seed is None else self.seed)
        if self.directivity is not None:
            weights = source_weights(
                self.directivity,
                self.axes[k],
                _ways(paths),
                paths.energy.shape[1],
                distances=paths.distance_m,
            )
            energy = np.divide(
                paths.energy, weights, out=np.zeros_like(paths.energy), where=weights > 0.0
            )
            paths = dataclasses.replace(paths, energy=energy)
        gains = synthesis.band_gains(self.n_fft, self.rate, response.bands)
        shaping = coherent.amplitude_spectra(paths, self.f, gains) * self.shift
        heads = to_head(paths.direction, response.orientation)
        if isinstance(self.fmt, Binaural):
            hrtf = self.fmt.at(self.rate)
            ears, onsets = hrtf.interpolate(heads, onsets=True)
            # Each ear's onset is in its delay (see :func:`_onsets`), so out of its
            # filter: the samples before it wrap round to the end, and the shift to the
            # middle of the kernel brings them back before it.
            ears = np.fft.rfft(ears, self.n_fft, axis=-1)
            ears *= np.exp(2j * np.pi * self.cycles * onsets[..., None])
            spectra = shaping[:, None, :] * ears
        elif isinstance(self.fmt, Microphone):  # its pattern may change with frequency
            spectra = shaping[:, None, :] * self.fmt.spectral_gains(heads, gains)
        else:
            spectra = shaping[:, None, :] * self.fmt.gains(heads)[:, :, None]
        return spectra.astype(np.complex64)


def _onsets(responses, track: Track, hrtf, rate: int) -> np.ndarray:
    """(N, 2) where each ear's response starts (s) on ``track``'s path at each place
    (zero where it is not heard), as :class:`Filters` takes it out of the ears."""
    from acoustic_spaces.spatial.directions import to_head

    onsets = np.zeros((len(track.present), 2))
    heard = np.flatnonzero(track.present)
    if len(heard):
        heads = np.array(
            [
                to_head(responses[k].early.direction[track.rows[k]], responses[k].orientation)
                for k in heard
            ]
        )
        onsets[heard] = hrtf.interpolate(heads, onsets=True)[1] / rate
    return onsets


def _ways(paths) -> np.ndarray:
    """(n, 3) the unit vectors the paths leave the source along, in room coordinates."""
    if len(paths.points) != len(paths):
        raise ValueError("the sound sent each way needs the paths' corners")
    first = np.array([p[1] - p[0] for p in paths.points], dtype=float).reshape(-1, 3)
    return first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-300)


@dataclass
class _Reading:
    """How a track is read: its places, the blocks it is heard in, its cutoff; with an
    emission, the way its path leaves the source at each place (in the source's frame)."""

    track: Track
    heard: np.ndarray
    first: int
    last: int
    cutoff: float
    ways: np.ndarray | None = None
    #: (N, 2) at the ears, where each ear's response starts at each place (s).
    onsets: np.ndarray | None = None

    def sent(self, emission: Emission, positions: np.ndarray, rate: int) -> np.ndarray:
        """``emission`` read at ``positions`` (its samples): at each, the sound sent the
        way the path leaves the source then, its way turning evenly from one place's to
        the next's in the time the sound leaves them."""
        assert self.ways is not None
        # The samples the sinc reaches, within the emission, as :func:`read` reads them.
        reach = _reach(min(self.cutoff, 1.0))
        lo = max(int(math.floor(positions.min())) - reach - 2, 0)
        hi = min(int(math.ceil(positions.max())) + reach + 3, emission.length)
        if lo >= hi:
            return np.zeros(len(positions))
        heard = self.heard
        left = (self.track.received - self.track.delays)[heard]
        at = np.arange(lo, hi) / rate
        ways = np.column_stack([np.interp(at, left, self.ways[heard, i]) for i in range(3)])
        ways /= np.maximum(np.linalg.norm(ways, axis=1, keepdims=True), 1e-300)
        return read(emission.sent(lo, hi, ways), positions - lo, self.cutoff)

    def onset(self, t: np.ndarray, ear: int) -> np.ndarray:
        """Where ``ear``'s response starts at times ``t`` (s): through the places as
        the delay goes, held past the path's ends."""
        assert self.onsets is not None
        return hermite(self.track.received[self.heard], self.onsets[self.heard, ear], t)

    def delays(self, t: np.ndarray, walk_last: int) -> np.ndarray:
        track, heard = self.track, self.heard
        delay = hermite(track.received[heard], track.delays[heard], t, hold=True)
        first_k, last_k = track.span
        if first_k > 0 or last_k < walk_last:
            # Past a path's own ends (it appears or goes) its delay carries on along
            # its slope.
            carried = hermite(track.received[heard], track.delays[heard], t, hold=False)
            ends = np.zeros(len(t), dtype=bool)
            if first_k > 0:
                ends |= t < track.received[first_k]
            if last_k < walk_last:
                ends |= t > track.received[last_k]
            delay[ends] = carried[ends]
        return delay


#: Blocks of :data:`HOP_S` whose delayed signals are read at once.
_BLOCKS = 128


def render(
    dry: np.ndarray | Emission,
    responses: list,
    times: np.ndarray,
    fmt,
    sample_rate: int,
    length: int,
    *,
    source_walks: bool,
    planes: np.ndarray,
    step_s: float,
    gusts=None,
    seed: int | None = None,
    axes: np.ndarray | None = None,
    directivity=None,
    span: tuple[int, int] | None = None,
    followers: Sequence[tuple[int, np.ndarray]] = (),
) -> np.ndarray:
    """(length, channels) the early paths of ``responses`` (the responses at the places
    of a walk, in order, the walker there at ``times``) heard as it goes, the mono
    ``dry`` signal through each path's changing delay and filter (see the module's
    description). ``step_s`` is the most a path's delay may jump from one place to the
    next; ``gusts`` gives the moments of the eddies in turbulence, ``seed`` the moment
    of each place's eddies without them. ``dry`` may be the sound the source sends each
    way, at ``sample_rate``: then ``axes`` (N, 3, 3) has the source's forward, left and
    up axes as columns at each place, and ``directivity`` (the source's, if it has one)
    is taken out of the paths' filters. With ``span`` (begin, end), samples begin to end
    alone: (end - begin, channels).

    ``followers`` are more mono signals going along the walk after the walker, each
    (lag, signal) ``lag`` samples behind it, as the wheels and the cars of a train
    follow its front: each goes through the same paths, places and filters, all of it
    ``lag`` samples later (``length`` counts them). They share what the paths cost
    to follow and filter, so each costs less than a rendering of its own."""
    from acoustic_spaces.spatial.render import Binaural

    rate = int(sample_rate)
    emission = dry if isinstance(dry, Emission) else None
    plain = dry if not isinstance(dry, Emission) else np.zeros(0)
    if followers and (emission is not None or span is not None):
        raise ValueError("signals follow the walker only as plain signals, rendered whole")
    signals = [(0, np.asarray(plain, dtype=float))]
    for lag, signal in followers:
        if int(lag) < 0:
            raise ValueError(f"a signal follows the walker, {lag} samples behind it is ahead")
        signals.append((int(lag), np.asarray(signal, dtype=float)))
    most = max(lag for lag, _ in signals)
    frames = np.zeros((0, 3, 3))
    if emission is not None:
        if emission.sample_rate != rate:
            raise ValueError(
                f"the sound sent each way is at {emission.sample_rate} Hz, the rendering "
                f"at {rate} Hz"
            )
        if axes is None:
            raise ValueError("the sound sent each way needs the source's axes at each place")
        frames = np.asarray(axes, dtype=float).reshape(len(times), 3, 3)
    places = [r.early for r in responses]
    followed = tracks(places, times, source_walks=source_walks, planes=planes, step_s=step_s)
    hop = max(int(round(HOP_S * rate)), 1)
    kernel = 1 << int(math.ceil(math.log2(KERNEL_S * rate)))
    half = kernel // 2
    taps = fmt.at(rate).taps if isinstance(fmt, Binaural) else 0
    n_fft = 1 << int(math.ceil(math.log2(2 * hop + kernel + taps)))
    blocks = int(math.ceil((length + half) / hop)) + 1
    last = len(times) - 1
    leaving = [_ways(paths) for paths in places] if emission is not None else []
    hrtf = fmt.at(rate) if isinstance(fmt, Binaural) else None
    readings = []
    for track in followed:
        first_k, last_k = track.span
        heard = np.flatnonzero(track.present)
        onsets = None if hrtf is None else _onsets(responses, track, hrtf, rate)
        # Heard from the place before its first to the place after its last, or to
        # either end of the sound where it is heard at the walk's.
        lo = 0.0 if first_k == 0 else track.received[first_k - 1]
        hi = math.inf if last_k == last else track.received[last_k + 1]
        j0 = max(int(math.floor(lo * rate / hop)), 0)
        j1 = (
            blocks - 1
            if not math.isfinite(hi)
            else min(int(math.ceil(hi * rate / hop)), blocks - 1)
        )
        speed = 1.0
        if len(heard) > 1:
            total = track.delays[heard, None] + (0.0 if onsets is None else onsets[heard])
            speed = float(
                np.max(1.0 - np.diff(total, axis=0) / np.diff(track.received[heard])[:, None])
            )
        ways = None
        if emission is not None:
            ways = np.zeros((len(times), 3))
            for k in heard:
                ways[k] = leaving[k][track.rows[k]] @ frames[k]
        readings.append(_Reading(track, heard, j0, j1, 0.97 / max(speed, 1.0), ways, onsets))
    # The places in use at once: for a walking source, those the paths' different
    # delays reach back to.
    spread = 2
    if source_walks and len(times) > 1:
        delays = np.concatenate([r.track.delays[r.heard] for r in readings] or [np.zeros(1)])
        interval = max(float(np.min(np.diff(times))), 1e-9)
        spread += int(math.ceil(float(np.ptp(delays)) / interval))
    filters = Filters(
        responses,
        fmt,
        rate,
        n_fft,
        half,
        gusts,
        seed,
        directivity if emission is not None else None,
        frames,
    )
    filters.size = 2 * (min(spread, len(times)) + 4)
    turbulent = any(r.early.wander is not None for r in responses)
    window = np.sin(np.pi * np.arange(2 * hop) / (2 * hop)) ** 2
    begin, end = (0, length) if span is None else (max(int(span[0]), 0), int(span[1]))
    end = min(end, length)
    # Block j adds to the samples from (j - 1) hops on, heard half a kernel later: the
    # blocks that reach samples begin to end, and ``out`` from block j_lo's first.
    j_lo = max((begin + half - n_fft) // hop + 1, 0)
    j_hi = min(-(-(end + half) // hop) + 1, blocks)
    base = (j_lo - 1) * hop
    out = np.zeros((max(j_hi - j_lo + 2, 0) * hop + n_fft + most, fmt.channels))
    # Fewer blocks at once with more signals, which each read every path's delays.
    chunk = max(_BLOCKS // len(signals), 1)
    for c0 in range(j_lo, j_hi, chunk):
        c1 = min(c0 + chunk, j_hi)
        n0, n1 = (c0 - 1) * hop, (c1 + 1) * hop
        t = np.arange(n0, n1) / rate
        active = []
        for reading in readings:
            if reading.first > c1 - 1 or reading.last < c0:
                continue
            delay = reading.delays(t, last)
            if turbulent:
                delay = delay + _moved(filters, reading.track, t, gusts)
            # At the ears, each ear's own delay (its onset added): (samples, 2).
            ears = (
                [delay] if reading.onsets is None else [delay + reading.onset(t, e) for e in (0, 1)]
            )
            heard = []
            for _, signal in signals:
                reads = []
                for each in ears:
                    positions = np.arange(n0, n1) - each * rate
                    if emission is not None:
                        reads.append(reading.sent(emission, positions, rate))
                    else:
                        reads.append(read(signal, positions, reading.cutoff))
                heard.append(np.column_stack(reads))
            active.append((reading, heard))
        if not active:
            continue
        for j in range(c0, c1):
            start = (j - 1) * hop
            centre = j * hop / rate
            totals: list = [None] * len(signals)
            for reading, heard in active:
                if j < reading.first or j > reading.last:
                    continue
                spectrum = _filter_at(filters, reading.track, centre, gusts)
                if spectrum is None:
                    continue
                for i, signal in enumerate(heard):
                    piece = signal[start - n0 : start - n0 + 2 * hop] * window[:, None]
                    if not np.any(piece):
                        continue
                    part = np.fft.rfft(piece, n_fft, axis=0).T * spectrum
                    totals[i] = part if totals[i] is None else totals[i] + part
            for (lag, _), total in zip(signals, totals, strict=True):
                if total is None:
                    continue
                wet = np.fft.irfft(total, n_fft, axis=-1).T
                at = start - base + lag
                if at < 0:
                    wet, at = wet[-at:], 0
                out[at : at + len(wet)] += wet[: len(out) - at]
    lo = begin + half - base
    return out[lo : lo + max(end - begin, 0)]


def _moved(filters: Filters, track: Track, t: np.ndarray, gusts) -> np.ndarray:
    """(len(t),) how far the eddies move the track's delay at times ``t`` (s): between
    the places on either side in proportion to how near they are, as
    :func:`_filter_at` takes the filter (the one that hears the path alone where the
    other does not), and between the moments on either side."""
    received = track.received
    last = len(received) - 1
    j = np.searchsorted(received, t, side="right") - 1
    k0, k1 = np.clip(j, 0, last), np.clip(j + 1, 0, last)
    span = received[k1] - received[k0]
    w = np.where(k1 > k0, (t - received[k0]) / np.where(span > 0.0, span, 1.0), 0.0)
    w = np.where(track.present[k1], np.where(track.present[k0], w, 1.0), 0.0)
    if gusts is None:
        m0 = m1 = np.zeros(len(t), dtype=np.int64)
        wm = np.zeros(len(t))
    else:
        m = t / gusts.step_s
        m0 = np.clip(np.floor(m), 0, gusts.count - 1).astype(np.int64)
        m1 = np.minimum(m0 + 1, gusts.count - 1)
        wm = np.where(m1 > m0, np.clip(m - m0, 0.0, 1.0), 0.0)
    out = np.zeros(len(t))
    for k in np.unique(np.concatenate([k0, k1])):
        if not track.present[k]:
            continue
        wk = np.where(k0 == k, 1.0 - w, 0.0) + np.where(k1 == k, w, 0.0)
        for moment in np.unique(np.concatenate([m0, m1])):
            weight = wk * (np.where(m0 == moment, 1.0 - wm, 0.0) + np.where(m1 == moment, wm, 0.0))
            if np.any(weight):
                out += weight * filters.moved(int(k), int(moment))[track.rows[k]]
    return out


def _filter_at(filters: Filters, track: Track, centre: float, gusts) -> np.ndarray | None:
    """(channels, bins) the track's filter at time ``centre`` (s): between the places on
    either side in proportion to how near they are (none where it is not heard), and
    in turbulence between the moments on either side."""
    received = track.received
    last = len(received) - 1
    j = int(np.searchsorted(received, centre, side="right")) - 1
    if j < 0:
        parts = [(0, 1.0)]
    elif j >= last:
        parts = [(last, 1.0)]
    else:
        w = (centre - received[j]) / (received[j + 1] - received[j])
        parts = [(j, 1.0 - w), (j + 1, w)]
    if gusts is None:
        moments = [(0, 1.0)]
    else:
        m = centre / gusts.step_s
        m0 = min(max(int(math.floor(m)), 0), gusts.count - 1)
        m1 = min(m0 + 1, gusts.count - 1)
        w = min(max(m - m0, 0.0), 1.0)
        moments = [(m0, 1.0 - w), (m1, w)] if m1 != m0 else [(m0, 1.0)]
    out = None
    for k, wk in parts:
        if wk <= 0.0 or not track.present[k]:
            continue
        for m, wm in moments:
            if wm <= 0.0:
                continue
            spectrum = filters.at(k, m)[track.rows[k]] * (wk * wm)
            out = spectrum if out is None else out + spectrum
    return out


__all__ = ["HOP_S", "KERNEL_S", "Track", "hermite", "plane_ids", "read", "render", "tracks"]
