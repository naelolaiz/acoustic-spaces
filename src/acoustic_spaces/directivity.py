"""Source directivity: how a source shares its sound power between directions, per
octave band (or third-octave band).

A directivity gives energy weights that average 1 over the sphere in every band, so a
source radiates its sound power (``level_db``) whatever its pattern; the pattern only
says where that power goes. Directions are in the source's own frame: x forward, y left,
z up, as :class:`acoustic_spaces.scene.Orientation` turns them; azimuth turns from the
front towards the left and elevation is positive upwards, in degrees.

- :class:`FirstOrder`: the patterns ``a + (1 - a) cos(theta)`` from omni (a = 1) to
  figure-8 (a = 0), one per band, so a loudspeaker can be omni in the bass and narrow
  in the treble.
- :class:`Balloon`: measured levels at any set of directions, read from a SOFA file
  (AES69, the ``FreeFieldDirectivityTF`` convention) or a CSV table, interpolated over
  the sphere; or the far-field pattern of a coherent array
  (:func:`array_directivity`).

``load_directivity`` turns a source's ``directivity`` entry of a scene file into one::

    directivity: cardioid                       # one pattern for every band
    directivity: {pattern: [omni, omni, subcardioid, cardioid, cardioid, supercardioid,
                            hypercardioid, hypercardioid]}
    directivity: {file: speaker.sofa}           # or speaker.csv
    directivity: {array: {count: 8, spacing_m: 0.15, steer_deg: -8}}
    directivity: {file: line_array.yaml}        # an array's fields in a YAML file

The CSV table has a header row ``azimuth, elevation`` followed by band centre
frequencies in Hz (octaves, thirds or any other layout), then one row per direction
with the level in dB in each band (any reference; only differences matter). Lines
starting with ``#`` are comments.

Measured data is averaged into octave bands. A simulation in third-octave bands takes
each octave band's directivity for its three third octaves (:meth:`Directivity.in_bands`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_spaces._extras import install_hint
from acoustic_spaces.air import speed_of_sound
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ
from acoustic_spaces.materials.bands import (
    Resolution,
    convert,
    crossover_frequencies,
    resolution_of,
)

#: The first-order patterns by name: the pressure is a + (1 - a) cos(theta).
PATTERNS: dict[str, float] = {
    "omni": 1.0,
    "subcardioid": 0.7,
    "cardioid": 0.5,
    "supercardioid": 0.37,
    "hypercardioid": 0.25,
    "figure8": 0.0,
}
#: Angular step of the grid measured data is interpolated on (degrees).
GRID_STEP_DEG = 5.0
#: Speed of sound (m/s) an array's phases take when no room gives one: air at 20 C.
SPEED_OF_SOUND = speed_of_sound(20.0)
#: Directions times pairs of elements an array's pattern is worked out for at a time.
_ARRAY_CHUNK = 1 << 18
#: Measured directions each grid point is interpolated from.
NEIGHBOURS = 4

_BANDS = len(OCTAVE_BANDS_HZ)
_BAND_COUNTS = (len(OCTAVE_BANDS_HZ), len(THIRD_OCTAVE_BANDS_HZ))
_SEPARATOR = re.compile(r"[,;\t]")
_SPACES = re.compile(r"\s+")


def _unit(directions) -> np.ndarray:
    v = np.asarray(directions, dtype=float).reshape(-1, 3)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)


def _from_angles(azimuth_deg, elevation_deg) -> np.ndarray:
    az, el = np.broadcast_arrays(
        np.radians(np.asarray(azimuth_deg, dtype=float)),
        np.radians(np.asarray(elevation_deg, dtype=float)),
    )
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def _sphere(n: int) -> np.ndarray:
    """``n`` near-uniform directions (a Fibonacci lattice)."""
    i = np.arange(n, dtype=float)
    z = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


class Directivity:
    """Energy weights per direction and octave band; see the module docstring."""

    #: Short description, for listings and reports.
    name: str = ""
    #: Octave bands (Hz) without data of their own, copied from the nearest band.
    extrapolated: tuple[int, ...] = ()

    def energy(self, directions) -> np.ndarray:
        """Energy weights (n, bands) for unit vectors (n, 3) in the source's frame;
        they average 1 over the sphere in each band."""
        raise NotImplementedError

    def index(self) -> np.ndarray:
        """Directivity index per band (dB): the on-axis level above the average."""
        on_axis = self.energy([[1.0, 0.0, 0.0]])[0]
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(on_axis)

    def native(self) -> tuple[np.ndarray, np.ndarray]:
        """What the compiled tracer takes: the first-order ``a`` per band, and a table of
        weights on the interpolation grid (elevations x azimuths x bands, empty for
        first-order patterns)."""
        raise NotImplementedError

    def in_bands(self, bands: Resolution) -> Directivity:
        """This directivity in octave or third-octave ``bands``: an octave band's
        weights for each of its three third octaves, or the mean of three third
        octaves' for their octave."""
        raise NotImplementedError

    @property
    def omni(self) -> bool:
        return False


@dataclass(frozen=True)
class FirstOrder(Directivity):
    """``a + (1 - a) cos(theta)`` patterns, one ``a`` per octave band."""

    a: tuple[float, ...]
    name: str = ""
    extrapolated: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.a) not in _BAND_COUNTS or not all(0.0 <= a <= 1.0 for a in self.a):
            raise ValueError(
                f"expected {_BANDS} values of a between 0 and 1 (or "
                f"{len(THIRD_OCTAVE_BANDS_HZ)} for third-octave bands), got {self.a}"
            )

    @classmethod
    def pattern(cls, name: str) -> FirstOrder:
        """The same named pattern in every band."""
        return cls((PATTERNS[name],) * _BANDS, name)

    def energy(self, directions) -> np.ndarray:
        a = np.asarray(self.a)
        p = a + (1.0 - a) * _unit(directions)[:, :1]
        return p**2 / (a**2 + (1.0 - a) ** 2 / 3.0)

    def native(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.a, dtype=float), np.zeros((0, 0, len(self.a)))

    def in_bands(self, bands: Resolution) -> FirstOrder:
        if resolution_of(len(self.a)) == bands:
            return self
        return FirstOrder(tuple(convert(self.a, bands)), self.name, self.extrapolated)

    @property
    def omni(self) -> bool:
        return all(a == 1.0 for a in self.a)


def _cell(shape: tuple[int, ...], directions) -> tuple[np.ndarray, ...]:
    """The cell of the grid of :func:`on_grid` (``shape``: elevations, azimuths) round
    each direction: its lower elevation's index, its two azimuths' indices, and how far
    across the cell the direction lies in elevation and in azimuth (0 to 1)."""
    v = _unit(directions)
    n_el, n_az = shape[:2]
    el = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    fi = (el + 90.0) / (180.0 / (n_el - 1))
    fj = az / (360.0 / n_az)
    i0 = np.clip(np.floor(fi).astype(int), 0, n_el - 2)
    j0 = np.floor(fj).astype(int) % n_az
    return i0, j0, (j0 + 1) % n_az, fi - i0, fj - np.floor(fj)


def on_grid(table: np.ndarray, directions) -> np.ndarray:
    """(n, ...) the values ``table`` (elevations, azimuths, ...) holds on a grid of
    elevations from -90 to 90 degrees and azimuths from 0 up to 360, each evenly spaced,
    read bilinearly at unit vectors (n, 3) in the source's frame."""
    i0, j0, j1, ti, tj = _cell(table.shape[:2], directions)
    shape = (-1,) + (1,) * (table.ndim - 2)
    ti, tj = ti.reshape(shape), tj.reshape(shape)
    t = table
    return (1 - ti) * ((1 - tj) * t[i0, j0] + tj * t[i0, j1]) + ti * (
        (1 - tj) * t[i0 + 1, j0] + tj * t[i0 + 1, j1]
    )


def sphere_mean(table: np.ndarray) -> np.ndarray:
    """(...) the mean over the sphere of what ``table`` holds on the grid of
    :func:`on_grid`, read bilinearly as it reads them."""
    return on_grid(table, _sphere(8192)).mean(axis=0)


def _grid(step: float) -> tuple[np.ndarray, np.ndarray]:
    elevation = np.linspace(-90.0, 90.0, int(round(180.0 / step)) + 1)
    azimuth = np.arange(0.0, 360.0, 360.0 / int(round(360.0 / step)))
    return elevation, azimuth


@dataclass(frozen=True)
class Balloon(Directivity):
    """Measured energy at some directions, interpolated over the sphere.

    The measurements are first spread onto a regular grid of elevations and azimuths
    (``GRID_STEP_DEG`` apart) by inverse-distance weighting of the ``NEIGHBOURS``
    nearest measured directions, then read off it bilinearly, which the compiled tracer
    does the same way. Each band is scaled to average 1 over the sphere.
    """

    #: Interpolation grid: energy weights (elevations, azimuths, bands).
    table: np.ndarray = field(repr=False)
    name: str = ""
    extrapolated: tuple[int, ...] = ()
    #: A coherent array's elements, for its sound added up element by element near it
    #: (``near_field: true``); None reads every weight off the pattern.
    near: NearField | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_measurements(
        cls,
        directions,
        energy,
        name: str = "",
        extrapolated: tuple[int, ...] = (),
        step: float = GRID_STEP_DEG,
    ) -> Balloon:
        """Interpolate energies (n, bands) measured at unit vectors (n, 3)."""
        measured = _unit(directions)
        energy = np.asarray(energy, dtype=float).reshape(len(measured), -1)
        if energy.shape[1] not in _BAND_COUNTS:
            raise ValueError(f"expected {_BANDS} bands, got {energy.shape[1]}")
        if len(measured) == 0 or not np.all(np.isfinite(energy)) or (energy < 0).any():
            raise ValueError("directivity data must be finite, non-negative energies")
        elevation, azimuth = _grid(step)
        points = _from_angles(azimuth[None, :], elevation[:, None]).reshape(-1, 3)
        k = min(NEIGHBOURS, len(measured))
        angle = np.arccos(np.clip(points @ measured.T, -1.0, 1.0))  # (grid, measured)
        nearest = np.argsort(angle, axis=1)[:, :k]
        near_angle = np.take_along_axis(angle, nearest, axis=1)
        weight = 1.0 / np.maximum(near_angle, 1e-9) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        values = np.einsum("gk,gkb->gb", weight, energy[nearest])
        table = values.reshape(len(elevation), len(azimuth), energy.shape[1])
        return cls.from_table(table, name, extrapolated)

    @classmethod
    def from_table(cls, table, name: str = "", extrapolated: tuple[int, ...] = ()) -> Balloon:
        """Energies on the grid of elevations (-90 to 90 degrees) and azimuths (0 up to
        360) that ``table`` (elevations, azimuths, bands) spans, each band scaled to
        average 1 over the sphere; silent bands become omnidirectional."""
        table = np.asarray(table, dtype=float)
        mean = sphere_mean(table)
        silent = mean <= 0.0
        table = np.where(silent, 1.0, table / np.where(silent, 1.0, mean))
        centres = OCTAVE_BANDS_HZ if table.shape[-1] == _BANDS else THIRD_OCTAVE_BANDS_HZ
        missing = tuple(f for f, s in zip(centres, silent, strict=True) if s)
        return cls(table, name, tuple(sorted(set(extrapolated) | set(missing))))

    def _interpolate(self, directions) -> np.ndarray:
        return on_grid(self.table, directions)

    def energy(self, directions) -> np.ndarray:
        return self._interpolate(directions)

    def near_energy(self, points) -> np.ndarray:
        """Energy weights (n, bands) at points (n, 3) in the source's frame: for an
        array with ``near``, its elements added up from where each is; otherwise, and
        far from the array alike, the pattern's weights towards the points."""
        if self.near is None:
            return self.energy(points)
        weights = self.near.weights(points)
        if self.table.shape[-1] == _BANDS:
            weights = weights.reshape(len(weights), _BANDS, 3).mean(axis=-1)
        return weights

    def native(self) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(self.table.shape[-1]), np.ascontiguousarray(self.table, dtype=float)

    def in_bands(self, bands: Resolution) -> Balloon:
        have = resolution_of(self.table.shape[-1])
        if have == bands:
            return self
        if bands == "third_octave":
            table = np.repeat(self.table, 3, axis=-1)
        else:
            table = self.table.reshape(*self.table.shape[:-1], -1, 3).mean(axis=-1)
        return Balloon(table, self.name, self.extrapolated, self.near)


@dataclass(frozen=True, eq=False)
class NearField:
    """A coherent array's elements (see :func:`array_directivity`), to add up their
    pressures at points near it, each from its own position with its own distance and
    direction, rather than read the far-field pattern off one point."""

    #: (elements, 3) positions in the source's frame (m).
    positions: np.ndarray
    #: (elements,) pressure gains, negative where the polarity is inverted.
    gains: np.ndarray
    #: (elements,) the feed's delays (s), steering included.
    delays: np.ndarray
    #: (elements, 3, 3) each element's forward, left and up axes as columns.
    turns: np.ndarray
    #: The elements' own pattern, in third-octave bands.
    element: Directivity
    speed_of_sound: float
    #: (25,) the third-octave bands' edges (Hz).
    edges: np.ndarray
    #: (24,) one over the pattern's mean energy over the sphere in each third-octave
    #: band, which scales it to the weights' average of 1; 0 where the band is silent.
    scale: np.ndarray

    def weights(self, points) -> np.ndarray:
        """Energy weights (n, 24) at points (n, 3) in the source's frame, per
        third-octave band, as the pattern's weights are: the elements' energy there,
        averaged over each band (of a flat spectrum) exactly, times the squared distance
        from the source's position. Far away they become the pattern's."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        reach = np.linalg.norm(points, axis=1)
        offsets = points[None] - self.positions[:, None]  # (elements, n, 3)
        spread = np.maximum(np.linalg.norm(offsets, axis=2), 1e-9)
        local = np.einsum("enk,ekj->enj", offsets / spread[..., None], self.turns)
        amplitude = _amplitudes(self.element, local) * (self.gains[:, None] / spread)[..., None]
        arrival = spread / self.speed_of_sound + self.delays[:, None]  # (elements, n)
        first_of, second_of = np.triu_indices(len(self.positions))
        twice = np.where(first_of == second_of, 1.0, 2.0)[:, None]
        # How much sooner one element of each pair is heard than the other (s).
        lag = arrival[first_of] - arrival[second_of]  # (pairs, n)
        still = np.abs(lag) < 1e-15
        energy = np.zeros((len(points), len(self.edges) - 1))
        below = np.sin(2.0 * math.pi * self.edges[0] * lag)
        for b in range(len(self.edges) - 1):
            above = np.sin(2.0 * math.pi * self.edges[b + 1] * lag)
            span = 2.0 * math.pi * (self.edges[b + 1] - self.edges[b]) * np.where(still, 1.0, lag)
            mean_cos = np.where(still, 1.0, (above - below) / span)
            pair = amplitude[first_of, :, b] * amplitude[second_of, :, b]
            energy[:, b] = np.sum(twice * pair * mean_cos, axis=0)
            below = above
        weights = energy * (reach**2)[:, None] * self.scale
        return np.where(self.scale > 0.0, weights, 1.0)


# -- reading ----------------------------------------------------------------------------


def _octaves(frequencies, energy) -> tuple[np.ndarray, tuple[int, ...]]:
    """Energies (..., frequencies) averaged into octave bands (..., bands); octaves
    without data copy the nearest octave that has some."""
    freqs = np.asarray(frequencies, dtype=float)
    energy = np.asarray(energy, dtype=float)
    out = np.full((*energy.shape[:-1], _BANDS), np.nan)
    have = np.zeros(_BANDS, dtype=bool)
    for k, fc in enumerate(OCTAVE_BANDS_HZ):
        inside = (freqs >= fc / math.sqrt(2.0)) & (freqs < fc * math.sqrt(2.0))
        if inside.any():
            out[..., k] = energy[..., inside].mean(axis=-1)
            have[k] = True
    known = np.flatnonzero(have)
    if len(known) == 0:
        raise ValueError("no directivity data between 44 Hz and 11 kHz")
    for k in np.flatnonzero(~have):
        out[..., k] = out[..., known[np.argmin(np.abs(known - k))]]
    return out, tuple(OCTAVE_BANDS_HZ[k] for k in np.flatnonzero(~have))


def read_directivity_csv(path: str | Path) -> Balloon:
    """A :class:`Balloon` from a CSV table of levels (see the module docstring)."""
    path = Path(path)
    header: list[str] | None = None
    rows: list[list[float]] = []
    with path.open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            separator = _SEPARATOR if _SEPARATOR.search(line) else _SPACES
            tokens = [t.strip() for t in separator.split(line) if t.strip()]
            if header is None:
                header = tokens
                continue
            try:
                rows.append([float(t) for t in tokens])
            except ValueError:
                raise ValueError(f"{path}:{lineno}: expected numbers, got {line!r}") from None
            if len(rows[-1]) != len(header):
                count = len(rows[-1])
                raise ValueError(f"{path}:{lineno}: {count} values for {len(header)} columns")
    if (
        header is None
        or len(header) < 3
        or [h.lower() for h in header[:2]]
        != [
            "azimuth",
            "elevation",
        ]
    ):
        raise ValueError(f"{path}: the header must be azimuth, elevation, then frequencies")
    try:
        frequencies = [float(h.lower().removesuffix("hz").strip()) for h in header[2:]]
    except ValueError:
        raise ValueError(f"{path}: band columns must be frequencies in Hz") from None
    if not rows:
        raise ValueError(f"{path}: no directions")
    data = np.asarray(rows)
    directions = _from_angles(data[:, 0], data[:, 1])
    energy, extrapolated = _octaves(frequencies, 10.0 ** (data[:, 2:] / 10.0))
    return Balloon.from_measurements(directions, energy, f"file {path.name}", extrapolated)


def write_directivity_csv(path: str | Path, balloon: Balloon, comment: str = "") -> Path:
    """Write ``balloon`` (in octave bands) as the CSV table :func:`read_directivity_csv`
    reads back: a row for each direction of its grid with its level in each band (dB
    over the sphere's mean), under ``comment``'s lines."""
    path = Path(path)
    table = balloon.in_bands("octave").table
    elevation, azimuth = _grid(180.0 / (table.shape[0] - 1))
    levels = 10.0 * np.log10(np.maximum(table, 1e-30))
    lines = [f"# {line}".rstrip() for line in comment.splitlines()]
    lines.append(", ".join(["azimuth", "elevation", *(f"{f:g}" for f in OCTAVE_BANDS_HZ)]))
    for i, el in enumerate(elevation):
        for j, az in enumerate(azimuth):
            values = ", ".join(f"{v:.2f}" for v in levels[i, j])
            lines.append(f"{az:g}, {el:g}, {values}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _frame(view: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation (3, 3) whose columns are forward, left and up for a view and up vector."""
    view = view / np.linalg.norm(view)
    up = up - (up @ view) * view
    up = up / np.linalg.norm(up)
    return np.column_stack([view, np.cross(up, view), up])


def _per_measurement(f, name: str, default, count: int) -> np.ndarray:
    """A SOFA position variable as Cartesian vectors (count, 3), one per measurement."""
    from acoustic_spaces.spatial.hrtf import _positions

    if name not in f:
        return np.broadcast_to(np.asarray(default, dtype=float), (count, 3))
    values = _positions(f[name])
    return np.broadcast_to(values, (count, 3)) if len(values) == 1 else values[:count]


def read_directivity_sofa(path: str | Path, measurement: int | None = None) -> Balloon:
    """A :class:`Balloon` from a SOFA file of source directivities (the
    ``FreeFieldDirectivityTF`` convention, or impulse responses in ``Data.IR``).

    Every receiver of every measurement gives one direction: where it sits relative to
    the source, in the source's frame (``SourceView`` and ``SourceUp``). ``measurement``
    picks one measurement (an instrument's notes, for example); by default their
    energies are averaged. Spectra are averaged in energy over each octave band.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(f"reading SOFA files needs h5py: {install_hint('sofa')}") from exc
    from acoustic_spaces.spatial.hrtf import _positions, _text

    path = Path(path)
    with h5py.File(path, "r") as f:
        if "Data.Real" in f:
            spectrum = np.asarray(f["Data.Real"][()], dtype=float)
            if "Data.Imag" in f:
                spectrum = spectrum + 1j * np.asarray(f["Data.Imag"][()], dtype=float)
            frequencies = np.asarray(f["N"][()], dtype=float).ravel()
        elif "Data.IR" in f:
            ir = np.asarray(f["Data.IR"][()], dtype=float)
            rate = float(np.asarray(f["Data.SamplingRate"][()]).ravel()[0])
            spectrum = np.fft.rfft(ir, axis=-1)
            frequencies = np.fft.rfftfreq(ir.shape[-1], 1.0 / rate)
        else:
            raise ValueError(f"{path}: no Data.Real or Data.IR; not a SOFA directivity file")
        if spectrum.ndim != 3 or spectrum.shape[-1] != len(frequencies):
            raise ValueError(f"{path}: expected data (measurements, receivers, frequencies)")
        m_count, r_count = spectrum.shape[:2]
        if measurement is not None and not 0 <= measurement < m_count:
            raise ValueError(f"{path}: measurement {measurement} of {m_count}")

        receivers = np.zeros((1, r_count, 3))
        if "ReceiverPosition" in f:
            dataset = f["ReceiverPosition"]
            raw = np.asarray(dataset[()], dtype=float)
            if raw.ndim == 3:  # (R, 3, M): receivers that move between measurements
                per_m = [_positions_array(raw[:, :, m], dataset) for m in range(raw.shape[2])]
                receivers = np.stack(per_m)
            else:
                receivers = _positions(dataset).reshape(1, -1, 3)
        listener = _per_measurement(f, "ListenerPosition", [0.0, 0.0, 0.0], m_count)
        listener_view = _per_measurement(f, "ListenerView", [1.0, 0.0, 0.0], m_count)
        listener_up = _per_measurement(f, "ListenerUp", [0.0, 0.0, 1.0], m_count)
        source = _per_measurement(f, "SourcePosition", [0.0, 0.0, 0.0], m_count)
        source_view = _per_measurement(f, "SourceView", [1.0, 0.0, 0.0], m_count)
        source_up = _per_measurement(f, "SourceUp", [0.0, 0.0, 1.0], m_count)
        title = _text(f.attrs.get("Title", "")) or path.stem

    chosen = range(m_count) if measurement is None else [measurement]
    directions, energy = [], []
    for m in chosen:
        local = receivers[min(m, len(receivers) - 1)]
        if len(local) != r_count:
            local = np.broadcast_to(local, (r_count, 3))
        world = listener[m] + local @ _frame(listener_view[m], listener_up[m]).T
        directions.append((world - source[m]) @ _frame(source_view[m], source_up[m]))
        energy.append(np.abs(spectrum[m]) ** 2)
    directions = np.concatenate(directions)
    if (np.linalg.norm(directions, axis=1) < 1e-9).any():
        raise ValueError(f"{path}: a receiver sits on the source; no direction to it")
    bands, extrapolated = _octaves(frequencies, np.concatenate(energy))
    # The same direction in several measurements (notes): average their energies.
    keys = np.round(_unit(directions), 6)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    summed = np.zeros((len(unique), _BANDS))
    np.add.at(summed, inverse, bands)
    counts = np.bincount(inverse, minlength=len(unique))[:, None]
    return Balloon.from_measurements(unique, summed / counts, f"SOFA {title}", extrapolated)


def _positions_array(values: np.ndarray, dataset) -> np.ndarray:
    """Like ``hrtf._positions`` for an array already read from ``dataset``."""
    from acoustic_spaces.spatial.hrtf import _text

    kind = _text(dataset.attrs.get("Type", "cartesian")).lower()
    if kind != "spherical":
        return values.reshape(-1, 3)
    units = _text(dataset.attrs.get("Units", "degree")).lower()
    az, el, r = values[:, 0], values[:, 1], values[:, 2]
    if "degree" not in units:
        az, el = np.degrees(az), np.degrees(el)
    return _from_angles(az, el) * r[:, None]


def read_directivity(
    path: str | Path,
    measurement: int | None = None,
    speed_of_sound: float = SPEED_OF_SOUND,
    near_field: bool = False,
) -> Balloon:
    """Measured directivity from a ``.sofa`` file or a CSV table, or an array's from a
    YAML file (``.yaml`` or ``.yml``; see :func:`read_array`), with its elements for
    ``near_field``."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"directivity file not found: {path}")
    if path.suffix.lower() == ".sofa":
        return read_directivity_sofa(path, measurement)
    if path.suffix.lower() in (".yaml", ".yml"):
        return array_directivity(read_array(path), speed_of_sound, near_field)
    return read_directivity_csv(path)


def read_array(path: str | Path):
    """A coherent array (:class:`~acoustic_spaces.scene.ArraySpec`) from a YAML file
    holding its fields; files named in it are found next to it."""
    import yaml
    from pydantic import ValidationError

    from acoustic_spaces.scene.model import ArraySpec, load_yaml

    path = Path(path)
    try:
        data = load_yaml(path.read_text(encoding="utf-8"))
        return ArraySpec.model_validate(data, context={"base_dir": path.parent})
    except (yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"{path.name}: not an array description: {exc}") from None


def directivity_files(spec, _seen: frozenset[Path] = frozenset()) -> list[Path]:
    """The files a source's directivity entry uses: its own file and, for a coherent
    array, its element's, whether the array is in the entry or in a YAML file (which
    names its element's file relative to itself), and so on down."""
    if spec is None or isinstance(spec, str):
        return []
    if getattr(spec, "file", None) is not None:
        return [Path(spec.file), *array_files(spec.file, _seen)]
    if getattr(spec, "array", None) is not None:
        return directivity_files(spec.array.element, _seen)
    return []


def array_files(path: str | Path, _seen: frozenset[Path] = frozenset()) -> list[Path]:
    """The files a coherent array's YAML file names for its element, and those they name
    in turn; none for other files, or a missing one."""
    path = Path(path)
    if path.suffix.lower() not in (".yaml", ".yml") or not path.is_file() or path in _seen:
        return []
    return directivity_files(read_array(path).element, _seen | {path})


def _amplitudes(element: Directivity, local: np.ndarray) -> np.ndarray:
    """Pressure amplitudes (elements, directions, bands) of an element pattern towards
    directions in each element's own frame (elements, directions, 3): first-order
    patterns with their sign, others by magnitude."""
    if isinstance(element, FirstOrder):
        a = np.asarray(element.a)
        return a + (1.0 - a) * local[..., :1]
    flat = element.energy(local.reshape(-1, 3))
    return np.sqrt(flat).reshape(*local.shape[:2], -1)


def array_directivity(
    spec, speed_of_sound: float = SPEED_OF_SOUND, near_field: bool = False
) -> Balloon:
    """The far-field pattern of a coherent array (an
    :class:`~acoustic_spaces.scene.ArraySpec`), per third-octave band; with
    ``near_field``, it also keeps the elements (:class:`NearField`), whose sound the
    early paths add up element by element.

    The elements' pressures add up with their phases: the path difference towards each
    direction, the feed's delays and the steering, the gains and polarities, and each
    element's own pattern along its aim (first-order patterns with their sign, measured
    ones by magnitude). Energies are exact averages over each band (of a flat spectrum),
    on a grid fine enough for the beam (1 to 5 degrees). The pattern holds in the far
    field, beyond about the array's length squared over the wavelength.
    """
    elements = spec.layout()
    c = float(speed_of_sound)
    positions = np.array([e.position for e in elements], dtype=float)
    gains = np.array([10.0 ** (e.gain_db / 20.0) * (-1 if e.invert else 1) for e in elements])
    delays = np.array([e.delay_ms * 1e-3 for e in elements])
    if spec.steer_deg:
        aim = _from_angles(0.0, spec.steer_deg).reshape(3)
        lead = positions @ aim
        delays = delays + (lead - lead.min()) / c
    turns = np.array([e.orientation.rotation() for e in elements], dtype=float)
    element = load_directivity(spec.element).in_bands("third_octave")
    length = float(np.max(np.linalg.norm(positions[:, None] - positions[None], axis=2)))
    # A grid fine enough for the beam at the top band (a quarter of its width), 1 to 5 deg.
    top = THIRD_OCTAVE_BANDS_HZ[-1] * 2 ** (1 / 6)
    width = math.degrees(c / top / max(length, 1e-9))
    step = next((s for s in (5.0, 2.5, 2.0, 1.0) if s <= width / 4), 1.0)
    elevation, azimuth = _grid(step)
    directions = _from_angles(azimuth[None, :], elevation[:, None]).reshape(-1, 3)
    # Band edges, as the filter banks split the spectrum.
    inner = crossover_frequencies("third_octave")
    edges = np.concatenate([[inner[0] / 2 ** (1 / 3)], inner, [inner[-1] * 2 ** (1 / 3)]])
    first_of, second_of = np.triu_indices(len(elements))
    twice = np.where(first_of == second_of, 1.0, 2.0)  # each pair of elements counts twice
    offsets = positions[first_of] - positions[second_of]
    shifts = delays[first_of] - delays[second_of]
    alike = bool(np.allclose(turns, turns[0]))
    if alike:
        # Elements aimed alike share a pattern: pairs the same distance and delay apart
        # (a straight column's, say) add up into one term.
        key = np.round(np.column_stack([offsets, shifts * c]) * 1e9)
        _, first_seen, group = np.unique(key, axis=0, return_index=True, return_inverse=True)
        pair_gains = np.bincount(group.ravel(), weights=gains[first_of] * gains[second_of] * twice)
        offsets, shifts = offsets[first_seen], shifts[first_seen]
    energy = np.zeros((len(directions), len(edges) - 1))
    chunk_size = max(64, _ARRAY_CHUNK // len(offsets))
    for first in range(0, len(directions), chunk_size):
        chunk = directions[first : first + chunk_size]
        # Each element's amplitude towards these directions, along its own aim.
        amplitude = _amplitudes(
            element, np.einsum("dk,nkj->ndj", chunk, turns[:1] if alike else turns)
        )
        # How much sooner one element of each pair is heard than the other (s).
        lag = chunk @ offsets.T / c - shifts[None, :]  # (directions, pairs)
        still = np.abs(lag) < 1e-15
        # The energy is the sum over pairs of w_i w_k cos(2 pi f lag); its mean over a
        # band, exactly: (sin(2 pi f2 lag) - sin(2 pi f1 lag)) / (2 pi (f2 - f1) lag).
        below = np.sin(2.0 * math.pi * edges[0] * lag)
        for b in range(len(edges) - 1):
            above = np.sin(2.0 * math.pi * edges[b + 1] * lag)
            span = 2.0 * math.pi * (edges[b + 1] - edges[b]) * np.where(still, 1.0, lag)
            mean_cos = np.where(still, 1.0, (above - below) / span)
            if alike:
                weights = amplitude[0, :, b, None] ** 2 * pair_gains[None, :]
            else:
                pair = amplitude[first_of, :, b] * amplitude[second_of, :, b]
                weights = (pair * (gains[first_of] * gains[second_of] * twice)[:, None]).T
            energy[first : first + len(chunk), b] = np.sum(weights * mean_cos, axis=1)
            below = above
    table = energy.reshape(len(elevation), len(azimuth), -1)
    balloon = Balloon.from_table(table, f"array of {len(elements)} elements")
    if not near_field:
        return balloon
    # The mean that from_table scales the pattern by.
    mean = Balloon(table)._interpolate(_sphere(8192)).mean(axis=0)
    scale = np.where(mean > 0.0, 1.0 / np.where(mean > 0.0, mean, 1.0), 0.0)
    near = NearField(positions, gains, delays, turns, element, c, edges, scale)
    return Balloon(balloon.table, balloon.name, balloon.extrapolated, near)


def lambert() -> Balloon:
    """A vibrating surface's radiation: energy in proportion to the cosine of the angle
    to the forward axis, nothing behind (Lambert's law), in every band."""
    elevation, azimuth = _grid(GRID_STEP_DEG)
    points = _from_angles(azimuth[None, :], elevation[:, None]).reshape(-1, 3)
    energy = np.repeat(np.maximum(points[:, :1], 0.0), _BANDS, axis=1)
    return Balloon.from_measurements(points, energy, "lambert")


#: The frequency (Hz) at which :func:`far_field_distance` is given in warnings.
FAR_FIELD_HZ = 1000.0


def array_length(spec) -> float | None:
    """The largest distance between two elements (m) of the coherent array a source's
    directivity entry describes, given in the scene or in a YAML file; None for other
    directivities."""
    if getattr(spec, "array", None) is not None:
        array = spec.array
    elif getattr(spec, "file", None) is not None and Path(spec.file).suffix.lower() in (
        ".yaml",
        ".yml",
    ):
        array = read_array(spec.file)
    else:
        return None
    positions = np.array([e.position for e in array.layout()], dtype=float)
    return float(np.max(np.linalg.norm(positions[:, None] - positions[None], axis=2)))


def far_field_distance(
    length_m: float, frequency_hz: float = FAR_FIELD_HZ, speed_of_sound: float = SPEED_OF_SOUND
) -> float:
    """From how far (m) an array ``length_m`` long acts as one point with its far-field
    pattern at ``frequency_hz``: its length squared over the wavelength. Nearer, its
    sound spreads as a line's and its beam has not formed yet."""
    return float(length_m) ** 2 * float(frequency_hz) / float(speed_of_sound)


def load_directivity(spec, speed_of_sound: float = SPEED_OF_SOUND) -> Directivity:
    """The directivity a scene's source entry describes: a pattern name (or
    ``lambert``), or an object with ``pattern`` (a name or ``a`` value per band),
    ``file`` or ``array``; an array's phases take ``speed_of_sound`` (m/s)."""
    if spec == "lambert":
        return lambert()
    if isinstance(spec, str):
        return FirstOrder.pattern(spec)
    near_field = bool(getattr(spec, "near_field", None))
    if getattr(spec, "array", None) is not None:
        return array_directivity(spec.array, speed_of_sound, near_field)
    if getattr(spec, "file", None) is not None:
        return read_directivity(spec.file, spec.measurement, speed_of_sound, near_field)
    values = [PATTERNS[v] if isinstance(v, str) else float(v) for v in spec.pattern]
    names = [v if isinstance(v, str) else f"a={float(v):g}" for v in spec.pattern]
    return FirstOrder(tuple(values), "per band: " + ", ".join(names))


__all__ = [
    "FAR_FIELD_HZ",
    "GRID_STEP_DEG",
    "PATTERNS",
    "SPEED_OF_SOUND",
    "Balloon",
    "Directivity",
    "FirstOrder",
    "NearField",
    "array_directivity",
    "array_files",
    "array_length",
    "directivity_files",
    "far_field_distance",
    "lambert",
    "load_directivity",
    "on_grid",
    "read_array",
    "read_directivity",
    "read_directivity_csv",
    "read_directivity_sofa",
    "sphere_mean",
    "write_directivity_csv",
]
