"""Source directivity: how a source shares its sound power between directions, per
octave band.

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
  the sphere.

``load_directivity`` turns a source's ``directivity`` entry of a scene file into one::

    directivity: cardioid                       # one pattern for every band
    directivity: {pattern: [omni, omni, subcardioid, cardioid, cardioid, supercardioid,
                            hypercardioid, hypercardioid]}
    directivity: {file: speaker.sofa}           # or speaker.csv

The CSV table has a header row ``azimuth, elevation`` followed by band centre
frequencies in Hz (octaves, thirds or any other layout), then one row per direction
with the level in dB in each band (any reference; only differences matter). Lines
starting with ``#`` are comments.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_spaces._extras import install_hint
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

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
#: Measured directions each grid point is interpolated from.
NEIGHBOURS = 4

_BANDS = len(OCTAVE_BANDS_HZ)
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
        if len(self.a) != _BANDS or not all(0.0 <= a <= 1.0 for a in self.a):
            raise ValueError(f"expected {_BANDS} values of a between 0 and 1, got {self.a}")

    @classmethod
    def pattern(cls, name: str) -> FirstOrder:
        """The same named pattern in every band."""
        return cls((PATTERNS[name],) * _BANDS, name)

    def energy(self, directions) -> np.ndarray:
        a = np.asarray(self.a)
        p = a + (1.0 - a) * _unit(directions)[:, :1]
        return p**2 / (a**2 + (1.0 - a) ** 2 / 3.0)

    def native(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.a, dtype=float), np.zeros((0, 0, _BANDS))

    @property
    def omni(self) -> bool:
        return all(a == 1.0 for a in self.a)


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
        if energy.shape[1] != _BANDS:
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
        table = values.reshape(len(elevation), len(azimuth), _BANDS)
        balloon = cls(table, name, tuple(extrapolated))
        mean = balloon._interpolate(_sphere(8192)).mean(axis=0)
        silent = mean <= 0.0
        table = np.where(silent, 1.0, table / np.where(silent, 1.0, mean))
        missing = tuple(f for f, s in zip(OCTAVE_BANDS_HZ, silent, strict=True) if s)
        return cls(table, name, tuple(sorted(set(extrapolated) | set(missing))))

    def _interpolate(self, directions) -> np.ndarray:
        v = _unit(directions)
        n_el, n_az = self.table.shape[:2]
        el = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
        az = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        fi = (el + 90.0) / (180.0 / (n_el - 1))
        fj = az / (360.0 / n_az)
        i0 = np.clip(np.floor(fi).astype(int), 0, n_el - 2)
        j0 = np.floor(fj).astype(int) % n_az
        ti, tj = (fi - i0)[:, None], (fj - np.floor(fj))[:, None]
        j1 = (j0 + 1) % n_az
        t = self.table
        return (1 - ti) * ((1 - tj) * t[i0, j0] + tj * t[i0, j1]) + ti * (
            (1 - tj) * t[i0 + 1, j0] + tj * t[i0 + 1, j1]
        )

    def energy(self, directions) -> np.ndarray:
        return self._interpolate(directions)

    def native(self) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(_BANDS), np.ascontiguousarray(self.table, dtype=float)


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


def read_directivity(path: str | Path, measurement: int | None = None) -> Balloon:
    """Measured directivity from a ``.sofa`` file or a CSV table (any other suffix)."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"directivity file not found: {path}")
    if path.suffix.lower() == ".sofa":
        return read_directivity_sofa(path, measurement)
    return read_directivity_csv(path)


def load_directivity(spec) -> Directivity:
    """The directivity a scene's source entry describes: a pattern name, or an object
    with ``pattern`` (a name or ``a`` value per band) or ``file``."""
    if isinstance(spec, str):
        return FirstOrder.pattern(spec)
    if getattr(spec, "file", None) is not None:
        return read_directivity(spec.file, spec.measurement)
    values = [PATTERNS[v] if isinstance(v, str) else float(v) for v in spec.pattern]
    names = [v if isinstance(v, str) else f"a={float(v):g}" for v in spec.pattern]
    return FirstOrder(tuple(values), "per band: " + ", ".join(names))


__all__ = [
    "GRID_STEP_DEG",
    "PATTERNS",
    "Balloon",
    "Directivity",
    "FirstOrder",
    "load_directivity",
    "read_directivity",
    "read_directivity_csv",
    "read_directivity_sofa",
]
